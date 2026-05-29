"""
USB I/O backend for the TI-84 Evo remote.

A single worker thread owns the serial link: the GUI enqueues scancodes and
transfer commands, and `UsbWorker` pumps them to the calculator and emits
decoded frames / status back via Qt signals. Also holds `capture_screen_rgb565`
(the raw framebuffer grab) and the small command objects the queue carries.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal

# evo_usb ships as a submodule (./evo_usb_py/evo_usb.py). Put it on the path
# before importing, so this works regardless of the current directory.
_EVO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "evo_usb_py"
)
if _EVO_PATH not in sys.path:
    sys.path.insert(0, _EVO_PATH)
import evo_usb  # noqa: E402  (path tweak must run first)

from keypad import SCREEN_BYTES  # noqa: E402


def capture_screen_rgb565(retries: int = 2) -> bytes:
    """Pull one framebuffer from the calc (sys/screen) as raw RGB565 bytes.

    The screen URL returns CBOR. The pixel blob is a definite-length byte
    string (major type 2) introduced by 0x5A + a 4-byte length, exactly as
    evo_usb.take_screenshot decodes it.

    The serial link occasionally returns a desynced/short first packet after
    an idle gap, so retry a couple of times before giving up.
    """
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            raw = evo_usb._get_request(evo_usb._screen_url(0))
            marker = raw.index(0x5A)
            return raw[marker + 5:marker + 5 + SCREEN_BYTES]
        except Exception as e:  # framing desync, short read, etc.
            last_err = e
            time.sleep(0.15)
    raise last_err


@dataclass(frozen=True)
class _SendCmd:
    path: str
    varname: str | None     # set -> send_file (Python); None -> send_var_file


@dataclass(frozen=True)
class _ListCmd:
    pass                    # request the variable directory


@dataclass(frozen=True)
class _RecvCmd:
    name: str
    type_id: int
    output: str


@dataclass(frozen=True)
class _DeleteCmd:
    name: str
    type_id: int


class UsbWorker(QObject):
    """Single-thread USB I/O pump.

    The GUI thread only ever enqueues scancodes (fast, non-blocking) and
    receives decoded frames via the `frame` signal. Everything that touches
    the serial port runs here.
    """

    frame = pyqtSignal(bytes)    # RGB565 framebuffer
    status = pyqtSignal(str)     # human-readable status / errors
    busy = pyqtSignal(bool)      # True when a transfer starts, False at end
    file_list = pyqtSignal(object)   # list[dict] from list_files()
    finished = pyqtSignal(bool, str)  # (success, human-readable message)
    connection = pyqtSignal(bool, str)  # (online, reason); edge-triggered

    # Queue sentinels (never valid scancodes) for non-scancode actions.
    _REFRESH = -1
    _BREAK = -2

    def __init__(self, poll_ms: int = 175):
        super().__init__()
        self.poll_ms = poll_ms
        self.poll_enabled = True
        # None until the first grab, so that first outcome always emits a
        # `connection` edge (drives the startup overlay either way).
        self._online: bool | None = None
        self._offline_probe_ms = 750    # reconnect probe cadence while offline
        self._q: "queue.Queue[int | _SendCmd | _ListCmd | _RecvCmd " \
            "| _DeleteCmd]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="usb-io"
        )

    # --- called from the GUI thread ---
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def press(self, code: int):
        self._q.put(code)

    def request_refresh(self):
        self._q.put(self._REFRESH)

    def press_break(self):
        """ON / quit-to-home, via hh01/sys/break, not a scancode."""
        self._q.put(self._BREAK)

    def request_file_list(self):
        self._q.put(_ListCmd())

    def send_file_op(self, path: str, varname: str | None = None):
        self._q.put(_SendCmd(path, varname))

    def recv_file_op(self, name: str, type_id: int, output: str):
        self._q.put(_RecvCmd(name, type_id, output))

    def delete_op(self, name: str, type_id: int):
        self._q.put(_DeleteCmd(name, type_id))

    # --- worker-thread internals ---
    def _set_online(self, online: bool, reason: str = ""):
        """Flip connection state, emitting `connection` only on a transition.

        The initial None state means the first call always emits, so the GUI
        learns the startup state before any frame has arrived.
        """
        if online == self._online:
            return
        self._online = online
        self.connection.emit(online, reason)

    @staticmethod
    def _reason_for(e) -> str:
        # Overlay text only; the detailed error goes to the status signal
        # (and so to the debug window). SystemExit == connect() found no
        # device; everything else is a link that was up and dropped.
        if isinstance(e, SystemExit):
            return "Searching for calculator…"
        return "Disconnected"

    def _dispatch(self, code):
        if isinstance(code, int):
            if code == self._REFRESH:
                return                   # fall through to the post-IO grab
            if code == self._BREAK:
                self._break()
                return
            self._send(code)
            return
        # Non-scancode command objects (file transfers, directory listing).
        if isinstance(code, _ListCmd):
            self._do_list()
        elif isinstance(code, _SendCmd):
            self._do_send(code)
        elif isinstance(code, _RecvCmd):
            self._do_recv(code)
        elif isinstance(code, _DeleteCmd):
            self._do_delete(code)

    def _send(self, code: int) -> bool:
        try:
            evo_usb.send_scancode(code)
            return True
        except (Exception, SystemExit) as e:
            self.status.emit(f"send 0x{code:02X} failed: {e}")
            self._set_online(False, self._reason_for(e))
            time.sleep(0.25)
            return False

    def _break(self) -> bool:
        try:
            evo_usb.send_break()
            return True
        except (Exception, SystemExit) as e:
            self.status.emit(f"break failed: {e}")
            self._set_online(False, self._reason_for(e))
            time.sleep(0.25)
            return False

    def _grab(self):
        try:
            t0 = time.time()
            rgb = capture_screen_rgb565()
            dt = (time.time() - t0) * 1000
            if len(rgb) == SCREEN_BYTES:
                self.frame.emit(rgb)
                self.status.emit(f"connected · screen {dt:.0f} ms")
                self._set_online(True)
            else:
                # Partial read: link is up but desynced (see capture_screen's
                # retry note), not a disconnect, so stay online.
                self.status.emit(f"short frame: {len(rgb)}B")
        except (Exception, SystemExit) as e:
            self.status.emit(f"screen read failed: {e}")
            self._set_online(False, self._reason_for(e))
            time.sleep(0.25)

    @contextmanager
    def _transfer_guard(self):
        """Run a blocking transfer with the background poll suppressed.

        Snapshots the user's poll_enabled choice, forces polling off for the
        duration, then restores it in finally (so an exception can't leave
        polling stuck off or the GUI's File menu stuck disabled). Grabs one
        fresh frame afterwards so the screen reflects the calc's new state.
        """
        prev_poll = self.poll_enabled
        self.poll_enabled = False
        self.busy.emit(True)
        try:
            yield
        finally:
            self.poll_enabled = prev_poll   # honor --no-poll / Live unchecked
            self.busy.emit(False)
            self._grab()

    def _do_list(self):
        with self._transfer_guard():
            try:
                entries = evo_usb.list_files()
                self.status.emit(f"directory: {len(entries)} entries")
                self.file_list.emit(entries)
            except (Exception, SystemExit) as e:
                self.status.emit(f"list failed: {e}")
                self._set_online(False, self._reason_for(e))
                self.finished.emit(
                    False, f"Could not read the calculator: {e}")

    def _do_send(self, cmd: _SendCmd):
        with self._transfer_guard():
            try:
                if cmd.varname is not None:
                    evo_usb.send_file(cmd.path, cmd.varname)
                    msg = f"Sent {os.path.basename(cmd.path)} as " \
                          f"'{cmd.varname}'."
                else:
                    evo_usb.send_var_file(cmd.path, "auto")
                    msg = f"Sent {os.path.basename(cmd.path)}."
                self.status.emit(msg)
                self.finished.emit(True, msg)
            except (Exception, SystemExit) as e:
                self.status.emit(f"send failed: {e}")
                self._set_online(False, self._reason_for(e))
                self.finished.emit(False, f"Send failed: {e}")

    def _do_recv(self, cmd: _RecvCmd):
        with self._transfer_guard():
            try:
                evo_usb.get_variable(cmd.name, cmd.type_id, cmd.output)
                msg = f"Saved {cmd.name} to {cmd.output}."
                self.status.emit(msg)
                self.finished.emit(True, msg)
            except (Exception, SystemExit) as e:
                self.status.emit(f"receive failed: {e}")
                self._set_online(False, self._reason_for(e))
                self.finished.emit(False, f"Receive failed: {e}")

    def _do_delete(self, cmd: _DeleteCmd):
        with self._transfer_guard():
            try:
                evo_usb.delete_variable(cmd.name, cmd.type_id)
                msg = f"Deleted {cmd.name}."
                self.status.emit(msg)
                self.finished.emit(True, msg)
            except (Exception, SystemExit) as e:
                self.status.emit(f"delete failed: {e}")
                self._set_online(False, self._reason_for(e))
                self.finished.emit(False, f"Delete failed: {e}")

    def _run(self):
        self._grab()  # initial frame
        last = time.monotonic()
        while not self._stop.is_set():
            did_io = False
            try:
                self._dispatch(self._q.get(timeout=0.05))
                did_io = True
                # Drain anything queued while we were busy, act on it all,
                # then take a single screenshot for the batch.
                while True:
                    try:
                        self._dispatch(self._q.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            now = time.monotonic()
            if did_io:
                time.sleep(0.02)         # let the calc render the keypress
                self._grab()
                last = time.monotonic()
            elif self._online is False:
                # Keep probing for reconnect even with --no-poll / Live off,
                # otherwise a non-polling user would never recover.
                if (now - last) * 1000 >= self._offline_probe_ms:
                    self._grab()
                    last = time.monotonic()   # re-read; _grab can block
            elif self.poll_enabled and (now - last) * 1000 >= self.poll_ms:
                self._grab()
                last = now
