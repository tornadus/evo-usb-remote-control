"""
TI-84 Evo remote control

Usage:
    python calc_remote.py [--poll-ms N] [--no-poll] [--scale N] [--debug]
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QLayout, QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

# evo_usb ships as a submodule (./evo_usb_py/evo_usb.py). Put it on the path
# before importing, so this works regardless of the current directory.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evo_usb_py")
)
import evo_usb  # noqa: E402  (path tweak must run first)

from keypad import (  # noqa: E402
    SCREEN_BYTES, SCREEN_H, SCREEN_W, SCANCODES,
    build_keypad, key_to_label,
)


def rgb565_to_rgb888(buf: bytes) -> bytes:
    a = np.frombuffer(buf, dtype="<u2")
    r = ((a >> 11) & 0x1F).astype(np.uint8)
    g = ((a >> 5) & 0x3F).astype(np.uint8)
    b = (a & 0x1F).astype(np.uint8)
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)
    return np.stack([r, g, b], axis=-1).tobytes()


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


def default_varname(path: str) -> str:
    """Suggest an AppVar name from a filename: drop the extension, keep only
    ASCII letters (send_file's name encoder handles A-Z only), uppercase, and
    cap at 8 chars. Falls back to FILE when nothing usable is left.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    letters = "".join(c for c in stem if c.isascii() and c.isalpha())
    return letters[:8].upper() or "FILE"


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

    # Queue sentinels (never valid scancodes) for non-scancode actions.
    _REFRESH = -1
    _BREAK = -2

    def __init__(self, poll_ms: int = 175):
        super().__init__()
        self.poll_ms = poll_ms
        self.poll_enabled = True
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
            time.sleep(0.25)
            return False

    def _break(self) -> bool:
        try:
            evo_usb.send_break()
            return True
        except (Exception, SystemExit) as e:
            self.status.emit(f"break failed: {e}")
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
            else:
                self.status.emit(f"short frame: {len(rgb)}B")
        except (Exception, SystemExit) as e:
            self.status.emit(f"screen read failed: {e}")
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
            elif self.poll_enabled and (now - last) * 1000 >= self.poll_ms:
                self._grab()
                last = now


class DebugWindow(QWidget):
    """Poll/refresh controls and a status readout, in a separate window shown
    when --debug is set. Fixed size with a word-wrapped status area, sized
    generously up front so even a long error message can't resize it.
    """

    def __init__(self, worker: UsbWorker):
        super().__init__()
        self.worker = worker
        self.setWindowTitle("TI-84 Evo - debug")

        outer = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.live_chk = QCheckBox("Live")
        self.live_chk.setChecked(worker.poll_enabled)
        self.live_chk.setToolTip(
            "Continuously poll the screen for blink/animation.")
        self.live_chk.toggled.connect(self._on_live_toggled)

        poll_lbl = QLabel("Poll (ms):")
        poll_lbl.setStyleSheet("color: #555;")
        self.poll_spin = QSpinBox()
        self.poll_spin.setRange(80, 3000)
        self.poll_spin.setSingleStep(50)
        self.poll_spin.setValue(worker.poll_ms)
        self.poll_spin.valueChanged.connect(self._on_poll_changed)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(worker.request_refresh)

        controls.addWidget(self.live_chk)
        controls.addWidget(poll_lbl)
        controls.addWidget(self.poll_spin)
        controls.addWidget(self.refresh_btn)
        controls.addStretch()
        outer.addLayout(controls)

        self.status_lbl = QLabel("connecting…")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #666;")
        self.status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self.status_lbl, stretch=1)

        self.setFixedSize(420, 260)

    def set_status(self, text: str):
        self.status_lbl.setText(text)

    def _on_live_toggled(self, on: bool):
        self.worker.poll_enabled = on

    def _on_poll_changed(self, val: int):
        self.worker.poll_ms = int(val)


class RemoteWindow(QMainWindow):
    def __init__(self, worker: UsbWorker, scale: int = 1, debug: bool = False):
        super().__init__()
        self.worker = worker
        self.scale = scale
        self.debug = debug
        self.debug_window = DebugWindow(worker) if debug else None
        # Whether a pending file_list is for a "recv" or "delete" interaction,
        # so a stray listing never pops a dialog. None when nothing is awaited.
        self._awaiting_list: str | None = None
        self.setWindowTitle("TI-84 Evo - USB remote")

        central = QWidget()
        # Skin only the body, not the whole window. Painting the background on
        # the QMainWindow bleeds under the transparent menu bar, leaving its
        # (system-themed) text unreadable on the light body; scoping it to the
        # central widget keeps the menu bar on the OS palette.
        central.setObjectName("body")
        central.setStyleSheet("#body { background: #f4f7fa; }")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        # Lock the window to its content size (no user resizing). This tracks
        # --scale automatically, since the screen label is sized from it.
        outer.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        # Model name, centered above the display, in a sans-serif face.
        model_lbl = QLabel("TI-84 Evo")
        model_font = QFont()
        model_font.setStyleHint(QFont.StyleHint.SansSerif)
        model_font.setFamily(model_font.defaultFamily())  # concrete family
        model_font.setBold(True)
        model_font.setPixelSize(14)
        model_lbl.setFont(model_font)
        model_lbl.setStyleSheet("color: #2b333a;")
        model_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(model_lbl, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Screen. The CSS border insets the content rect, so size the label
        # to the pixmap plus the border on each side, otherwise the outer
        # 2px of the framebuffer (e.g. the battery icon) gets clipped.
        border = 2
        self.screen_label = QLabel()
        self.screen_label.setFixedSize(SCREEN_W * scale + 2 * border,
                                       SCREEN_H * scale + 2 * border)
        self.screen_label.setStyleSheet(
            f"border: {border}px solid #aab; background: #000;"
        )
        self.screen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.screen_label,
                        alignment=Qt.AlignmentFlag.AlignCenter)

        # Virtual keypad.
        outer.addWidget(build_keypad(self.press_key),
                        alignment=Qt.AlignmentFlag.AlignHCenter)

        self.setCentralWidget(central)

        # File menu for transfers. The menu bar lives in its own reserved
        # region, so it doesn't disturb the central widget's fixed-size layout;
        # build it before setFixedSize(sizeHint()) so its height is counted.
        m = self.menuBar().addMenu("File")
        self.send_action = m.addAction("Send File…", self._on_send)
        self.recv_action = m.addAction("Receive File…", self._on_receive)
        self.delete_action = m.addAction("Delete Variable…", self._on_delete)

        # SetFixedSize above only pins the window's minimum; lock the maximum
        # too so it can't be resized at all. sizeHint already accounts for
        # --scale (the screen label is sized from it).
        self.setFixedSize(self.sizeHint())
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        worker.frame.connect(self.on_frame)
        worker.status.connect(self._on_status)
        worker.busy.connect(self._on_busy)
        worker.file_list.connect(self._on_file_list)
        worker.finished.connect(self._on_finished)

    def _on_status(self, text: str):
        # Route to the debug window if open. Otherwise only surface real
        # failures (to stderr) and drop the routine per-frame chatter.
        if self.debug_window is not None:
            self.debug_window.set_status(text)
        elif "fail" in text.lower() or "error" in text.lower():
            print(text, file=sys.stderr)

    # --- file transfer (File menu) ---
    def _on_send(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Send File to Calculator", "",
            "Calculator files (*.8x* *.8c* *.py);;All files (*)")
        if not path:
            return
        if path.lower().endswith(".py"):
            varname, ok = QInputDialog.getText(
                self, "Send Python Script",
                "AppVar name (≤8 letters, A–Z):", text=default_varname(path))
            if not ok:
                return
            varname = varname.strip().upper()
            valid = (1 <= len(varname) <= 8
                     and varname.isascii() and varname.isalpha())
            if not valid:
                QMessageBox.warning(self, "Send File",
                                    "Name must be 1–8 letters (A–Z).")
                return
            self.worker.send_file_op(path, varname)
        else:
            self.worker.send_file_op(path)

    def _on_receive(self):
        self._awaiting_list = "recv"
        self.worker.request_file_list()

    def _on_delete(self):
        self._awaiting_list = "delete"
        self.worker.request_file_list()

    def _on_file_list(self, entries):
        mode, self._awaiting_list = self._awaiting_list, None
        if mode is None:
            return
        if not entries:
            QMessageBox.information(
                self, "File", "No variables on calculator.")
            return

        labels = [
            f"{e['name']}  (type {e['type']}, {e['size']}B, "
            f"{'RAM' if not e['mem'] else 'Arc'})"
            for e in entries
        ]
        title = "Receive File" if mode == "recv" else "Delete Variable"
        prompt = "Variable to download:" if mode == "recv" \
            else "Variable to delete:"
        label, ok = QInputDialog.getItem(
            self, title, prompt, labels, 0, editable=False)
        if not ok:
            return
        entry = entries[labels.index(label)]

        if mode == "recv":
            ext = evo_usb.EVO_TYPE_EXTENSIONS.get(entry["type"], "bin")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Variable As", f"{entry['name']}.{ext}")
            if not path:
                return
            self.worker.recv_file_op(entry["name"], entry["type"], path)
        else:
            if QMessageBox.question(
                    self, "Delete Variable",
                    f"Delete {entry['name']} from the calculator?"
            ) == QMessageBox.StandardButton.Yes:
                self.worker.delete_op(entry["name"], entry["type"])

    def _on_busy(self, busy: bool):
        self.send_action.setEnabled(not busy)
        self.recv_action.setEnabled(not busy)
        self.delete_action.setEnabled(not busy)

    def _on_finished(self, success: bool, msg: str):
        if success:
            QMessageBox.information(self, "Transfer", msg)
        else:
            QMessageBox.warning(self, "Transfer", msg)

    # --- input ---
    def press_key(self, label: str):
        if label == "ON":
            # ON isn't a normal key-matrix scancode on hardware. The calc's
            # break command is its real "quit to home" behavior over USB.
            self.worker.press_break()
            return
        code = SCANCODES.get(label)
        if not code:
            self._on_status(f"no scancode for {label}")
            return
        self.worker.press(code)

    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        label = key_to_label(event)
        if label is None:
            return super().keyPressEvent(event)
        self.press_key(label)

    # --- output ---
    def on_frame(self, rgb565: bytes):
        rgb888 = rgb565_to_rgb888(rgb565)
        img = QImage(
            rgb888, SCREEN_W, SCREEN_H, SCREEN_W * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        pix = QPixmap.fromImage(img).scaled(
            SCREEN_W * self.scale, SCREEN_H * self.scale,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.screen_label.setPixmap(pix)

    def closeEvent(self, event):
        self.worker.stop()
        if self.debug_window is not None:
            self.debug_window.close()
        super().closeEvent(event)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--poll-ms", type=int, default=175,
                    help="background screen poll period in ms (default 175)")
    ap.add_argument("--no-poll", action="store_true",
                    help="only refresh after input / manual Refresh")
    ap.add_argument("--scale", type=int, default=1,
                    help="screen magnification (default 1)")
    ap.add_argument("--debug", action="store_true",
                    help="open a separate window with poll/refresh controls "
                         "and a live status readout")
    args = ap.parse_args()

    worker = UsbWorker(poll_ms=args.poll_ms)
    worker.poll_enabled = not args.no_poll

    app = QApplication(sys.argv)
    win = RemoteWindow(worker, scale=args.scale, debug=args.debug)
    win.show()
    if win.debug_window is not None:
        win.debug_window.show()

    worker.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
