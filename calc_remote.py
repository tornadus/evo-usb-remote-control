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

import numpy as np
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QLayout, QMainWindow,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
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


class UsbWorker(QObject):
    """Single-thread USB I/O pump.

    The GUI thread only ever enqueues scancodes (fast, non-blocking) and
    receives decoded frames via the `frame` signal. Everything that touches
    the serial port runs here.
    """

    frame = pyqtSignal(bytes)    # RGB565 framebuffer
    status = pyqtSignal(str)     # human-readable status / errors

    # Queue sentinels (never valid scancodes) for non-scancode actions.
    _REFRESH = -1
    _BREAK = -2

    def __init__(self, poll_ms: int = 175):
        super().__init__()
        self.poll_ms = poll_ms
        self.poll_enabled = True
        self._q: "queue.Queue[int]" = queue.Queue()
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

    # --- worker-thread internals ---
    def _dispatch(self, code: int):
        if code == self._REFRESH:
            return                       # fall through to the post-IO grab
        if code == self._BREAK:
            self._break()
            return
        self._send(code)

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
        self.setWindowTitle("TI-84 Evo - USB remote")
        self.setStyleSheet("QMainWindow { background: #f4f7fa; }")

        central = QWidget()
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
        # SetFixedSize above only pins the window's minimum; lock the maximum
        # too so it can't be resized at all. sizeHint already accounts for
        # --scale (the screen label is sized from it).
        self.setFixedSize(self.sizeHint())
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        worker.frame.connect(self.on_frame)
        worker.status.connect(self._on_status)

    def _on_status(self, text: str):
        # Route to the debug window if open. Otherwise only surface real
        # failures (to stderr) and drop the routine per-frame chatter.
        if self.debug_window is not None:
            self.debug_window.set_status(text)
        elif "fail" in text.lower() or "error" in text.lower():
            print(text, file=sys.stderr)

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
