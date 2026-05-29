"""
PyQt6 UI for the TI-84 Evo remote.

`RemoteWindow` is the main window: model label, live screen, the virtual
keypad (from `keypad.py`), and a File menu for transfers. `Overlay` is the
offline veil drawn over a frozen screen, and `DebugWindow` is the optional
poll/refresh/status panel (--debug). All of these consume a `UsbWorker`
(built in calc_remote.main) but never create one, so this module has no
runtime dependency on the backend.
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QLayout, QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

# evo_usb ships as a submodule (./evo_usb_py/evo_usb.py). Put it on the path
# before importing, so this works regardless of the current directory.
_EVO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "evo_usb_py"
)
if _EVO_PATH not in sys.path:
    sys.path.insert(0, _EVO_PATH)
import evo_usb  # noqa: E402  (path tweak must run first)

from keypad import (  # noqa: E402
    SCREEN_H, SCREEN_W, SCANCODES, build_keypad, key_to_label,
)

if TYPE_CHECKING:            # avoids a runtime import cycle with worker.py
    from worker import UsbWorker


def rgb565_to_rgb888(buf: bytes) -> bytes:
    a = np.frombuffer(buf, dtype="<u2")
    r = ((a >> 11) & 0x1F).astype(np.uint8)
    g = ((a >> 5) & 0x3F).astype(np.uint8)
    b = (a & 0x1F).astype(np.uint8)
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)
    return np.stack([r, g, b], axis=-1).tobytes()


def default_varname(path: str) -> str:
    """Suggest an AppVar name from a filename: drop the extension, keep only
    ASCII letters (send_file's name encoder handles A-Z only), uppercase, and
    cap at 8 chars. Falls back to FILE when nothing usable is left.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    letters = "".join(c for c in stem if c.isascii() and c.isalpha())
    return letters[:8].upper() or "FILE"


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


class Overlay(QWidget):
    """Translucent veil over the screen label, shown while offline.

    Dims the last (frozen) frame with a dark wash and draws centered status
    text on top. Mouse-transparent so the keypad underneath stays usable.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._text = "Searching for calculator…"
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def set_text(self, text: str):
        self._text = text
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 150))    # ~60% dark veil
        p.setPen(QColor(235, 238, 242))
        f = self.font()
        f.setPixelSize(14)
        f.setBold(True)
        p.setFont(f)
        flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
        p.drawText(self.rect().adjusted(10, 10, -10, -10), flags, self._text)


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
        # Mirror of the worker's connection state; gates input while offline.
        self._online = False
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

        # Disconnect veil, parented to the same widget that owns the screen
        # label's layout, so its geometry maps cleanly over the label.
        self.overlay = Overlay(central)

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
        worker.connection.connect(self._on_connection)

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

    # --- connection state / overlay ---
    def _sync_overlay(self):
        """Align the overlay to the screen label's current geometry."""
        parent = self.overlay.parentWidget()
        top_left = self.screen_label.mapTo(
            parent, self.screen_label.rect().topLeft())
        self.overlay.setGeometry(
            top_left.x(), top_left.y(),
            self.screen_label.width(), self.screen_label.height())
        self.overlay.raise_()

    def _on_connection(self, online: bool, reason: str):
        self._online = online
        # A dead link can't accept transfers; mirror the busy disabling.
        self.send_action.setEnabled(online)
        self.recv_action.setEnabled(online)
        self.delete_action.setEnabled(online)
        if online:
            self.overlay.hide()
        else:
            self.overlay.set_text(reason)
            self._sync_overlay()
            self.overlay.show()
            self.overlay.raise_()

    def showEvent(self, event):
        # The layout only finalizes the label's position once shown; sync the
        # veil here and bring it up as "Searching…" until the first frame.
        super().showEvent(event)
        self._sync_overlay()
        if not self._online:
            self.overlay.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_overlay()

    # --- input ---
    def press_key(self, label: str):
        if not self._online:
            # Drop keystrokes while offline so they can't pile up behind
            # failing sends and replay as a burst on reconnect.
            self._on_status("offline · keystroke ignored")
            return
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
