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
from PyQt6.QtGui import (
    QAction, QActionGroup, QColor, QFont, QImage, QKeyEvent, QPainter, QPixmap,
)
from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QLayout, QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
    QWidget, QWIDGETSIZE_MAX,
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
        # Keypad position relative to the screen and the last frame received,
        # both adjustable on the fly from the View menu. The frame is kept so a
        # scale change can re-render a frozen screen without a fresh frame.
        self.keypad_side = "below"
        self._last_img = None
        # CSS border around the screen; insets the content rect, so the label
        # must be sized to the pixmap plus the border on each side.
        self.border = 2
        self.setWindowTitle("TI-84 Evo - USB remote")

        # Persistent leaf widgets. Built once and reparented onto a freshly
        # built central widget whenever the keypad position changes, so their
        # state (and the keypad's signal wiring) survives a relayout.
        self.model_lbl = QLabel("TI-84 Evo")     # centered above the display
        model_font = QFont()
        model_font.setStyleHint(QFont.StyleHint.SansSerif)
        model_font.setFamily(model_font.defaultFamily())  # concrete family
        model_font.setBold(True)
        model_font.setPixelSize(14)
        self.model_lbl.setFont(model_font)
        self.model_lbl.setStyleSheet("color: #2b333a;")
        self.model_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.screen_label = QLabel()
        self.screen_label.setFixedSize(SCREEN_W * scale + 2 * self.border,
                                       SCREEN_H * scale + 2 * self.border)
        self.screen_label.setStyleSheet(
            f"border: {self.border}px solid #aab; background: #000;"
        )
        self.screen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.keypad = build_keypad(self.press_key)

        # Disconnect veil, parented to self for now; _apply_layout reparents it
        # onto each central it builds so its geometry maps over the screen.
        self.overlay = Overlay(self)

        # Menus live in the menu bar's own reserved region, so they don't
        # disturb the central widget's fixed-size layout. Build them before the
        # first _apply_layout so their height is counted in sizeHint().
        m = self.menuBar().addMenu("File")
        self.send_action = m.addAction("Send File…", self._on_send)
        self.recv_action = m.addAction("Receive File…", self._on_receive)
        self.delete_action = m.addAction("Delete Variable…", self._on_delete)
        self._build_view_menu()

        # Arrange screen + keypad and lock the window to that content size.
        self._apply_layout()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        worker.frame.connect(self.on_frame)
        worker.status.connect(self._on_status)
        worker.busy.connect(self._on_busy)
        worker.file_list.connect(self._on_file_list)
        worker.finished.connect(self._on_finished)
        worker.connection.connect(self._on_connection)

    # --- view (scale / keypad position) ---
    def _build_view_menu(self):
        """View menu: live screen scale and keypad position, as two exclusive
        radio groups. Selections are session-only (not persisted)."""
        view = self.menuBar().addMenu("View")

        self._scale_group = QActionGroup(self)
        self._scale_group.setExclusive(True)
        view.addSection("Scale")
        for n, text in ((1, "1x"), (2, "2x")):
            act = QAction(text, self, checkable=True)
            act.setChecked(n == self.scale)
            act.triggered.connect(lambda _checked, k=n: self.set_scale(k))
            self._scale_group.addAction(act)
            view.addAction(act)

        view.addSeparator()

        self._side_group = QActionGroup(self)
        self._side_group.setExclusive(True)
        view.addSection("Keypad position")
        for side, text in (("below", "Below"), ("left", "Left"),
                           ("right", "Right")):
            act = QAction(text, self, checkable=True)
            act.setChecked(side == self.keypad_side)
            act.triggered.connect(
                lambda _checked, s=side: self.set_keypad_side(s))
            self._side_group.addAction(act)
            view.addAction(act)

    def _apply_layout(self):
        """(Re)build the central widget for the current keypad position,
        reusing the persistent screen/keypad/overlay widgets, then re-lock the
        window to fit. Called on startup and on each keypad-position change."""
        self.setUpdatesEnabled(False)

        central = QWidget()
        # Skin only the body, not the whole window: painting the background on
        # the QMainWindow bleeds under the transparent menu bar, leaving its
        # (system-themed) text unreadable on the light body.
        central.setObjectName("body")
        central.setStyleSheet("#body { background: #f4f7fa; }")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        # Lock the window to its content size (no user resizing).
        outer.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        outer.addWidget(self.model_lbl,
                        alignment=Qt.AlignmentFlag.AlignHCenter)

        if self.keypad_side == "below":
            outer.addWidget(self.screen_label,
                            alignment=Qt.AlignmentFlag.AlignCenter)
            outer.addWidget(self.keypad,
                            alignment=Qt.AlignmentFlag.AlignHCenter)
        else:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            if self.keypad_side == "left":
                widgets = (self.keypad, self.screen_label)
            else:
                widgets = (self.screen_label, self.keypad)
            for w in widgets:
                row.addWidget(w, alignment=Qt.AlignmentFlag.AlignVCenter)
            outer.addLayout(row)

        # The overlay isn't in any layout, so reparent it onto the new central
        # before setCentralWidget deletes the old one (else it dies with it).
        self.overlay.setParent(central)
        self.overlay.hide()

        self.setCentralWidget(central)
        self._refit()
        self._render_screen()
        self._sync_overlay()
        self.setFocus()             # keep receiving physical-keyboard keys
        if not self._online:
            self.overlay.show()
            self.overlay.raise_()

        self.setUpdatesEnabled(True)

    def _refit(self):
        """Re-lock the window to its current content size. The existing fixed
        pin is released first, then both layouts re-activated, so sizeHint()
        reflects the new content rather than the stale pinned size."""
        self.setMinimumSize(0, 0)
        self.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
        cw = self.centralWidget()
        if cw is not None and cw.layout() is not None:
            cw.layout().invalidate()
            cw.layout().activate()
        self.layout().invalidate()      # QMainWindowLayout (menu + central)
        self.layout().activate()
        self.setFixedSize(self.sizeHint())

    def set_scale(self, n: int):
        if n == self.scale:
            return
        self.scale = n
        self.screen_label.setFixedSize(SCREEN_W * n + 2 * self.border,
                                       SCREEN_H * n + 2 * self.border)
        self._render_screen()   # re-scale the frozen frame to the new size
        self._refit()           # window grows/shrinks to the new label
        self._sync_overlay()    # veil tracks the resized label

    def set_keypad_side(self, side: str):
        if side == self.keypad_side:
            return
        self.keypad_side = side
        self._apply_layout()

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
            f"{e['name']}  "
            f"({evo_usb.EVO_TYPE_EXTENSIONS.get(e['type'], 'bin')}, "
            f"{e['size']}B, {'RAM' if not e['mem'] else 'Arc'})"
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
            if entry["type"] == 15:
                # Python scripts download as readable source, not raw AppVar.
                default = f"{entry['name']}.py"
                filt = "Python source (*.py)"
            else:
                ext = evo_usb.EVO_TYPE_EXTENSIONS.get(entry["type"], "bin")
                default = f"{entry['name']}.{ext}"
                filt = ""
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Variable As", default, filt)
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
    def _render_screen(self):
        """Scale the last received frame to the current zoom and show it. A
        no-op until the first frame arrives; called again on a scale change so
        a frozen screen re-renders immediately, with no fresh frame needed."""
        if self._last_img is None:
            return
        pix = QPixmap.fromImage(self._last_img).scaled(
            SCREEN_W * self.scale, SCREEN_H * self.scale,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.screen_label.setPixmap(pix)

    def on_frame(self, rgb565: bytes):
        rgb888 = rgb565_to_rgb888(rgb565)
        # .copy() detaches from the local buffer so the image stays valid for
        # re-rendering after the next scale change, not just this frame.
        self._last_img = QImage(
            rgb888, SCREEN_W, SCREEN_H, SCREEN_W * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        self._render_screen()

    def closeEvent(self, event):
        self.worker.stop()
        if self.debug_window is not None:
            self.debug_window.close()
        super().closeEvent(event)
