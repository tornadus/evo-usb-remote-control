"""
Virtual TI-84 Evo keypad: the PyQt6 widgets and lookup tables for the
remote-control window.

`calc_remote.py` builds its window around `build_keypad()` and uses the maps
below to translate button labels and physical keystrokes into Evo scancodes.

The layout mirrors the hardware: 5 columns wide, with the D-pad spanning
rows 1-2, cols 3-4.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGridLayout, QLayout, QPushButton, QWidget,
)

SCREEN_W = 320
SCREEN_H = 240
SCREEN_BYTES = SCREEN_W * SCREEN_H * 2  # RGB565


# Scancode to calc function, found empirically. SCANCODES below is the
# inverse map (button label to scancode).
DISCOVERED: dict[int, str] = {
    0x01: "DOWN",  0x02: "LEFT",  0x03: "RIGHT",  0x04: "UP",
    0x09: "ENTER",
    0x0A: "<> (precision)",
    0x0B: "+",
    0x0C: "-",
    0x0D: "*",          # multiplication (×)
    0x0E: "÷",
    0x0F: "CLEAR",
    0x10: "ON",
    0x11: "(-)",        # negate
    0x12: "3",
    0x13: "6",
    0x14: "9",
    0x15: ")",
    0x16: "TAN",
    0x17: "VARS",
    0x19: ".",
    0x1A: "2",
    0x1B: "5",
    0x1C: "8",
    0x1D: "(",
    0x1E: "COS",
    0x1F: "PRGM",
    0x20: "STAT",
    0x21: "0",
    0x22: "1",
    0x23: "4",
    0x24: "7",
    0x25: ",",
    0x26: "SIN",
    0x27: "n/d (frac)",
    0x28: "X,T,θ,n",
    0x2A: "STO→",
    0x2B: "LN",
    0x2C: "LOG",
    0x2D: "X^2",
    0x2E: "X^□",
    0x2F: "MATH",
    0x30: "ALPHA",
    0x31: "GRAPH",
    0x32: "TRACE",
    0x33: "ZOOM",
    0x34: "WINDOW",
    0x35: "Y=",
    0x36: "2ND",
    0x37: "MODE",
    0x38: "DEL",
}


# Inverse map: button label to scancode.
SCANCODES: dict[str, int] = {
    # F-row
    "Y=": 0x35, "WINDOW": 0x34, "ZOOM": 0x33, "TRACE": 0x32, "GRAPH": 0x31,
    # Mod row
    "2ND": 0x36, "MODE": 0x37, "DEL": 0x38,
    # Alpha row
    "ALPHA": 0x30, "X,T,θ,n": 0x28, "STAT": 0x20,
    # Math row
    "MATH": 0x2F, "n/d": 0x27, "PRGM": 0x1F, "VARS": 0x17, "CLEAR": 0x0F,
    # x^□ / sin / cos / tan / ÷
    "X^□": 0x2E, "SIN": 0x26, "COS": 0x1E, "TAN": 0x16, "÷": 0x0E,
    # x² / , / ( / ) / ×
    "X^2": 0x2D, ",": 0x25, "(": 0x1D, ")": 0x15, "×": 0x0D,
    # log / 7 / 8 / 9 / −
    "LOG": 0x2C, "7": 0x24, "8": 0x1C, "9": 0x14, "−": 0x0C,
    # ln / 4 / 5 / 6 / +
    "LN": 0x2B, "4": 0x23, "5": 0x1B, "6": 0x13, "+": 0x0B,
    # sto / 1 / 2 / 3 / <>
    "STO→": 0x2A, "1": 0x22, "2": 0x1A, "3": 0x12, "<>": 0x0A,
    # on / 0 / . / (-) / enter
    "ON": 0x10, "0": 0x21, ".": 0x19, "(-)": 0x11, "ENTER": 0x09,
    # D-pad
    "UP": 0x04, "DOWN": 0x01, "LEFT": 0x02, "RIGHT": 0x03,
}


# Stylesheets per button category. Keeps the visual hierarchy similar to
# the reference even without per-key second-function micro labels.
STYLE_FN = """
QPushButton {
    background: white; color: #1c4587; font-weight: 600;
    border: 1px solid #aab; border-radius: 4px; padding-top: 8px; }
QPushButton:pressed { background: #d8e2ee; }
"""
STYLE_2ND = """
QPushButton {
    background: #1e526b; color: white; font-weight: 700;
    border: 1px solid #144156; border-radius: 4px; padding-top: 8px; }
QPushButton:pressed { background: #163d52; }
"""
STYLE_ALPHA = """
QPushButton {
    background: #4a7c44; color: white; font-weight: 700;
    border: 1px solid #335a30; border-radius: 4px; padding-top: 8px; }
QPushButton:pressed { background: #3b6638; }
"""
STYLE_LIGHT = """
QPushButton {
    background: #e6ebef; color: #222; font-weight: 600;
    border: 1px solid #bbc; border-radius: 4px; padding-top: 8px; }
QPushButton:pressed { background: #ccd2d9; }
"""
STYLE_DARK = """
QPushButton {
    background: #2b333a; color: white; font-weight: 700;
    border: 1px solid #1a2026; border-radius: 4px; padding-top: 8px; }
QPushButton:pressed { background: #1a2026; }
"""
STYLE_BLUE = """
QPushButton {
    background: #1e6fb8; color: white; font-weight: 700;
    border: 1px solid #144d80; border-radius: 4px; padding-top: 8px; }
QPushButton:pressed { background: #144d80; }
"""
# (The D-pad arrows are painted by the Dpad widget, not via a stylesheet.)

# Physical-keyboard to calculator-key mapping, mirroring TI's official
# TI-84 Evo keyboard map so muscle memory carries over:
#   https://education.ti.com/en/product-resources/eguides/eguide-84-evo/keyboard-mapping
# key_to_label resolves in three tiers (the map is modifier-aware, e.g.
# m -> ×, Shift+M -> MODE): SPECIAL_KEY_MAP for non-character keys, the
# LETTER maps for A-Z, then CHAR_KEY_MAP keyed on the text the OS typed.
SPECIAL_KEY_MAP = {
    Qt.Key.Key_F1: "Y=",   Qt.Key.Key_F2: "WINDOW", Qt.Key.Key_F3: "ZOOM",
    Qt.Key.Key_F4: "TRACE", Qt.Key.Key_F5: "GRAPH",
    Qt.Key.Key_F6: "2ND",  Qt.Key.Key_F7: "ALPHA",
    Qt.Key.Key_Return: "ENTER", Qt.Key.Key_Enter: "ENTER",
    Qt.Key.Key_Backspace: "CLEAR",   # TI: BACKSPACE -> CLEAR
    Qt.Key.Key_Delete: "DEL",        # TI: DELETE -> DEL
    Qt.Key.Key_Up: "UP",   Qt.Key.Key_Down: "DOWN",
    Qt.Key.Key_Left: "LEFT", Qt.Key.Key_Right: "RIGHT",
}

# Unshifted letter to primary calc function (TI's A-Z assignment).
LETTER_KEY_MAP = {
    Qt.Key.Key_A: "MATH", Qt.Key.Key_B: "n/d",  Qt.Key.Key_C: "PRGM",
    Qt.Key.Key_D: "X^□",  Qt.Key.Key_E: "SIN",  Qt.Key.Key_F: "COS",
    Qt.Key.Key_G: "TAN",  Qt.Key.Key_H: "÷",    Qt.Key.Key_I: "X^2",
    Qt.Key.Key_J: ",",    Qt.Key.Key_K: "(",    Qt.Key.Key_L: ")",
    Qt.Key.Key_M: "×",    Qt.Key.Key_N: "LOG",  Qt.Key.Key_O: "7",
    Qt.Key.Key_P: "8",    Qt.Key.Key_Q: "9",    Qt.Key.Key_R: "−",
    Qt.Key.Key_S: "LN",   Qt.Key.Key_T: "4",    Qt.Key.Key_U: "5",
    Qt.Key.Key_V: "6",    Qt.Key.Key_W: "+",    Qt.Key.Key_X: "STO→",
    Qt.Key.Key_Y: "1",    Qt.Key.Key_Z: "2",
}

# Shift+letter to second function (overrides LETTER_KEY_MAP when held).
SHIFT_LETTER_KEY_MAP = {
    Qt.Key.Key_M: "MODE", Qt.Key.Key_X: "X,T,θ,n", Qt.Key.Key_S: "STAT",
    Qt.Key.Key_V: "VARS", Qt.Key.Key_O: "<>",
    Qt.Key.Key_H: "ON",   # Mac eGuide: Shift+H -> ON/HOME
}

# Printable character to calc function. Digits map to themselves, and
# symbols cover the operators and the shifted-digit alternates.
CHAR_KEY_MAP = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "/": "÷", "*": "×", "-": "−", "+": "+",
    ".": ".", ":": ".", ",": ",", "(": "(", ")": ")",
    "_": "(-)", "?": "(-)",   # Shift+- and Shift+? -> (-)
    "~": "ON",               # Shift+` (US layout) -> ON/HOME
}


def key_to_label(event) -> str | None:
    """Translate a Qt key press into a calculator button label, mirroring TI's
    official TI-84 Evo keyboard map. Returns None when the key isn't mapped
    (so the caller can defer to the default Qt handling).
    """
    mods = event.modifiers()
    # Leave Ctrl/Alt/Meta combos to the OS (copy/paste, window shortcuts, …).
    if mods & (Qt.KeyboardModifier.ControlModifier
               | Qt.KeyboardModifier.AltModifier
               | Qt.KeyboardModifier.MetaModifier):
        return None

    key = event.key()
    if key in SPECIAL_KEY_MAP:
        return SPECIAL_KEY_MAP[key]

    shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
    if Qt.Key.Key_A.value <= int(key) <= Qt.Key.Key_Z.value:
        if shift and key in SHIFT_LETTER_KEY_MAP:
            return SHIFT_LETTER_KEY_MAP[key]
        return LETTER_KEY_MAP.get(key)

    return CHAR_KEY_MAP.get(event.text())


# Per-key 2nd-function (top-left, yellow/orange) and alpha (top-right,
# green) labels. A best guess from a stock TI-84 CE layout, adjusted as
# we learn what the Evo actually displays.
LEGENDS: dict[str, tuple[str, str]] = {
    # F-row
    "Y=":      ("plot",   "f1"),
    "WINDOW":  ("tblset", "f2"),
    "ZOOM":    ("format", "f3"),
    "TRACE":   ("calc",   "f4"),
    "GRAPH":   ("table",  "f5"),
    # Mod row
    "2ND":     ("",       ""),
    "MODE":    ("quit",   ""),
    "DEL":     ("ins",    ""),
    # Alpha row
    "ALPHA":   ("lock",   ""),
    "X,T,θ,n": ("link",   ""),
    "STAT":    ("list",  "distr"),
    # Math row
    "MATH":    ("test",   "A"),
    "n/d":     ("angle",  "B"),
    "PRGM":    ("draw",  "C"),
    "VARS":    ("matrix",   ""),
    "CLEAR":   ("<- clr",       ""),
    # x^□ row
    "X^□":     ("√x",      "D"),
    "SIN":     ("sin⁻¹",  "E"),
    "COS":     ("cos⁻¹",  "F"),
    "TAN":     ("tan⁻¹",  "G"),
    "÷":       ("π",      "H"),
    # x² row
    "X^2":     ("√",      "I"),
    ",":       ("EE",     "J"),
    "(":       ("{",      "K"),
    ")":       ("}",      "L"),
    "×":       ("e",      "M"),
    # log row
    "LOG":     ("10ˣ",    "N"),
    "7":       ("u",      "O"),
    "8":       ("v",      "P"),
    "9":       ("w",      "Q"),
    "−":       ("[",      "R"),
    # ln row
    "LN":      ("eˣ",     "S"),
    "4":       ("L4",     "T"),
    "5":       ("L5",     "U"),
    "6":       ("L6",     "V"),
    "+":       ("]",      "W"),
    # sto row
    "STO→":    ("rcl",    "X"),
    "1":       ("L1",     "Y"),
    "2":       ("L2",     "Z"),
    "3":       ("L3",     "θ"),
    "<>":      ("mem",    ""),
    # bottom
    "ON":      ("off",    ""),
    "0":       ("cat.", "␣"),   # ␣ = the calc's space glyph (U+2423 open box)
    ".":       ("i",      ":"),
    "(-)":     ("Ans",    "?"),
    "ENTER":   ("entry",  ""),
    # d-pad
    "UP":      ("",       ""),
    "DOWN":    ("",       ""),
    "LEFT":    ("",       ""),
    "RIGHT":   ("",       ""),
}


class CalcButton(QPushButton):
    """Button with main label centered + small 2nd-layer (top-left,
    orange/yellow) and ALPHA-layer (top-right, green) legends painted on
    top. Sized to a 1:1.375 height:width ratio."""

    BASE_HEIGHT = 44
    BASE_WIDTH = int(BASE_HEIGHT * 1.375)  # 60
    LEGEND_FONT_PT = 8

    LEGEND_2ND_COLOR_LIGHT = QColor("#d8851a")  # orange on light backgrounds
    LEGEND_2ND_COLOR_DARK = QColor("#f0c060")   # yellow on dark backgrounds
    LEGEND_ALPHA_COLOR_LIGHT = QColor("#2b6024")
    LEGEND_ALPHA_COLOR_DARK = QColor("#7adc6b")

    def __init__(self, label: str, second: str = "", alpha: str = "",
                 dark: bool = False, parent=None):
        super().__init__(label, parent)
        self.second_label = second
        self.alpha_label = alpha
        self.dark = dark
        self.setFixedSize(self.BASE_WIDTH, self.BASE_HEIGHT)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not (self.second_label or self.alpha_label):
            return
        p = QPainter(self)
        f = QFont()
        f.setPointSize(self.LEGEND_FONT_PT)
        f.setBold(True)
        p.setFont(f)
        w = self.width()
        fm = p.fontMetrics()
        # Slot height = line height + headroom. Tall glyphs like [ ] { }
        # overshoot the nominal line height and were getting clipped.
        slot_h = fm.height() + 4
        # A lone legend takes the full width. If both are present, give each
        # its natural width when they fit side by side, else split evenly
        # (keeps long labels like "matrix" from clipping).
        full_w = w - 6
        adv = fm.horizontalAdvance
        sw = adv(self.second_label) if self.second_label else 0
        aw = adv(self.alpha_label) if self.alpha_label else 0
        if self.second_label and self.alpha_label:
            if sw + aw + 4 <= full_w:
                second_w, alpha_w = sw + 1, aw + 1
            else:
                second_w = alpha_w = w // 2 - 4
        else:
            second_w = alpha_w = full_w
        if self.second_label:
            p.setPen(self.LEGEND_2ND_COLOR_DARK if self.dark
                     else self.LEGEND_2ND_COLOR_LIGHT)
            p.drawText(QRect(3, 1, second_w, slot_h),
                       int(Qt.AlignmentFlag.AlignLeft
                           | Qt.AlignmentFlag.AlignTop),
                       self.second_label)
        if self.alpha_label:
            p.setPen(self.LEGEND_ALPHA_COLOR_DARK if self.dark
                     else self.LEGEND_ALPHA_COLOR_LIGHT)
            p.drawText(QRect(w - alpha_w - 3, 1, alpha_w, slot_h),
                       int(Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignTop),
                       self.alpha_label)
        p.end()


# Keypad layout: (style, label, row, col, rowspan, colspan). 5 columns wide,
# right column is operators (÷ × − +) and ENTER. The D-pad spans rows 1-2.
KEYPAD = [
    # row 0: F-keys
    (STYLE_FN,    "Y=",      0, 0, 1, 1),
    (STYLE_FN,    "WINDOW",  0, 1, 1, 1),
    (STYLE_FN,    "ZOOM",    0, 2, 1, 1),
    (STYLE_FN,    "TRACE",   0, 3, 1, 1),
    (STYLE_FN,    "GRAPH",   0, 4, 1, 1),
    # row 1 / 2: mod keys + d-pad (placed below)
    (STYLE_2ND,   "2ND",     1, 0, 1, 1),
    (STYLE_LIGHT, "MODE",    1, 1, 1, 1),
    (STYLE_LIGHT, "DEL",     1, 2, 1, 1),
    (STYLE_ALPHA, "ALPHA",   2, 0, 1, 1),
    (STYLE_LIGHT, "X,T,θ,n", 2, 1, 1, 1),
    (STYLE_LIGHT, "STAT",    2, 2, 1, 1),
    # row 3: math / frac / PRGM / VARS / CLEAR (frac in slot 2)
    (STYLE_LIGHT, "MATH",    3, 0, 1, 1),
    (STYLE_LIGHT, "n/d",     3, 1, 1, 1),
    (STYLE_LIGHT, "PRGM",    3, 2, 1, 1),
    (STYLE_LIGHT, "VARS",    3, 3, 1, 1),
    (STYLE_LIGHT, "CLEAR",   3, 4, 1, 1),
    # row 4: x^□ / sin / cos / tan / ÷
    (STYLE_LIGHT, "X^□",     4, 0, 1, 1),
    (STYLE_LIGHT, "SIN",     4, 1, 1, 1),
    (STYLE_LIGHT, "COS",     4, 2, 1, 1),
    (STYLE_LIGHT, "TAN",     4, 3, 1, 1),
    (STYLE_FN,    "÷",       4, 4, 1, 1),
    # row 5: x^2 / , / ( / ) / ×
    (STYLE_LIGHT, "X^2",     5, 0, 1, 1),
    (STYLE_LIGHT, ",",       5, 1, 1, 1),
    (STYLE_LIGHT, "(",       5, 2, 1, 1),
    (STYLE_LIGHT, ")",       5, 3, 1, 1),
    (STYLE_FN,    "×",       5, 4, 1, 1),
    # row 6: log / 7 / 8 / 9 / −
    (STYLE_LIGHT, "LOG",     6, 0, 1, 1),
    (STYLE_DARK,  "7",       6, 1, 1, 1),
    (STYLE_DARK,  "8",       6, 2, 1, 1),
    (STYLE_DARK,  "9",       6, 3, 1, 1),
    (STYLE_FN,    "−",       6, 4, 1, 1),
    # row 7: ln / 4 / 5 / 6 / +
    (STYLE_LIGHT, "LN",      7, 0, 1, 1),
    (STYLE_DARK,  "4",       7, 1, 1, 1),
    (STYLE_DARK,  "5",       7, 2, 1, 1),
    (STYLE_DARK,  "6",       7, 3, 1, 1),
    (STYLE_FN,    "+",       7, 4, 1, 1),
    # row 8: sto / 1 / 2 / 3 / <> (precision change, right col)
    (STYLE_LIGHT, "STO→",    8, 0, 1, 1),
    (STYLE_DARK,  "1",       8, 1, 1, 1),
    (STYLE_DARK,  "2",       8, 2, 1, 1),
    (STYLE_DARK,  "3",       8, 3, 1, 1),
    (STYLE_FN,    "<>",      8, 4, 1, 1),
    # row 9: on / 0 / . / (-) / enter
    (STYLE_LIGHT, "ON",      9, 0, 1, 1),
    (STYLE_DARK,  "0",       9, 1, 1, 1),
    (STYLE_DARK,  ".",       9, 2, 1, 1),
    (STYLE_DARK,  "(-)",     9, 3, 1, 1),
    (STYLE_BLUE,  "ENTER",   9, 4, 1, 1),
]


def _rounded_polygon_path(points, radii) -> QPainterPath:
    """Closed path through `points`, rounding each vertex by the matching
    radius (0 = a sharp corner). Each rounded corner is a quadratic bend with
    the vertex as control point. The radius is clamped to half the shorter
    adjacent edge."""
    path = QPainterPath()
    n = len(points)
    for i in range(n):
        cur, prev, nxt = points[i], points[(i - 1) % n], points[(i + 1) % n]
        if radii[i] <= 0:
            (path.moveTo if i == 0 else path.lineTo)(cur)
            continue
        v1 = QPointF(prev.x() - cur.x(), prev.y() - cur.y())
        v2 = QPointF(nxt.x() - cur.x(), nxt.y() - cur.y())
        l1, l2 = math.hypot(v1.x(), v1.y()), math.hypot(v2.x(), v2.y())
        rr = min(radii[i], l1 / 2, l2 / 2)
        a = QPointF(cur.x() + v1.x() / l1 * rr, cur.y() + v1.y() / l1 * rr)
        b = QPointF(cur.x() + v2.x() / l2 * rr, cur.y() + v2.y() / l2 * rr)
        (path.moveTo if i == 0 else path.lineTo)(a)
        path.quadTo(cur, b)
    path.closeSubpath()
    return path


class Dpad(QWidget):
    """The four arrow keys as one widget. Each arm pins its outer edge to the
    d-pad boundary and tapers to a rounded point near the center (sharp
    shoulders at the rectangle-to-triangle transition), so keys nest closely.
    One widget rather than four overlapping ones lets the points interlock
    cleanly. Clicking a shape calls on_press(direction)."""

    OUTER_R = 4    # outer corners, matching the rectangular keys
    TIP_R = 5      # tip toward the center (kept fairly tight)
    TIP_GAP = 7    # half the gap between opposing tips at the center
    UD_EDGE = CalcButton.BASE_WIDTH - 5   # Up/Down outer-edge length
    LR_EDGE = 40   # Left/Right outer-edge length (shorter, per the HW)

    def __init__(self, on_press, parent=None):
        super().__init__(parent)
        self._on_press = on_press
        bw, bh = CalcButton.BASE_WIDTH, CalcButton.BASE_HEIGHT
        gap = bw // 4
        w, h = 2 * bw + gap, 2 * bh + gap
        self.setFixedSize(w, h)

        ud, lr = self.UD_EDGE, self.LR_EDGE
        R, T, g = self.OUTER_R, self.TIP_R, self.TIP_GAP
        cxu = (w - ud) / 2     # Up/Down arms, centered (top/bottom pinned)
        cyl = (h - lr) / 2     # Left/Right arms, centered (sides pinned)
        # Shoulders sit at each arm's midpoint (outer edge to tip), so a key
        # is half rectangle, half triangle, and longer arms get a longer body
        # rather than a thin point.
        sh_v = (h / 2 - g) / 2     # UP/DOWN shoulder inset from the outer edge
        sh_h = (w / 2 - g) / 2     # LEFT/RIGHT shoulder inset
        P = QPointF
        # (direction, glyph, vertices clockwise, per-vertex radii, glyph rect).
        # radii pattern: outer corners = R, center tip = T, shoulders = 0.
        self._keys = [
            ("UP", "▲",
             [P(cxu, 0), P(cxu + ud, 0), P(cxu + ud, sh_v),
              P(w / 2, h / 2 - g), P(cxu, sh_v)],
             [R, R, 0, T, 0], QRect(round(cxu), 0, ud, round(sh_v))),
            ("DOWN", "▼",
             [P(w / 2, h / 2 + g), P(cxu + ud, h - sh_v),
              P(cxu + ud, h), P(cxu, h), P(cxu, h - sh_v)],
             [T, 0, R, R, 0],
             QRect(round(cxu), round(h - sh_v), ud, round(sh_v))),
            ("LEFT", "◀",
             [P(0, cyl), P(sh_h, cyl), P(w / 2 - g, h / 2),
              P(sh_h, cyl + lr), P(0, cyl + lr)],
             [R, 0, T, 0, R], QRect(0, round(cyl), round(sh_h), lr)),
            ("RIGHT", "▶",
             [P(w, cyl), P(w, cyl + lr), P(w - sh_h, cyl + lr),
              P(w / 2 + g, h / 2), P(w - sh_h, cyl)],
             [R, R, 0, T, 0],
             QRect(round(w - sh_h), round(cyl), round(sh_h), lr)),
        ]
        self._paths = {d: _rounded_polygon_path(v, r)
                       for d, _glyph, v, r, _gr in self._keys}
        self._pressed = None

    def _hit(self, pos) -> str | None:
        pt = QPointF(pos)
        for d in self._paths:
            if self._paths[d].contains(pt):
                return d
        return None

    def mousePressEvent(self, e):
        self._pressed = self._hit(e.position())
        if self._pressed:
            self.update()

    def mouseReleaseEvent(self, e):
        if self._pressed and self._hit(e.position()) == self._pressed:
            self._on_press(self._pressed)
        if self._pressed:
            self._pressed = None
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = self.font()
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        # Normalize every arrow to the L/R glyph's ink height and center it by
        # its ink box: the ▲▼ and ◀▶ glyphs differ in size and aren't centered
        # in their line box.
        target = fm.tightBoundingRect("◀").height()
        for d, glyph, _v, _r, grect in self._keys:
            path = self._paths[d]
            fill = QColor("#d8e2ee") if self._pressed == d else QColor("white")
            p.fillPath(path, QBrush(fill))
            p.strokePath(path, QPen(QColor("#aab"), 1))
            ink = fm.tightBoundingRect(glyph)
            if ink.height() <= 0:
                continue
            p.save()
            p.setPen(QColor("#1c4587"))
            p.translate(grect.center().x() + 0.5, grect.center().y() + 0.5)
            p.scale(target / ink.height(), target / ink.height())
            p.drawText(QPointF(-(ink.left() + ink.right()) / 2,
                               -(ink.top() + ink.bottom()) / 2), glyph)
            p.restore()
        p.end()


def build_dpad(on_press) -> QWidget:
    """The d-pad: four arrow keys tapering to points near the center, spanning
    two button rows + one gap. `on_press(label)` fires with the arrow label."""
    return Dpad(on_press)


# Faces that aren't just label.lower(): the precision-change key (<>) prints as
# the same left/right arrow glyphs the D-pad uses, with a space between them.
FACE_OVERRIDES = {
    "<>": "◀ ▶",
}

# Digits and arithmetic operators get a larger face than the function keys,
# matching the calculator and giving the number pad more presence.
BIG_LABEL_KEYS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "−", "×", "÷",
}
BIG_LABEL_POINT_BUMP = 4   # points added on top of the default button font


def build_keypad(on_press) -> QWidget:
    """Build the full keypad widget. `on_press(label)` fires on every key.

    Uniform spacing in both axes (a quarter button width). Locking the
    layout size constraint stops Qt from stretching the columns.
    """
    gap = CalcButton.BASE_WIDTH // 4

    wrap = QWidget()
    grid = QGridLayout(wrap)
    grid.setHorizontalSpacing(gap)
    grid.setVerticalSpacing(gap)
    grid.setContentsMargins(gap, gap, gap, gap)
    grid.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

    dark_styles = {STYLE_DARK, STYLE_2ND, STYLE_ALPHA, STYLE_BLUE}

    for style, label, r, c, rs, cs in KEYPAD:
        second, alpha = LEGENDS.get(label, ("", ""))
        # `label` stays the canonical id (the SCANCODES/LEGENDS/key_to_label
        # key). The face is the calc's lowercase print, with glyph overrides.
        face = FACE_OVERRIDES.get(label, label.lower())
        btn = CalcButton(face, second=second, alpha=alpha,
                         dark=(style in dark_styles))
        btn.setStyleSheet(style)
        if label in BIG_LABEL_KEYS:
            f = btn.font()
            if f.pointSize() > 0:
                f.setPointSize(f.pointSize() + BIG_LABEL_POINT_BUMP)
            else:                       # font defined in pixels, not points
                f.setPixelSize(f.pixelSize() + 5)
            btn.setFont(f)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.clicked.connect(lambda _, k=label: on_press(k))
        grid.addWidget(btn, r, c, rs, cs)

    # D-pad spans rows 1-2 cols 3-4 (next to mod / alpha)
    grid.addWidget(build_dpad(on_press), 1, 3, 2, 2)
    return wrap
