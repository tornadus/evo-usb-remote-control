"""
TI-84 Evo remote control

Usage:
    python calc_remote.py [--poll-ms N] [--no-poll] [--scale N] [--debug]
"""
from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from worker import UsbWorker
from window import RemoteWindow


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
