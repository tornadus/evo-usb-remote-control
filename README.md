# TI-84 Evo USB remote control

Drive a physical TI-84 Evo calculator from your desktop over its USB link.
The app shows the calculator's live screen and gives you a
clickable virtual keypad as well as keybinds for your physical keyboard!


## Layout

| Path            | Description                                                                |
| --------------- | -------------------------------------------------------------------------- |
| `calc_remote.py`| Entry point: CLI args and app launch                                       |
| `worker.py`     | USB I/O worker thread, screen capture, file-transfer commands              |
| `window.py`     | Main/debug windows, offline overlay, screen rendering, File-menu wiring    |
| `keypad.py`     | The virtual keypad widget and keymapping                                   |
| `evo_usb_py/`   | Submodule: the low-level USB/Kermit transport (`evo_usb`), pinned upstream.|


## Setup

Make sure to clone **with the submodule**.

```bash
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init
```

Install the dependencies (a venv + pip is suggested, but your distro may package the requirements):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Plug in the calculator over USB, then:

```bash
python calc_remote.py [--poll-ms N] [--no-poll] [--scale N] [--debug]
```

- `--poll-ms N`: background screen-poll period in ms (default 175).
- `--no-poll`: only refresh after input or a manual refresh.
- `--scale N`: screen magnification (default 1).
- `--debug`: show the live/poll/refresh controls and a status readout in a separate window.

## Keyboard mapping

The keyboard bindings mirror TI's
[official TI-84 Evo emulator keyboard map](https://education.ti.com/en/product-resources/eguides/eguide-84-evo/keyboard-mapping),
so muscle memory carries over.
