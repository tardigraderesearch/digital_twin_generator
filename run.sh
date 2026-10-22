#!/bin/sh
# Digital Twin Generator launcher for Linux and macOS.
#
#   ./run.sh                       open the app
#   ./run.sh health SCAN.obj       any dtg command works too (dtg --help)
#
# First run: finds Python 3.11+, makes a private .venv next to this file and
# installs everything into it. Later runs skip straight to launching unless
# the dependencies changed. Nothing is installed system-wide and sudo is
# never used. Set DTG_PYTHON=/path/to/python to pick a Python.

set -u

HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
VENV="$HERE/.venv"
VPY="$VENV/bin/python"
STAMP="$VENV/.digital-twin-generator-installed"

say() { printf '%s\n' "[digital-twin-generator] $*"; }
fail() {
    printf '\n%s\n' "[digital-twin-generator] PROBLEM: $1" >&2
    shift
    for line in "$@"; do printf '%s\n' "$line" | sed 's/^/    /' >&2; done
    printf '\n' >&2
    exit 1
}

os_name=$(uname -s 2>/dev/null || echo unknown)

python_fix() {
    if [ "$os_name" = "Darwin" ]; then
        echo "Install Python 3.12 from https://www.python.org/downloads/macos/"
        echo "(or with Homebrew: brew install python@3.12 python-tk@3.12), then run ./run.sh again."
    else
        echo "Install Python 3.11 or newer with your package manager, for example:"
        echo "  Debian/Ubuntu: sudo apt install python3 python3-venv python3-tk"
        echo "  Fedora:        sudo dnf install python3 python3-tkinter"
        echo "  Arch:          sudo pacman -S python tk"
        echo "  openSUSE:      sudo zypper install python3 python3-tk"
        echo "  Alpine:        sudo apk add python3 py3-tkinter"
        echo "Then run ./run.sh again."
    fi
}

tk_fix() {
    if [ "$os_name" = "Darwin" ]; then
        echo "Install Python from https://www.python.org/downloads/macos/ (it includes Tk),"
        echo "or with Homebrew: brew install python-tk"
    else
        echo "Debian/Ubuntu: sudo apt install python3-tk"
        echo "Fedora:        sudo dnf install python3-tkinter"
        echo "Arch:          sudo pacman -S tk"
        echo "openSUSE:      sudo zypper install python3-tk"
        echo "Alpine:        sudo apk add py3-tkinter"
    fi
}

VERSION_OK='import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'

# 1. Find a Python 3.11+ ----------------------------------------------------
PY=""
for candidate in "${DTG_PYTHON:-}" python3 python3.13 python3.12 python3.11 python3.14 python; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$VERSION_OK" >/dev/null 2>&1; then
        PY=$(command -v "$candidate")
        break
    fi
done
if [ -z "$PY" ]; then
    fail "Python 3.11 or newer was not found." "$(python_fix)"
fi
say "Using $("$PY" -c 'import sys; print("Python %d.%d.%d" % sys.version_info[:3])') at $PY"

# 2. Tk is needed for the window (not for the command-line tools) -------------
needs_window=1
case "${1:-}" in
    health|synth|wiggle|build|servos|-h|--help|--version) needs_window=0 ;;
esac
if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
    if [ "$needs_window" -eq 1 ]; then
        fail "This Python has no tkinter, which the app window needs." \
            "Fix it with:" "$(tk_fix)" \
            "(Without Tk you can still use the tools: ./run.sh health scan.obj)"
    fi
    say "Note: tkinter is missing; fine for '$1', but the window will not open."
fi

# 3. Private virtual environment ---------------------------------------------
if [ ! -x "$VPY" ] || ! "$VPY" -c "import sys" >/dev/null 2>&1; then
    say "First run: creating a private Python environment in .venv (one time only)..."
    rm -rf "$VENV"
    if ! "$PY" -m venv "$VENV"; then
        rm -rf "$VENV"
        fail "Could not create the virtual environment." \
            "Debian/Ubuntu: sudo apt install python3-venv" \
            "Other systems: reinstall Python from python.org or your package manager."
    fi
fi

# 4. Install / update only when pyproject.toml or Python changed ------------
WANT=$("$VPY" -c 'import hashlib, sys
data = open(sys.argv[1], "rb").read() + sys.version.encode()
print(hashlib.sha256(data).hexdigest())' "$HERE/pyproject.toml") || fail "Could not read pyproject.toml."
HAVE=$(cat "$STAMP" 2>/dev/null || true)
if [ "$WANT" != "$HAVE" ] || ! "$VPY" -c "import dtg, cv2, numpy, scipy, trimesh, matplotlib, PIL" >/dev/null 2>&1; then
    say "Installing digital-twin-generator and its libraries (numpy, SciPy, trimesh, matplotlib, OpenCV, Pillow)..."
    say "This needs internet once and takes a minute or two."
    "$VPY" -m pip install --disable-pip-version-check --quiet --upgrade pip \
        || say "Could not upgrade pip; continuing with the bundled one."
    if ! "$VPY" -m pip install --disable-pip-version-check --quiet --upgrade --editable "$HERE"; then
        fail "Installing the Python libraries failed (see the pip messages above)." \
            "Check your internet connection and run ./run.sh again." \
            "To start over completely, delete the .venv folder."
    fi
    if ! "$VPY" -c "import dtg, cv2, numpy, scipy, trimesh, matplotlib, PIL" ; then
        fail "The libraries installed but do not import (see the error above)." \
            "Delete the .venv folder and run ./run.sh again."
    fi
    printf '%s\n' "$WANT" > "$STAMP"
    say "Install complete."
else
    say "Everything is already installed."
fi

# 5. Launch -------------------------------------------------------------------
if [ $# -eq 0 ]; then
    say "Opening the Digital Twin Generator window..."
fi
exec "$VPY" -m dtg "$@"
