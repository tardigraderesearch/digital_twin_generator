#!/bin/sh
# Digital Twin Generator launcher for Linux and macOS.
#
#   ./run.sh                       open the app
#   ./run.sh health SCAN.obj       any dtg command works too (dtg --help)
#
# This script only finds Python 3.11+ and hands over to launcher.py, which
# does the rest on every OS: a private .venv next to this file (first run
# only), installing the dependencies when they change, and launching. Nothing
# is installed system-wide and sudo is never used.
# Set DTG_PYTHON=/path/to/python to pick a Python.

HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
VERSION_OK='import sys; sys.exit(sys.version_info < (3, 11))'

PY=""
for candidate in "${DTG_PYTHON:-}" python3 python3.13 python3.12 python3.11 python3.14 python; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$VERSION_OK" >/dev/null 2>&1; then
        PY=$(command -v "$candidate")
        break
    fi
done

if [ -z "$PY" ]; then
    {
        printf '\n%s\n' "[digital-twin-generator] PROBLEM: Python 3.11 or newer was not found."
        if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
            echo "    Install Python 3.12 from https://www.python.org/downloads/macos/"
            echo "    (or with Homebrew: brew install python@3.12 python-tk@3.12), then run ./run.sh again."
        else
            echo "    Install Python 3.11 or newer with your package manager, for example:"
            echo "      Debian/Ubuntu: sudo apt install python3 python3-venv python3-tk"
            echo "      Fedora:        sudo dnf install python3 python3-tkinter"
            echo "      Arch:          sudo pacman -S python tk"
            echo "      openSUSE:      sudo zypper install python3 python3-tk"
            echo "      Alpine:        sudo apk add python3 py3-tkinter"
            echo "    Then run ./run.sh again."
        fi
        echo
    } >&2
    exit 1
fi

exec "$PY" "$HERE/launcher.py" "$@"
