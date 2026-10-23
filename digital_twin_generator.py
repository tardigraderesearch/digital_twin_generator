"""Digital Twin Generator. To use:  python digital_twin_generator.py  [command]

One file, any OS. It checks Python and Tk, and if the libraries are missing it asks
before installing them into a private .venv next to this file (nothing system-wide).
When everything is there it says so and starts. With no command it opens the app
window; any dtg command (doctor, health, wiggle, rom, build, servos, ...) passes straight through.

-y / --yes, or a non-interactive run, installs without asking.
robo-prober (the open hardware library DTG drives the controller with) is installed from a
sibling checkout (../robo_prober) when there is one, else from its public repository;
DTG_ROBO_PROBER overrides the source (a path or a pip requirement).
Standard library only above the install step.
"""

import hashlib
import os
import platform
import shutil
import subprocess
import sys

APP = "digital-twin-generator"
MODULE = "dtg"
DISPLAY_NAME = "Digital Twin Generator"
LIBRARIES = "robo-prober, numpy, SciPy, trimesh, matplotlib, OpenCV, Pillow"
IMPORTS = "import dtg, robo_prober, cv2, numpy, scipy, trimesh, matplotlib, PIL"
NO_WINDOW_COMMANDS = {"doctor", "health", "synth", "tags", "wiggle", "rom", "register", "segment", "parts",
                      "build", "servos", "session", "estop", "-h", "--help", "--version"}
ROBO_PROBER_GIT = "robo-prober @ git+https://github.com/tardigraderesearch/robo_prober.git"
MIN_PYTHON = (3, 11)

HERE = os.path.dirname(os.path.abspath(__file__))
SIBLING_ROBO_PROBER = os.path.join(os.path.dirname(HERE), "robo_prober")
VENV = os.path.join(HERE, ".venv")
WINDOWS = os.name == "nt"
VPY = os.path.join(VENV, "Scripts", "python.exe") if WINDOWS else os.path.join(VENV, "bin", "python")
STAMP = os.path.join(VENV, ".%s-installed" % APP)
SYSTEM = platform.system()
RUN = "python digital_twin_generator.py"


class Problem(Exception):
    def __init__(self, message, *lines):
        super().__init__(message)
        self.message = message
        self.lines = lines


def say(message):
    print("[%s] %s" % (APP, message), file=sys.stderr, flush=True)


def python_fix():
    if WINDOWS:
        return [
            "Install it from https://www.python.org/downloads/windows/",
            'In the installer, tick "Add python.exe to PATH" and keep "tcl/tk and IDLE" ticked.',
            "Then run %s again." % RUN,
        ]
    if SYSTEM == "Darwin":
        return [
            "Install Python 3.12 from https://www.python.org/downloads/macos/",
            "(or with Homebrew: brew install python@3.12 python-tk@3.12), then run python3 digital_twin_generator.py again.",
        ]
    return [
        "Install Python 3.11 or newer with your package manager, for example:",
        "  Debian/Ubuntu: sudo apt install python3 python3-venv python3-tk",
        "  Fedora:        sudo dnf install python3 python3-tkinter",
        "  Arch:          sudo pacman -S python tk",
        "  openSUSE:      sudo zypper install python3 python3-tk",
        "  Alpine:        sudo apk add python3 py3-tkinter",
        "Then run python3 digital_twin_generator.py again.",
    ]


def tk_fix():
    if WINDOWS:
        return [
            "Re-run the Python installer from https://www.python.org/downloads/windows/",
            'choose "Modify" and tick "tcl/tk and IDLE". Then run python digital_twin_generator.py again.',
        ]
    if SYSTEM == "Darwin":
        return [
            "Install Python from https://www.python.org/downloads/macos/ (it includes Tk),",
            "or with Homebrew: brew install python-tk",
        ]
    return [
        "Debian/Ubuntu: sudo apt install python3-tk",
        "Fedora:        sudo dnf install python3-tkinter",
        "Arch:          sudo pacman -S tk",
        "openSUSE:      sudo zypper install python3-tk",
        "Alpine:        sudo apk add py3-tkinter",
    ]


def venv_fix():
    if WINDOWS:
        return ["Reinstall Python from https://www.python.org/downloads/windows/ and try again."]
    return [
        "Debian/Ubuntu: sudo apt install python3-venv",
        "Other systems: reinstall Python from python.org or your package manager.",
    ]


def quiet_ok(cmd):
    """Run cmd silently; True when it exits 0."""
    try:
        return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def check_python():
    if sys.version_info < MIN_PYTHON:
        raise Problem(
            "Python %d.%d or newer is needed (this is %s)." % (MIN_PYTHON + (platform.python_version(),)),
            *python_fix()
        )
    say("Using Python %s at %s" % (platform.python_version(), sys.executable))


def check_tk(args):
    needs_window = not args or args[0] not in NO_WINDOW_COMMANDS
    try:
        import tkinter  # noqa: F401
    except ImportError:
        if needs_window:
            raise Problem(
                "This Python has no tkinter, which the app window needs.",
                "Fix it with:",
                *(tk_fix() + ["(Without Tk you can still use the tools: %s health scan.obj)" % RUN])
            )
        say("Note: tkinter is missing; fine for '%s', but the window will not open." % args[0])


ASSUME_YES = False


def confirm(what):
    """Ask before installing. Yes by default; non-interactive runs and -y skip the question."""
    if ASSUME_YES or not sys.stdin.isatty():
        say("%s (auto-yes)." % what)
        return
    try:
        answer = input("[%s] %s Install now? [Y/n] " % (APP, what)).strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("", "y", "yes"):
        say("Nothing installed. Run %s again when you're ready." % RUN)
        sys.exit(0)


def ensure_venv():
    if os.path.isfile(VPY) and quiet_ok([VPY, "-c", "import sys"]):
        return
    confirm("%s needs its Python libraries (%s) in a private .venv." % (DISPLAY_NAME, LIBRARIES))
    say("First run: creating a private Python environment in .venv (one time only)...")
    shutil.rmtree(VENV, ignore_errors=True)
    if subprocess.call([sys.executable, "-m", "venv", VENV]) != 0 or not os.path.isfile(VPY):
        shutil.rmtree(VENV, ignore_errors=True)
        raise Problem("Could not create the virtual environment.", *venv_fix())


def robo_prober_source():
    """Where robo-prober comes from: DTG_ROBO_PROBER, a sibling checkout, or the public repo."""
    override = os.environ.get("DTG_ROBO_PROBER")
    if override:
        return override
    if os.path.isfile(os.path.join(SIBLING_ROBO_PROBER, "pyproject.toml")):
        return SIBLING_ROBO_PROBER
    return ROBO_PROBER_GIT


def wanted_stamp():
    try:
        with open(os.path.join(HERE, "pyproject.toml"), "rb") as handle:
            data = handle.read()
    except OSError:
        raise Problem("Could not read pyproject.toml next to %s." % RUN)
    # The venv's own Python version, so a rebuilt interpreter triggers a reinstall.
    version = subprocess.run(
        [VPY, "-c", "import sys; sys.stdout.write(sys.version)"],
        stdout=subprocess.PIPE, check=False,
    ).stdout
    # The venv's Python version and robo-prober's source too: changing either reinstalls.
    return hashlib.sha256(data + version + robo_prober_source().encode()).hexdigest()


def ensure_installed():
    want = wanted_stamp()
    try:
        with open(STAMP, encoding="utf-8") as handle:
            have = handle.read().strip()
    except OSError:
        have = ""
    if want == have and quiet_ok([VPY, "-P", "-c", IMPORTS]):
        say("All dependencies present. Good to go.")
        return
    if have:
        confirm("%s's libraries need updating." % DISPLAY_NAME)
    say("Installing %s and its libraries (%s)..." % (APP, LIBRARIES))
    say("This needs internet once and takes a minute or two.")
    pip = [VPY, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", "--upgrade"]
    if subprocess.call(pip + ["pip"]) != 0:
        say("Could not upgrade pip; continuing with the bundled one.")
    source = robo_prober_source()
    say("robo-prober from %s" % source)
    rp = ["--editable", source] if os.path.isdir(source) else [source]
    if subprocess.call(pip + rp) != 0 or subprocess.call(pip + ["--editable", HERE]) != 0:
        raise Problem(
            "Installing the Python libraries failed (see the pip messages above).",
            "Check your internet connection and run %s again." % RUN,
            "To start over completely, delete the .venv folder.",
        )
    if subprocess.call([VPY, "-P", "-c", IMPORTS]) != 0:
        raise Problem(
            "The libraries installed but do not import (see the error above).",
            "Delete the .venv folder and run %s again." % RUN,
        )
    with open(STAMP, "w", encoding="utf-8") as handle:
        handle.write(want + "\n")
    say("Install complete. Good to go.")


def launch(args):
    if not args:
        say("Opening the %s window..." % DISPLAY_NAME)
    # -P: never import from the current folder (nothing here may shadow an installed package).
    cmd = [VPY, "-P", "-m", MODULE] + args
    if not WINDOWS:
        os.execv(VPY, cmd)
    code = subprocess.call(cmd)
    if code != 0:
        print("", file=sys.stderr, flush=True)
        say("The app exited with code %d." % code)
    return code


def main(args):
    global ASSUME_YES
    if args[:1] in (["-y"], ["--yes"]):
        ASSUME_YES, args = True, args[1:]
    try:
        check_python()
        check_tk(args)
        ensure_venv()
        ensure_installed()
    except Problem as problem:
        sys.stdout.flush()
        lines = ["", "[%s] PROBLEM: %s" % (APP, problem.message)]
        lines += ["    " + line for line in problem.lines]
        print("\n".join(lines) + "\n", file=sys.stderr, flush=True)
        return 1
    return launch(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
