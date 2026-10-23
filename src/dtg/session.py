"""A twin-building session: one folder that holds every step's inputs and results.

The GUI and the command line work on the same folder, so any step can be
run (or re-run) from either. Sessions are saved automatically under
``~/digital_twin_generator_maps/<robot>-<timestamp>/`` and never overwritten:

    session.json        what each step did (status, summary, inputs)
    markers.json        copy of aruco-tag-maker's markers.json (step 2)
    camera.json         camera intrinsics used for the live capture
    observations.jsonl  camera log: reference, wiggle and ROM captures (steps 3-4)
    health/             scan health check: health.json, health.png, cleaned copy (step 1)
    twin/               URDF, meshes, comms map, report (steps 5-8)

The original scan is only ever read; its path is recorded, it is never copied
over or modified.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "dtg.session/1"
MAPS_DIR = Path.home() / "digital_twin_generator_maps"


def maps_dir() -> Path:
    """Where sessions are saved (DTG_MAPS_DIR overrides, e.g. for tests)."""
    return Path(os.environ.get("DTG_MAPS_DIR") or MAPS_DIR)

# (key, tab title, one line saying what to do next)
STEPS = [
    ("scan", "1. Scan", "Load the scan and run the health check. The original file is never changed."),
    ("tags", "2. Tags", "Load the markers.json that aruco-tag-maker wrote for the tags on the robot."),
    ("wiggle", "3. Wiggle", "Connect, arm, then wiggle each output a little: the camera sees which joint moves."),
    ("rom", "4. Range of motion", "Move each joint to its two safe ends and press Set ROM 1 / Set ROM 2."),
    ("register", "5. Register", "Line the camera model up with the scan: tag centres or 3 matching points."),
    ("segment", "6. Cut links", "Cut the scan into links by what moved together."),
    ("parts", "7. Missing parts", "Add servo bodies from the library, stand-ins and CAD where the scan has nothing."),
    ("export", "8. Export", "Write the URDF, the comms map (channel -> joint) and the report."),
]
STEP_KEYS = [k for k, _, _ in STEPS]


class SessionError(ValueError):
    pass


def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-.")[:60]
    return name or "robot"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Session:
    root: Path
    data: dict

    # ------------------------------------------------------------ create / open
    @classmethod
    def new(cls, name: str = "robot", base: Path | None = None) -> "Session":
        base = Path(base or maps_dir())
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = base / f"{safe_name(name)}-{stamp}"
        n = 2
        while root.exists():                      # never overwrite a session
            root = base / f"{safe_name(name)}-{stamp}-{n}"
            n += 1
        root.mkdir(parents=True)
        s = cls(root, {"schema": SCHEMA, "name": safe_name(name), "created": _now(),
                       "steps": {k: {"status": "pending"} for k in STEP_KEYS},
                       "inputs": {}, "confirmations": {"channels": {}, "joints": {}}})
        s.save()
        return s

    @classmethod
    def open(cls, path: str | Path) -> "Session":
        path = Path(path)
        if str(path) == "latest":
            path = latest()
        if path.is_file():
            path = path.parent
        f = path / "session.json"
        if not f.is_file():
            raise SessionError(f"{path} is not a DTG session (no session.json)")
        data = json.loads(f.read_text())
        if data.get("schema") != SCHEMA:
            raise SessionError(f"{f}: unknown schema {data.get('schema')!r}")
        for k in STEP_KEYS:
            data.setdefault("steps", {}).setdefault(k, {"status": "pending"})
        data.setdefault("inputs", {})
        data.setdefault("confirmations", {"channels": {}, "joints": {}})
        return cls(path, data)

    def save(self) -> Path:
        f = self.root / "session.json"
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        os.replace(tmp, f)
        return f

    # ------------------------------------------------------------ steps
    def step(self, key: str) -> dict:
        if key not in STEP_KEYS:
            raise SessionError(f"unknown step {key!r}")
        return self.data["steps"][key]

    def set_step(self, key: str, status: str, summary: str = "", **info) -> None:
        if status not in ("pending", "done", "flagged", "skipped", "failed"):
            raise SessionError(f"bad step status {status!r}")
        self.data["steps"][key] = {"status": status, "summary": summary, "at": _now(), **info}
        self.save()

    def set_input(self, key: str, value) -> None:
        self.data["inputs"][key] = value
        self.save()

    def input(self, key: str, default=None):
        return self.data["inputs"].get(key, default)

    # ------------------------------------------------------------ files
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def markers_path(self) -> Path:
        return self.path("markers.json")

    @property
    def log_path(self) -> Path:
        return self.path("observations.jsonl")

    @property
    def camera_path(self) -> Path:
        return self.path("camera.json")

    @property
    def twin_dir(self) -> Path:
        return self.path("twin")

    @property
    def scan_path(self) -> Path | None:
        p = self.input("scan")
        return Path(p) if p else None

    def has_log(self) -> bool:
        return self.log_path.is_file() and self.log_path.stat().st_size > 0

    def import_markers(self, src: str | Path) -> Path:
        from .markers import load_markers
        load_markers(src)                         # validate before copying
        if Path(src).resolve() != self.markers_path.resolve():
            shutil.copyfile(src, self.markers_path)
        self.set_input("markers_source", str(Path(src).resolve()))
        return self.markers_path

    def import_log(self, src: str | Path) -> Path:
        """Use a recorded camera log (a synthetic or earlier capture) for steps 3-4."""
        from .observations import load_log
        load_log(src)
        if Path(src).resolve() != self.log_path.resolve():
            shutil.copyfile(src, self.log_path)
        self.set_input("log_source", str(Path(src).resolve()))
        return self.log_path

    # ------------------------------------------------------------ operator answers
    def confirm_channel(self, channel: int, verdict: str, note: str = "") -> None:
        """Operator's answer for a wiggle result: confirmed | rejected | (None clears)."""
        if verdict not in ("confirmed", "rejected", None):
            raise SessionError("verdict is confirmed or rejected")
        table = self.data["confirmations"]["channels"]
        if verdict is None:
            table.pop(str(channel), None)
        else:
            table[str(channel)] = {"verdict": verdict, "note": note, "at": _now()}
        self.save()

    def confirm_joint(self, joint: str, verdict: str, note: str = "") -> None:
        """Operator's answer for a flagged ROM fit (axis / limits look right on the robot)."""
        if verdict not in ("confirmed", "rejected", None):
            raise SessionError("verdict is confirmed or rejected")
        table = self.data["confirmations"]["joints"]
        if verdict is None:
            table.pop(joint, None)
        else:
            table[joint] = {"verdict": verdict, "note": note, "at": _now()}
        self.save()

    @property
    def confirmations(self) -> dict:
        return self.data["confirmations"]

    def summary_lines(self) -> list[str]:
        out = [f"session {self.root}"]
        for key, title, _ in STEPS:
            st = self.step(key)
            out.append(f"  {title:<20} {st['status']:<8} {st.get('summary', '')}")
        return out


def latest(base: Path | None = None) -> Path:
    base = Path(base or maps_dir())
    found = sorted((p for p in base.glob("*/session.json")), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SessionError(f"no sessions in {base}; start one with: dtg session new NAME")
    return found[-1].parent
