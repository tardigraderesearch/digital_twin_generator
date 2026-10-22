"""Camera observation log: the data that the wiggle and ROM steps produce.

The camera is the only sensor used during wiggle and ROM (the 3D scanner is
never used while anything moves). Each settled capture becomes one Frame:

    {"t": 12.3, "phase": "wiggle", "channel": 4, "pulse_us": 1580,
     "event": "wiggle_plus", "poses": {"0": [[...4x4...]], "1": ...},
     "reproj_px": {"0": 0.21, "1": 0.35}}

* poses are T_camera_tag (4x4, metres), from OpenCV ArUco pose estimation
  with markers.json tag_size_mm, averaged over the settled frames.
* phase "reference": the robot in the pose it was scanned in (Scan A). This
  defines joint zero in the exported model.
* phase "wiggle": one channel at a time, tiny pulse change around the current
  pulse. events: wiggle_base, wiggle_plus, wiggle_minus.
* phase "rom": per discovered joint, the operator's slider sweep. events:
  sample (slider settled), rom1, rom2 (Set ROM 1 / Set ROM 2 pressed).

A log is JSON Lines: line 1 is a header, every other line a Frame. The same
file is written live (dtg.camera.ObservationRecorder) and read back here for
replay, so inference code cannot tell live from replayed data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SCHEMA = "dtg.observations/1"
PHASES = ("reference", "wiggle", "rom")
EVENTS = ("reference", "wiggle_base", "wiggle_plus", "wiggle_minus", "sample", "rom1", "rom2")


class ObservationError(ValueError):
    pass


@dataclass
class Frame:
    t: float
    phase: str
    event: str
    poses: dict[int, np.ndarray]
    channel: int | None = None
    pulse_us: float | None = None
    reproj_px: dict[int, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"t": float(self.t), "phase": self.phase, "event": self.event, "channel": self.channel,
                "pulse_us": None if self.pulse_us is None else float(self.pulse_us),
                "poses": {str(k): np.asarray(v).round(9).tolist() for k, v in self.poses.items()},
                "reproj_px": {str(k): float(v) for k, v in self.reproj_px.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "Frame":
        if d.get("phase") not in PHASES or d.get("event") not in EVENTS:
            raise ObservationError(f"bad frame phase/event: {d.get('phase')}/{d.get('event')}")
        poses = {}
        for k, v in d.get("poses", {}).items():
            m = np.asarray(v, dtype=float)
            if m.shape != (4, 4) or not np.isfinite(m).all():
                raise ObservationError(f"tag {k}: pose must be a finite 4x4 matrix")
            poses[int(k)] = m
        return cls(t=float(d["t"]), phase=d["phase"], event=d["event"], poses=poses,
                   channel=None if d.get("channel") is None else int(d["channel"]),
                   pulse_us=None if d.get("pulse_us") is None else float(d["pulse_us"]),
                   reproj_px={int(k): float(v) for k, v in d.get("reproj_px", {}).items()})


@dataclass
class ObservationLog:
    header: dict
    frames: list[Frame]

    def select(self, phase: str | None = None, channel: int | None = None, event: str | None = None) -> list[Frame]:
        return [f for f in self.frames if (phase is None or f.phase == phase)
                and (channel is None or f.channel == channel) and (event is None or f.event == event)]

    @property
    def channels(self) -> list[int]:
        return sorted({f.channel for f in self.frames if f.channel is not None})

    @property
    def tag_ids(self) -> list[int]:
        return sorted({k for f in self.frames for k in f.poses})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            out.write(json.dumps(self.header) + "\n")
            for f in self.frames:
                out.write(json.dumps(f.to_json()) + "\n")
        return path


def new_header(source: str, markers: dict | None = None, camera: dict | None = None, **extra) -> dict:
    return {"schema": SCHEMA, "source": source, "units": "m", "pose_convention": "T_camera_tag",
            "markers": markers or {}, "camera": camera or {}, **extra}


def load_log(path: str | Path) -> ObservationLog:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ObservationError("empty observation log")
    header = json.loads(lines[0])
    if header.get("schema") != SCHEMA:
        raise ObservationError(f"not a {SCHEMA} log")
    return ObservationLog(header, [Frame.from_json(json.loads(x)) for x in lines[1:] if x.strip()])


def replay(log: ObservationLog, *, realtime: bool = False) -> Iterator[Frame]:
    """Yield frames in time order (optionally at recorded speed) so code that
    consumes a live stream can be driven by a recorded log."""
    import time
    frames = sorted(log.frames, key=lambda f: f.t)
    start_wall, start_t = time.monotonic(), frames[0].t if frames else 0.0
    for f in frames:
        if realtime:
            delay = (f.t - start_t) - (time.monotonic() - start_wall)
            if delay > 0:
                time.sleep(delay)
        yield f


# ---------------------------------------------------------------- rigid-motion helpers

def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def mean_pose(poses: Iterable[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rigid transforms (rotation projected back to SO(3))."""
    poses = list(poses)
    if not poses:
        raise ObservationError("no poses to average")
    Rs = np.stack([p[:3, :3] for p in poses]).sum(axis=0)
    u, _, vt = np.linalg.svd(Rs)
    R = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    return out
