"""Live camera feed -> observation log (API boundary for the hardware steps).

NOT exercised on hardware yet. Pose estimation itself is tested on rendered
marker images; capture, wiggle driving and ROM hooks are written against
robo-prober v0.2's Controller API (set_pulse / position / armed) and will
be validated when a camera and an armed controller are on the bench.

    tracker  = ArucoTracker(0, markers, CameraIntrinsics.load("camera.json"))
    recorder = ObservationRecorder(tracker, "runs/bot1/observations.jsonl", markers)
    recorder.reference()                          # robot in the scanned pose
    for ch in range(16):                          # WIGGLE, controller already armed by the operator
        recorder.wiggle_channel(controller, ch, start_us=1500, delta_us=80)
    # ROM: robo-prober's ROM window calls these as the operator works
    recorder.rom_sample(ch, pulse_us)            # slider settled
    recorder.rom_mark(ch, 1, pulse_us)           # Set ROM 1 pressed (2 for Set ROM 2)

Safety: this module never arms anything. wiggle_channel refuses an unarmed
controller, moves one channel at a time by a small delta (default 80 us,
clamped to 150 us), waits for the controller's slew-limited move to settle
and always returns the channel to its start pulse, also on error.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .markers import Markers
from .observations import Frame, mean_pose, new_header

MAX_WIGGLE_US = 150


@dataclass
class CameraIntrinsics:
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int] | None = None

    @classmethod
    def load(cls, path) -> "CameraIntrinsics":
        d = json.loads(Path(path).read_text())
        return cls(np.asarray(d["camera_matrix"], float), np.asarray(d.get("dist_coeffs", [0] * 5), float),
                   tuple(d["image_size"]) if d.get("image_size") else None)

    def to_dict(self) -> dict:
        return {"camera_matrix": self.K.tolist(), "dist_coeffs": self.dist.ravel().tolist(),
                "image_size": list(self.size) if self.size else None}


def _dictionary(name: str):
    import cv2
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def tag_object_points(size_m: float) -> np.ndarray:
    """Marker corners in the tag frame (OpenCV order: TL, TR, BR, BL; z out of the face)."""
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def estimate_poses(image, markers: Markers, intrinsics: CameraIntrinsics, detector=None):
    """Detect the markers.json tags in one image. Returns ({id: T_camera_tag}, {id: reprojection px})."""
    import cv2
    if markers.tag_size_mm is None:
        raise ValueError("markers.json has no tag_size_mm; pose needs the printed black-square size")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = detector or cv2.aruco.ArucoDetector(_dictionary(markers.dictionary), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    poses, errors = {}, {}
    if ids is None:
        return poses, errors
    obj = tag_object_points(markers.tag_size_mm / 1000.0)
    wanted = set(markers.tags.values())
    for c, i in zip(corners, ids.ravel()):
        if int(i) not in wanted:
            continue
        img = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, intrinsics.K, intrinsics.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics.K, intrinsics.dist)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        poses[int(i)] = T
        errors[int(i)] = float(np.sqrt(((proj.reshape(4, 2) - img) ** 2).sum(axis=1).mean()))
    return poses, errors


class ArucoTracker:
    """Grab settled frames from an OpenCV camera and average tag poses."""

    def __init__(self, source, markers: Markers, intrinsics: CameraIntrinsics, *, max_reproj_px: float = 1.5):
        import cv2
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {source!r}")
        self.markers, self.intrinsics, self.max_reproj_px = markers, intrinsics, max_reproj_px
        self.detector = cv2.aruco.ArucoDetector(_dictionary(markers.dictionary), cv2.aruco.DetectorParameters())

    def capture(self, frames: int = 8):
        per_tag: dict[int, list] = {}
        errs: dict[int, list] = {}
        for _ in range(frames):
            ok, image = self.cap.read()
            if not ok:
                continue
            poses, e = estimate_poses(image, self.markers, self.intrinsics, self.detector)
            for k, T in poses.items():
                if e[k] <= self.max_reproj_px:   # drop high-reprojection (likely flipped) detections
                    per_tag.setdefault(k, []).append(T)
                    errs.setdefault(k, []).append(e[k])
        need = max(1, frames // 2)
        return ({k: mean_pose(v) for k, v in per_tag.items() if len(v) >= need},
                {k: float(np.mean(errs[k])) for k, v in per_tag.items() if len(v) >= need})

    def close(self):
        self.cap.release()


class ObservationRecorder:
    """Append settled captures to a JSON Lines observation log as they happen."""

    def __init__(self, tracker, path, markers: Markers, *, settle_s: float = 0.6, camera: dict | None = None):
        self.tracker, self.path, self.settle_s = tracker, Path(path), settle_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        header = new_header("live", markers=markers.to_dict(),
                            camera=camera or (tracker.intrinsics.to_dict() if hasattr(tracker, "intrinsics") else {}))
        self.path.write_text(json.dumps(header) + "\n")

    def _write(self, phase, event, channel=None, pulse=None) -> Frame:
        poses, errs = self.tracker.capture()
        frame = Frame(t=time.monotonic() - self.t0, phase=phase, event=event, poses=poses,
                      channel=channel, pulse_us=pulse, reproj_px=errs)
        with self.path.open("a") as out:
            out.write(json.dumps(frame.to_json()) + "\n")
        return frame

    def reference(self) -> Frame:
        """Capture with the robot in the pose it was scanned in (joint zero)."""
        return self._write("reference", "reference")

    def wiggle_channel(self, controller, channel: int, *, start_us: int, delta_us: int = 80,
                       slew_us_s: float = 200.0) -> list[Frame]:
        if not getattr(controller, "armed", False):
            raise RuntimeError("controller is not armed: the operator arms it (robo-prober) before the wiggle")
        delta = int(min(abs(delta_us), MAX_WIGGLE_US))
        wait = delta / slew_us_s + self.settle_s
        frames = []
        try:
            for event, pulse in (("wiggle_base", start_us), ("wiggle_plus", start_us + delta),
                                 ("wiggle_base", start_us), ("wiggle_minus", start_us - delta),
                                 ("wiggle_base", start_us)):
                controller.set_pulse(channel, pulse)
                time.sleep(wait)
                frames.append(self._write("wiggle", event, channel, pulse))
        finally:
            controller.set_pulse(channel, start_us)
        return frames

    def rom_sample(self, channel: int, pulse_us: float) -> Frame:
        time.sleep(self.settle_s)
        return self._write("rom", "sample", channel, pulse_us)

    def rom_mark(self, channel: int, which: int, pulse_us: float) -> Frame:
        if which not in (1, 2):
            raise ValueError("ROM mark is 1 or 2")
        time.sleep(self.settle_s)
        return self._write("rom", f"rom{which}", channel, pulse_us)


def render_marker_view(markers: Markers, tag_id: int, T_camera_tag, intrinsics: CameraIntrinsics,
                       size=(640, 480), px: int = 400):
    """Synthetic camera image of one tag at a known pose (for tests / demos)."""
    import cv2
    side = markers.tag_size_mm / 1000.0
    # marker image with a 1-module white quiet zone so the detector sees the border
    bits = int(markers.dictionary.split("_")[1].split("X")[0]) + 2
    module = px // bits
    mk = cv2.aruco.generateImageMarker(_dictionary(markers.dictionary), tag_id, module * bits)
    canvas = np.full((module * (bits + 2),) * 2, 255, np.uint8)
    canvas[module:module + module * bits, module:module + module * bits] = mk
    h = side / 2 * (bits + 2) / bits
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    R, t = np.asarray(T_camera_tag)[:3, :3], np.asarray(T_camera_tag)[:3, 3]
    rvec = cv2.Rodrigues(R)[0]
    img_pts, _ = cv2.projectPoints(obj, rvec, t, intrinsics.K, intrinsics.dist)
    n = canvas.shape[0] - 1
    src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
    H = cv2.getPerspectiveTransform(src, img_pts.reshape(4, 2).astype(np.float32))
    out = cv2.warpPerspective(canvas, H, size, borderValue=255, flags=cv2.INTER_LINEAR)
    return out
