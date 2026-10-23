"""Live camera feed -> observation log (API boundary for the hardware steps).

NOT exercised on hardware yet. Pose estimation itself is tested on rendered
marker images; capture, wiggle driving and ROM hooks are written against
robo-prober v0.2's Controller API (set_pulse / position / armed) and will
be validated when a camera and an armed controller are on the bench.

    tracker  = ArucoTracker(0, markers, CameraIntrinsics.load("camera.json"))
    tracker.lock_focus()                          # manual focus sweep; autofocus is not trusted
    recorder = ObservationRecorder(tracker, "runs/bot1/observations.jsonl", markers)
    recorder.reference()                          # robot in the scanned pose
    for ch in range(16):                          # WIGGLE, controller already armed by the operator
        recorder.wiggle_channel(controller, ch, start_us=1500, delta_us=80)
    # ROM: robo-prober's ROM window calls these as the operator works
    recorder.rom_sample(ch, pulse_us)            # slider settled
    recorder.rom_mark(ch, 1, pulse_us)           # Set ROM 1 pressed (2 for Set ROM 2)

Focus: on the real bench (OAK-D, 2026-10-22) autofocus locked on the
background and blurred every tag below detection. lock_focus() sweeps the
lens, scores Laplacian-variance sharpness on the robot region, locks manual
focus at the peak and refuses to continue without a clear peak. Detection
uses CLAHE and sub-pixel corner refinement, as on that bench.

Safety: this module never arms anything. wiggle_channel refuses an unarmed
controller, moves one channel at a time by a small delta (default 80 us,
clamped to 150 us), waits for the controller's slew-limited move to settle
and ends at the start pulse. On any error it stops the controller (all
outputs released) instead of attempting a recovery move.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .markers import Markers
from .observations import Frame, new_header

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


def make_detector(dictionary: str, *, subpix: bool = True):
    """ArUco detector; sub-pixel corner refinement on by default (it was on for the real bench data)."""
    import cv2
    params = cv2.aruco.DetectorParameters()
    if subpix:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(_dictionary(dictionary), params)


def preprocess(image, *, clahe: bool = False):
    """Greyscale, optionally contrast-equalised (CLAHE) for dim, noisy rooms."""
    import cv2
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def tag_object_points(size_m: float) -> np.ndarray:
    """Marker corners in the tag frame (OpenCV order: TL, TR, BR, BL; z out of the face)."""
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def detect_corners(image, markers: Markers, detector=None, *, clahe: bool = False) -> dict[int, np.ndarray]:
    """Image corners (4x2, OpenCV order) of every markers.json tag found in one image."""
    gray = preprocess(image, clahe=clahe)
    detector = detector or make_detector(markers.dictionary)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    wanted = set(markers.tags.values())
    return {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.ravel()) if int(i) in wanted}


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def refine_corners_edges(image, corners: np.ndarray, *, gamma: float = 2.2, samples: int = 16,
                         half_width: float | None = None) -> np.ndarray | None:
    """Unbiased tag corners from the four edges of the black square.

    The detector's corners (and cornerSubPix, which is made for chessboard
    saddle points) sit inside a blurred square's true corners, so the tag
    looks smaller and therefore farther away: on the virtual bench that is a
    1-2% depth error that differs per tag with its defocus (5-10 mm at
    0.6 m), and it moves every fitted pivot along the camera ray. Here each
    edge is located where the intensity crosses halfway between the black
    border and the white quiet zone, which a symmetric blur does not move
    (computed on linearised intensity, ``gamma``), a line is fitted through
    each edge and the corners are the line intersections.

    Returns the refined 4x2 corners (same order), or None when an edge
    cannot be measured (too small, no contrast, clipped by the image).
    """
    gray = image if image.ndim == 2 else __import__("cv2").cvtColor(image, __import__("cv2").COLOR_BGR2GRAY)
    lin = (gray.astype(np.float32) / 255.0) ** gamma
    c = np.asarray(corners, float).reshape(4, 2)
    centre = c.mean(axis=0)
    side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    if side < 12:
        return None
    hw = half_width or max(2.0, min(6.0, side / 14.0))
    steps = np.linspace(-hw, hw, int(hw * 4) + 1)
    lines = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        if n @ (centre - (a + b) / 2) > 0:
            n = -n                                  # n points outward (dark -> light)
        pts = []
        for s in np.linspace(0.2, 0.8, samples):
            p = a + (b - a) * s
            prof = _bilinear(lin, p[0] + steps * n[0], p[1] + steps * n[1])
            dark, light = prof[: len(prof) // 4].mean(), prof[-(len(prof) // 4):].mean()
            if light - dark < 0.05:
                continue
            level = (dark + light) / 2
            above = np.nonzero(prof >= level)[0]
            if not len(above) or above[0] == 0:
                continue
            k = above[0]
            t = steps[k - 1] + (level - prof[k - 1]) / (prof[k] - prof[k - 1]) * (steps[k] - steps[k - 1])
            pts.append(p + t * n)
        if len(pts) < samples // 2:
            return None
        pts = np.asarray(pts)
        m = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - m)
        lines.append((m, vt[0]))
    out = np.zeros((4, 2))
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]       # corner i joins edge i-1 and edge i
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        s = np.linalg.solve(A, p2 - p1)
        out[i] = p1 + s[0] * d1
    if np.abs(out - c).max() > hw + 2:
        return None                                 # refinement ran away: keep the detector's corners
    return out


def pose_from_corners(img: np.ndarray, markers: Markers, intrinsics: CameraIntrinsics):
    """(T_camera_tag, reprojection px) from one tag's four image corners, or None."""
    import cv2
    if markers.tag_size_mm is None:
        raise ValueError("markers.json has no tag_size_mm; pose needs the printed black-square size")
    obj = tag_object_points(markers.tag_size_mm / 1000.0)
    ok, rvec, tvec = cv2.solvePnP(obj, img, intrinsics.K, intrinsics.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics.K, intrinsics.dist)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, float(np.sqrt(((proj.reshape(4, 2) - img) ** 2).sum(axis=1).mean()))


def pose_candidates(img: np.ndarray, markers: Markers, intrinsics: CameraIntrinsics) -> list:
    """Both IPPE solutions for one tag, best first: [(T_camera_tag, reprojection px), ...].

    A square tag seen with noisy corners has two poses that fit almost
    equally well (mirror-like tilts, often 20-40 deg apart); which one wins
    can flip from frame to frame."""
    import cv2
    obj = tag_object_points(markers.tag_size_mm / 1000.0)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, intrinsics.K, intrinsics.dist,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return []
    out = []
    for i in range(int(n)):
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvecs[i])[0]
        T[:3, 3] = np.asarray(tvecs[i]).ravel()
        out.append((T, float(np.ravel(errs)[i]) if errs is not None else float("nan")))
    return sorted(out, key=lambda x: x[1])


def estimate_poses(image, markers: Markers, intrinsics: CameraIntrinsics, detector=None, *, clahe: bool = False):
    """Detect the markers.json tags in one image. Returns ({id: T_camera_tag}, {id: reprojection px})."""
    if markers.tag_size_mm is None:
        raise ValueError("markers.json has no tag_size_mm; pose needs the printed black-square size")
    poses, errors = {}, {}
    for i, img in detect_corners(image, markers, detector, clahe=clahe).items():
        fit = pose_from_corners(img, markers, intrinsics)
        if fit is not None:
            poses[i], errors[i] = fit
    return poses, errors


def _rot_deg(R) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def sharpness(image, roi=None, *, presmooth: float = 1.5) -> float:
    """Focus measure: variance of the Laplacian of a lightly smoothed image
    (LoG) over roi=(x, y, w, h), default the middle half of the frame.

    The smoothing matters in dim rooms: at high gain, plain Laplacian
    variance is dominated by sensor noise and shows no focus peak at all.
    """
    import cv2
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x, y, rw, rh = roi or (w // 4, h // 4, w // 2, h // 2)
    crop = gray[y:y + rh, x:x + rw].astype(np.float32)
    if presmooth > 0:
        crop = cv2.GaussianBlur(crop, (0, 0), presmooth)
    return float(cv2.Laplacian(crop, cv2.CV_32F).var())


@dataclass
class FocusResult:
    lens: int
    scores: dict
    peak_ratio: float          # best score / median score: a clear peak is well above 1

    def to_dict(self) -> dict:
        return {"lens": self.lens, "peak_ratio": self.peak_ratio, "scores": {str(k): v for k, v in self.scores.items()}}


def focus_sweep(cap, *, positions=range(80, 250, 15), refine_step: int = 5, roi=None, frames: int = 3,
                settle_frames: int = 2, set_focus=None, min_peak_ratio: float = 1.5) -> FocusResult:
    """Sweep the lens, score sharpness on the robot region, lock manual focus at the peak.

    ``cap`` is anything with read() (cv2.VideoCapture, or an adapter for a
    camera SDK). Lens control defaults to the OpenCV properties
    (CAP_PROP_AUTOFOCUS=0, CAP_PROP_FOCUS); pass ``set_focus(position)`` for
    cameras that need their own call (e.g. an OAK-D CameraControl
    setManualFocus). Raises RuntimeError when no position stands out, because
    measuring with blurred tags gives wrong poses or none.
    """
    import cv2
    if set_focus is None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        def set_focus(position):
            if not cap.set(cv2.CAP_PROP_FOCUS, position):
                raise RuntimeError("camera refused manual focus (is autofocus still on?)")

    def score(position) -> float:
        set_focus(position)
        images = []
        for i in range(settle_frames + frames):
            ok, image = cap.read()
            if ok and i >= settle_frames:
                images.append(preprocess(image).astype(np.float32))
        if not images:
            raise RuntimeError("camera returned no frames during the focus sweep")
        return sharpness(np.mean(images, axis=0), roi)   # averaging frames also beats noise down

    scores = {int(p): score(int(p)) for p in positions}
    best = max(scores, key=scores.get)
    step = int(np.median(np.diff(sorted(scores)))) if len(scores) > 1 else refine_step
    for p in range(best - step + refine_step, best + step, refine_step):
        if p not in scores and 0 <= p <= 255:
            scores[p] = score(p)
    best = max(scores, key=scores.get)
    ratio = scores[best] / max(float(np.median(list(scores.values()))), 1e-9)
    set_focus(best)
    if ratio < min_peak_ratio:
        raise RuntimeError(f"focus sweep found no clear peak (best/median {ratio:.2f}); check the region and light")
    return FocusResult(lens=best, scores=dict(sorted(scores.items())), peak_ratio=float(ratio))


class ArucoTracker:
    """Grab settled frames from a camera and average tag poses.

    ``source`` is an OpenCV device index / URL, or an already-open capture
    object with read() (a camera SDK adapter, or a virtual camera).
    """

    def __init__(self, source, markers: Markers, intrinsics: CameraIntrinsics, *, max_reproj_px: float = 1.5,
                 clahe: bool = True, subpix: bool = True):
        import cv2
        self.cap = source if hasattr(source, "read") else cv2.VideoCapture(source)
        if hasattr(self.cap, "isOpened") and not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {source!r}")
        self.markers, self.intrinsics, self.max_reproj_px = markers, intrinsics, max_reproj_px
        self.clahe = clahe
        self.edge_refine = True    # unbiased edge-based corners (see refine_corners_edges)
        self.gamma = 2.2           # camera ISP gamma used to linearise intensities for the edge fit
        self.last_refined: set = set()
        self.detector = make_detector(markers.dictionary, subpix=subpix)
        self.focus: FocusResult | None = None
        self.ambiguity_ratio = 1.5
        self._last: dict[int, np.ndarray] = {}

    def lock_focus(self, **kwargs) -> FocusResult:
        """Manual focus sweep + lock (see focus_sweep). Run before the reference capture."""
        self.focus = focus_sweep(self.cap, **kwargs)
        return self.focus

    def capture(self, frames: int = 8):
        """Settled capture: each tag's corners are averaged (per-corner median)
        over the frames it was seen in, then its pose is solved once.

        Averaging corners rather than poses matters with noisy images, and so
        does the square-tag ambiguity (see pose_candidates): the solution kept
        is the one closest to the tag's last accepted pose; a tag seen for the
        first time is accepted only if one solution clearly fits better
        (``ambiguity_ratio``). A tag must be seen in at least half the frames.
        """
        per_tag: dict[int, list] = {}
        refined: dict[int, list] = {}
        for _ in range(frames):
            ok, image = self.cap.read()
            if not ok:
                continue
            found = detect_corners(image, self.markers, self.detector, clahe=self.clahe)
            gray = preprocess(image) if found and self.edge_refine else None
            for k, img in found.items():
                per_tag.setdefault(k, []).append(img)
                if gray is not None:
                    r = refine_corners_edges(gray, img, gamma=self.gamma)
                    if r is not None:
                        refined.setdefault(k, []).append(r)
        need = max(1, frames // 2)
        poses, errs = {}, {}
        self.last_refined = set()
        for k, seen in per_tag.items():
            if len(seen) < need:
                continue
            if len(refined.get(k, ())) >= need:      # never mix refined and raw corners in one median
                seen = refined[k]
                self.last_refined.add(k)
            fit = self._choose(k, pose_candidates(np.median(np.stack(seen), axis=0), self.markers, self.intrinsics))
            if fit is not None and fit[1] <= self.max_reproj_px:   # drop poor fits (partial, smeared)
                poses[k], errs[k] = fit
                self._last[k] = fit[0]
        return poses, errs

    def _choose(self, tag: int, candidates: list):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        (Ta, ea), (Tb, eb) = candidates[:2]
        last = self._last.get(tag)
        if last is not None:
            da = _rot_deg(last[:3, :3].T @ Ta[:3, :3])
            db = _rot_deg(last[:3, :3].T @ Tb[:3, :3])
            if abs(da - db) > 5.0:
                return (Ta, ea) if da < db else (Tb, eb)
        if eb >= self.ambiguity_ratio * max(ea, 1e-6):
            return Ta, ea
        return None        # ambiguous with no history: skip this capture rather than guess

    def close(self):
        self.cap.release()


class ObservationRecorder:
    """Append settled captures to a JSON Lines observation log as they happen."""

    def __init__(self, tracker, path, markers: Markers, *, settle_s: float = 0.6, camera: dict | None = None):
        self.tracker, self.path, self.settle_s = tracker, Path(path), settle_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        camera = camera or (tracker.intrinsics.to_dict() if hasattr(tracker, "intrinsics") else {})
        if getattr(tracker, "focus", None) is not None:
            camera = {**camera, "focus": tracker.focus.to_dict()}
        header = new_header("live", markers=markers.to_dict(), camera=camera)
        self.path.write_text(json.dumps(header) + "\n")

    expect: dict | None = None      # channel -> {"child": [tag ids], "parent": [tag ids]}
    last_visibility: dict | None = None

    def _capture(self, phase, event, channel=None, pulse=None, frames: int | None = None) -> Frame:
        if frames is None:
            poses, errs = self.tracker.capture()
        else:
            try:
                poses, errs = self.tracker.capture(frames=frames)
            except TypeError:       # a tracker without a frame count
                poses, errs = self.tracker.capture()
        return Frame(t=time.monotonic() - self.t0, phase=phase, event=event, poses=poses,
                     channel=channel, pulse_us=pulse, reproj_px=errs)

    def _append(self, frame: Frame) -> Frame:
        with self.path.open("a") as out:
            out.write(json.dumps(frame.to_json()) + "\n")
        return frame

    def _write(self, phase, event, channel=None, pulse=None) -> Frame:
        return self._append(self._capture(phase, event, channel, pulse))

    def expect_from_wiggle(self, wiggle) -> dict:
        """Remember which tags each joint's parent and child links carry (from
        the wiggle result), so rom_mark can check the child is in view."""
        self.expect = expected_tags(wiggle)
        return self.expect

    def reference(self) -> Frame:
        """Capture with the robot in the pose it was scanned in (joint zero)."""
        return self._write("reference", "reference")

    def wiggle_channel(self, controller, channel: int, *, start_us: int, delta_us: int = 80,
                       slew_us_s: float = 200.0) -> list[Frame]:
        """base, +delta, base, -delta, base on one channel, a settled capture at each.

        A channel that is not driven yet is started at ``start_us`` first
        (robo-prober v0.2 START: the first pulse is explicit, so choose the
        pulse the servo is already at). The wiggle ends at ``start_us``. On
        any error the controller is stopped (every output released) and the
        error re-raised: no recovery move after a fault, as in robo-prober.
        """
        if not getattr(controller, "armed", False):
            raise RuntimeError("controller is not armed: the operator arms it (robo-prober) before the wiggle")
        delta = int(min(abs(delta_us), MAX_WIGGLE_US))
        wait = delta / slew_us_s + self.settle_s
        frames = []
        try:
            if hasattr(controller, "position") and controller.position(channel) is None:
                controller.start(channel, start_us)
            for event, pulse in (("wiggle_base", start_us), ("wiggle_plus", start_us + delta),
                                 ("wiggle_base", start_us), ("wiggle_minus", start_us - delta),
                                 ("wiggle_base", start_us)):
                controller.set_pulse(channel, pulse)
                time.sleep(wait)
                frames.append(self._write("wiggle", event, channel, pulse))
        except BaseException:
            try:
                controller.stop()
            except Exception:  # noqa: BLE001 - the firmware watchdog releases outputs too
                pass
            raise
        return frames

    def rom_sample(self, channel: int, pulse_us: float) -> Frame:
        time.sleep(self.settle_s)
        return self._write("rom", "sample", channel, pulse_us)

    def rom_mark(self, channel: int, which: int, pulse_us: float, *, expect: dict | list | None = None,
                 retries: int = 2, retry_frames: int = 16) -> Frame:
        """Capture at Set ROM 1 / Set ROM 2.

        Near the ends of a large range a tag often turns away or slips
        behind another link. When the child link's tags (``expect``, or what
        ``expect_from_wiggle`` stored for this channel) are not seen, the
        capture is retried up to ``retries`` times with ``retry_frames``
        frames each (a hand or flicker passes; a longer average detects a
        marginal tag), and the best capture is kept. ``last_visibility``
        then says what was seen (see ``mark_visibility``) so the window can
        tell the operator to back off a little or add a second tag; the fit
        still extrapolates a hidden end, flagged."""
        if which not in (1, 2):
            raise ValueError("ROM mark is 1 or 2")
        time.sleep(self.settle_s)
        want = expect if expect is not None else (self.expect or {}).get(channel)
        event = f"rom{which}"
        frame = self._capture("rom", event, channel, pulse_us)
        vis = mark_visibility(frame, want) if want is not None else None
        tries = 0
        while vis is not None and not vis["child_visible"] and tries < retries:
            tries += 1
            again = self._capture("rom", event, channel, pulse_us, frames=retry_frames)
            again_vis = mark_visibility(again, want)
            if len(again_vis["child_seen"]) + len(again_vis["parent_seen"]) > \
                    len(vis["child_seen"]) + len(vis["parent_seen"]):
                frame, vis = again, again_vis
        if vis is not None:
            vis["retries"] = tries
            vis["mark"] = f"ROM {which}"
            vis["message"] = _visibility_message(vis)
        self.last_visibility = vis
        return self._append(frame)


def expected_tags(wiggle) -> dict:
    """channel -> {"child": [tag ids], "parent": [tag ids]} from a WiggleResult."""
    out = {}
    for c in wiggle.channels:
        if c.child is None or c.parent is None:
            continue
        try:
            out[c.channel] = {"child": list(wiggle.link(c.child).tags), "parent": list(wiggle.link(c.parent).tags)}
        except StopIteration:
            continue
    return out


def mark_visibility(frame: Frame, expect) -> dict:
    """Which of a joint's tags a capture saw.

    ``expect`` is {"child": [ids], "parent": [ids]}, a plain list of child
    tag ids, or a WiggleResult (looked up by the frame's channel). Returns
    {"channel", "visible", "child_seen", "child_hidden", "parent_seen",
    "child_visible", "parent_visible", "ok", "message"}. ``ok`` needs at
    least one child and one parent tag: then the joint position at this
    capture is measured, not extrapolated."""
    if hasattr(expect, "channels"):
        expect = expected_tags(expect).get(frame.channel, {})
    if isinstance(expect, (list, tuple, set)):
        expect = {"child": list(expect), "parent": None}
    child = [int(k) for k in (expect or {}).get("child") or []]
    parent = (expect or {}).get("parent")
    seen = sorted(int(k) for k in frame.poses)
    out = {"channel": frame.channel, "visible": seen,
           "child_seen": [k for k in child if k in frame.poses],
           "child_hidden": [k for k in child if k not in frame.poses],
           "parent_seen": [int(k) for k in parent if k in frame.poses] if parent is not None else []}
    out["child_visible"] = bool(out["child_seen"])
    out["parent_visible"] = bool(out["parent_seen"]) if parent is not None else True
    out["ok"] = out["child_visible"] and out["parent_visible"]
    out["message"] = _visibility_message(out)
    return out


def _visibility_message(vis: dict) -> str:
    mark = vis.get("mark") or "this capture"
    if vis["ok"]:
        return f"{mark}: child tag(s) {vis['child_seen']} seen"
    if not vis["child_visible"]:
        return (f"{mark}: child tag(s) {vis['child_hidden']} hidden at this end. Back off a little and mark again, "
                "or stick a second tag on that link facing the camera at this end. If you keep this mark, the end "
                "is extrapolated from the pulse fit and the joint is flagged.")
    return (f"{mark}: no parent tag seen; the joint position here cannot be measured. Check the camera view "
            "of the parent link.")


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
