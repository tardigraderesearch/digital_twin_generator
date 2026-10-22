"""Virtual camera: renders the virtual robot and degrades it like a real webcam.

Calibrated against a real bench capture (OAK-D RGB CAM_A at 1080p, factory
fx ~1541 px, dim room, ISO 1600, 60 ms exposure, DICT_4X4_100 tags at desk
distance, 2026-10-22):

* tags 42-107 px across; detection 29-30/30 frames for big/frontal tags,
  15-22/30 for small/oblique ones; tag-centre jitter 0.12-0.6 px (frontal)
  and 1.5-2.8 px (small/oblique) on a static scene;
* autofocus locked on the background and blurred every tag below detection;
  a manual lens sweep found one sharp peak (lens 140).

The model: linear radiance from sim.render -> exposure and mains flicker ->
depth-dependent defocus (lens position vs subject distance) -> shot + read
noise at high gain -> ISP gamma -> 8-bit, with an occasional hand passing in
front of the robot. ``read()`` / ``set()`` / ``get()`` follow
cv2.VideoCapture, so DTG's live tracker consumes it unchanged.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .render import Renderer

WIDTH, HEIGHT = 1920, 1080
FX = 1541.0
LENS_MIN, LENS_MAX = 0, 255


def lens_for_distance(d_m: float) -> float:
    """Sharp lens position for a subject distance (OAK-like: ~140 at 0.58 m, ~110 at 1.25 m)."""
    return 84.0 + 32.5 / max(d_m, 0.05)


_GAMMA = (255.0 * (np.arange(4096) / 4095.0) ** (1 / 2.2)).round().astype(np.uint8)


def _gamma(lin: np.ndarray) -> np.ndarray:
    """ISP gamma to 8 bit through a 12-bit table."""
    return _GAMMA[(lin * 4095).astype(np.int16)]


def default_intrinsics(rng=None) -> np.ndarray:
    cx, cy = WIDTH / 2 - 0.5, HEIGHT / 2 - 0.5
    if rng is not None:
        cx += rng.normal(0, 4)
        cy += rng.normal(0, 4)
    return np.array([[FX, 0, cx], [0, FX, cy], [0, 0, 1.0]])


class VirtualCamera:
    """A cv2.VideoCapture look-alike over a live VirtualRobot."""

    def __init__(self, robot, renderer: Renderer, *, rng, clock=time.monotonic, noise: dict | None = None,
                 occlusion_rate: float = 0.02, autofocus_target: str = "background"):
        self.robot, self.renderer, self.rng, self.clock = robot, renderer, rng, clock
        # Dim room at high gain: few electrons per pixel (calibrated with sim.calibrate
        # against the OAK-D bench numbers above; the light level varies per run).
        self.noise = {"full_well": float(rng.uniform(18, 40)), "read_e": 3.0, "exposure": float(rng.uniform(0.55, 0.85)),
                      "flicker": 0.02, "optics_sigma": 1.0, "blur_per_lens": 0.06, **(noise or {})}
        self.occlusion_rate = occlusion_rate
        self.autofocus = True
        # Autofocus on a webcam/OAK in this scene latches onto the busy wall.
        # (it hunts past the wall to near infinity in a dim room: see the bench notes above)
        self.lens = float(lens_for_distance(3.0 if autofocus_target == "background" else 0.58))
        self._occluder: dict | None = None
        self._cache_key = None
        self._cache = None
        self._lock = threading.Lock()
        self.frames_read = 0
        self.last_clean = None   # last noise-free radiance (for the showcase video)
        self.tap = None          # optional tap(t, frame) for recording a video

    # ---------------------------------------------------- cv2.VideoCapture API
    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def set(self, prop, value) -> bool:
        if prop == cv2.CAP_PROP_AUTOFOCUS:
            self.autofocus = bool(value)
            return True
        if prop == cv2.CAP_PROP_FOCUS:
            if self.autofocus:
                return False            # like UVC: manual focus is ignored while AF is on
            self.lens = float(np.clip(value, LENS_MIN, LENS_MAX))
            return True
        return False

    def get(self, prop):
        return {cv2.CAP_PROP_FOCUS: self.lens, cv2.CAP_PROP_AUTOFOCUS: float(self.autofocus),
                cv2.CAP_PROP_FRAME_WIDTH: WIDTH, cv2.CAP_PROP_FRAME_HEIGHT: HEIGHT}.get(prop, 0.0)

    def read(self):
        with self._lock:
            frame = self._frame()
            self.frames_read += 1
        if self.tap is not None:
            self.tap(self.clock(), frame)
        return True, frame

    # ---------------------------------------------------- image formation
    def _clean(self):
        from .robot import link_motions
        q = self.robot.snapshot(self.clock())
        key = tuple(round(v, 3) for v in q.values())
        if key != self._cache_key:
            self._cache = self.renderer.render(link_motions(self.robot.truth, q))
            self._cache_key = key
        return self._cache

    def _update_occluder(self):
        if self._occluder is None:
            if self.rng.random() < self.occlusion_rate:
                W, H = self.renderer.size
                self._occluder = {"frames": int(self.rng.integers(3, 25)),
                                  "c": np.array([self.rng.uniform(0.3, 0.7) * W, self.rng.uniform(0.25, 0.8) * H]),
                                  "v": self.rng.normal(0, 25, 2), "ax": (int(self.rng.integers(40, 110)),
                                                                         int(self.rng.integers(80, 200))),
                                  "ang": float(self.rng.uniform(0, 180))}
        elif self._occluder["frames"] <= 0:
            self._occluder = None
        else:
            self._occluder["frames"] -= 1
            self._occluder["c"] += self._occluder["v"]

    def _frame(self) -> np.ndarray:
        rad, depth = self._clean()
        self._update_occluder()
        o = self._occluder
        key = (self._cache_key, round(self.lens, 2),
               None if o is None else (tuple(np.round(o["c"], 1)), o["ax"], round(o["ang"], 1)))
        if key != getattr(self, "_blur_key", None):
            rad = rad.copy()
            if o is not None:   # a hand / forearm close to the lens
                depth = depth.copy()
                centre = tuple(int(v) for v in o["c"])
                cv2.ellipse(rad, centre, o["ax"], o["ang"], 0, 360, (0.38, 0.47, 0.62), -1)
                cv2.ellipse(depth, centre, o["ax"], o["ang"], 0, 360, 0.30, -1)
            self.last_clean = rad
            self._blurred = self._defocus(rad, depth)
            self._blur_key = key
        n = self.noise
        # sensor: shot noise + read noise in electrons at high gain
        e = self._blurred * np.float32(n["exposure"] * (1 + self.rng.normal(0, n["flicker"])) * n["full_well"])
        np.clip(e, 0, None, out=e)
        noise = np.empty_like(e)
        cv2.randn(noise, 0.0, 1.0)
        e += noise * np.sqrt(e + np.float32(n["read_e"] ** 2))
        lin = np.clip(e * np.float32(1.0 / n["full_well"]), 0, 1)
        return _gamma(lin)

    def _defocus(self, img, depth):
        n = self.noise
        finite = np.where(np.isfinite(depth), depth, 3.0)
        bins = np.array([0.30, 0.45, 0.55, 0.65, 0.80, 1.0, 1.3, 1.8, 3.0])
        idx = np.abs(finite[..., None] - bins).argmin(axis=-1)
        out = np.empty_like(img)
        for b in np.unique(idx):
            sigma = n["optics_sigma"] + n["blur_per_lens"] * abs(self.lens - lens_for_distance(bins[b]))
            blurred = cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0.05 else img
            mask = idx == b
            out[mask] = blurred[mask]
        return out
