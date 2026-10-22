"""Assemble the virtual bench for one seed: robot + scene + camera + firmware.

    world = build_world(seed)          # hidden truth, robot, camera (no firmware yet)
    with world.firmware() as bench:    # the real firmware core behind a pty, PWM -> robot
        ctl = robo_prober.api.Controller(SerialProtocolTransport(bench.port)).open()
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import HEIGHT, WIDTH, VirtualCamera, default_intrinsics, lens_for_distance
from .render import Renderer
from .robot import NEUTRAL_US, RobotTruth, VirtualRobot, link_motions, random_robot
from .scene import TARGET, build_scene, camera_pose, candidate_tags, markers_manifest, select_tags


@dataclass
class World:
    seed: int
    truth: RobotTruth
    robot: VirtualRobot
    camera: VirtualCamera
    T_world_cam: np.ndarray
    K: np.ndarray                 # true intrinsics (renderer)
    K_calibrated: np.ndarray      # what DTG is told (a calibration with small errors)
    manifest: dict
    sharp_lens: float
    t0: float = field(default_factory=time.monotonic)
    log: list = field(default_factory=list)

    def clock(self) -> float:
        return time.monotonic() - self.t0

    @property
    def markers(self):
        from dtg.markers import parse_markers
        return parse_markers(self.manifest)

    def event(self, kind: str, **data) -> None:
        self.log.append({"t": round(self.clock(), 4), "event": kind, **data})

    @contextmanager
    def firmware(self):
        """Start the real firmware core behind a pty; its PWM outputs drive the robot."""
        from robo_prober.firmware_bench import FirmwareBench

        def on_pwm(_t_fw, channel, pulse):
            self.robot.set_pulse(self.clock(), channel, pulse)

        bench = FirmwareBench(on_pwm=on_pwm)
        bench.open()
        self.event("firmware_started", port=bench.port)
        try:
            yield bench
        finally:
            bench.close()

    def write_markers(self, path: Path) -> Path:
        path.write_text(json.dumps(self.manifest, indent=2) + "\n")
        return path

    def intrinsics_dict(self) -> dict:
        return {"camera_matrix": self.K_calibrated.tolist(), "dist_coeffs": [0.0] * 5,
                "image_size": [WIDTH, HEIGHT]}


def _weak_tags(camera, truth, lens, frames: int = 6) -> list[int]:
    """Tags detected in fewer than 4 of ``frames`` preview frames at good focus."""
    import cv2
    from dtg.camera import detect_corners, make_detector
    from dtg.markers import parse_markers
    names = [t["name"] for _, t in sorted(truth.tags.items())]
    markers = parse_markers(markers_manifest(names))
    det = make_detector(markers.dictionary)
    af, old_lens, rate = camera.autofocus, camera.lens, camera.occlusion_rate
    camera.autofocus, camera.lens, camera.occlusion_rate = False, lens, 0.0
    counts = {k: 0 for k in truth.tags}
    for _ in range(frames):
        for k in detect_corners(camera.read()[1], markers, det, clahe=True):
            counts[k] = counts.get(k, 0) + 1
    camera.autofocus, camera.lens, camera.occlusion_rate = af, old_lens, rate
    camera.frames_read = 0
    return [k for k, n in counts.items() if n < 4]


def build_world(seed: int, *, calibration_error: float = 0.004, camera_noise: dict | None = None,
                occlusion_rate: float = 0.004) -> World:
    rng = np.random.default_rng(10_000 + seed)
    T_world_cam = camera_pose(rng)
    K = default_intrinsics(rng)
    truth = random_robot(seed)
    # Stick tags where the camera sees them with the robot at rest (as a person would).
    rest = {j.name: j.joint_deg_at(NEUTRAL_US) + j.rest_offset_deg for j in truth.joints}
    rest_motions = link_motions(truth, rest)
    candidates = candidate_tags(rng, truth.dims, rest_motions, T_world_cam[:3, 3])
    bare = Renderer(build_scene(truth, T_world_cam, np.random.default_rng(0)), K, (WIDTH, HEIGHT),
                    np.linalg.inv(T_world_cam), supersample=1)
    _, depth_rest = bare.render(rest_motions)
    world_box = {}
    exclude: set = set()
    for _ in range(4):
        # The operator sticks tags on, looks at the camera preview and moves any
        # tag the detector does not pick up reliably.
        truth.tags = select_tags(candidates, depth_rest, T_world_cam, K, rng, exclude=exclude)
        scene = build_scene(truth, T_world_cam, rng)
        renderer = Renderer(scene, K, (WIDTH, HEIGHT), np.linalg.inv(T_world_cam))
        robot = VirtualRobot(truth)
        camera = VirtualCamera(robot, renderer, rng=rng, clock=lambda: world_box["w"].clock() if "w" in world_box
                               else 0.0, noise=camera_noise, occlusion_rate=occlusion_rate)
        weak = _weak_tags(camera, truth, lens_for_distance(float(np.linalg.norm(T_world_cam[:3, 3] - TARGET))))
        if not weak:
            break
        exclude |= {truth.tags[k]["cand"] for k in weak}
    # a decent but imperfect calibration: focal length and principal point off a little
    Kc = K.copy()
    Kc[0, 0] *= 1 + rng.normal(0, calibration_error)
    Kc[1, 1] = Kc[0, 0] * (1 + rng.normal(0, calibration_error / 4))
    Kc[0, 2] += rng.normal(0, 2.0)
    Kc[1, 2] += rng.normal(0, 2.0)
    dist = float(np.linalg.norm(T_world_cam[:3, 3] - TARGET))
    names = [t["name"] for _, t in sorted(truth.tags.items())]
    world = World(seed=seed, truth=truth, robot=robot, camera=camera, T_world_cam=T_world_cam, K=K,
                  K_calibrated=Kc, manifest=markers_manifest(names), sharp_lens=lens_for_distance(dist))
    world_box["w"] = world
    return world
