import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dtg.camera import CameraIntrinsics, ObservationRecorder, estimate_poses, render_marker_view
from dtg.markers import parse_markers
from dtg.parts import load_servo_library, servo_body_mesh, servo_spec

REQUIRED = {"SG90", "MG90S", "MG996R", "DS3218", "DS3225", "LD-20MG"}


def test_servo_library_entries_have_sources():
    lib = load_servo_library()
    models = {s["model"] for s in lib["servos"]}
    assert REQUIRED <= models and len(models) >= 9
    for s in lib["servos"]:
        assert s["sources"], s["model"]
        assert isinstance(s["dims_verified"], bool) and isinstance(s["mass_verified"], bool)
        if s["dims_verified"]:
            assert s["body_mm"] and "manufacturer" in s["dims_source"]


def test_servo_spec_and_placement():
    spec = servo_spec("MG996R")
    assert spec.effort_nm == pytest.approx(11 * 0.0980665)
    assert any("not manufacturer-verified" in a for a in spec.assumptions)
    pivot, axis = np.array([0.1, 0.0, 0.2]), np.array([0, 1.0, 0])
    body = servo_body_mesh(spec, pivot, axis, [1, 0, 0])
    along = (body.vertices - pivot) @ axis
    assert along.max() == pytest.approx(0, abs=1e-9)                  # spline top on the pivot
    assert -along.min() == pytest.approx(spec.height_with_spline_mm / 1000, rel=1e-6)
    radial = np.linalg.norm((body.vertices - pivot) - np.outer(along, axis), axis=1)
    assert radial.max() > 0.02                                          # case extends sideways


def test_unknown_servo():
    with pytest.raises(KeyError):
        servo_spec("NOPE-1")


MARKERS = parse_markers({"dictionary": "DICT_4X4_50", "tag_size_mm": 25, "tags": {"base": {"id": 3}}})
INTR = CameraIntrinsics(np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]]), np.zeros(5))


@pytest.mark.parametrize("angles", [[25, -15, 10], [-30, 20, 70]])
def test_pose_estimation_on_rendered_marker(angles):
    T = np.eye(4)
    T[:3, :3] = (Rotation.from_euler("x", 180, degrees=True) * Rotation.from_euler("xyz", angles, degrees=True)).as_matrix()
    T[:3, 3] = [0.01, -0.005, 0.25]
    poses, errs = estimate_poses(render_marker_view(MARKERS, 3, T, INTR), MARKERS, INTR)
    D = np.linalg.inv(T) @ poses[3]
    assert np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))) < 2.0
    assert np.linalg.norm(D[:3, 3]) < 0.005 and errs[3] < 1.0


class FakeTracker:
    intrinsics = INTR

    def __init__(self):
        self.calls = 0

    def capture(self, frames=8):
        self.calls += 1
        return {3: np.eye(4)}, {3: 0.2}


class FakeController:
    def __init__(self, armed=True, fail_at=None):
        self.armed, self.sent, self.fail_at = armed, [], fail_at

    def set_pulse(self, channel, us):
        self.sent.append((channel, us))
        if self.fail_at is not None and len(self.sent) == self.fail_at:
            raise RuntimeError("link lost")

    def stop(self):
        self.sent.append("STOP")


def test_recorder_wiggle_safety(tmp_path):
    rec = ObservationRecorder(FakeTracker(), tmp_path / "obs.jsonl", MARKERS, settle_s=0)
    with pytest.raises(RuntimeError, match="not armed"):
        rec.wiggle_channel(FakeController(armed=False), 0, start_us=1500)
    ctl = FakeController()
    frames = rec.wiggle_channel(ctl, 2, start_us=1500, delta_us=500, slew_us_s=1e9)
    assert max(abs(us - 1500) for _, us in ctl.sent) == 150      # delta clamped
    assert ctl.sent[-1] == (2, 1500) and len(frames) == 5
    ctl = FakeController(fail_at=2)
    with pytest.raises(RuntimeError):
        rec.wiggle_channel(ctl, 3, start_us=1400, slew_us_s=1e9)
    assert ctl.sent[-2:] == [(3, 1480), "STOP"]                    # stopped on error, no recovery move
    from dtg.observations import load_log
    log = load_log(tmp_path / "obs.jsonl")
    assert log.header["source"] == "live" and len(log.select(phase="wiggle")) == 6  # 5 + 1 before the abort
