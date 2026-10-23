"""ROM-end occlusion: rom_mark retries a capture that misses the child tag and
says what it saw, so the window can tell the operator; edge-based corners
remove the defocus depth bias of single-tag poses."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dtg import camera as cam
from dtg.markers import parse_markers
from dtg.observations import Frame, load_log

MARKERS = parse_markers({"dictionary": "DICT_4X4_50", "tag_size_mm": 30.0,
                         "tags": {"base": {"id": 0}, "arm": {"id": 1}, "arm_b": {"id": 2}}})


class ScriptedTracker:
    """Returns the scripted tag sets in order (the last one repeats)."""

    def __init__(self, *seen):
        self.seen, self.calls = list(seen), []

    def capture(self, frames=8):
        self.calls.append(frames)
        ids = self.seen.pop(0) if len(self.seen) > 1 else self.seen[0]
        return {k: np.eye(4) for k in ids}, {k: 0.2 for k in ids}


EXPECT = {"child": [1, 2], "parent": [0]}


def test_mark_retries_until_the_child_tag_is_seen(tmp_path):
    tr = ScriptedTracker([0], [0], [0, 2])
    rec = cam.ObservationRecorder(tr, tmp_path / "log.jsonl", MARKERS, settle_s=0, camera={})
    frame = rec.rom_mark(4, 1, 1800, expect=EXPECT)
    assert tr.calls == [8, 16, 16]                       # first capture, then two longer retries
    assert sorted(frame.poses) == [0, 2]
    vis = rec.last_visibility
    assert vis["ok"] and vis["child_seen"] == [2] and vis["child_hidden"] == [1] and vis["retries"] == 2
    log = load_log(tmp_path / "log.jsonl")
    assert [f.event for f in log.frames] == ["rom1"]     # only the kept capture is written


def test_mark_hidden_child_warns_and_keeps_the_best_capture(tmp_path):
    tr = ScriptedTracker([], [0], [])
    rec = cam.ObservationRecorder(tr, tmp_path / "log.jsonl", MARKERS, settle_s=0, camera={})
    frame = rec.rom_mark(4, 2, 1200, expect=EXPECT)
    vis = rec.last_visibility
    assert not vis["ok"] and not vis["child_visible"] and vis["parent_visible"]
    assert sorted(frame.poses) == [0]                    # the retry that saw the parent beats the empty one
    assert "ROM 2" in vis["message"] and "hidden" in vis["message"] and "second tag" in vis["message"]


def test_mark_without_expectation_does_not_retry(tmp_path):
    tr = ScriptedTracker([])
    rec = cam.ObservationRecorder(tr, tmp_path / "log.jsonl", MARKERS, settle_s=0, camera={})
    rec.rom_mark(4, 1, 1500)
    assert tr.calls == [8] and rec.last_visibility is None


def test_expectation_from_the_wiggle_result(tmp_path, synth):
    log, _, wiggle, _ = synth
    rec = cam.ObservationRecorder(ScriptedTracker([]), tmp_path / "log.jsonl", MARKERS, settle_s=0, camera={})
    expect = rec.expect_from_wiggle(wiggle)
    joint = next(c for c in wiggle.channels if c.child)
    assert expect[joint.channel]["child"] == wiggle.link(joint.child).tags
    frame = Frame(t=0, phase="rom", event="rom1", poses={}, channel=joint.channel, pulse_us=1500)
    vis = cam.mark_visibility(frame, wiggle)
    assert not vis["ok"] and vis["child_hidden"] == wiggle.link(joint.child).tags
    frame.poses = {k: np.eye(4) for k in wiggle.link(joint.child).tags + wiggle.link(joint.parent).tags}
    assert cam.mark_visibility(frame, wiggle)["ok"]
    assert cam.mark_visibility(frame, [99])["child_hidden"] == [99]


# ---------------------------------------------------------------- edge-based corners

K = np.array([[1541.0, 0, 960], [0, 1541.0, 540], [0, 0, 1]])
INTR = cam.CameraIntrinsics(K, np.zeros(5), (1920, 1080))


def _blurred_view(T, sigma_px, gamma=2.2):
    """A defocused camera image: blur in linear light, then the ISP gamma."""
    img = cam.render_marker_view(MARKERS, 1, T, INTR, size=(1920, 1080), px=600).astype(np.float32) / 255.0
    lin = cv2.GaussianBlur(img, (0, 0), sigma_px)
    return np.clip(255.0 * lin ** (1 / gamma), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("sigma", [0.8, 1.2])
def test_edge_corners_remove_defocus_depth_bias(sigma):
    T = np.eye(4)
    T[:3, :3] = (Rotation.from_euler("x", 180, degrees=True) * Rotation.from_euler("xy", [25, -15], degrees=True)).as_matrix()
    T[:3, 3] = [0.03, -0.02, 0.6]
    image = _blurred_view(T, sigma)
    raw = cam.detect_corners(image, MARKERS, clahe=True)[1]
    fixed = cam.refine_corners_edges(cam.preprocess(image), raw)
    assert fixed is not None
    ray = T[:3, 3] / np.linalg.norm(T[:3, 3])

    def depth_err_mm(corners):
        Tm = cam.pose_candidates(corners, MARKERS, INTR)[0][0]
        return float((Tm[:3, 3] - T[:3, 3]) @ ray * 1000)

    assert abs(depth_err_mm(raw)) > 2.0                  # the detector's corners: tag looks too far
    assert abs(depth_err_mm(fixed)) < 0.6


def test_edge_corners_give_up_cleanly():
    flat = np.full((200, 200), 128, np.uint8)
    corners = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], float)
    assert cam.refine_corners_edges(flat, corners) is None
    assert cam.refine_corners_edges(flat, corners / 20) is None   # too small to measure
