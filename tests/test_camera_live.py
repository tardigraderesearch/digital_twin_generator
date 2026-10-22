"""Live-camera boundary without hardware: wiggle against a robo-prober-like
controller, focus sweep against a camera whose sharpness depends on the lens."""
import json

import cv2
import numpy as np
import pytest

from dtg import camera as cam
from dtg.markers import parse_markers

MARKERS = parse_markers({"dictionary": "DICT_4X4_50", "tag_size_mm": 30.0, "tags": {"base": {"id": 0}}})


class FakeController:
    """robo-prober v0.2 semantics: nothing is driven until start(); set_pulse needs a started channel."""

    def __init__(self, fail_at=None):
        self.armed, self.calls, self.fail_at = True, [], fail_at
        self._pos: dict[int, int] = {}

    def position(self, ch):
        return self._pos.get(ch)

    def start(self, ch, pulse):
        assert ch not in self._pos
        self.calls.append(("start", ch, pulse))
        self._pos[ch] = pulse

    def set_pulse(self, ch, pulse):
        if ch not in self._pos:
            raise RuntimeError(f"channel {ch} is not started")
        if self.fail_at is not None and len(self.calls) >= self.fail_at:
            raise RuntimeError("timeout waiting for MOVE")
        self.calls.append(("move", ch, pulse))
        self._pos[ch] = pulse

    def stop(self):
        self.calls.append(("stop",))
        self.armed = False


class StillTracker:
    def capture(self):
        return {}, {}


@pytest.fixture
def recorder(tmp_path):
    return cam.ObservationRecorder(StillTracker(), tmp_path / "log.jsonl", MARKERS, settle_s=0.0, camera={})


def test_wiggle_starts_an_undriven_channel_explicitly(recorder):
    ctl = FakeController()
    frames = recorder.wiggle_channel(ctl, 3, start_us=1500, delta_us=80, slew_us_s=1e9)
    assert ctl.calls[0] == ("start", 3, 1500)
    assert [c[2] for c in ctl.calls[1:]] == [1500, 1580, 1500, 1420, 1500]
    assert [f.event for f in frames] == ["wiggle_base", "wiggle_plus", "wiggle_base", "wiggle_minus", "wiggle_base"]


def test_wiggle_fault_stops_instead_of_moving_back(recorder):
    ctl = FakeController(fail_at=3)
    with pytest.raises(RuntimeError, match="timeout"):
        recorder.wiggle_channel(ctl, 0, start_us=1500, slew_us_s=1e9)
    assert ctl.calls[-1] == ("stop",) and not ctl.armed


class LensCamera:
    """Texture that blurs with distance from the sharp lens position."""

    def __init__(self, sharp=140, autofocus_lens=95):
        rng = np.random.default_rng(0)
        blocks = (rng.random((60, 80)) > 0.5).astype(np.uint8) * 255
        self.base = cv2.resize(blocks, (320, 240), interpolation=cv2.INTER_NEAREST)
        self.sharp, self.af, self.lens = sharp, True, autofocus_lens

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_AUTOFOCUS:
            self.af = bool(value)
            return True
        if prop == cv2.CAP_PROP_FOCUS and not self.af:
            self.lens = value
            return True
        return False

    def read(self):
        sigma = 0.3 + 0.08 * abs(self.lens - self.sharp)
        return True, cv2.GaussianBlur(self.base, (0, 0), sigma)


def test_focus_sweep_locks_on_the_peak():
    c = LensCamera(sharp=142)
    result = cam.focus_sweep(c, positions=range(80, 250, 15), refine_step=5)
    assert abs(result.lens - 142) <= 3 and c.lens == result.lens and not c.af
    assert result.peak_ratio > 3
    json.dumps(result.to_dict())


def test_focus_sweep_refuses_without_a_peak():
    class Flat(LensCamera):
        def read(self):
            return True, self.base
    with pytest.raises(RuntimeError, match="no clear peak"):
        cam.focus_sweep(Flat(), positions=range(80, 250, 30))


def test_tracker_accepts_a_capture_object():
    c = LensCamera()
    tracker = cam.ArucoTracker(c, MARKERS, cam.CameraIntrinsics(np.eye(3), np.zeros(5)))
    assert tracker.cap is c
    tracker.lock_focus(positions=range(80, 250, 20))
    assert tracker.focus.lens == c.lens


def test_focus_sweep_survives_high_gain_noise():
    """Dim room, high ISO: plain Laplacian variance is all noise; the LoG measure still peaks."""
    class Noisy(LensCamera):
        rng = np.random.default_rng(1)

        def read(self):
            ok, img = super().read()
            noisy = img.astype(np.float32) * 0.03 + 60 + self.rng.normal(0, 12, img.shape)
            return True, np.clip(noisy, 0, 255).astype(np.uint8)
    c = Noisy(sharp=150)
    plain = []
    for lens in (90, 150, 210):
        c.af, c.lens = False, lens
        plain.append(cv2.Laplacian(c.read()[1], cv2.CV_64F).var())
    assert max(plain) / min(plain) < 1.5                      # no usable peak without smoothing
    result = cam.focus_sweep(Noisy(sharp=150), positions=range(80, 250, 15))
    assert abs(result.lens - 150) <= 5
