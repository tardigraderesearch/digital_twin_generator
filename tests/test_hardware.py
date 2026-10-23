"""Bench safety rules, against robo-prober's in-process firmware model (no hardware)."""
import numpy as np
import pytest

from dtg.hardware import Bench, BenchError, format_channels, parse_channels


class StillCamera:
    """A camera that sees nothing (the emulator has no robot)."""

    def read(self):
        return True, np.zeros((240, 320), np.uint8)

    def release(self):
        pass


@pytest.fixture()
def bench():
    from dtg.camera import CameraIntrinsics
    b = Bench("emulator", camera=StillCamera(),
              intrinsics=CameraIntrinsics(np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]]), np.zeros(5)))
    yield b
    b.close()


def test_connect_reads_only(bench):
    info = bench.connect()
    assert info["mode"] == "disarmed" and not bench.armed
    targets = bench.discover()
    assert any(t.cls == "pca9685" for t in targets)
    assert all(t.channel != 107 for t in targets)          # GPIO 7 is driven by something: never a target
    assert not bench.armed and bench.controller.positions() == {}


def test_arming_needs_the_cutoff_and_estop_latches(bench):
    bench.connect()
    with pytest.raises(BenchError) as err:
        bench.arm(cutoff_confirmed=False)
    assert err.value.reason == "cutoff_unconfirmed" and not bench.armed
    bench.arm(cutoff_confirmed=True)
    assert bench.armed
    bench.estop()
    assert bench.estop_latched and not bench.armed
    with pytest.raises(BenchError) as err:
        bench.arm(cutoff_confirmed=True)
    assert err.value.reason == "estop_latched"
    with pytest.raises(BenchError):
        bench.clear_estop(cutoff_confirmed=False)
    bench.clear_estop(cutoff_confirmed=True)
    assert not bench.estop_latched and bench.controller.hello()["mode"] == "disarmed"


def test_estop_latch_is_detected_on_connect(monkeypatch):
    """A latch left on the device (HELLO mode=estop) is seen on connect; nothing can arm until Clear."""
    import robo_prober.api as api
    real = api.connect

    def latched(*a, **k):
        ctl = real(*a, **k)
        ctl.info = {**ctl.info, "mode": "estop"}
        return ctl
    monkeypatch.setattr(api, "connect", latched)
    b = Bench("emulator")
    b.connect()
    try:
        assert b.estop_latched
        with pytest.raises(BenchError) as err:
            b.arm(cutoff_confirmed=True)
        assert err.value.reason == "estop_latched"
    finally:
        b.close()


def test_wiggle_rules(bench, tmp_path):
    from dtg.markers import parse_markers
    bench.connect()
    bench.discover()
    m = parse_markers({"dictionary": "DICT_4X4_50", "tag_size_mm": 25, "tags": {"base": {"id": 0}}})
    bench.open_camera(m, focus=False)
    bench.start_recording(tmp_path / "obs.jsonl", m, settle_s=0.0)
    with pytest.raises(BenchError) as err:
        bench.wiggle(0, start_us=1500)
    assert err.value.reason == "disarmed"
    bench.arm(cutoff_confirmed=True)
    with pytest.raises(BenchError) as err:
        bench.wiggle(0, start_us=None)
    assert err.value.reason == "bad_start"
    with pytest.raises(BenchError) as err:
        bench.wiggle(107, start_us=1500)
    assert err.value.reason == "pin_driven"
    frames = bench.wiggle(3, start_us=1500, delta_us=500, slew_us_s=2000)
    pulses = [f.pulse_us for f in frames]
    assert pulses == [1500, 1650, 1500, 1350, 1500]           # delta clamped to 150 us
    state = bench.controller.status(3)
    assert state["current_us"] == 1500 and state["active"]
    # the firmware holds the band itself: a move outside start +/- delta is refused by the device
    from robo_prober.transport import ProtocolError
    with pytest.raises(ProtocolError, match="out_of_limits"):
        bench.controller.move(3, 1700)
    bench.disarm()
    assert not bench.armed and not bench.controller.status(3)["active"]


def test_channel_lists():
    assert parse_channels("0-3, 5,104") == [0, 1, 2, 3, 5, 104]
    assert format_channels([0, 1, 2, 3, 5, 104]) == "0-3, 5, 104"
