import numpy as np
import pytest

from dtg.kinematics import infer_kinematics
from dtg.linkage import infer_linkage
from dtg.markers import parse_markers
from dtg.synthetic import bot1_like_arm, generate


def score(log, kin):
    truth = {t["channel"]: t for t in log.header["truth"]["joints"].values()}
    rows = []
    for j in kin.joints:
        t = truth[j.channel]
        ta = np.asarray(t["axis_root"])
        sign = np.sign(np.dot(j.axis_model, ta))
        axis_err = np.degrees(np.arccos(min(1.0, abs(np.dot(j.axis_model, ta)))))
        d = np.asarray(j.pivot_model) - np.asarray(t["pivot_root"])
        pivot_err = np.linalg.norm(d - ta * (d @ ta)) * 1000
        lo, hi = sorted([sign * t["rom1_rad"], sign * t["rom2_rad"]])
        rows.append(dict(j=j, t=t, sign=sign, axis=axis_err, pivot=pivot_err,
                         lim=np.degrees(max(abs(j.lower - lo), abs(j.upper - hi)))))
    return rows


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_axis_pivot_limits_accuracy(seed):
    log = generate(seed=seed)
    kin = infer_kinematics(log, infer_linkage(log, parse_markers(log.header["markers"])))
    rows = score(log, kin)
    assert len(rows) == 3
    for r in rows:
        assert r["axis"] < 2.0, r
        assert r["pivot"] < 3.0, r
        assert r["lim"] < 2.0, r
        assert r["j"].rad_per_us == pytest.approx(r["t"]["rad_per_us"], rel=0.03)
        assert r["j"].status == "ok", r["j"].reasons


def test_terminal_ranges_and_pulse_map(synth):
    log, _, _, kin = synth
    for r in score(log, kin):
        j, t = r["j"], r["t"]
        assert j.terminal_range_1.pulse_us == t["rom1_us"] and j.terminal_range_2.pulse_us == t["rom2_us"]
        assert j.terminal_range_1.position == pytest.approx(r["sign"] * t["rom1_rad"], abs=np.radians(2))
        assert j.zero_pulse_us == pytest.approx(t["zero_us"], abs=25)
        # direction: positive pulse turns +axis when truth direction (in the fitted axis sign) is positive
        assert j.direction == int(np.sign(r["sign"] * t["direction_about_axis"]))
        assert len(j.terminal_range_1.child_pose_in_parent) == 4


def test_noise_free_is_near_exact():
    log = generate(seed=0, rot_noise_deg=0, trans_noise_mm=0, depth_noise_mm=0, dropout=0, wobble_deg=0)
    kin = infer_kinematics(log, infer_linkage(log, parse_markers(log.header["markers"])))
    for r in score(log, kin):
        assert r["axis"] < 1e-4 and r["pivot"] < 1e-3 and r["lim"] < 1e-4


def test_end_stop_is_flagged():
    arm = bot1_like_arm()
    log = generate(arm, seed=0)
    # simulate a hard stop: the forearm stops moving 10 deg before ROM 1 although the pulse keeps going
    from dtg.observations import inv
    from scipy.spatial.transform import Rotation
    j = next(x for x in arm.joints if x.channel == 7)
    rom1 = [f for f in log.select(phase="rom", channel=7) if f.event == "rom1"][0]
    base, tag = rom1.poses[2], rom1.poses[4]
    rel = inv(base) @ tag
    R = Rotation.from_rotvec(np.radians(12) * np.array([0, 1, 0])).as_matrix()  # undo some motion
    rel[:3, :3] = R @ rel[:3, :3]
    rom1.poses[4] = base @ rel
    kin = infer_kinematics(log, infer_linkage(log, parse_markers(log.header["markers"])))
    elbow = next(x for x in kin.joints if x.channel == j.channel)
    assert elbow.status == "flagged"
    assert any("ROM 1" in r for r in elbow.reasons)
