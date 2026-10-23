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


def test_mark_with_hidden_tags_uses_same_pulse_samples():
    """Found by the virtual bench: tags hidden only in the ROM 1 capture dropped the whole joint."""
    import copy
    log = generate(seed=0)
    rom1 = next(f for f in log.select(phase="rom", channel=7) if f.event == "rom1")
    sample = copy.deepcopy(rom1)                 # the settled slider sample at the same pulse, just before
    sample.event, sample.t = "sample", rom1.t - 0.1
    log.frames.append(sample)
    rom1.poses.pop(4)                            # forearm tag hidden at the mark itself
    kin = infer_kinematics(log, infer_linkage(log, parse_markers(log.header["markers"])))
    elbow = next(j for j in kin.joints if j.channel == 7)
    assert elbow.terminal_range_1.pulse_us == rom1.pulse_us
    assert any("same pulse" in r for r in elbow.reasons)


def test_hidden_rom_ends_are_extrapolated_and_flagged():
    """Found by the virtual bench: a one-tag link seen only in the middle of its
    range (the tag turns away near the ends) used to drop the joint entirely."""
    log = generate(seed=0)
    truth = log.header["truth"]["joints"]["j_elbow"]
    for f in log.select(phase="rom", channel=7):
        if abs(f.pulse_us - truth["zero_us"]) > 250:
            f.poses.pop(4, None)                               # forearm tag invisible beyond +-250 us
    kin = infer_kinematics(log, infer_linkage(log, parse_markers(log.header["markers"])))
    elbow = next(j for j in kin.joints if j.channel == 7)
    assert elbow.status == "flagged" and any("extrapolated" in r for r in elbow.reasons)
    r = next(x for x in score(log, kin) if x["j"].channel == 7)
    assert r["lim"] < 3.0 and r["axis"] < 3.0


def test_pivot_cross_check_flags_a_pivot_the_tag_centres_disagree_with():
    """Two independent pivots: one from tag orientations (pose pairs), one from
    the tag centres (circle fit). Orientations that over-rotate the forearm tag
    about its own centre move the first but not the second: flagged, not ok."""
    from scipy.spatial.transform import Rotation
    log = generate(seed=0)
    kin = infer_kinematics(log, infer_linkage(log, parse_markers(log.header["markers"])))
    elbow = next(j for j in kin.joints if j.channel == 7)
    assert elbow.quality["pivot_disagreement_mm"] < 5.0 and elbow.quality["circle_pivot_sigma_mm"] > 0
    zero = log.header["truth"]["joints"]["j_elbow"]["zero_us"]
    for f in log.frames:
        if f.channel == 7 and 4 in f.poses and f.pulse_us is not None:
            extra = Rotation.from_rotvec(np.radians(0.04 * (f.pulse_us - zero)) * np.array([0, 0, 1.0])).as_matrix()
            f.poses[4] = f.poses[4].copy()
            f.poses[4][:3, :3] = f.poses[4][:3, :3] @ extra       # tag turns about its own centre
    kin = infer_kinematics(log, infer_linkage(log, parse_markers(log.header["markers"])))
    elbow = next(j for j in kin.joints if j.channel == 7)
    assert elbow.status == "flagged"
    assert any("tag centres differ" in r for r in elbow.reasons), elbow.reasons
