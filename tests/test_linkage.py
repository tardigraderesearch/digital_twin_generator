import numpy as np

from dtg.linkage import infer_linkage
from dtg.markers import parse_markers
from dtg.synthetic import bot1_like_arm, generate


def mapping(result):
    return {c.channel: (c.status, c.child, c.direction) for c in result.channels}


def test_channel_to_joint_map(synth):
    log, _, wiggle, _ = synth
    truth = log.header["truth"]
    got = mapping(wiggle)
    for name, j in truth["joints"].items():
        status, child, _ = got[j["channel"]]
        assert status == "mapped" and child == j["child"], name
    for c in truth["unused_channels"]:
        assert got[c][0] == "no_response"


def test_links_from_signatures(synth):
    _, _, wiggle, _ = synth
    groups = sorted(sorted(l.tags) for l in wiggle.links)
    assert groups == [[0, 5], [1], [2, 3], [4]]
    assert wiggle.link("base").signature == []


def test_direction_consistent_with_rom(synth):
    _, _, wiggle, kin = synth
    for j in kin.joints:
        assert j.quality["wiggle_direction_agrees"], j.name


def test_mapping_stable_across_seeds():
    for seed in range(1, 6):
        log = generate(seed=seed)
        result = infer_linkage(log, parse_markers(log.header["markers"]))
        got = mapping(result)
        for j in log.header["truth"]["joints"].values():
            status, child, _ = got[j["channel"]]
            # never a wrong silent answer: either right, or flagged
            assert (status == "mapped" and child == j["child"]) or status == "needs_confirmation"


def test_two_channels_same_joint_flagged():
    """Channel 6 replays exactly what channel 7 does (two servos on one joint)."""
    import copy
    log = generate(seed=0)
    log.frames = [f for f in log.frames if f.channel != 6]
    for f in [f for f in log.frames if f.phase == "wiggle" and f.channel == 7]:
        twin = copy.deepcopy(f)
        twin.channel = 6
        log.frames.append(twin)
    got = mapping(infer_linkage(log, parse_markers(log.header["markers"])))
    assert got[7][0] == "needs_confirmation" and got[6][0] == "needs_confirmation"


def test_missing_intermediate_tag_flagged():
    arm = bot1_like_arm()
    del arm.tags[1]  # turret untagged
    log = generate(arm, seed=0)
    got = mapping(infer_linkage(log, parse_markers(log.header["markers"])))
    # base yaw moves everything; with no turret tag the shoulder's parent is unknown
    assert got[0][0] == "needs_confirmation"


def test_hidden_link_is_not_called_no_response():
    log = generate(seed=0)
    for f in log.select(phase="wiggle", channel=7):
        f.poses.pop(4, None)  # forearm tag hidden for the whole elbow wiggle
    got = mapping(infer_linkage(log, parse_markers(log.header["markers"])))
    assert got[7][0] == "needs_confirmation"


def test_canonical_axis():
    from dtg.linkage import canonical_axis
    assert np.allclose(canonical_axis([0, -2, 0.1]), -np.array([0, -2, 0.1]) / np.linalg.norm([0, -2, 0.1]))


def test_one_sided_noise_spike_on_unused_channel_is_not_a_joint():
    """Found by the virtual bench: one capture of an unused channel just over the
    noise threshold made it 'mapped'. A real wiggle moves the child both ways."""
    from scipy.spatial.transform import Rotation
    log = generate(seed=0)
    plus = next(f for f in log.select(phase="wiggle", channel=6) if f.event == "wiggle_plus")
    kick = np.eye(4)
    kick[:3, :3] = Rotation.from_rotvec(np.radians(3.0) * np.array([0, 0, 1])).as_matrix()
    plus.poses[4] = plus.poses[4] @ kick                    # a glint on the forearm tag, one side only
    got = mapping(infer_linkage(log, parse_markers(log.header["markers"])))
    assert got[6][0] == "no_response"


def test_big_one_sided_move_is_flagged():
    from scipy.spatial.transform import Rotation
    log = generate(seed=0)
    plus = next(f for f in log.select(phase="wiggle", channel=6) if f.event == "wiggle_plus")
    kick = np.eye(4)
    kick[:3, :3] = Rotation.from_rotvec(np.radians(9) * np.array([0, 0, 1])).as_matrix()
    plus.poses[4] = plus.poses[4] @ kick
    assert mapping(infer_linkage(log, parse_markers(log.header["markers"])))[6][0] == "needs_confirmation"


def test_hidden_root_does_not_merge_moving_tags_into_the_base():
    """Found by the virtual bench: the root tag dropped out of one capture of the
    base-yaw and shoulder wiggles; tags above them were then 'grouped by their
    other channels' into the static base and the elbow came back mapped with
    parent=base. A wrong silent answer is never acceptable."""
    log = generate(seed=0)
    truth = log.header["truth"]["joints"]
    for ch in (truth["j_base_yaw"]["channel"], truth["j_shoulder"]["channel"]):
        frame = next(f for f in log.select(phase="wiggle", channel=ch) if f.event == "wiggle_plus")
        frame.poses.pop(0, None)
        frame.poses.pop(5, None)                            # both base tags hidden in that capture
    result = infer_linkage(log, parse_markers(log.header["markers"]))
    for j in truth.values():
        c = next(c for c in result.channels if c.channel == j["channel"])
        assert c.status == "needs_confirmation" or (c.child == j["child"] and c.parent == j["parent"]), c


def test_second_base_tag_stands_in_for_a_hidden_root():
    """Root tag hidden in one capture of two wiggles: the other base tag (proven
    rigid with it in every channel's wiggle) keeps the map exact."""
    log = generate(seed=0)
    truth = log.header["truth"]["joints"]
    for ch in (truth["j_base_yaw"]["channel"], truth["j_shoulder"]["channel"]):
        frame = next(f for f in log.select(phase="wiggle", channel=ch) if f.event == "wiggle_plus")
        frame.poses.pop(0, None)
    result = infer_linkage(log, parse_markers(log.header["markers"]))
    got = mapping(result)
    for j in truth.values():
        assert got[j["channel"]][:2] == ("mapped", j["child"]), got
