import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dtg.export import comms_map, export_all, validate_urdf


def test_urdf_valid_with_terminal_range_limits(synth, tmp_path):
    _, _, _, kin = synth
    report = export_all(kin, tmp_path)
    root = ET.parse(tmp_path / "robot.urdf").getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    assert set(joints) == {j.name for j in kin.joints}
    for j in kin.joints:
        lim = joints[j.name].find("limit")
        assert float(lim.get("lower")) == pytest.approx(j.lower, abs=1e-8)
        assert float(lim.get("upper")) == pytest.approx(j.upper, abs=1e-8)
        assert np.allclose([float(x) for x in joints[j.name].find("axis").get("xyz").split()], j.axis_model)
    assert report["urdf"]["root"] == "base"
    assert report["verified"] is False


def test_urdf_independent_loader(synth, tmp_path):
    yourdfpy = pytest.importorskip("yourdfpy")
    _, _, _, kin = synth
    export_all(kin, tmp_path)
    robot = yourdfpy.URDF.load(str(tmp_path / "robot.urdf"), load_meshes=False)
    assert set(robot.actuated_joint_names) == {j.name for j in kin.joints}
    # forward kinematics at zero: each link frame sits on its fitted pivot
    for j in kin.joints:
        T = robot.get_transform(j.child, "base")
        assert np.allclose(T[:3, 3], j.pivot_model, atol=1e-9)
    # rotating a joint keeps its own pivot fixed (axis passes through the frame origin)
    j = kin.joints[0]
    robot.update_cfg({j.name: j.upper})
    assert np.allclose(robot.get_transform(j.child, "base")[:3, 3], j.pivot_model, atol=1e-9)


def test_validate_rejects_bad_limits(tmp_path):
    bad = tmp_path / "bad.urdf"
    bad.write_text('<robot name="r"><link name="a"/><link name="b"/><joint name="j" type="revolute">'
                   '<parent link="a"/><child link="b"/><axis xyz="0 0 1"/><limit lower="1" upper="0" effort="1" velocity="1"/>'
                   '</joint></robot>')
    with pytest.raises(ValueError, match="lower < upper"):
        validate_urdf(bad)


def test_comms_map(synth):
    log, _, _, kin = synth
    cmap = comms_map(kin, {"j_upper_arm": "MG996R"})
    rows = {r["channel"]: r for r in cmap["channels"]}
    truth = log.header["truth"]
    for t in truth["joints"].values():
        r = rows[t["channel"]]
        assert r["status"] == "mapped" and r["safe_range_us"] == sorted([t["rom1_us"], t["rom2_us"]])
        assert r["us_per_rad"] == pytest.approx(1 / t["rad_per_us"], rel=0.03)
    for c in truth["unused_channels"]:
        assert rows[c]["status"] == "no_response" and "joint" in rows[c]
    assert rows[0]["servo_model"] == "MG996R"
    json.dumps(cmap)


def test_joint_with_unfitted_parent_is_left_out_not_a_second_root(tmp_path, synth):
    """Found by the virtual bench: the turret joint failed to fit, the forearm
    joint (parent turret) did, and validate_urdf raised 'expected one root
    link', so build_twin wrote nothing at all."""
    import copy
    from dtg.export import export_all
    _, _, _, kin = synth
    broken = copy.deepcopy(kin)
    first = next(j for j in broken.joints if j.parent == broken.root)
    broken.joints = [j for j in broken.joints if j is not first]
    report = export_all(broken, tmp_path / "twin")
    assert report["urdf"]["root"] == broken.root
    assert any("not in the URDF" in f for f in report["flags"])
