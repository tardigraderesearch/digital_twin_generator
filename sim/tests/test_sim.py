"""Tests for the virtual bench itself (the harness, not DTG or robo-prober)."""
import os

import cv2
import numpy as np
import pytest

from sim.robot import NEUTRAL_US, VirtualRobot, link_motions, random_robot


def test_truth_is_randomized_and_consistent():
    a, b = random_robot(1), random_robot(2)
    for t in (a, b):
        chans = [j.channel for j in t.joints]
        assert len(set(chans)) == 3 and sorted(chans + t.unused_channels) == list(range(8))
        for j in t.joints:
            assert j.stop_lo_deg < j.joint_deg_at(NEUTRAL_US) < j.stop_hi_deg
            assert abs(j.pulse_for(j.joint_deg_at(1700)) - 1700) < 1e-6
    assert [j.channel for j in a.joints] != [j.channel for j in b.joints] or \
        [j.horn_offset_deg for j in a.joints] != [j.horn_offset_deg for j in b.joints]


def test_servo_lag_slew_release_and_start_jump():
    t = random_robot(3)
    robot = VirtualRobot(t)
    j = t.joints[1]
    q0 = robot.joint_angles()[j.name]
    robot.set_pulse(0.0, j.channel, 1500)
    assert robot.starts[0]["jump_deg"] == pytest.approx(abs(j.rest_offset_deg), abs=1e-6)
    robot.set_pulse(1.0, j.channel, 1600)
    robot.advance(1.01)
    assert robot.joint_angles()[j.name] != pytest.approx(j.joint_deg_at(1600), abs=0.5)   # lag
    robot.advance(3.0)
    assert robot.joint_angles()[j.name] == pytest.approx(j.joint_deg_at(1600), abs=j.backlash_deg / 2 + 0.05)
    robot.set_pulse(3.0, j.channel, None)
    held = robot.joint_angles()[j.name]
    robot.advance(5.0)
    assert robot.joint_angles()[j.name] == held and held != q0


def test_hard_stop_stalls_and_is_logged_once():
    t = random_robot(4)
    robot = VirtualRobot(t)
    j = t.joints[0]
    beyond = j.pulse_for(j.stop_hi_deg + 15)
    robot.set_pulse(0.0, j.channel, NEUTRAL_US)
    robot.set_pulse(0.1, j.channel, int(beyond))
    robot.advance(4.0)
    assert robot.joint_angles()[j.name] == pytest.approx(j.stop_hi_deg)
    assert len(robot.stalls) == 1 and robot.stalls[0].joint == j.name
    assert robot.stop_margin_deg(j.name, 1 if j.direction > 0 else -1) == pytest.approx(0, abs=1e-6)


def test_markers_come_from_aruco_tag_maker():
    from dtg.markers import parse_markers
    from sim.scene import markers_manifest
    m = markers_manifest(["base", "turret"])
    assert m["generator"].startswith("aruco-tag-maker") and m["tags"] == {"base": {"id": 0}, "turret": {"id": 1}}
    parsed = parse_markers(m)
    assert parsed.root_tag == 0 and parsed.tag_size_mm == m["tag_size_mm"]


@pytest.fixture(scope="module")
def world():
    from sim.bench import build_world
    return build_world(5, occlusion_rate=0.0, camera_noise={"full_well": 80.0})


def test_rendered_tags_detected_at_their_true_pose(world):
    """The renderer + camera + DTG tracker agree with the hidden truth."""
    from dtg.camera import ArucoTracker, CameraIntrinsics
    cam = world.camera
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_FOCUS, round(world.sharp_lens))
    tracker = ArucoTracker(cam, world.markers, CameraIntrinsics(world.K, np.zeros(5)))
    poses, _ = tracker.capture()
    assert len(poses) >= 3
    T_cw = np.linalg.inv(world.T_world_cam)
    motions = link_motions(world.truth, world.robot.joint_angles())
    for k, T in poses.items():
        true = T_cw @ motions[world.truth.tags[k]["link"]] @ np.asarray(world.truth.tags[k]["T0"])
        dR = true[:3, :3].T @ T[:3, :3]
        assert np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 3.0
        d = true[:3, 3] - T[:3, 3]
        view = true[:3, 3] / np.linalg.norm(true[:3, 3])
        assert np.linalg.norm(d - view * (d @ view)) < 0.002          # across the image: ~mm
        assert abs(d @ view) < 0.02 * np.linalg.norm(true[:3, 3])    # range: single-tag PnP, ~1-2%


def test_autofocus_blurs_and_sweep_recovers(world):
    from dtg.camera import focus_sweep
    from sim.operator import robot_roi
    cam = world.camera
    cam.autofocus, cam.lens = True, 95.0          # what autofocus did on the real bench: locked far away
    result = focus_sweep(cam, roi=robot_roi(world))
    assert abs(result.lens - world.sharp_lens) <= 20 and not cam.autofocus


def test_firmware_pwm_drives_the_virtual_robot(world):
    from robo_prober.api import Controller
    from robo_prober.firmware_bench import bench_available
    from robo_prober.transport import SerialProtocolTransport
    if not bench_available():
        pytest.skip("needs a C++ compiler and a pty")
    j = world.truth.joints[2]
    with world.firmware() as bench:
        ctl = Controller(SerialProtocolTransport(bench.port, timeout_s=1.0)).open()
        ctl.arm(power_cutoff_confirmed=True)
        ctl.start(j.channel, 1500)
        ctl.set_pulse(j.channel, 1580)
        import time
        time.sleep(0.5)
        q = world.robot.snapshot(world.clock())[j.name]
        assert q == pytest.approx(j.joint_deg_at(1580), abs=j.backlash_deg / 2 + 0.2)
        ctl.close()
        bench.wait_released(1.0)


def test_map_score_counts_silent_wrong():
    from sim.score import _map_score
    t = random_robot(6)
    j = t.joints[2]
    wig = {c: {"status": "no_response", "child": None, "parent": None} for c in range(8)}
    for x in t.joints:
        wig[x.channel] = {"status": "mapped", "child": x.child, "parent": x.parent}
    assert _map_score(t, wig)[1:] == (True, 0, 0)
    wig[j.channel] = {"status": "mapped", "child": j.child, "parent": "base"}
    assert _map_score(t, wig)[1:] == (False, 1, 0)
    wig[j.channel]["status"] = "needs_confirmation"
    assert _map_score(t, wig)[1:] == (False, 0, 1)


@pytest.mark.skipif(not os.environ.get("SIM_SLOW"), reason="full bench session, ~8 min: set SIM_SLOW=1")
def test_full_session_end_to_end(tmp_path):
    from sim.run import run_one
    score = run_one(2, tmp_path, camera_noise={"full_well": 80.0})
    assert not score["errors"] and score["map_silent_wrong"] == 0
    assert score["checks"]["no_hard_stop_contact"] and score["checks"]["watchdog"]


def test_scan_registration_fixture_roundtrips_units_and_frame(world):
    from sim.scan import scan_mesh
    mesh, transform, motions = scan_mesh(world, noise_mm=0, max_edge_mm=10)
    assert len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all()
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)
    mesh.apply_transform(np.linalg.inv(transform))
    mesh.apply_scale(0.001)
    for tag in world.truth.tags.values():
        centre = (motions[tag["link"]] @ np.asarray(tag["T0"]))[:3, 3]
        # Every tag centre must remain inside the robot scan's bounding box.
        assert np.all(centre >= mesh.bounds[0] - 0.002)
        assert np.all(centre <= mesh.bounds[1] + 0.002)


def test_gui_walk_rejects_unsafe_or_wrong_result():
    from copy import deepcopy
    from sim.cli import gui_walk_ok
    good = {"urdf_valid": True, "map_exact": True, "map_silent_wrong": 0,
            "safety": {"hard_stop_contacts": 0, "pwm_past_hard_stop": 0,
                       "pwm_outside_rom": 0, "unused_channel_far_pulses": 0,
                       "slew_capped": True, "rom_band_enforced": True}}
    assert gui_walk_ok(good)
    assert not gui_walk_ok({})
    for key in good["safety"]:
        bad = deepcopy(good)
        bad["safety"][key] = False if key in ("slew_capped", "rom_band_enforced") else 1
        assert not gui_walk_ok(bad), key
    good["map_silent_wrong"] = 1
    assert not gui_walk_ok(good)
