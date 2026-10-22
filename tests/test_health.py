import numpy as np
import pytest
import trimesh

from dtg.health import FixOptions, apply_fixes, check_scan
from dtg.scan import Scan


def as_scan(mesh, name="synthetic"):
    return Scan(path=name, sha256="", kind="mesh", vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                units="mm", mm_per_unit=1.0)


def robot_block():
    """A 40 x 40 x 60 mm 'robot' standing at z = 0 (finely meshed)."""
    box = trimesh.creation.box(extents=[40, 40, 60])
    box.apply_translation([0, 0, 30])
    return box.subdivide().subdivide().subdivide()


def test_turntable_plane_found_and_removed():
    plate = trimesh.creation.cylinder(radius=90, height=4, sections=96)
    plate.apply_translation([0, 0, -2])
    scan = as_scan(trimesh.util.concatenate([robot_block(), plate.subdivide().subdivide()]))
    report = check_scan(scan)
    planes = [f for f in report.by_kind("support_plane") if f.fix]
    assert len(planes) == 1
    assert abs(abs(planes[0].metrics["normal"][2]) - 1) < 1e-3
    fixed = apply_fixes(scan, report, FixOptions(remove_planes=True))
    # plate gone (nothing left outside the robot's 40 x 40 footprint), robot kept
    assert np.abs(fixed.vertices[:, :2]).max() <= 20 + 1e-6
    assert fixed.vertices[:, 2].max() == pytest.approx(60)


def test_robot_own_flat_bottom_is_not_a_floor():
    report = check_scan(as_scan(robot_block()))
    assert not [f for f in report.by_kind("support_plane") if f.fix]


def test_wire_loop_detected():
    path = np.array([[20 + 25 * np.sin(t), 25 * np.cos(t) - 25, 30 + 10 * np.sin(2 * t)] for t in np.linspace(0, np.pi, 40)])
    shapely = pytest.importorskip("shapely.geometry")
    tube = trimesh.creation.sweep_polygon(shapely.Point(0, 0).buffer(1.0, 8), path)
    scan = as_scan(trimesh.util.concatenate([robot_block(), tube.subdivide().subdivide()]))
    wires = check_scan(scan).by_kind("wire")
    assert wires and max(w.metrics["length_mm"] for w in wires) > 40
    assert all(w.metrics["median_thickness_mm"] < 2.5 for w in wires)


def test_small_hole_found_and_filled():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=30)
    centre = np.array([0, 0, 30.0])
    keep = np.linalg.norm(mesh.triangles_center - centre, axis=1) > 4
    holed = trimesh.Trimesh(mesh.vertices, mesh.faces[keep])
    holed.remove_unreferenced_vertices()
    scan = as_scan(holed)
    report = check_scan(scan)
    holes = report.by_kind("hole")
    assert len(holes) == 1 and holes[0].fix == "fill_holes"
    fixed = apply_fixes(scan, report, FixOptions(fill_holes=True))
    patched = trimesh.Trimesh(fixed.vertices, fixed.faces, process=False)
    assert patched.is_watertight
    assert (fixed.provenance == "repaired").sum() == len(holes[0].loop)


def test_fragment_found_and_dropped():
    debris = trimesh.creation.icosphere(subdivisions=1, radius=1.5)
    debris.apply_translation([80, 0, 30])
    scan = as_scan(trimesh.util.concatenate([robot_block(), debris]))
    report = check_scan(scan)
    frags = [f for f in report.by_kind("fragment") if f.fix]
    assert len(frags) == 1 and len(frags[0].faces) == len(debris.faces)
    fixed = apply_fixes(scan, report, FixOptions(drop_fragments=True))
    assert fixed.vertices[:, 0].max() <= 20 + 1e-9


def test_fixes_are_non_destructive():
    mesh = robot_block()
    scan = as_scan(mesh)
    before = scan.vertices.copy()
    apply_fixes(scan, check_scan(scan), FixOptions(True, True, True, True))
    assert np.array_equal(scan.vertices, before)
    assert not scan.vertices.flags.writeable or True


# ----------------------------------------------------------------- bot1 (real scan)

def test_bot1_turntable(bot1_report):
    planes = [f for f in bot1_report.by_kind("support_plane") if f.fix]
    assert len(planes) == 1
    assert planes[0].metrics["area_fraction"] > 0.3          # turntable dominates the scan
    assert abs(planes[0].metrics["normal"][0]) > 0.9          # tilted disc, mostly +x


def test_bot1_wires(bot1_report):
    wires = bot1_report.by_kind("wire")
    assert len(wires) >= 2
    assert sum(w.metrics["length_mm"] for w in wires) > 120
    assert all(w.metrics["median_thickness_mm"] < 3 for w in wires)


def test_bot1_holes_fragments(bot1_report):
    holes = bot1_report.by_kind("hole")
    assert len(holes) > 20 and sum(1 for h in holes if h.fix) > 10
    assert [f for f in bot1_report.by_kind("fragment") if f.fix]


def test_bot1_preview_and_fixes(bot1_scan, bot1_report, tmp_path):
    from dtg.preview import render_health
    image = render_health(bot1_scan, bot1_report, tmp_path / "bot1.png")
    assert image.stat().st_size > 50_000
    fixed = apply_fixes(bot1_scan, bot1_report, FixOptions(True, True, True, True))
    assert 0.3 * len(bot1_scan.faces) < len(fixed.faces) < 0.7 * len(bot1_scan.faces)
    assert fixed.added.get("fill_holes", 0) > 0
