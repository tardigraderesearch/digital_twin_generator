"""A simulated 3D scan of the virtual robot, for steps 1, 5 and 6 on the bench.

The robot is "scanned" in its rest pose (the same pose as the reference
capture): every robot part, tag tile and standoff of the scene is posed,
fused into one surface (the way a scanner welds touching parts), subdivided
to a scanner-like density, jittered with scan noise, and written in
millimetres in its own frame (a random rigid transform away from the bench
frame), so registration has to solve scale, rotation and translation.

It also writes what the operator would pick for step 5: the centre of every
tag tile in scan coordinates, with a little pick noise.

What it is not: no texture, no holes, no wire loops, no shadowed regions.
The real scanner's artefacts are only covered by the bot1 scan tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import link_motions
from .scene import build_scene

SKIP = ("table", "wall")


def scan_mesh(world, *, noise_mm: float = 0.25, max_edge_mm: float = 3.0, seed: int = 0):
    import trimesh
    rng = np.random.default_rng(40_000 + world.seed + seed)
    scene = build_scene(world.truth, world.T_world_cam, np.random.default_rng(0))
    motions = link_motions(world.truth, world.robot.joint_angles())
    parts = []
    for it in scene.items:
        if it.name in SKIP or it.name.startswith("tag") and not it.name.startswith("tile"):
            continue                                 # flat textures carry no geometry of their own
        M = motions.get(it.link, np.eye(4))
        v = it.verts @ M[:3, :3].T + M[:3, 3]
        mesh = trimesh.Trimesh(v, it.tris, process=True)
        if not mesh.is_volume:
            mesh.fix_normals()
        # grow each part a hair so touching parts overlap and fuse cleanly (no slivers)
        c = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-c)
        mesh.apply_scale(1.0 + 0.0006 / max(float(mesh.extents.min()), 1e-4))
        mesh.apply_translation(c)
        parts.append(mesh)
    # A concatenated mesh is not the fused scan this fixture promises.
    # Fail visibly if the required manifold engine cannot produce it.
    fused = trimesh.boolean.union(parts, engine="manifold")
    mm = fused.copy()
    mm.apply_scale(1000.0)
    v, f = trimesh.remesh.subdivide_to_size(mm.vertices, mm.faces, max_edge=max_edge_mm)
    mesh = trimesh.Trimesh(v, f, process=True)
    mesh.vertices += rng.normal(0, noise_mm, mesh.vertices.shape)
    T_scan_world = np.eye(4)
    T_scan_world[:3, :3] = Rotation.from_euler("zyx", rng.uniform([-180, -10, -10], [180, 10, 10]),
                                               degrees=True).as_matrix()
    T_scan_world[:3, 3] = rng.uniform(-200, 200, 3)
    mesh.apply_transform(T_scan_world)
    return mesh, T_scan_world, motions


def write_scan(world, out: Path, *, pick_noise_mm: float = 0.5) -> dict:
    """Write scan.stl (mm) and scan_tags.json (tag centres the operator would pick)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mesh, T_scan_world, motions = scan_mesh(world)
    path = out / f"sim-robot-{world.seed}.stl"
    mesh.export(str(path))
    rng = np.random.default_rng(50_000 + world.seed)
    centres = {}
    for tag_id, tag in world.truth.tags.items():
        T = motions[tag["link"]] @ np.asarray(tag["T0"])
        p_mm = (T_scan_world[:3, :3] @ (T[:3, 3] * 1000.0)) + T_scan_world[:3, 3]
        centres[str(tag_id)] = (p_mm + rng.normal(0, pick_noise_mm, 3)).round(2).tolist()
    tags_path = out / "scan_tags.json"
    tags_path.write_text(json.dumps(centres, indent=2) + "\n")
    return {"scan": str(path), "scan_tags": str(tags_path), "faces": int(len(mesh.faces)),
            "T_scan_world_mm": T_scan_world.tolist()}
