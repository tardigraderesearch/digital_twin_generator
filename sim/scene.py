"""The bench scene: the virtual robot's link geometry, its ArUco tiles, a table,
a textured wall behind, and the fixed camera.

Tags come from aruco-tag-maker (markers.json manifest + marker bitmaps for the
real dictionary and printed size). They are stuck on standoffs while the
robot sits at rest, facing the camera at 15-35 degrees of obliquity (DTG's
README recommends ~30 degrees to avoid the frontal pose ambiguity).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from .render import Item, Scene, Texture, box, cylinder, item, quad

TAG_NAMES = ["base", "base_2", "turret", "turret_2", "upper_arm", "upper_arm_2", "forearm", "forearm_2"]
TAG_SIZE_MM = 30.0           # aruco-tag-maker default black-square size
STANDOFF_M = 0.012
TARGET = np.array([0.04, -0.03, 0.08])


def tag_settings():
    from aruco_tag_maker.settings import Settings
    return Settings(tag_size_mm=TAG_SIZE_MM)


def markers_manifest(names=TAG_NAMES) -> dict:
    """markers.json exactly as aruco-tag-maker exports it for these tag names."""
    from aruco_tag_maker.exporter import manifest
    return manifest(list(names), tag_settings())


def tag_texture(marker_id: int) -> np.ndarray:
    from aruco_tag_maker.marker import marker_image
    s = tag_settings()
    img = marker_image(s.dictionary, marker_id, s.quiet_zone_modules, pixels_per_module=32)
    return (0.06 + 0.82 * (img.astype(np.float32) / 255.0)).astype(np.float32)


def look_at(eye, target) -> np.ndarray:
    """T_world_camera, OpenCV convention (z forward, x right, y down)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = eye
    return T


def camera_pose(rng) -> np.ndarray:
    az = math.radians(-65 + rng.uniform(-8, 8))
    el = math.radians(28 + rng.uniform(-5, 5))
    dist = rng.uniform(0.52, 0.62)
    eye = TARGET + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return look_at(eye, TARGET + rng.normal(0, 0.005, 3))


def _frame(origin, normal, spin_deg) -> np.ndarray:
    z = np.asarray(normal, float) / np.linalg.norm(normal)
    x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.9 else np.cross([0, 1.0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z] @ Rotation.from_euler("z", spin_deg, degrees=True).as_matrix()
    T[:3, 3] = origin
    return T


def candidate_tags(rng, dims, motions, camera_eye) -> list[dict]:
    """Possible tag mounts on every moving link and the base, at the rest pose.

    Each faces mostly the camera (75%) and partly away from the surface, then
    tilts 15-35 deg off-axis with a random spin, like tags stuck on by hand.
    """
    bh, th, L1, L2 = dims["base_h"], dims["turret_h"], dims["upper_len"], dims["fore_len"]
    r, ysh, zsh = dims["base_r"], dims["shoulder_y"], dims["shoulder"][2]
    yel = dims["elbow"][1]
    anchors = [("base", (r * math.cos(a), r * math.sin(a), bh * h), (math.cos(a), math.sin(a), 0.15))
               for a in np.radians([-150, -120, -90, -60, -30, 0, 30]) for h in (0.75,)]
    anchors += [("turret", (0.026, y, bh + th * h), (1, 0, 0)) for y in (-0.012, 0.012) for h in (0.5,)]
    anchors += [("turret", (x, 0.030, bh + th * 0.5), (0, 1, 0)) for x in (-0.012, 0.012)]
    anchors += [("turret", (x, 0.0, bh + th), (0, 0, 1)) for x in (-0.012, 0.012)]
    anchors += [("turret", (x, -0.030, bh + th * 0.25), (0, -1, -0.2)) for x in (-0.015, 0.015)]
    anchors += [("turret", (0.026, 0.0, bh + th * h), (1, -0.5, 0)) for h in (0.25, 0.8)]
    anchors += [("upper_arm", (f * L1, ysh - 0.011, zsh), (0, -1, 0)) for f in (0.25, 0.5, 0.75, 0.95)]
    anchors += [("upper_arm", (f * L1, ysh, zsh + 0.010), (0, 0, 1)) for f in (0.4, 0.8)]
    anchors += [("forearm", (L1, yel - 0.011, zsh - f * L2), (0, -1, 0)) for f in (0.3, 0.6, 0.85)]
    anchors += [("forearm", (L1 + 0.010, yel, zsh - f * L2), (1, 0, 0)) for f in (0.4, 0.8)]
    anchors += [("forearm", (L1 - 0.010, yel, zsh - f * L2), (-1, 0, 0)) for f in (0.5,)]
    out = []
    for link, anchor, outward in anchors:
        M = motions[link]
        p_rest = M[:3, :3] @ np.asarray(anchor) + M[:3, 3]
        out_rest = M[:3, :3] @ (np.asarray(outward, float) / np.linalg.norm(outward))
        centre = p_rest + out_rest * STANDOFF_M
        to_cam = np.asarray(camera_eye) - centre
        to_cam /= np.linalg.norm(to_cam)
        if to_cam @ out_rest < -0.2:      # nobody sticks a tag on the far side
            continue
        base = to_cam * 0.75 + out_rest * 0.25
        base /= np.linalg.norm(base)
        perp = np.cross(base, rng.normal(size=3))
        perp /= np.linalg.norm(perp)
        normal = Rotation.from_rotvec(perp * math.radians(rng.uniform(15, 35))).apply(base)
        T_rest = _frame(centre, normal, rng.uniform(0, 360))
        # lengthen the standoff until no tile corner digs into the surface
        h = tag_settings().tile_size_mm / 2000.0
        corners = T_rest[:3, :3] @ np.array([[-h, -h, -0.003], [h, -h, -0.003], [h, h, -0.003], [-h, h, -0.003]]).T
        dig = -(corners.T @ out_rest).min() - STANDOFF_M + 0.003
        if dig > 0:
            T_rest[:3, 3] += out_rest * dig
        out.append({"link": link, "T0": np.linalg.inv(M) @ T_rest, "T_rest": T_rest, "anchor0": list(anchor),
                    "cand": len(out)})
    return out


def select_tags(candidates, depth_rest, T_world_cam, K, rng, per_link: int = 2, exclude=()) -> dict:
    """Keep up to ``per_link`` tags per link that the camera sees whole at rest
    (nothing in front of any part of the tile, no overlap with another chosen
    tile, black square at least 45 px). Names and ids follow aruco-tag-maker:
    ids in name order, names <link>, <link>_2."""
    import cv2
    tile = tag_settings().tile_size_mm / 1000.0
    T_cw = np.linalg.inv(T_world_cam)
    g = np.linspace(-tile / 2, tile / 2, 7)
    grid = np.array([[x, y, 0.0, 1.0] for x in g for y in g])
    H, W = depth_rest.shape
    scored = []
    for c in candidates:
        if c["cand"] in exclude:
            continue
        pc = (T_cw @ c["T_rest"] @ grid.T).T[:, :3]
        uv = pc @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        ui, vi = np.round(uv).astype(int).T
        if (ui < 2).any() or (vi < 2).any() or (ui >= W - 2).any() or (vi >= H - 2).any():
            continue
        visible = np.mean(pc[:, 2] <= depth_rest[vi, ui] + 0.002)
        corners = uv[[0, 6, 48, 42]]
        side = np.mean(np.linalg.norm(np.roll(corners, 1, axis=0) - corners, axis=1)) * 6 / 8
        view = -pc.mean(axis=0) / np.linalg.norm(pc.mean(axis=0))
        facing = float((T_cw[:3, :3] @ c["T_rest"][:3, 2]) @ view)
        if visible < 0.85 or side < 40 or facing < 0.35:
            continue
        good = visible >= 0.97 and side >= 45 and facing >= 0.5
        hull = cv2.convexHull(corners.astype(np.float32))
        scored.append((rng.random(), c, hull, good, visible))
    chosen: dict[str, list] = {}
    hulls = []

    def take(c, hull):
        if any(cv2.intersectConvexConvex(hull, h)[0] > 1.0 for h in hulls):
            return
        chosen.setdefault(c["link"], []).append(c)
        hulls.append(hull)

    ordered = sorted(scored, key=lambda x: x[0])
    for round_ in range(per_link):            # one tag per link first, hardest links first
        for link in ("turret", "forearm", "upper_arm", "base"):
            for _, c, hull, good, _ in ordered:
                if c["link"] == link and good and len(chosen.get(link, [])) == round_ \
                        and all(c is not x for x in chosen.get(link, [])):
                    take(c, hull)
    # every link gets a tag if at all possible (a careful operator insists)
    for _, c, hull, good, _ in sorted(scored, key=lambda x: -x[4]):
        if not chosen.get(c["link"]):
            take(c, hull)
    names, picked = [], []
    for link in ("base", "turret", "upper_arm", "forearm"):
        for i, c in enumerate(chosen.get(link, [])):
            names.append(link if i == 0 else f"{link}_{i + 1}")
            picked.append(c)
    from aruco_tag_maker.exporter import manifest
    ids = manifest(names, tag_settings())["tags"]
    return {ids[n]["id"]: {"link": c["link"], "T0": c["T0"], "name": n, "anchor0": c["anchor0"], "cand": c["cand"]}
            for n, c in zip(names, picked)}


def _texture_noise(rng, shape, scale, lo, hi):
    small = rng.uniform(lo, hi, (max(2, shape[0] // scale), max(2, shape[1] // scale))).astype(np.float32)
    import cv2
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def build_scene(truth, T_world_cam, rng) -> Scene:
    import cv2
    d = truth.dims
    bh, th, L1, L2 = d["base_h"], d["turret_h"], d["upper_len"], d["fore_len"]
    r, ysh, zsh, yel = d["base_r"], d["shoulder_y"], d["shoulder"][2], d["elbow"][1]
    w, t = d["arm_w"], d["arm_t"]
    pla = rng.choice([(0.80, 0.80, 0.82), (0.15, 0.45, 0.85), (0.70, 0.70, 0.30), (0.25, 0.25, 0.27)])
    items: list[Item] = []
    # table with a faint wood grain, and a busy wall behind (autofocus bait)
    grain = _texture_noise(rng, (256, 256), 4, 0.35, 0.55)
    grain = np.dstack([grain * 0.75, grain * 0.9, grain]).astype(np.float32)
    items.append(item("world", quad(np.eye(4), 0.9, 0.9), (0.4, 0.4, 0.4),
                      Texture(grain, np.eye(4), (0.9, 0.9)), "table"))
    view = T_world_cam[:3, 2].copy()
    view[2] = 0
    view /= np.linalg.norm(view)
    wall_c = TARGET + view * 0.70
    wall_c[2] = 0.30
    Tw = _frame(wall_c, -view, 0)
    Tw[:3, :3] = np.c_[np.cross([0, 0, 1.0], -view), [0, 0, 1.0], -view]
    wall = np.full((512, 768, 3), 0.55, np.float32)
    for _ in range(90):  # posters, books, cables: high-contrast edges far away
        x0, y0 = rng.integers(0, 740), rng.integers(0, 490)
        cv2.rectangle(wall, (int(x0), int(y0)), (int(x0 + rng.integers(8, 120)), int(y0 + rng.integers(8, 90))),
                      tuple(float(c) for c in rng.uniform(0.05, 0.95, 3)), -1)
    items.append(item("world", quad(Tw, 1.6, 0.9), (0.5, 0.5, 0.5), Texture(wall, Tw, (1.6, 0.9)), "wall"))
    # robot
    items.append(item("base", cylinder(0, 0, 0.09, 0.0, 0.004, 32), (0.85, 0.55, 0.20), name="turntable"))
    items.append(item("base", cylinder(0, 0, r, 0.004, bh, 24), (0.28, 0.28, 0.30), name="base"))
    items.append(item("turret", box((-0.026, -0.030, bh), (0.026, 0.030, bh + th)), pla, name="turret"))
    items.append(item("turret", box((-0.010, ysh + 0.010, zsh - 0.010), (0.010, -0.028, zsh + 0.010)),
                      (0.2, 0.2, 0.22), name="shoulder_servo"))
    items.append(item("upper_arm", box((-0.014, ysh - w / 2, zsh - t / 2), (L1 + 0.014, ysh + w / 2, zsh + t / 2)),
                      pla, name="upper_arm"))
    items.append(item("forearm", box((L1 - t / 2, yel - w / 2, zsh - L2), (L1 + t / 2, yel + w / 2, zsh + 0.012)),
                      pla, name="forearm"))
    # tag tiles on standoffs
    s = tag_settings()
    tile = s.tile_size_mm / 1000.0
    for tag_id, tag in truth.tags.items():
        T0 = np.asarray(tag["T0"])
        items.append(item(tag["link"], box((-tile / 2, -tile / 2, -0.003), (tile / 2, tile / 2, -0.0004), T0),
                          (0.85, 0.85, 0.85), name=f"tile{tag_id}"))
        anchor = np.asarray(tag["anchor0"])
        post_len = float(np.linalg.norm(T0[:3, 3] - anchor))
        post = np.eye(4)
        post[:3, :3] = T0[:3, :3]
        post[:3, 3] = T0[:3, 3]
        items.append(item(tag["link"], box((-0.004, -0.004, -post_len), (0.004, 0.004, -0.003), post),
                          (0.3, 0.3, 0.3), name=f"post{tag_id}"))
        items.append(item(tag["link"], quad(T0, tile, tile), (0.8, 0.8, 0.8),
                          Texture(tag_texture(tag_id), T0, (tile, tile)), name=f"tag{tag_id}"))
    light = rng.normal(size=3)
    light[2] = abs(light[2]) + 1.0
    return Scene(items=items, light_dir=light, ambient=float(rng.uniform(0.35, 0.55)))
