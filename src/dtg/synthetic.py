"""Synthetic camera observations for a bot1-like desktop arm (no hardware).

Generates exactly what the live recorder would write: a reference capture,
a wiggle of every controller channel (one at a time, small pulse change, back
to start) and a ROM sweep per joint with Set ROM 1 / Set ROM 2 marks; tag
poses come from forward kinematics, a fixed camera, pose noise, a little base
wobble and random tag dropouts. The ground truth goes in the header under
"truth" so tests can score the inference; inference code never reads it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .observations import Frame, ObservationLog, inv, new_header


@dataclass
class SynthJoint:
    name: str
    parent: str
    child: str
    axis: tuple            # world axis at the zero pose
    pivot: tuple           # world point on the axis at the zero pose (m)
    channel: int
    direction: int         # +1 / -1: sign of angle per pulse about `axis`
    rad_per_us: float
    zero_us: float         # pulse at which the joint is at its zero pose
    rom1_us: int
    rom2_us: int
    start_us: int = 1500
    type: str = "revolute"


@dataclass
class SynthArm:
    joints: list[SynthJoint]
    root: str
    tags: dict[int, tuple[str, np.ndarray]]   # id -> (link, T_world_tag at zero pose)
    unused_channels: list[int] = field(default_factory=list)


def _tag_pose(position, normal_towards, spin_deg=0.0) -> np.ndarray:
    z = np.asarray(normal_towards, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.9 else np.cross([0, 1.0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.c_[x, y, z] @ Rotation.from_euler("z", spin_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def bot1_like_arm() -> SynthArm:
    """Turntable-base yaw + shoulder pitch + elbow pitch, metres, z up.
    Channel numbers and directions are deliberately scrambled, and channels
    1 and 6 are wired to nothing (they must come back as 'no response')."""
    cam_dir = np.array([1.0, -0.4, 0.5])
    joints = [
        SynthJoint("j_base_yaw", "base", "turret", (0, 0, 1), (0.0, 0.0, 0.06), channel=4, direction=+1,
                   rad_per_us=np.radians(0.09), zero_us=1500, rom1_us=1050, rom2_us=1950),
        SynthJoint("j_shoulder", "turret", "upper_arm", (0, 1, 0), (0.0, 0.0, 0.11), channel=0, direction=-1,
                   rad_per_us=np.radians(0.095), zero_us=1480, rom1_us=1150, rom2_us=1800),
        SynthJoint("j_elbow", "upper_arm", "forearm", (0, 1, 0), (0.10, 0.0, 0.11), channel=7, direction=+1,
                   rad_per_us=np.radians(0.085), zero_us=1520, rom1_us=1900, rom2_us=1200),
    ]
    tags = {
        0: ("base", _tag_pose((0.03, -0.045, 0.03), cam_dir, 10)),
        5: ("base", _tag_pose((-0.03, -0.05, 0.02), cam_dir, -20)),
        1: ("turret", _tag_pose((0.025, -0.03, 0.085), cam_dir, 30)),
        2: ("upper_arm", _tag_pose((0.045, -0.052, 0.111), (0.3, -1.0, 0.3), 0)),
        3: ("upper_arm", _tag_pose((0.085, -0.052, 0.111), (0.5, -1.0, 0.2), 45)),
        4: ("forearm", _tag_pose((0.10, -0.072, 0.06), (0.6, -1.0, 0.1), 15)),
    }
    return SynthArm(joints=joints, root="base", tags=tags, unused_channels=[1, 6])


def _joint_motion(j: SynthJoint, pulse: float) -> np.ndarray:
    q = j.direction * j.rad_per_us * (pulse - j.zero_us)
    G = np.eye(4)
    a = np.asarray(j.axis, float) / np.linalg.norm(j.axis)
    if j.type == "prismatic":
        G[:3, 3] = a * q
        return G
    R = Rotation.from_rotvec(a * q).as_matrix()
    p = np.asarray(j.pivot, float)
    G[:3, :3] = R
    G[:3, 3] = p - R @ p
    return G


def _link_motion(arm: SynthArm, pulses: dict[int, float]) -> dict[str, np.ndarray]:
    by_child = {j.child: j for j in arm.joints}
    cache = {arm.root: np.eye(4)}

    def motion(link):
        if link not in cache:
            j = by_child[link]
            cache[link] = motion(j.parent) @ _joint_motion(j, pulses[j.channel])
        return cache[link]
    for j in arm.joints:
        motion(j.child)
    return cache


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


def generate(arm: SynthArm | None = None, *, seed: int = 0, rot_noise_deg: float = 0.4,
             trans_noise_mm: float = 0.5, depth_noise_mm: float = 1.2, dropout: float = 0.01,
             wobble_deg: float = 0.15, wiggle_us: int = 80, rom_step_us: int = 40) -> ObservationLog:
    arm = arm or bot1_like_arm()
    rng = np.random.default_rng(seed)
    T_world_cam = look_at((0.42, -0.30, 0.30), (0.04, 0.0, 0.10))
    T_cam_world = inv(T_world_cam)
    channels = sorted([j.channel for j in arm.joints] + arm.unused_channels)
    pulses = {c: 1500.0 for c in channels}
    for j in arm.joints:
        pulses[j.channel] = float(j.start_us)
    frames: list[Frame] = []
    clock = [0.0]

    def capture(phase, event, channel=None):
        motions = _link_motion(arm, pulses)
        wob = np.eye(4)
        wob[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.radians(wobble_deg), 3)).as_matrix()
        poses, reproj = {}, {}
        for tag_id, (link, T0) in arm.tags.items():
            if event != "reference" and rng.random() < dropout:
                continue
            T = T_cam_world @ wob @ motions[link] @ T0
            noise = np.eye(4)
            noise[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.radians(rot_noise_deg), 3)).as_matrix()
            T = T @ noise
            T[:3, 3] += rng.normal(0, [trans_noise_mm / 1000, trans_noise_mm / 1000, depth_noise_mm / 1000])
            poses[tag_id] = T
            reproj[tag_id] = float(abs(rng.normal(0.3, 0.1)))
        frames.append(Frame(t=clock[0], phase=phase, event=event, poses=poses, channel=channel,
                            pulse_us=None if channel is None else pulses[channel], reproj_px=reproj))
        clock[0] += 0.5

    capture("reference", "reference")
    for c in channels:  # wiggle: one channel at a time, back to start after each move
        start = pulses[c]
        for event, value in (("wiggle_base", start), ("wiggle_plus", start + wiggle_us), ("wiggle_base", start),
                             ("wiggle_minus", start - wiggle_us), ("wiggle_base", start)):
            pulses[c] = value
            capture("wiggle", event, c)
        pulses[c] = start
    for j in arm.joints:  # ROM sweep per joint
        c, start = j.channel, pulses[j.channel]
        for target, mark in ((j.rom1_us, "rom1"), (j.rom2_us, "rom2")):
            step = rom_step_us if target > pulses[c] else -rom_step_us
            while abs(target - pulses[c]) > rom_step_us:
                pulses[c] += step
                capture("rom", "sample", c)
            pulses[c] = float(target)
            capture("rom", mark, c)
        step = rom_step_us if start > pulses[c] else -rom_step_us
        while abs(start - pulses[c]) > rom_step_us:
            pulses[c] += step
            capture("rom", "sample", c)
        pulses[c] = start
    header = new_header("synthetic", markers={"dictionary": "DICT_4X4_50", "tag_size_mm": 25.0,
                                              "tags": {_tag_name(arm, k): {"id": k} for k in arm.tags}},
                        camera={"model": "synthetic pinhole", "note": "poses generated, not detected"},
                        truth=truth(arm), noise={"rot_deg": rot_noise_deg, "trans_mm": trans_noise_mm,
                                                 "depth_mm": depth_noise_mm, "dropout": dropout,
                                                 "wobble_deg": wobble_deg})
    return ObservationLog(header, frames)


def _tag_name(arm: SynthArm, tag_id: int) -> str:
    link = arm.tags[tag_id][0]
    same = sorted(k for k, (l, _) in arm.tags.items() if l == link)
    return link if same.index(tag_id) == 0 else f"{link}_{same.index(tag_id) + 1}"


def truth(arm: SynthArm) -> dict:
    """Ground truth in the root tag's frame at the reference (start-pulse) pose."""
    root_tag = min(k for k, (l, _) in arm.tags.items() if l == arm.root)
    start = {j.channel: float(j.start_us) for j in arm.joints}
    motions = _link_motion(arm, start)
    T_root = motions[arm.root] @ arm.tags[root_tag][1]
    W = inv(T_root)
    joints = {}
    for j in arm.joints:
        M_parent = motions[j.parent]  # world motion of the parent at the reference pose
        axis_world = M_parent[:3, :3] @ (np.asarray(j.axis, float) / np.linalg.norm(j.axis))
        pivot_world = (M_parent @ np.r_[j.pivot, 1.0])[:3]
        q_ref = j.direction * j.rad_per_us * (j.start_us - j.zero_us)
        q1 = j.direction * j.rad_per_us * (j.rom1_us - j.zero_us) - q_ref
        q2 = j.direction * j.rad_per_us * (j.rom2_us - j.zero_us) - q_ref
        joints[j.name] = {"channel": j.channel, "parent": j.parent, "child": j.child, "type": j.type,
                          "axis_root": (W[:3, :3] @ axis_world).tolist(),
                          "pivot_root": (W @ np.r_[pivot_world, 1.0])[:3].tolist(),
                          "direction_about_axis": j.direction, "rad_per_us": j.rad_per_us,
                          "rom1_rad": q1, "rom2_rad": q2, "rom1_us": j.rom1_us, "rom2_us": j.rom2_us,
                          "zero_us": j.start_us}
    return {"root_tag": root_tag, "joints": joints, "unused_channels": arm.unused_channels,
            "tags": {str(k): l for k, (l, _) in arm.tags.items()}}


# ------------------------------------------------------------------ synthetic scan

# Link primitives at the zero pose, world frame, metres: ("box", lo, hi) or ("cyl", centre_xy, r, z0, z1)
BOT1_LIKE_SHAPES = {
    "base": [("cyl", (0.0, 0.0), 0.05, 0.0, 0.06)],
    "turret": [("box", (-0.025, -0.032, 0.06), (0.025, 0.03, 0.105))],
    "upper_arm": [("box", (-0.015, -0.052, 0.098), (0.115, -0.032, 0.124))],
    "forearm": [("box", (0.088, -0.072, 0.03), (0.112, -0.052, 0.122))],
}


def _inside(shape, pts):
    if shape[0] == "box":
        lo, hi = np.asarray(shape[1]), np.asarray(shape[2])
        return ((pts >= lo) & (pts <= hi)).all(axis=1)
    _, c, r, z0, z1 = shape
    return (np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) <= r) & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)


def synthetic_scan(arm: SynthArm | None = None, shapes: dict | None = None, *, pitch: float = 0.003,
                   seed: int = 0):
    """A fused one-piece 'scan' of the arm at its reference pose.

    Returns dict with vertices (mm, in a scan frame unrelated to the model
    frame), faces, true vertex link labels, tag centres in scan coordinates
    (with 0.5 mm pick noise), and T_scan_world (incl. the x1000 scale)."""
    import trimesh
    arm = arm or bot1_like_arm()
    shapes = shapes or BOT1_LIKE_SHAPES
    rng = np.random.default_rng(seed)
    start = {j.channel: float(j.start_us) for j in arm.joints}
    motions = _link_motion(arm, start)
    allpts = []
    for link, prims in shapes.items():
        for s in prims:
            if s[0] == "box":
                corners = np.array(np.meshgrid(*zip(s[1], s[2]))).reshape(3, -1).T
            else:
                corners = np.array([[s[1][0] + dx * s[2], s[1][1] + dy * s[2], z]
                                    for dx in (-1, 1) for dy in (-1, 1) for z in (s[3], s[4])])
            allpts.append(corners @ motions[link][:3, :3].T + motions[link][:3, 3])
    allpts = np.vstack(allpts)
    lo, hi = allpts.min(axis=0) - 2 * pitch, allpts.max(axis=0) + 2 * pitch
    axes = [np.arange(lo[i], hi[i], pitch) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    owner = np.full(len(grid), -1)
    names = list(shapes)
    for i, link in enumerate(names):
        local = (grid - motions[link][:3, 3]) @ motions[link][:3, :3]  # world -> zero-pose link coords
        for s in shapes[link]:
            owner[_inside(s, local) & (owner < 0)] = i
    shape = [len(a) for a in axes]
    occ = (owner >= 0).reshape(shape)
    verts_world, faces = _voxel_surface(occ, lo, pitch)
    # true label = owner of the nearest occupied voxel centre
    from scipy.spatial import cKDTree
    occ_idx = np.flatnonzero(owner >= 0)
    _, nn = cKDTree(grid[occ_idx]).query(verts_world)
    truth_labels = np.array(names, dtype=object)[owner[occ_idx][nn]]
    # an arbitrary scan frame, millimetres
    R = Rotation.from_euler("zyx", [40, -15, 120], degrees=True).as_matrix()
    T_scan_world = np.eye(4)
    T_scan_world[:3, :3] = 1000.0 * R
    T_scan_world[:3, 3] = [12.0, -40.0, 85.0]
    verts_scan = verts_world @ T_scan_world[:3, :3].T + T_scan_world[:3, 3]
    tags_scan = {}
    for k, (link, T0) in arm.tags.items():
        centre_world = (motions[link] @ T0)[:3, 3]
        tags_scan[k] = centre_world @ T_scan_world[:3, :3].T + T_scan_world[:3, 3] + rng.normal(0, 0.5, 3)
    return {"vertices": verts_scan, "faces": faces, "truth_labels": truth_labels,
            "tag_centres_scan": tags_scan, "T_scan_world": T_scan_world}


def _voxel_surface(occ: np.ndarray, origin, pitch):
    """Exposed faces of an occupancy grid as one connected triangle mesh
    (shared corner vertices, outward winding)."""
    pad = np.pad(occ, 1)
    quads = []
    # (axis, sign): corner offsets of the face, counter-clockwise seen from outside
    for axis in range(3):
        for sign in (1, -1):
            nb = np.roll(pad, -sign, axis=axis)
            exposed = pad & ~nb
            idx = np.argwhere(exposed[1:-1, 1:-1, 1:-1])
            if not len(idx):
                continue
            u, v = (axis + 1) % 3, (axis + 2) % 3
            base = idx.copy()
            if sign > 0:
                base[:, axis] += 1
            c = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                q = base.copy()
                q[:, u] += du
                q[:, v] += dv
                c.append(q)
            if sign < 0:
                c = c[::-1]
            quads.append(np.stack(c, axis=1))
    quads = np.concatenate(quads)
    corners, inv = np.unique(quads.reshape(-1, 3), axis=0, return_inverse=True)
    q = inv.reshape(-1, 4)
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return np.asarray(origin) + corners * pitch - pitch / 2, faces
