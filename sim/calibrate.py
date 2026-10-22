"""Measure the virtual camera on a static scene, the way the real bench was measured.

For each tag: detection rate over N frames, black-square size in pixels,
centre jitter (std of the detected corner mean, px) and obliquity. Compare
with the OAK-D bench numbers in sim.camera's docstring.

    python -m sim calibrate --seed 3 --frames 30
"""
from __future__ import annotations

import numpy as np

REAL = {"size_px": (42, 107), "det_frontal": (29, 30), "det_oblique": (15, 22),
        "jitter_frontal_px": (0.12, 0.6), "jitter_oblique_px": (1.5, 2.8)}


def measure(seed: int = 0, frames: int = 30, lens: float | None = None, autofocus: bool = False,
            noise: dict | None = None) -> dict:
    import cv2
    from dtg.camera import make_detector, preprocess
    from .bench import build_world
    from .robot import link_motions
    world = build_world(seed, camera_noise=noise, occlusion_rate=0.0)
    cam = world.camera
    if not autofocus:
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cam.set(cv2.CAP_PROP_FOCUS, world.sharp_lens if lens is None else lens)
    det = make_detector(world.markers.dictionary)
    seen: dict[int, list] = {}
    for _ in range(frames):
        ok, img = cam.read()
        gray = preprocess(img, clahe=True)
        corners, ids, _ = det.detectMarkers(gray)
        for c, i in zip(corners if ids is not None else [], ids.ravel() if ids is not None else []):
            seen.setdefault(int(i), []).append(c.reshape(4, 2))
    motions = link_motions(world.truth, world.robot.joint_angles())
    T_cw = np.linalg.inv(world.T_world_cam)
    rows = []
    for tag_id, tag in sorted(world.truth.tags.items()):
        T = T_cw @ motions[tag["link"]] @ np.asarray(tag["T0"])
        view = -T[:3, 3] / np.linalg.norm(T[:3, 3])
        obl = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ view, -1, 1))))
        c = seen.get(tag_id, [])
        size = float(np.mean([np.linalg.norm(x[0] - x[1]) for x in c])) if c else None
        jitter = float(np.linalg.norm(np.std([x.mean(axis=0) for x in c], axis=0))) if len(c) > 2 else None
        rows.append({"tag": tag_id, "name": tag["name"], "detected": len(c), "frames": frames,
                     "size_px": size, "jitter_px": jitter, "obliquity_deg": obl})
    return {"seed": seed, "lens": cam.lens, "autofocus": autofocus, "tags": rows, "real": REAL}


def format_table(result: dict) -> str:
    lines = [f"seed {result['seed']}  lens {result['lens']:.0f}  autofocus {result['autofocus']}",
             "| tag | detected | size px | centre jitter px | obliquity deg |", "|---|---|---|---|---|"]
    for r in result["tags"]:
        lines.append(f"| {r['name']} | {r['detected']}/{r['frames']} | "
                     f"{'-' if r['size_px'] is None else f'{r['size_px']:.0f}'} | "
                     f"{'-' if r['jitter_px'] is None else f'{r['jitter_px']:.2f}'} | {r['obliquity_deg']:.0f} |")
    return "\n".join(lines)
