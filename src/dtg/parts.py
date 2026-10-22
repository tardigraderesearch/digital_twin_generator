"""Missing-part handling: servo library, clean servo bodies, stand-ins, CAD.

Every part carries a provenance label so the report says where geometry
came from:

    scanned       straight from the scan
    repaired      scan with automatic fixes (filled holes)
    library-servo clean servo body from the servo library on a fitted joint axis
    stand-in      primitive placeholder between joints / around tags
    CAD           a user-supplied STL snapped in by a transform or 3 points
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import trimesh

PROVENANCE = ("scanned", "repaired", "library-servo", "stand-in", "CAD")
KGCM_TO_NM = 0.0980665


def load_servo_library(path: str | Path | None = None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    return json.loads(resources.files("dtg").joinpath("data/servos.json").read_text())


@dataclass
class ServoSpec:
    model: str
    body_mm: list
    height_with_spline_mm: float
    length_with_tabs_mm: float
    tab_height_mm: float
    shaft_offset_from_end_mm: float
    mass_g: float | None
    effort_nm: float | None
    velocity_rad_s: float | None
    assumptions: list
    entry: dict

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "entry"} | {
            "dims_verified": self.entry.get("dims_verified"), "mass_verified": self.entry.get("mass_verified"),
            "sources": self.entry.get("sources")}


def servo_spec(model: str, library: dict | None = None) -> ServoSpec:
    lib = library or load_servo_library()
    entry = next((s for s in lib["servos"] if s["model"].lower() == model.lower()), None)
    if entry is None:
        raise KeyError(f"servo {model!r} not in library; known: {', '.join(s['model'] for s in lib['servos'])}")
    defaults = lib["class_defaults"][entry.get("size_class") or "standard"]
    assumptions = []
    vals = {}
    for key in ("body_mm", "height_with_spline_mm", "length_with_tabs_mm", "tab_height_mm", "shaft_offset_from_end_mm"):
        if entry.get(key) is None:
            vals[key] = defaults[key]
            assumptions.append(f"{key} = {defaults[key]} ({entry.get('size_class')} class default, not datasheet)")
        else:
            vals[key] = entry[key]
    if not entry.get("dims_verified"):
        assumptions.append(f"dimensions not manufacturer-verified: {entry.get('dims_source') or 'no source'}")
    torque = entry.get("stall_torque_kgcm") or {}
    speed = entry.get("speed_s_per_60deg") or {}
    effort = max(torque.values()) * KGCM_TO_NM if torque else None
    # use the speed at the same (highest) voltage as the torque where possible
    velocity = (np.pi / 3) / min(speed.values()) if speed else None
    return ServoSpec(model=entry["model"], mass_g=entry.get("mass_g"), effort_nm=effort, velocity_rad_s=velocity,
                     assumptions=assumptions, entry=entry, **vals)


def _frame_from_axis(axis, length_dir_hint):
    z = np.asarray(axis, float)
    z /= np.linalg.norm(z)
    x = np.asarray(length_dir_hint, float) - z * np.dot(length_dir_hint, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def servo_body_mesh(spec: ServoSpec, pivot, axis, length_dir_hint, *, units_per_mm: float = 0.001) -> trimesh.Trimesh:
    """A clean servo: case box + mounting-tab plate + spline cylinder.

    Placed so the output shaft lies on the joint axis, the top of the spline
    is at `pivot`, the case hangs on the -axis side and its long side points
    along `length_dir_hint` (e.g. towards the parent link's centre).
    Output in model units (metres by default)."""
    L, W, H = spec.body_mm
    s = units_per_mm
    off = spec.shaft_offset_from_end_mm
    spline_h = max(spec.height_with_spline_mm - H, 1.0)
    # servo-local frame: origin at the shaft axis on the case top; x along length
    case = trimesh.creation.box(extents=[L * s, W * s, H * s])
    case.apply_translation([(L / 2 - off) * s, 0, -H / 2 * s])
    tab_t = 2.5
    tabs = trimesh.creation.box(extents=[spec.length_with_tabs_mm * s, W * s, tab_t * s])
    tabs.apply_translation([(L / 2 - off) * s, 0, (spec.tab_height_mm - H) * s])
    spline = trimesh.creation.cylinder(radius=3.0 * s, height=spline_h * s, sections=24)
    spline.apply_translation([0, 0, spline_h / 2 * s])
    body = trimesh.util.concatenate([case, tabs, spline])
    T = np.eye(4)
    T[:3, :3] = _frame_from_axis(axis, length_dir_hint)
    T[:3, 3] = np.asarray(pivot, float) - T[:3, :3] @ np.array([0, 0, spline_h * s])
    body.apply_transform(T)
    return body


def stand_in(start, end, radius: float = 0.008) -> trimesh.Trimesh:
    """Capsule from `start` to `end` (model units): placeholder for a link part
    that the scan missed."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    if length < 1e-9:
        return trimesh.creation.icosphere(radius=radius, subdivisions=2).apply_translation(start)
    cap = trimesh.creation.capsule(height=length, radius=radius, count=[16, 8])
    T = np.eye(4)
    T[:3, :3] = _frame_from_axis(end - start, [1.0, 0, 0])
    T[:3, 3] = (start + end) / 2
    cap.apply_transform(T)
    return cap


def tag_block(T_model_tag, size: float = 0.03, thickness: float = 0.004) -> trimesh.Trimesh:
    """Thin box at a tag pose: minimal stand-in for a link known only by its tag."""
    box = trimesh.creation.box(extents=[size, size, thickness])
    box.apply_translation([0, 0, -thickness / 2])
    box.apply_transform(np.asarray(T_model_tag))
    return box


def cad_part(path: str | Path, transform=None, *, points_part=None, points_model=None, scale_to_model: float = 0.001):
    """Snap a CAD STL into the model frame, either with a known 4x4 transform
    (applied after scaling mm -> m) or by matching >=3 points picked on the
    part (part units) to the same points in the model frame."""
    from .registration import similarity_from_points
    mesh = trimesh.load(str(path), force="mesh")
    if transform is None and points_part is not None and points_model is not None:
        T, _, _ = similarity_from_points(np.asarray(points_part) * scale_to_model, points_model, with_scale=False)
    elif transform is not None:
        T = np.asarray(transform, float)
    else:
        raise ValueError("give a transform or matching point lists")
    mesh.apply_scale(scale_to_model)
    mesh.apply_transform(T)
    return mesh
