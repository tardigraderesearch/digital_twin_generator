"""Run the pipeline from files: camera log (+ markers, + scan) -> twin.

Order (2026-10-22): load scan + health check -> tag placement
(markers.json) -> WIGGLE (camera: channel -> joint map, direction) -> ROM
(camera: axis, origin, terminal ranges, us <-> angle) -> register tags to scan
-> cut scan by linkage -> missing parts -> export.

``build_twin`` runs every stage; the GUI and the step commands run the
stages one at a time on a ``TwinState``:

    st = analyse(opt)            # wiggle + ROM fits (+ operator confirmations)
    scan_stage(st)               # health check + chosen fixes on a copy
    register_stage(st)           # tag centres / 3 points -> model frame
    segment_stage(st)            # scan cut into links
    parts_stage(st)              # library servos, stand-ins, CAD
    report = export_stage(st)    # URDF, comms map, report
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .export import LinkGeometry, export_all, link_frames
from .health import FixOptions, apply_fixes, check_scan
from .kinematics import RomParams, infer_kinematics
from .linkage import WiggleParams, infer_linkage
from .markers import load_markers, parse_markers
from .observations import load_log
from .parts import cad_part, servo_body_mesh, servo_spec, stand_in, tag_block
from .registration import register_points, register_tags, tag_positions_model
from .scan import load_scan
from .segment import SegmentParams, link_meshes, segment_by_linkage


@dataclass
class BuildOptions:
    log: Path
    out: Path
    markers: Path | None = None
    scan: Path | None = None
    scan_tags: Path | dict | None = None      # JSON {tag_id: [x, y, z]} in scan coordinates
    manual_points: Path | dict | None = None  # JSON {"scan": [[x,y,z]x3+], "model": [[x,y,z]x3+]}
    fixes: FixOptions = field(default_factory=FixOptions)
    servo_models: dict = field(default_factory=dict)  # joint name or "ch<N>" -> library model / servo_db id
    cad_parts: list = field(default_factory=list)     # [{"link", "path", "transform" | "points_part"+"points_model"}]
    stand_ins: bool = True
    name: str = "robot"
    preview: bool = True
    confirmations: dict = field(default_factory=dict)  # {"channels": {ch: {verdict}}, "joints": {name: {verdict}}}


@dataclass
class TwinState:
    opt: BuildOptions
    out: Path
    log: object = None
    markers: object = None
    wiggle: object = None
    kin: object = None
    tag_poses: dict = field(default_factory=dict)
    scan: object = None
    health: object = None
    fixed: object = None
    registration: object = None
    seg: object = None
    V: np.ndarray | None = None
    geometry: dict = field(default_factory=dict)
    servo_specs: dict = field(default_factory=dict)
    servo_names: dict = field(default_factory=dict)
    health_summary: dict | None = None
    extra: dict = field(default_factory=lambda: {"pipeline": {}})


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def _json_or_path(value):
    if value is None or isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text())


def _verdict(table: dict, key) -> str | None:
    item = (table or {}).get(str(key))
    return item.get("verdict") if isinstance(item, dict) else item


def apply_channel_confirmations(wiggle, confirmations: dict) -> list[str]:
    """Operator answers on the wiggle result. A rejected channel is never used as a joint."""
    notes = []
    table = (confirmations or {}).get("channels", {})
    for c in wiggle.channels:
        verdict = _verdict(table, c.channel)
        if verdict == "rejected":
            if c.joint:
                notes.append(f"channel {c.channel}: operator rejected {c.parent} -> {c.child}; left out")
            c.joint, c.status = None, "rejected"
            c.reasons.append("operator rejected this mapping")
        elif verdict == "confirmed" and c.status == "needs_confirmation":
            c.reasons.append("operator confirmed this mapping on the robot")
    return notes


def analyse(opt: BuildOptions, progress=print) -> TwinState:
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    st = TwinState(opt=opt, out=out)
    st.log = load_log(opt.log)
    st.markers = load_markers(opt.markers) if opt.markers else (
        parse_markers(st.log.header["markers"]) if st.log.header.get("markers") else None)
    progress("wiggle: mapping channels to joints")
    st.wiggle = infer_linkage(st.log, st.markers, WiggleParams())
    notes = apply_channel_confirmations(st.wiggle, opt.confirmations)
    _save(out / "wiggle.json", st.wiggle.to_dict())
    progress("ROM: fitting axes, origins and terminal ranges")
    st.kin = infer_kinematics(st.log, st.wiggle, RomParams())
    st.kin.flags.extend(notes)
    rejected = {name for name, item in (opt.confirmations or {}).get("joints", {}).items()
                if _verdict({name: item}, name) == "rejected"}
    if rejected:
        st.kin.flags.extend(f"{j.name} (channel {j.channel}): operator rejected the fit; left out"
                            for j in st.kin.joints if j.name in rejected)
        st.kin.joints = [j for j in st.kin.joints if j.name not in rejected]
    _save(out / "kinematics.json", st.kin.to_dict())
    st.tag_poses = tag_positions_model(st.kin)
    st.extra["confirmations"] = opt.confirmations or {}
    return st


def scan_stage(st: TwinState, progress=print) -> None:
    opt = st.opt
    if not opt.scan:
        st.extra["pipeline"]["scan"] = "no scan given: links get stand-ins and library parts only"
        return
    progress("scan: health check")
    st.scan = load_scan(opt.scan)
    st.health = check_scan(st.scan)
    _save(st.out / "health.json", st.health.to_dict())
    st.health_summary = {"checklist": st.health.checklist(), "file": "health.json"}
    if opt.preview:
        from .preview import render_health
        render_health(st.scan, st.health, st.out / "health.png")
    st.fixed = apply_fixes(st.scan, st.health, opt.fixes)
    st.extra["pipeline"]["fixes"] = st.fixed.summary()


def register_stage(st: TwinState, progress=print):
    opt = st.opt
    if st.fixed is None:
        return None
    reg = None
    tags = _json_or_path(opt.scan_tags)
    points = _json_or_path(opt.manual_points)
    if tags:
        centres = {int(k): np.asarray(v, float) for k, v in tags.items()}
        reg = register_tags(centres, st.tag_poses)
    elif points:
        reg = register_points(points["scan"], points["model"])
    st.registration = reg
    if reg is not None:
        progress(f"registration: rms {reg.rms_m * 1000:.1f} mm, scale {reg.scale:.5g}")
        st.extra["pipeline"]["registration"] = reg.to_dict()
    return reg


def segment_stage(st: TwinState, progress=print):
    opt, fixed, reg = st.opt, st.fixed, st.registration
    if fixed is None:
        return None
    if reg is None or fixed.faces is None:
        st.extra["pipeline"]["segmentation"] = ("skipped: no registration (give tag centres or 3 points)"
                                                if reg is None else "skipped: point cloud scan, mesh it first")
        return None
    progress("scan: cutting into links by linkage")
    import trimesh
    V = fixed.vertices @ reg.T_model_scan[:3, :3].T + reg.T_model_scan[:3, 3]
    st.V = V
    seg = segment_by_linkage(V, fixed.faces, st.kin, st.tag_poses, SegmentParams())
    st.seg = seg
    st.extra["pipeline"]["segmentation"] = {"stats": seg.stats, "guides": seg.guides}
    meshes = link_meshes(V, fixed.faces, seg)
    for link, mesh in meshes.items():
        if link == "unassigned":
            mesh.export(str(st.out / "unassigned.stl"))
            continue
        idx = seg.link_faces(link)          # split each link's faces by provenance (scanned vs repaired)
        prov = fixed.provenance[idx]
        for label in ("scanned", "repaired"):
            sel = idx[prov == label]
            if len(sel):
                used, inv = np.unique(fixed.faces[sel].ravel(), return_inverse=True)
                part = trimesh.Trimesh(V[used], inv.reshape(-1, 3), process=False)
                st.geometry.setdefault(link, LinkGeometry(link)).parts.append(
                    (part, label, f"{len(sel)} faces from the scan"))
    if opt.preview:
        from .preview import render_labels
        render_labels(V, fixed.faces, seg.face_labels, seg.links, st.out / "segmentation.png",
                      "scan cut by tag linkage (model frame)")
    return seg


def parts_stage(st: TwinState, progress=print) -> None:
    opt, kin = st.opt, st.kin
    progress("missing parts: library servos, CAD, stand-ins")
    origins = link_frames(kin)
    for j in kin.joints:
        model = opt.servo_models.get(j.name) or opt.servo_models.get(f"ch{j.channel}")
        if not model:
            continue
        spec = servo_spec(model)
        st.servo_specs[j.name] = spec.to_dict()
        st.servo_names[j.name] = spec.model
        hint = origins[j.parent] - np.asarray(j.pivot_model)
        body = servo_body_mesh(spec, j.pivot_model, j.axis_model, hint)
        st.geometry.setdefault(j.parent, LinkGeometry(j.parent)).parts.append(
            (body, "library-servo", f"{spec.model} on {j.name} axis; " + "; ".join(spec.assumptions)))
    for item in opt.cad_parts or []:
        mesh = cad_part(item["path"], item.get("transform"), points_part=item.get("points_part"),
                        points_model=item.get("points_model"))
        st.geometry.setdefault(item["link"], LinkGeometry(item["link"])).parts.append(
            (mesh, "CAD", f"{Path(item['path']).name} snapped in by the operator"))
    if opt.stand_ins:
        for link in kin.link_tags:
            if link in st.geometry and any(p[1] in ("scanned", "repaired", "CAD") for p in st.geometry[link].parts):
                continue
            ends = [np.asarray(j.pivot_model) for j in kin.joints if j.parent == link]
            parts = [stand_in(origins.get(link, np.zeros(3)), e) for e in ends]
            parts += [tag_block(st.tag_poses[k]) for k in kin.link_tags[link] if k in st.tag_poses]
            if parts:
                import trimesh
                st.geometry.setdefault(link, LinkGeometry(link)).parts.append(
                    (trimesh.util.concatenate(parts), "stand-in", "capsules between joint pivots + tag blocks; "
                     "no scan geometry for this link"))


def export_stage(st: TwinState, progress=print) -> dict:
    opt = st.opt
    progress("export: URDF, comms map, report")
    report = export_all(st.kin, st.out, name=opt.name, geometry=st.geometry, servo_models=st.servo_names,
                        servo_specs=st.servo_specs, health=st.health_summary, extra=st.extra)
    joints_table = (opt.confirmations or {}).get("joints", {})
    channels_table = (opt.confirmations or {}).get("channels", {})
    unconfirmed = []
    for j in report["joints"]:
        answer = _verdict(joints_table, j["name"]) or (
            "confirmed (wiggle map)" if _verdict(channels_table, j["channel"]) == "confirmed" else None)
        j["operator"] = answer or ("not asked" if j["status"] == "ok" else "NOT CONFIRMED")
        if j["status"] != "ok" and not answer:
            unconfirmed.append(j["name"])
    report["unconfirmed_flagged_joints"] = unconfirmed
    (st.out / "report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.tolist()
                                                   if hasattr(o, "tolist") else str(o)) + "\n")
    return report


def build_twin(opt: BuildOptions, progress=print) -> dict:
    st = analyse(opt, progress)
    scan_stage(st, progress)
    register_stage(st, progress)
    segment_stage(st, progress)
    parts_stage(st, progress)
    return export_stage(st, progress)
