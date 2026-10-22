"""Run the whole pipeline from files: camera log (+ markers, + scan) -> twin.

Order (2026-10-22): load scan + health check -> tag placement
(markers.json) -> WIGGLE (camera: channel -> joint map, direction) -> ROM
(camera: axis, origin, terminal ranges, us <-> angle) -> register tags to scan
-> cut scan by linkage -> missing parts -> export.
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
from .parts import servo_body_mesh, servo_spec, stand_in, tag_block
from .registration import register_points, register_tags, tag_positions_model
from .scan import load_scan
from .segment import SegmentParams, link_meshes, segment_by_linkage


@dataclass
class BuildOptions:
    log: Path
    out: Path
    markers: Path | None = None
    scan: Path | None = None
    scan_tags: Path | None = None        # JSON {tag_id: [x, y, z]} in scan coordinates
    manual_points: Path | None = None    # JSON {"scan": [[x,y,z]x3+], "model": [[x,y,z]x3+]}
    fixes: FixOptions = field(default_factory=FixOptions)
    servo_models: dict = field(default_factory=dict)  # joint name or "ch<N>" -> library model
    stand_ins: bool = True
    name: str = "robot"
    preview: bool = True


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def build_twin(opt: BuildOptions, progress=print) -> dict:
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    log = load_log(opt.log)
    markers = load_markers(opt.markers) if opt.markers else (
        parse_markers(log.header["markers"]) if log.header.get("markers") else None)
    progress("wiggle: mapping channels to joints")
    wiggle = infer_linkage(log, markers, WiggleParams())
    _save(out / "wiggle.json", wiggle.to_dict())
    progress("ROM: fitting axes, origins and terminal ranges")
    kin = infer_kinematics(log, wiggle, RomParams())
    _save(out / "kinematics.json", kin.to_dict())
    tag_poses = tag_positions_model(kin)
    geometry: dict[str, LinkGeometry] = {}
    health_summary = None
    extra: dict = {"pipeline": {}}
    if opt.scan:
        progress("scan: health check")
        scan = load_scan(opt.scan)
        health = check_scan(scan)
        _save(out / "health.json", health.to_dict())
        health_summary = {"checklist": health.checklist(), "file": "health.json"}
        if opt.preview:
            from .preview import render_health
            render_health(scan, health, out / "health.png")
        fixed = apply_fixes(scan, health, opt.fixes)
        extra["pipeline"]["fixes"] = fixed.summary()
        reg = None
        if opt.scan_tags:
            centres = {int(k): np.asarray(v, float) for k, v in json.loads(Path(opt.scan_tags).read_text()).items()}
            reg = register_tags(centres, tag_poses)
        elif opt.manual_points:
            pts = json.loads(Path(opt.manual_points).read_text())
            reg = register_points(pts["scan"], pts["model"])
        if reg is None or fixed.faces is None:
            extra["pipeline"]["segmentation"] = ("skipped: no registration (give --scan-tags or --manual-points)"
                                                 if reg is None else "skipped: point cloud scan, mesh it first")
        else:
            progress("scan: registering and cutting by linkage")
            extra["pipeline"]["registration"] = reg.to_dict()
            V = fixed.vertices @ reg.T_model_scan[:3, :3].T + reg.T_model_scan[:3, 3]
            seg = segment_by_linkage(V, fixed.faces, kin, tag_poses, SegmentParams())
            extra["pipeline"]["segmentation"] = {"stats": seg.stats, "guides": seg.guides}
            meshes = link_meshes(V, fixed.faces, seg)
            for link, mesh in meshes.items():
                if link == "unassigned":
                    mesh.export(str(out / "unassigned.stl"))
                    continue
                # split each link's faces by provenance (scanned vs repaired)
                idx = seg.link_faces(link)
                prov = fixed.provenance[idx]
                for label in ("scanned", "repaired"):
                    sel = idx[prov == label]
                    if len(sel):
                        used, inv = np.unique(fixed.faces[sel].ravel(), return_inverse=True)
                        import trimesh
                        part = trimesh.Trimesh(V[used], inv.reshape(-1, 3), process=False)
                        geometry.setdefault(link, LinkGeometry(link)).parts.append(
                            (part, label, f"{len(sel)} faces from the scan"))
            if opt.preview:
                from .preview import render_labels
                render_labels(V, fixed.faces, seg.face_labels, seg.links, out / "segmentation.png",
                              "scan cut by tag linkage (model frame)")
    # missing parts: library servos on fitted axes
    servo_specs, servo_names = {}, {}
    for j in kin.joints:
        model = opt.servo_models.get(j.name) or opt.servo_models.get(f"ch{j.channel}")
        if not model:
            continue
        spec = servo_spec(model)
        servo_specs[j.name] = spec.to_dict()
        servo_names[j.name] = spec.model
        parent_origin = link_frames(kin)[j.parent]
        hint = parent_origin - np.asarray(j.pivot_model)
        body = servo_body_mesh(spec, j.pivot_model, j.axis_model, hint)
        geometry.setdefault(j.parent, LinkGeometry(j.parent)).parts.append(
            (body, "library-servo", f"{spec.model} on {j.name} axis; " + "; ".join(spec.assumptions)))
    if opt.stand_ins:
        origins = link_frames(kin)
        for link in kin.link_tags:
            if link in geometry and any(p[1] in ("scanned", "repaired") for p in geometry[link].parts):
                continue
            ends = [np.asarray(j.pivot_model) for j in kin.joints if j.parent == link]
            parts = [stand_in(origins.get(link, np.zeros(3)), e) for e in ends]
            parts += [tag_block(tag_poses[k]) for k in kin.link_tags[link] if k in tag_poses]
            if parts:
                import trimesh
                geometry.setdefault(link, LinkGeometry(link)).parts.append(
                    (trimesh.util.concatenate(parts), "stand-in", "capsules between joint pivots + tag blocks; "
                     "no scan geometry for this link"))
    progress("export: URDF, comms map, report")
    report = export_all(kin, out, name=opt.name, geometry=geometry, servo_models=servo_names,
                        servo_specs=servo_specs, health=health_summary, extra=extra)
    return report
