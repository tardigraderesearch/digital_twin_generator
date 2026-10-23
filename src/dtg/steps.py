"""The eight pipeline steps on a session folder. The GUI and the CLI both call these.

    1 scan      run_health(session, scan, fixes)
    2 tags      load_tags(session, markers.json)
    3 wiggle    wiggle_live(session, bench, channels, start_us)   or analyse_wiggle(session) on a recorded log
    4 rom       (live: Bench.start_rom / rom_move / rom_mark)      analyse_rom(session)
    5 register  run_register(session)
    6 segment   run_segment(session)
    7 parts     run_parts(session)
    8 export    run_export(session)

Steps 5-8 rebuild what they need from the session (``twin_state``), so any
step can be re-run after an earlier one changes. The GUI keeps the state in
memory between steps (``cache``).
"""
from __future__ import annotations

import json
from pathlib import Path

from .session import Session

FIX_NAMES = {"planes": "remove_planes", "wires": "strip_wires", "holes": "fill_holes", "fragments": "drop_fragments"}


def _noop(*_a, **_k):
    return None


# ---------------------------------------------------------------- 1. scan
def run_health(session: Session, scan_path, fixes=None, progress=_noop, preview: bool = True):
    """Health check on the scan (read-only), preview image, and a cleaned copy when fixes are chosen."""
    from .health import FixOptions, apply_fixes, check_scan, save_fixed
    from .scan import load_scan
    scan_path = Path(scan_path).resolve()
    progress(f"loading {scan_path.name}")
    scan = load_scan(scan_path)
    progress("checking: floor/turntable planes, wire loops, holes, fragments")
    report = check_scan(scan)
    out = session.path("health")
    out.mkdir(exist_ok=True)
    (out / "health.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    image = None
    if preview:
        from .preview import render_health
        image = render_health(scan, report, out / "health.png")
    fixed_path = None
    chosen = sorted(fixes or [])
    if chosen:
        opts = FixOptions(**{v: (k in chosen or v in chosen) for k, v in FIX_NAMES.items()})
        result = apply_fixes(scan, report, opts)
        fixed_path = out / f"{scan_path.stem}.fixed{'.ply' if result.faces is None else '.stl'}"
        save_fixed(result, fixed_path)
        progress(f"cleaned copy: {fixed_path.name} ({result.summary()}); original untouched")
    session.set_input("scan", str(scan_path))
    session.set_input("fixes", [k for k, v in FIX_NAMES.items() if k in chosen or v in chosen])
    n = len(report.findings)
    session.set_step("scan", "done", f"{n} finding(s); fixes: {', '.join(chosen) or 'none'}",
                     health="health/health.json", cleaned=str(fixed_path) if fixed_path else None)
    return scan, report, image, fixed_path


# ---------------------------------------------------------------- 2. tags
def load_tags(session: Session, markers_path=None, manifest: dict | None = None):
    """Copy aruco-tag-maker's markers.json into the session (or a manifest dict, e.g. the sim's)."""
    from .markers import load_markers
    if manifest is not None:
        session.markers_path.write_text(json.dumps(manifest, indent=2) + "\n")
        session.set_input("markers_source", "simulated bench (aruco-tag-maker manifest)")
    else:
        session.import_markers(markers_path)
    m = load_markers(session.markers_path)
    session.set_step("tags", "done", f"{len(m.tags)} tag(s), {m.dictionary}, {m.tag_size_mm} mm; root tag {m.root_tag}")
    return m


def markers(session: Session):
    from .markers import load_markers, parse_markers
    if session.markers_path.is_file():
        return load_markers(session.markers_path)
    if session.has_log():
        from .observations import load_log
        header = load_log(session.log_path).header
        if header.get("markers"):
            return parse_markers(header["markers"])
    return None


PLACEMENT_TIPS = [
    "Every moving link needs at least one tag; two at different angles keep it visible at both ROM ends.",
    "Tilt tags about 30 degrees to the camera: a tag seen face-on has an ambiguous pose.",
    "The base (root) tag never moves: put it on the frame, not the turntable.",
    "Tags must be in the scan too (step 5 registers the scan to them): stick them on before scanning.",
]


# ---------------------------------------------------------------- 3. wiggle
def wiggle_live(session: Session, bench, channels, *, start_us: int, delta_us: int = 80, progress=_noop,
                stop=lambda: False, reference: bool = True):
    """Reference capture, then one output at a time. The controller must already be armed."""
    m = markers(session)
    if m is None:
        raise ValueError("step 2 first: load markers.json")
    if bench.recorder is None:
        bench.start_recording(session.log_path, m)
        session.camera_path.write_text(json.dumps(bench.camera_dict(), indent=2) + "\n")
    if reference:
        progress("reference capture (the robot in its scanned pose)")
        frame = bench.reference()
        progress(f"reference: {len(frame.poses)} tag(s) seen")
    for ch in channels:
        if stop():
            break
        progress(f"wiggling output {ch}")
        bench.wiggle(ch, start_us=start_us, delta_us=delta_us)
    session.set_input("wiggle_start_us", start_us)
    return analyse_wiggle(session)


def analyse_wiggle(session: Session):
    """Channel -> joint map from the camera log. Flags stay flagged until the operator answers."""
    from .linkage import infer_linkage
    from .observations import load_log
    from .pipeline import apply_channel_confirmations
    if not session.has_log():
        raise ValueError("no camera log yet: wiggle (step 3) or use a recorded log")
    log = load_log(session.log_path)
    res = infer_linkage(log, markers(session))
    apply_channel_confirmations(res, session.confirmations)
    (session.path("wiggle.json")).write_text(json.dumps(res.to_dict(), indent=2, default=str) + "\n")
    answered = session.confirmations.get("channels", {})
    open_flags = [c.channel for c in res.channels if c.status == "needs_confirmation" and str(c.channel) not in answered]
    mapped = [c for c in res.channels if c.joint]
    session.set_step("wiggle", "flagged" if open_flags else "done",
                     f"{len(mapped)} joint(s) mapped; {len(open_flags)} need your confirmation"
                     + (f" (outputs {', '.join(map(str, open_flags))})" if open_flags else ""),
                     open_flags=open_flags)
    return res


# ---------------------------------------------------------------- 4..8 on a TwinState
def build_options(session: Session, out: Path | None = None):
    from .health import FixOptions
    from .pipeline import BuildOptions
    fixes = session.input("fixes") or []
    return BuildOptions(log=session.log_path, out=Path(out or session.twin_dir),
                        markers=session.markers_path if session.markers_path.is_file() else None,
                        scan=session.input("scan"),
                        scan_tags=session.input("scan_tags"), manual_points=session.input("manual_points"),
                        fixes=FixOptions(**{v: k in fixes for k, v in FIX_NAMES.items()}),
                        servo_models=session.input("servo_models") or {},
                        cad_parts=session.input("cad_parts") or [],
                        stand_ins=session.input("stand_ins", True), name=session.data.get("name", "robot"),
                        confirmations=session.confirmations)


def analyse_rom(session: Session, progress=_noop):
    from .pipeline import analyse
    if not session.has_log():
        raise ValueError("no camera log yet")
    st = analyse(build_options(session), progress)
    flagged = [j.name for j in st.kin.joints if j.status != "ok"
               and j.name not in session.confirmations.get("joints", {})]
    session.set_step("rom", "flagged" if flagged else "done",
                     f"{len(st.kin.joints)} joint(s) fitted; {len(flagged)} flagged fit(s) need your confirmation",
                     open_flags=flagged)
    return st


def twin_state(session: Session, upto: str, progress=_noop, cache: dict | None = None):
    """TwinState with every stage up to ``upto`` (rom, scan, register, segment, parts) done."""
    from . import pipeline
    order = ["rom", "scan", "register", "segment", "parts"]
    cache = cache if cache is not None else {}
    key = json.dumps(session.data["inputs"], sort_keys=True, default=str) + json.dumps(
        session.confirmations, sort_keys=True) + str(session.log_path.stat().st_mtime if session.has_log() else 0)
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
    st = cache.get("rom")
    if st is None:
        st = cache["rom"] = analyse_rom(session, progress)
    funcs = {"scan": pipeline.scan_stage, "register": pipeline.register_stage,
             "segment": pipeline.segment_stage, "parts": pipeline.parts_stage}
    for stage in order[1:order.index(upto) + 1]:
        if not cache.get(f"did_{stage}"):
            if stage == "parts":         # parts are added to geometry: rebuild from the segment result
                st.servo_specs.clear()
                st.servo_names.clear()
                for g in st.geometry.values():
                    g.parts[:] = [p for p in g.parts if p[1] in ("scanned", "repaired")]
            funcs[stage](st, progress)
            cache[f"did_{stage}"] = True
    return st


def run_register(session: Session, progress=_noop, cache=None):
    st = twin_state(session, "register", progress, cache)
    reg = st.registration
    if st.fixed is None:
        session.set_step("register", "skipped", "no scan: the twin uses stand-ins and library parts")
    elif reg is None:
        session.set_step("register", "pending", "give tag centres in the scan (3+) or 3 matching points")
    else:
        session.set_step("register", "done", f"{reg.method}: {reg.pairs} pairs, rms {reg.rms_m * 1000:.1f} mm")
    return st


def run_segment(session: Session, progress=_noop, cache=None):
    st = twin_state(session, "segment", progress, cache)
    seg = st.extra["pipeline"].get("segmentation")
    if st.seg is None:
        session.set_step("segment", "skipped", str(seg or "no scan"))
    else:
        stats = st.seg.stats
        session.set_step("segment", "done", ", ".join(f"{k}: {v}" for k, v in list(stats.items())[:4]))
    return st


def run_parts(session: Session, progress=_noop, cache=None):
    st = twin_state(session, "parts", progress, cache)
    counts: dict = {}
    for g in st.geometry.values():
        for _, prov, _ in g.parts:
            counts[prov] = counts.get(prov, 0) + 1
    session.set_step("parts", "done", ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "no parts")
    return st


def parts_table(st) -> list[dict]:
    rows = []
    for link, g in sorted(st.geometry.items()):
        for mesh, prov, note in g.parts:
            rows.append({"link": link, "provenance": prov, "faces": int(len(mesh.faces)), "note": note})
    return rows


def run_export(session: Session, progress=_noop, cache=None) -> dict:
    from .pipeline import export_stage
    st = twin_state(session, "parts", progress, cache)
    report = export_stage(st, progress)
    unconfirmed = report.get("unconfirmed_flagged_joints") or []
    session.set_step("export", "flagged" if unconfirmed else "done",
                     f"{len(report['joints'])} joint(s) in robot.urdf"
                     + (f"; flagged and not confirmed: {', '.join(unconfirmed)}" if unconfirmed else ""),
                     out=str(session.twin_dir))
    return report
