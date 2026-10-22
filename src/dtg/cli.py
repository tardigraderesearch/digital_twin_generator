"""Command line. `dtg` with no arguments opens the window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _print(data) -> None:
    print(json.dumps(data, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def cmd_health(a) -> int:
    from .health import FixOptions, apply_fixes, check_scan, save_fixed
    from .preview import render_health
    from .scan import load_scan
    scan = load_scan(a.scan, mm_per_unit=a.mm_per_unit)
    report = check_scan(scan)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "health.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print("\n".join(report.checklist()))
    if not a.no_preview:
        print(f"preview: {render_health(scan, report, out / 'health.png')}")
    if a.fix:
        names = {"planes": "remove_planes", "wires": "strip_wires", "holes": "fill_holes", "fragments": "drop_fragments"}
        chosen = {names[x] for x in a.fix}
        opts = FixOptions(**{v: v in chosen for v in names.values()})
        result = apply_fixes(scan, report, opts)
        path = out / f"{Path(a.scan).stem}.fixed{'.ply' if result.faces is None else '.stl'}"
        save_fixed(result, path)
        print(f"cleaned copy (original untouched): {path}  {result.summary()}")
    return 0


def cmd_synth(a) -> int:
    from .synthetic import generate
    log = generate(seed=a.seed, rot_noise_deg=a.rot_noise, trans_noise_mm=a.trans_noise, dropout=a.dropout)
    path = log.save(a.out)
    markers = Path(a.out).with_name("markers.json")
    markers.write_text(json.dumps(log.header["markers"], indent=2) + "\n")
    print(f"synthetic log: {path} ({len(log.frames)} frames); markers: {markers}")
    return 0


def cmd_wiggle(a) -> int:
    from .linkage import infer_linkage
    from .markers import load_markers, parse_markers
    from .observations import load_log
    log = load_log(a.log)
    markers = load_markers(a.markers) if a.markers else parse_markers(log.header["markers"]) if log.header.get("markers") else None
    res = infer_linkage(log, markers)
    for c in res.channels:
        print(f"channel {c.channel:>2}: {c.status:<18} {c.joint or '':<16} {c.parent or ''} -> {c.child or ''}"
              f"  dir {c.direction if c.direction is not None else '-'}  {'; '.join(c.reasons)}")
    for f in res.flags:
        print(f"flag: {f}")
    return 0


def cmd_build(a) -> int:
    from .health import FixOptions
    from .pipeline import BuildOptions, build_twin
    servos = dict(x.split("=", 1) for x in (a.servo or []))
    names = {"planes": "remove_planes", "wires": "strip_wires", "holes": "fill_holes", "fragments": "drop_fragments"}
    fixes = FixOptions(**{v: (k in (a.fix or [])) for k, v in names.items()})
    report = build_twin(BuildOptions(log=Path(a.log), out=Path(a.out), markers=a.markers, scan=a.scan,
                                     scan_tags=a.scan_tags, manual_points=a.manual_points, fixes=fixes,
                                     servo_models=servos, stand_ins=not a.no_stand_ins, name=a.name))
    print(f"wrote {a.out}/robot.urdf, comms_map.json, report.json")
    for j in report["joints"]:
        print(f"  {j['name']:<16} ch {j['channel']:>2}  {j['status']:<8} limits {j['limits'][0]:+.3f} .. {j['limits'][1]:+.3f}"
              + (f"  ({'; '.join(j['reasons'])})" if j["reasons"] else ""))
    return 0


def cmd_servos(a) -> int:
    from .parts import load_servo_library
    for s in load_servo_library()["servos"]:
        print(f"{s['model']:<10} {s['manufacturer']:<22} body {s['body_mm']}  mass {s['mass_g']} g  "
              f"dims {'verified' if s['dims_verified'] else 'UNVERIFIED'}  mass {'verified' if s['mass_verified'] else 'UNVERIFIED'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dtg", description="Digital Twin Generator: built robot -> verified, sim-ready model")
    ap.add_argument("--version", action="version", version=f"dtg {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="open the window (default)")
    h = sub.add_parser("health", help="scan health check with preview and optional cleaned copy")
    h.add_argument("scan")
    h.add_argument("--out", default="out/health")
    h.add_argument("--mm-per-unit", type=float, help="override the units guess")
    h.add_argument("--fix", nargs="*", choices=["planes", "wires", "holes", "fragments"])
    h.add_argument("--no-preview", action="store_true")
    s = sub.add_parser("synth", help="write a synthetic camera log (bot1-like arm) for testing")
    s.add_argument("--out", default="out/synthetic/observations.jsonl")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rot-noise", type=float, default=0.4)
    s.add_argument("--trans-noise", type=float, default=0.5)
    s.add_argument("--dropout", type=float, default=0.01)
    w = sub.add_parser("wiggle", help="channel -> joint map from a camera log")
    w.add_argument("log")
    w.add_argument("--markers")
    b = sub.add_parser("build", help="full pipeline: log (+ scan) -> URDF, comms map, report")
    b.add_argument("log")
    b.add_argument("--out", default="out/twin")
    b.add_argument("--markers")
    b.add_argument("--scan")
    b.add_argument("--scan-tags", help="JSON {tag_id: [x,y,z]} tag centres in scan coordinates")
    b.add_argument("--manual-points", help='JSON {"scan": [[x,y,z]...], "model": [[x,y,z]...]}')
    b.add_argument("--fix", nargs="*", choices=["planes", "wires", "holes", "fragments"])
    b.add_argument("--servo", nargs="*", help="joint=MODEL or chN=MODEL, e.g. ch0=MG996R")
    b.add_argument("--no-stand-ins", action="store_true")
    b.add_argument("--name", default="robot")
    sub.add_parser("servos", help="list the servo library")
    a = ap.parse_args(argv)
    if a.cmd in (None, "gui"):
        from .gui import run
        return run()
    return {"health": cmd_health, "synth": cmd_synth, "wiggle": cmd_wiggle, "build": cmd_build,
            "servos": cmd_servos}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
