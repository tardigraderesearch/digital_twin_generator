"""Command line. ``dtg`` with no arguments opens the window.

Every pipeline step runs on a session folder (``--session PATH``, or
``--session latest``), the same folder the window uses:

    dtg session new bot1                          -> ~/digital_twin_generator_maps/bot1-<time>/
    dtg health SCAN.obj --session latest --fix planes wires
    dtg tags markers.json --session latest
    dtg wiggle --session latest --port sim --start-us 1500 --channels 0-7 --power-cutoff-confirmed
    dtg confirm --session latest --channel 6 yes
    dtg rom --session latest --port sim --power-cutoff-confirmed      (interactive: + - 1 2 w n q)
    dtg register --session latest --scan-tags tags.json
    dtg segment --session latest
    dtg parts --session latest --servo ch1=MG996R
    dtg export --session latest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _say(msg: str) -> None:
    print(f"[dtg] {msg}", file=sys.stderr, flush=True)


def _session(a):
    from .session import Session
    s = Session.open(a.session)
    _say(f"session {s.root}")
    return s


# ---------------------------------------------------------------- doctor / session
def cmd_doctor(a) -> int:
    import importlib.util
    import platform
    print(f"digital-twin-generator {__version__} on Python {platform.python_version()} ({sys.platform})")
    ok = True
    for module, why, required in (("numpy", "maths", True), ("scipy", "fits", True), ("trimesh", "scans", True),
                                  ("cv2", "camera / ArUco", True), ("PIL", "window theme", True),
                                  ("robo_prober", "controller, discovery, ROM, servo library", True),
                                  ("tkinter", "window", False), ("aruco_tag_maker", "simulated bench", False)):
        found = importlib.util.find_spec(module) is not None
        ok = ok and (found or not required)
        print(f"  {'ok' if found else '--'}  {module:<16} {why}{'' if found or not required else ' (REQUIRED)'}")
    from .hardware import sim_available
    sim, why = sim_available()
    print(f"  {'ok' if sim else '--'}  simulated bench  {why}")
    try:
        import robo_prober as rp
        ports = [p["device"] for p in rp.discover()]
        print(f"  ok  serial ports      {', '.join(ports) if ports else 'none found'} (listed, never opened)")
    except Exception as exc:  # noqa: BLE001
        print(f"  --  serial port listing failed: {exc}")
    from .session import maps_dir
    print(f"  ok  sessions          {maps_dir()}")
    return 0 if ok else 1


def cmd_session(a) -> int:
    from .session import Session
    if a.action == "new":
        s = Session.new(a.name)
        print(s.root)
        return 0
    s = Session.open(a.path or "latest")
    print("\n".join(s.summary_lines()))
    return 0


# ---------------------------------------------------------------- 1. scan
def cmd_health(a) -> int:
    from .health import FixOptions, apply_fixes, check_scan, save_fixed
    from .preview import render_health
    from .scan import load_scan
    if a.session:
        from .steps import run_health
        s = _session(a)
        _, report, image, fixed = run_health(s, a.scan, a.fix or [], progress=_say, preview=not a.no_preview)
        print("\n".join(report.checklist()))
        if image:
            print(f"preview: {image}")
        if fixed:
            print(f"cleaned copy (original untouched): {fixed}")
        return 0
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


# ---------------------------------------------------------------- 2. tags
def cmd_tags(a) -> int:
    from .steps import PLACEMENT_TIPS, load_tags
    s = _session(a)
    m = load_tags(s, a.markers)
    for name, tid in m.tags.items():
        print(f"  tag {tid:>3}  {name}{'  (root)' if tid == m.root_tag else ''}")
    print(f"{m.dictionary}, black square {m.tag_size_mm} mm")
    for tip in PLACEMENT_TIPS:
        print(f"  - {tip}")
    return 0


# ---------------------------------------------------------------- hardware helpers
def _bench(a):
    from .camera import CameraIntrinsics
    from .hardware import Bench
    intr = CameraIntrinsics.load(a.intrinsics) if getattr(a, "intrinsics", None) else None
    cam = getattr(a, "camera", None)
    if cam is not None and str(cam).isdigit():
        cam = int(cam)
    bench = Bench(a.port, camera=cam, intrinsics=intr)
    info = bench.connect()
    _say(f"connected: {info.get('id')} fw {info.get('fw')} (read-only so far)")
    if bench.estop_latched:
        bench.close()
        raise SystemExit("[dtg] estop_latched: the controller's e-stop is latched. Clear it: "
                         f"dtg estop --port {a.port} --clear --power-cutoff-confirmed")
    return bench


def _print_wiggle(res) -> None:
    for c in res.channels:
        print(f"channel {c.channel:>3}: {c.status:<18} {c.joint or '':<16} {c.parent or ''} -> {c.child or ''}"
              f"  dir {c.direction if c.direction is not None else '-'}  {'; '.join(c.reasons)}")
    for f in res.flags:
        print(f"flag: {f}")


# ---------------------------------------------------------------- 3. wiggle
def cmd_wiggle(a) -> int:
    if a.log and not a.session:                 # analysis of a recorded log, no session
        from .linkage import infer_linkage
        from .markers import load_markers, parse_markers
        from .observations import load_log
        log = load_log(a.log)
        markers = load_markers(a.markers) if a.markers else parse_markers(log.header["markers"]) \
            if log.header.get("markers") else None
        _print_wiggle(infer_linkage(log, markers))
        return 0
    from .steps import analyse_wiggle, load_tags, markers, wiggle_live
    s = _session(a)
    if a.log:
        s.import_log(a.log)
    if a.port:
        from .hardware import parse_channels
        if a.start_us is None:
            raise SystemExit("[dtg] bad_start: give --start-us, the pulse each servo is most likely at now "
                             "(the first pulse moves it there from wherever it is; horns may be mounted off-centre)")
        if not a.power_cutoff_confirmed:
            raise SystemExit("[dtg] cutoff_unconfirmed: arming needs --power-cutoff-confirmed (a physical power "
                             "cutoff within reach)")
        bench = _bench(a)
        try:
            if markers(s) is None and bench.markers_manifest():
                load_tags(s, manifest=bench.markers_manifest())
            channels = parse_channels(a.channels) if a.channels else [t.channel for t in bench.discover()
                                                                      if t.cls == "pca9685"]
            bench.arm(cutoff_confirmed=True)
            res = wiggle_live(s, bench, channels, start_us=a.start_us, delta_us=a.delta_us, progress=_say)
        finally:
            bench.close()
    else:
        res = analyse_wiggle(s)
    _print_wiggle(res)
    print(s.step("wiggle")["summary"])
    return 0


def cmd_confirm(a) -> int:
    s = _session(a)
    verdict = {"yes": "confirmed", "y": "confirmed", "no": "rejected", "n": "rejected", "clear": None}[a.answer]
    if a.channel is not None:
        s.confirm_channel(a.channel, verdict, a.note or "")
        from .steps import analyse_wiggle
        analyse_wiggle(s)
        print(s.step("wiggle")["summary"])
    elif a.joint:
        s.confirm_joint(a.joint, verdict, a.note or "")
        print(f"{a.joint}: {verdict or 'cleared'}")
    else:
        raise SystemExit("[dtg] give --channel N or --joint NAME")
    return 0


# ---------------------------------------------------------------- 4. ROM
ROM_HELP = ("keys: + / - step, +N / -N move N us, 1 = Set ROM 1, 2 = Set ROM 2, w = widen band, "
            "n = next joint, q = stop (disarm)")


def cmd_rom(a) -> int:
    from .steps import analyse_rom, analyse_wiggle
    s = _session(a)
    if a.port:
        if not sys.stdin.isatty() and not a.script:
            raise SystemExit("[dtg] the ROM sweep needs an operator at the keyboard (or --script FILE)")
        if not a.power_cutoff_confirmed:
            raise SystemExit("[dtg] cutoff_unconfirmed: arming needs --power-cutoff-confirmed")
        wig = analyse_wiggle(s)
        start = int(a.start_us or s.input("wiggle_start_us") or 0)
        if not start:
            raise SystemExit("[dtg] bad_start: give --start-us (where the wiggle left each servo)")
        joints = [(c.channel, c.joint, start) for c in wig.channels if c.joint
                  and (c.status == "mapped" or (s.confirmations["channels"].get(str(c.channel)) or {}).get("verdict")
                       == "confirmed")]
        if not joints:
            raise SystemExit("[dtg] no mapped or confirmed joints: run the wiggle (step 3) and answer its flags")
        lines = iter(Path(a.script).read_text().splitlines()) if a.script else None
        bench = _bench(a)
        try:
            from .steps import markers
            bench.start_recording(s.log_path, markers(s))
            bench.start_rom(joints, cutoff_confirmed=True)
            print(ROM_HELP)
            for ch, name, st in joints:
                bench.rom_enable(ch)
                while True:
                    pos = bench.rom.position(ch)
                    prompt = f"[{name} ch {ch}] {pos - st:+d} us (abs {pos}) > "
                    if lines is not None:
                        cmd = next(lines, "q")
                        print(prompt + cmd)
                    else:
                        cmd = input(prompt)
                    cmd = cmd.strip().lower()
                    if cmd in ("q", "quit"):
                        raise KeyboardInterrupt
                    if cmd == "n":
                        break
                    if cmd in ("1", "2"):
                        pulse, frame = bench.rom_mark(ch, int(cmd))
                        print(f"  ROM {cmd} = {pulse} us; tags seen: {sorted(frame.poses) if frame else '-'}")
                    elif cmd == "w":
                        print(f"  band now {bench.rom_widen(ch)}")
                    elif cmd in ("+", "-") or cmd[:1] in "+-" and cmd[1:].isdigit():
                        step = int(cmd[1:] or a.step_us)
                        bench.rom_move(ch, pos + (step if cmd[0] == "+" else -step))
                    elif cmd:
                        print(ROM_HELP)
        except (KeyboardInterrupt, EOFError):
            _say("stopped")
        finally:
            bench.close()
    st = analyse_rom(s, _say)
    for j in st.kin.joints:
        import math
        print(f"  {j.name:<16} ch {j.channel:>3}  {j.status:<8} {math.degrees(j.lower):+.1f}..{math.degrees(j.upper):+.1f} deg"
              + (f"  ({'; '.join(j.reasons)})" if j.reasons else ""))
    print(s.step("rom")["summary"])
    return 0


# ---------------------------------------------------------------- 5-8
def cmd_register(a) -> int:
    from .steps import run_register
    s = _session(a)
    if a.scan_tags:
        s.set_input("scan_tags", json.loads(Path(a.scan_tags).read_text()))
        s.set_input("manual_points", None)
    if a.manual_points:
        s.set_input("manual_points", json.loads(Path(a.manual_points).read_text()))
        s.set_input("scan_tags", None)
    run_register(s, _say)
    print(s.step("register")["summary"])
    return 0


def cmd_segment(a) -> int:
    from .steps import run_segment
    s = _session(a)
    run_segment(s, _say)
    print(s.step("segment")["summary"])
    return 0


def cmd_parts(a) -> int:
    from .steps import parts_table, run_parts
    s = _session(a)
    if a.servo:
        models = dict(s.input("servo_models") or {})
        models.update(dict(x.split("=", 1) for x in a.servo))
        s.set_input("servo_models", models)
    if a.no_stand_ins:
        s.set_input("stand_ins", False)
    st = run_parts(s, _say)
    for row in parts_table(st):
        print(f"  {row['link']:<16} {row['provenance']:<14} {row['faces']:>7} faces  {row['note'][:90]}")
    return 0


def cmd_export(a) -> int:
    from .steps import run_export
    s = _session(a)
    report = run_export(s, _say)
    print(f"wrote {s.twin_dir}: robot.urdf, comms_map.json, report.json")
    for j in report["joints"]:
        print(f"  {j['name']:<16} ch {j['channel']:>3}  {j['status']:<8} operator: {j['operator']}")
    print(s.step("export")["summary"])
    return 0


def cmd_estop(a) -> int:
    from .hardware import Bench
    bench = Bench(a.port)
    bench.connect()
    try:
        if a.clear:
            bench.clear_estop(cutoff_confirmed=a.power_cutoff_confirmed)
            print("e-stop cleared (nothing armed)")
        else:
            bench.estop()
            print("E-STOP latched: every output released")
    finally:
        bench.controller.transport.close()
    return 0


# ---------------------------------------------------------------- files-only tools
def cmd_synth(a) -> int:
    from .synthetic import generate
    log = generate(seed=a.seed, rot_noise_deg=a.rot_noise, trans_noise_mm=a.trans_noise, dropout=a.dropout)
    path = log.save(a.out)
    markers = Path(a.out).with_name("markers.json")
    markers.write_text(json.dumps(log.header["markers"], indent=2) + "\n")
    print(f"synthetic log: {path} ({len(log.frames)} frames); markers: {markers}")
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
    servos = load_servo_library()["servos"]
    if a.search:
        from robo_prober import servo_db
        ids = [e["id"] for e in servo_db.search(a.search, limit=15)]
        servos = sorted((s for s in servos if s["id"] in ids), key=lambda s: ids.index(s["id"]))
    for s in servos:
        print(f"{s['id']:<34} {s['model']:<14} body {s['body_mm']}  mass {s['mass_g']} g  "
              f"dims {'verified' if s['dims_verified'] else 'UNVERIFIED'}  mass {'verified' if s['mass_verified'] else 'UNVERIFIED'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dtg", description="Digital Twin Generator: built robot -> sim-ready model. "
                                 "Run with no command for the window.",
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"dtg {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="open the window (default)")
    g.add_argument("--session", help="open this session folder (or 'latest')")
    g.add_argument("--port", help="preselect a controller: a serial port, 'emulator' or 'sim[:seed]'")
    sub.add_parser("doctor", help="what this installation can do")
    se = sub.add_parser("session", help="new NAME | show [PATH]")
    se.add_argument("action", choices=["new", "show"])
    se.add_argument("name", nargs="?", default="robot")
    se.add_argument("path", nargs="?")

    def session_arg(p, required=True):
        p.add_argument("--session", required=required, help="session folder, or 'latest'")

    def hw_args(p):
        p.add_argument("--port", help="serial port, 'emulator' or 'sim[:seed]' (the virtual bench)")
        p.add_argument("--camera", help="camera index or URL (the sim brings its own)")
        p.add_argument("--intrinsics", help="camera calibration JSON (camera_matrix, dist_coeffs)")
        p.add_argument("--power-cutoff-confirmed", action="store_true",
                       help="a physical power cutoff is within reach (needed to arm)")

    h = sub.add_parser("health", help="1. scan health check with preview and optional cleaned copy")
    h.add_argument("scan")
    session_arg(h, required=False)
    h.add_argument("--out", default="out/health")
    h.add_argument("--mm-per-unit", type=float, help="override the units guess")
    h.add_argument("--fix", nargs="*", choices=["planes", "wires", "holes", "fragments"])
    h.add_argument("--no-preview", action="store_true")
    t = sub.add_parser("tags", help="2. load aruco-tag-maker's markers.json into the session")
    t.add_argument("markers")
    session_arg(t)
    w = sub.add_parser("wiggle", help="3. channel -> joint map: live (--port) or from a camera log")
    w.add_argument("log", nargs="?", help="recorded camera log (.jsonl)")
    w.add_argument("--markers")
    session_arg(w, required=False)
    hw_args(w)
    w.add_argument("--channels", help="outputs to wiggle, e.g. 0-7 (default: every PCA9685 channel found)")
    w.add_argument("--start-us", type=int, help="start pulse: where each servo most likely is now (required live)")
    w.add_argument("--delta-us", type=int, default=80)
    c = sub.add_parser("confirm", help="answer a flagged wiggle channel or ROM joint: yes | no | clear")
    session_arg(c)
    c.add_argument("--channel", type=int)
    c.add_argument("--joint")
    c.add_argument("answer", choices=["yes", "y", "no", "n", "clear"])
    c.add_argument("--note")
    r = sub.add_parser("rom", help="4. range of motion: live sweep (--port, interactive) and the fit")
    session_arg(r)
    hw_args(r)
    r.add_argument("--start-us", type=int, help="start pulse (default: the wiggle's)")
    r.add_argument("--step-us", type=int, default=40)
    r.add_argument("--script", help="read the keys from a file instead of the keyboard (tests)")
    rg = sub.add_parser("register", help="5. register the scan to the tags")
    session_arg(rg)
    rg.add_argument("--scan-tags", help="JSON {tag_id: [x,y,z]} tag centres in scan coordinates")
    rg.add_argument("--manual-points", help='JSON {"scan": [[x,y,z]...], "model": [[x,y,z]...]}')
    sg = sub.add_parser("segment", help="6. cut the scan into links")
    session_arg(sg)
    pa = sub.add_parser("parts", help="7. missing parts: library servos, stand-ins")
    session_arg(pa)
    pa.add_argument("--servo", nargs="*", help="joint=MODEL or chN=MODEL (model or servo_db id)")
    pa.add_argument("--no-stand-ins", action="store_true")
    ex = sub.add_parser("export", help="8. URDF, comms map, report")
    session_arg(ex)
    es = sub.add_parser("estop", help="latch the controller's e-stop, or clear it")
    es.add_argument("--port", required=True)
    es.add_argument("--clear", action="store_true")
    es.add_argument("--power-cutoff-confirmed", action="store_true")
    s = sub.add_parser("synth", help="write a synthetic camera log (bot1-like arm) for testing")
    s.add_argument("--out", default="out/synthetic/observations.jsonl")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rot-noise", type=float, default=0.4)
    s.add_argument("--trans-noise", type=float, default=0.5)
    s.add_argument("--dropout", type=float, default=0.01)
    b = sub.add_parser("build", help="all steps from files: log (+ scan) -> URDF, comms map, report")
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
    sv = sub.add_parser("servos", help="list (or search) the servo library")
    sv.add_argument("search", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd in (None, "gui"):
        from .gui import run
        return run(session=getattr(a, "session", None), port=getattr(a, "port", None))
    if a.cmd == "session" and a.action == "show":
        a.path = a.path or (a.name if a.name != "robot" else None)
    return {"doctor": cmd_doctor, "session": cmd_session, "health": cmd_health, "tags": cmd_tags,
            "wiggle": cmd_wiggle, "confirm": cmd_confirm, "rom": cmd_rom, "register": cmd_register,
            "segment": cmd_segment, "parts": cmd_parts, "export": cmd_export, "estop": cmd_estop,
            "synth": cmd_synth, "build": cmd_build, "servos": cmd_servos}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
