"""Write sim/REPORT.md from a Monte-Carlo summary (python -m sim report RUN_DIR)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .score import THRESHOLDS

HERE = Path(__file__).resolve().parent

NOTES = 'The virtual bench is an end-to-end, simulated hardware test for robo-prober v0.2 and DTG. It runs the real\ncode paths with no robot and no person present. Regenerate this report with `make sim-mc`\n(`python -m sim montecarlo --n 50 --report`).\n\n## Bugs found and fixed (local commits)\n\n**Safety**\n\n1. **robo-prober rom-gui: first-move snap after E-STOP / Disarm with no map loaded** (`b15926e`). The\n   "Start at" box kept the old value, so the next Enable jumped the servo 200 us (~20 deg). The box is now\n   always refreshed; the rom-gui smoke (`make sim-gui`) now shows a 0.99 deg start (rest offset) and passes.\n2. **DTG wiggle did a recovery move after any error**, transport faults included (`cd6f6b3`). It now stops\n   the controller (all outputs released) and re-raises, as in robo-prober.\n\n**Correctness: silent wrong channel->joint maps** (each was seen on the bench)\n\n3. One noisy capture on an unused channel mapped it to a joint. Motion must now appear on both sides of the\n   wiggle, go back and forth, and have a similar size on each side (`202d127`, `cf6d1ef`).\n4. A hidden tag was placed on the static base by a 90th-percentile rigidity test that ignored the frames\n   where it moved. The test now uses the worst frame (`202d127`).\n5. The filter for "cannot be placed" placeholder links compared an int with frozenset keys and never\n   filtered; a placeholder could become a silent parent (`202d127`).\n\n**Robustness: real-camera conditions**\n\n6. DTG\'s wiggle broke on a freshly armed v0.2 controller (no START) (`cd6f6b3`).\n7. Square-tag PnP flipped between its two solutions (a static base tag "moved" 30-45 deg). Corners are now\n   averaged, then one solve, and the solution is picked by continuity (`db235fd`, `be37c7f`).\n8. Focus: plain Laplacian variance is pure noise at high gain. The sweep uses LoG on averaged frames\n   (`658c813`). The focus sweep + lock is new, from the real OAK-D finding (`cd6f6b3`).\n9. A hidden root tag blanked whole channels. Another proven-rigid base tag now stands in (`3dbe95d`).\n10. Tags hidden at a Set ROM mark dropped the joint. The fit uses the same-pulse sample (`6069249`), or\n    extrapolates from the pulse fit, flagged (`02e4c49`).\n11. A joint whose parent joint was dropped made a two-root URDF, and `build_twin` crashed with no output.\n    The joint is now left out and reported (`93ea96c`).\n\n## What is simulated\n\n| piece | how |\n|---|---|\n| robot | A 3-joint arm modelled on bot1: cylindrical base with turntable, yaw turret, shoulder pitch, hanging forearm. Link sizes are around the bot1 scan. Hidden truth is drawn per seed: which PCA9685 channel (0-7) drives which joint and which are unused, direction, pulse->angle gain (0.085-0.105 deg/us), centre trim, horn mis-mount (whole spline teeth plus a few degrees), speed, first-order lag, gear backlash, and hard stops 400-850 us either side of 1500 us. Commanding past a stop stalls the servo and logs a "damage risk" event. Released servos go limp where they are. Pure kinematics. |\n| controller | robo-prober\'s **real C++ firmware core** (`rp_core.cpp`), built natively with `bench_main.cpp`, behind a pseudo terminal. The unmodified host API connects to it like a USB serial device. Its PWM outputs drive the virtual servos (`robo_prober.firmware_bench`, added on v0.2). |\n| camera | A z-buffer software renderer, 1920x1080 with fx 1541 like the OAK-D at 1080p. Tags come from aruco-tag-maker (DICT_4X4_100, 30 mm black square, its markers.json and bitmaps), stuck on standoffs where the camera sees them at rest. The model adds dim-room shot and read noise at high gain, light varying per run, mains flicker, depth-dependent defocus driven by a lens position, an autofocus that locks on the far wall, and an occasional hand in front. It exposes a `cv2.VideoCapture`-like object, so frames go through DTG\'s **real** `ArucoTracker`. The noise is calibrated against the real OAK-D bench numbers. |\n| operator | Locks focus with DTG\'s sweep, captures the reference, arms through robo-prober, and runs WIGGLE through DTG\'s `wiggle_channel` on channels 0-7. It re-runs flagged wiggles (as DTG asks) and confirms only what it can see. ROM runs through robo-prober\'s `RomSession` exactly as the sliders drive it: Enable at the shown start, slow targets, Widen when the band runs out, and Set ROM 1 / Set ROM 2 4-10 deg short of the stops it can see. DTG\'s `rom_sample` / `rom_mark` record along the way. Then it probes the firmware\'s safety behaviour. The rom-gui window gets a separate event-driven smoke test (`make sim-gui`). |\n| scoring | DTG\'s full `build_twin` runs on the recorded log. Scoring compares it with the hidden truth in the true root-tag frame: map, direction, unused channels, axes, pivots, terminal ranges, us<->angle, URDF, and a twin check (random pulses inside the measured safe ranges; tag positions from URDF + comms map vs the robot). Safety is scored too. |\n\n## Observations (not changed)\n\n- Firmware LIMIT bands persist across sessions until reboot. A band widened in rom-gui and never confirmed\n  by ROM is still active for the next client; DISARM does not restore 1400-1600 us.\n- Esc E-STOP needs the ROM window to have keyboard focus. The red button and the watchdog (a frozen window\n  releases in about 1 s) do not.\n- Wiring DTG\'s ROM hooks into a `heartbeat="manual"` Tk thread puts settle + capture close to the 1 s\n  watchdog. Run them off the UI thread.\n- With one tag per link, the ends of a large ROM are often unseen (the tag turns away or goes behind\n  another link). Use 2+ tags per link at different angles.\n\n## What the bench cannot validate\n\n- Real optics: distortion (zero here), rolling shutter, motion blur, glare on printed tiles, real ISP and\n  auto-exposure. The noise matches the measured jitter only roughly.\n- Real servos: backlash is a simple dead band. No sag, gravity, stall current, heating or horn slip, so "no\n  hard-stop contact" only covers stops the operator can see.\n- Scan registration on real tags, segmentation of a real scan, and the health check (no scan here).\n- Humans: the scripted operator sees the stops perfectly and confirms flagged items from the truth.\n- Replays rebuild the same robot, scene and noise draws, but thread timing differs.'


def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def _pct(k, n):
    return f"{100 * k / n:.0f}% ({k}/{n})" if n else "-"


def build(summary: dict, calibration: str | None = None, notes: str | None = None, artifacts: dict | None = None) -> str:
    results = summary["results"]
    n = len(results)
    ok = [r for r in results if r.get("pass")]
    lines = ["# Virtual bench report", ""]
    safety = [("arming (cutoff + token; nothing before ARM; ARM alone drives nothing)", ["arming_guarded", "arm_outputs_nothing"]),
              ("ROM limits never exceeded (firmware rejects past marks; no PWM outside ROM)", ["rom_limits_held"]),
              ("never commanded past a hard stop / no stop contact", ["pwm_never_past_stops", "no_hard_stop_contact"]),
              ("slew capped at the firmware limit", ["slew_capped"]),
              ("watchdog releases on host silence and on unplug", ["watchdog"]),
              ("e-stop (command + input) releases and latches", ["estop_latch"]),
              ("out-of-band / bad token / bad value rejected", ["out_of_band_rejected"]),
              ("no first-move snap at START", ["no_first_move_snap"]),
              ("wrong IO: unused channels only wiggled; no silent wrong map", ["unused_channels_only_wiggled", "no_silent_wrong_map"]),
              ("released servos hold still", ["released_servos_hold"])]
    lines += ["## Safety verdict", "", f"{n} randomized robots (random wiring, inverted joints, mis-mounted horns).", "",
              "| safety property | result |", "|---|---|"]
    for label, keys in safety:
        good = sum(all((r.get("checks") or {}).get(k) for k in keys) for r in results)
        lines.append(f"| {label} | {'PASS' if good == n else 'FAIL'} ({good}/{n}) |")
    lines.append("")
    if notes:
        lines += [notes.strip(), ""]
    lines += ["## Monte-Carlo run", "",
              f"{n} randomized robots, light level {summary.get('light')}"
              + (f", wall time {summary['wall_s'] / 60:.0f} min" if summary.get("wall_s") else "")
              + (f" (batch capped early: {summary['partial']})" if summary.get("partial") else "")
              + (f". {summary['note']}" if summary.get("note") else "")
              + ". Real-time sessions in parallel on one desktop (load ~40 on 32 cores).", "",
              f"**Overall pass (every check below): {_pct(len(ok), n)}**", ""]
    # per-check pass rates
    names = sorted({k for r in results for k in (r.get("checks") or {})})
    lines += ["### Checks", "", "| check | pass rate |", "|---|---|"]
    for k in names:
        vals = [r["checks"][k] for r in results if k in (r.get("checks") or {})]
        lines.append(f"| {k} | {_pct(sum(bool(v) for v in vals), len(vals))} |")
    # channel map
    first_exact = sum(bool((r.get("map_first_pass") or {}).get("exact")) for r in results)
    exact = sum(bool(r.get("map_exact")) for r in results)
    wrong = sum(int(r.get("map_silent_wrong") or 0) + int((r.get("map_first_pass") or {}).get("silent_wrong") or 0)
                for r in results)
    flagged_runs = sum(bool(r.get("map_flagged")) for r in results)
    rewiggled = sum(bool((r.get("map_first_pass") or {}).get("rewiggles")) for r in results)
    lines += ["", "### Channel -> joint map (wiring, direction, unused channels)", "",
              "| measure | result |", "|---|---|",
              f"| exact on the first wiggle | {_pct(first_exact, n)} |",
              f"| exact after re-running flagged channels (operator follows DTG's advice) | {_pct(exact, n)} |",
              f"| runs still with flagged channels (operator must confirm) | {_pct(flagged_runs, n)} |",
              f"| runs where the operator re-ran a wiggle | {_pct(rewiggled, n)} |",
              f"| **silent wrong mappings** (mapped, wrong child/parent) | **{wrong}** |", ""]
    # accuracy
    joints = [j for r in results for j in (r.get("joints") or {}).values() if j.get("found")]
    total_joints = sum(len(r.get("joints") or {}) for r in results)
    lines += ["### Joint accuracy vs ground truth", "",
              f"{len(joints)} of {total_joints} joints fitted ({_pct(len(joints), total_joints)}).", "",
              "| metric | n | median | p95 | max | threshold |", "|---|---|---|---|---|---|"]
    metrics = [("axis error (deg)", "axis_deg", lambda j: j.get("axis_deg"), THRESHOLDS["axis_deg"]),
               ("pivot error, perpendicular to axis (mm)", "pivot_mm", lambda j: j.get("pivot_mm"),
                THRESHOLDS["pivot_mm"]),
               ("terminal range error (deg)", "terminal_deg", lambda j: j.get("terminal_deg"),
                THRESHOLDS["terminal_deg"]),
               ("us -> angle gain error (%)", "gain_pct", lambda j: abs(j["gain_pct"]) if "gain_pct" in j else None,
                THRESHOLDS["gain_pct"]),
               ("zero pulse error (us)", "zero_pulse_us",
                lambda j: abs(j["zero_pulse_us"]) if "zero_pulse_us" in j else None, THRESHOLDS["zero_pulse_us"])]
    for label, _, fn, thr in metrics:
        s = _stats(fn(j) for j in joints)
        if s:
            lines.append(f"| {label} | {s['n']} | {s['median']:.2f} | {s['p95']:.2f} | {s['max']:.2f} | {thr} |")
    lines += ["", "Split by what DTG itself said about the joint (a large error on a joint DTG marked `ok` "
              "would be the dangerous case):", "",
              "| DTG status | joints | axis deg (median / max) | pivot mm (median / max) | terminal deg "
              "(median / max) | gain % (median / max) | wrong direction |", "|---|---|---|---|---|---|---|"]
    for st in ("ok", "flagged"):
        sel = [j for j in joints if j.get("status") == st]
        if not sel:
            continue
        def mm(key, f=lambda v: v):
            v = [f(j[key]) for j in sel if j.get(key) is not None]
            return f"{np.median(v):.2f} / {max(v):.2f}" if v else "-"
        lines.append(f"| {st} | {len(sel)} | {mm('axis_deg')} | {mm('pivot_mm')} | {mm('terminal_deg')} | "
                     f"{mm('gain_pct', abs)} | {sum(not j.get('direction_ok', True) for j in sel)} |")
    lines.append("")
    dir_ok = [j.get("direction_ok") for j in joints if "direction_ok" in j]
    lines.append(f"| direction correct | {len(dir_ok)} | {_pct(sum(dir_ok), len(dir_ok))} | | | all |")
    twin = _stats((r.get("twin") or {}).get("p95_mm") for r in results)
    twin_med = _stats((r.get("twin") or {}).get("median_mm") for r in results)
    if twin:
        lines += ["", "**Twin check** (command random pulses inside the measured safe ranges; where URDF + comms map "
                  "put every tag vs where the robot really puts it):", "",
                  f"per-run median {twin_med['median']:.1f} mm; per-run p95: median {twin['median']:.1f} mm, worst "
                  f"{twin['max']:.1f} mm (threshold p95 < {THRESHOLDS['twin_p95_mm']} mm; {twin['n']} runs)."]
    urdf = [r.get("urdf_valid") for r in results if "urdf_valid" in r]
    lines += ["", f"URDF valid (DTG validator + yourdfpy load): {_pct(sum(bool(u) for u in urdf), len(urdf))}.", ""]
    # safety
    sf = [r.get("safety") or {} for r in results]
    lines += ["### Safety under simulation (real firmware core, real host)", "",
              "| behaviour | result |", "|---|---|"]
    contacts = sum(int(s.get("hard_stop_contacts") or 0) for s in sf)
    lines.append(f"| hard-stop contacts (damage risk) in all sessions | **{contacts}** |")
    js = _stats(s.get("max_start_jump_deg") for s in sf)
    if js:
        lines.append(f"| first-move jump at START (deg), per run max | median {js['median']:.1f}, worst {js['max']:.1f} "
                     f"(threshold {THRESHOLDS['start_jump_deg']}) |")
    for label, key in (("watchdog release on host silence (s)", "watchdog_release_s"),
                       ("release after host unplug / crash (s)", "unplug_release_s"),
                       ("E-STOP release (s)", "estop_release_s")):
        s = _stats(x.get(key) for x in sf)
        if s:
            lines.append(f"| {label} | median {s['median']:.3f}, max {s['max']:.3f} |")
    for label, key in (("E-STOP latches until cleared", "estop_latched"),
                       ("hardware e-stop input latches", "estop_pin_latched"),
                       ("out-of-band / bad-token / bad-value moves rejected, output unchanged", "out_of_band_rejected"),
                       ("released servos stay put (no motion after release)", "holds_still_after_release")):
        vals = [x.get(key) for x in sf if key in x]
        lines.append(f"| {label} | {_pct(sum(bool(v) for v in vals), len(vals))} |")
    arm = [x.get("arm_alone_outputs") for x in sf if "arm_alone_outputs" in x]
    lines.append(f"| ARM alone drives no output | {_pct(sum(a == 0 for a in arm), len(arm))} |")
    vals = [x.get("arming_guarded") for x in sf if "arming_guarded" in x]
    lines.append(f"| ARM refused without cutoff / with bad token; START, MOVE refused before ARM | "
                 f"{_pct(sum(bool(v) for v in vals), len(vals))} |")
    edges = sum(int(x.get("pwm_edges") or 0) for x in sf)
    lines.append(f"| PWM edges audited (every output change the firmware made) | {edges} |")
    for label, key in (("PWM pulses commanding a joint past its hard stop", "pwm_past_hard_stop"),
                       ("PWM pulses outside the marked ROM range after Set ROM 1+2", "pwm_outside_rom"),
                       ("pulses on unused channels beyond the +-80 us wiggle / default band",
                        "unused_channel_far_pulses")):
        lines.append(f"| {label} | **{sum(int(x.get(key) or 0) for x in sf)}** |")
    rates = _stats(x.get("max_slew_us_s") for x in sf)
    if rates:
        lines.append(f"| firmware slew (us/s, worst 50 ms window per run; cap 250) | median {rates['median']:.0f}, "
                     f"worst {rates['max']:.0f} |")
    vals = [x.get("rom_band_enforced") for x in sf if x.get("rom_band_enforced") is not None]
    lines.append(f"| MOVE 20 us past a marked ROM end rejected by the firmware, output unchanged | "
                 f"{_pct(sum(bool(v) for v in vals), len(vals))} |")
    hw = [r.get("hardware_cases") or {} for r in results]
    inv = sum(h.get("inverted_joints", 0) for h in hw)
    horn = sum(h.get("horn_off_by_teeth", 0) for h in hw)
    lines += ["", f"Hardware cases covered: every run has its own random wiring (3 of channels 0-7, 5 unused); "
              f"{inv} inverted joints and {horn} horns mounted one or more spline teeth off across {len(hw)} runs."]
    # light level
    bins = [("dim (full well < 35 e-, like the OAK-D bench)", 0, 35), ("medium (35-80 e-)", 35, 80),
            ("bright (> 80 e-)", 80, 1e9)]
    lines += ["", "### By light level", "", "| light | runs | overall pass | map exact | all joints fitted |",
              "|---|---|---|---|---|"]
    for label, lo, hi in bins:
        rs = [r for r in results if lo <= (r.get("camera") or {}).get("full_well", -1) < hi]
        if rs:
            lines.append(f"| {label} | {len(rs)} | {_pct(sum(bool(r.get('pass')) for r in rs), len(rs))} | "
                         f"{_pct(sum(bool(r.get('map_exact')) for r in rs), len(rs))} | "
                         f"{_pct(sum(bool((r.get('checks') or {}).get('all_joints_fitted')) for r in rs), len(rs))} |")
    focus = [r.get("focus") or {} for r in results]
    af = [f.get("autofocus_tags_detected") for f in focus if "autofocus_tags_detected" in f]
    lens_err = _stats(abs(f["lens"] - f["sharp_lens_truth"]) for f in focus if "lens" in f)
    if af:
        lines += ["", f"Focus: with autofocus (locked on the wall) a median of {np.median(af):.0f} tags were detected; "
                  f"after DTG's sweep + lock the lens was within {lens_err['median']:.0f} (median) / "
                  f"{lens_err['max']:.0f} (worst) steps of the subject's sharp position." if lens_err else ""]
    # failures
    failing = [r for r in results if not r.get("pass")]
    if failing:
        lines += ["", "### Failing seeds (replay: `python -m sim replay --seed N`)", "",
                  "| seed | light (e-) | failed checks |", "|---|---|---|"]
        for r in failing:
            failed = [k for k, v in (r.get("checks") or {}).items() if not v]
            extra = f" untagged: {r['untagged_links']}" if r.get("untagged_links") else ""
            lines.append(f"| {r['seed']} | {(r.get('camera') or {}).get('full_well', '-')} | "
                         f"{', '.join(failed)}{extra} |")
    if calibration:
        lines += ["", "## Virtual camera vs the real OAK-D bench", "", calibration.strip()]
    if artifacts:
        lines += ["", "## Artifacts", ""] + [f"- {k}: `{v}`" for k, v in artifacts.items()]
    text = "\n".join(lines) + "\n"
    # safety first: move the simulated-safety table right under the run summary
    a, b = text.find("### Safety under simulation"), text.find("### By light level")
    c = text.find("### Checks")
    if -1 not in (a, b, c):
        text = text[:c] + text[a:b] + text[c:a] + text[b:]
    return text


def write(run_dir: Path, out: Path | None = None, **kwargs) -> Path:
    run_dir = Path(run_dir)
    if (run_dir / "summary.json").is_file():
        summary = json.loads((run_dir / "summary.json").read_text())
    else:   # a batch stopped early: report the seeds that finished, and say so
        results = [json.loads(p.read_text()) for p in sorted(run_dir.glob("seed-*/score.json"))]
        started = len(list(run_dir.glob("seed-*")))
        summary = {"results": results, "light": "mixed (dim..bright per seed)",
                   "wall_s": None, "partial": f"{len(results)} finished of {started} started"}
    notes = NOTES
    out = out or HERE / "REPORT.md"
    out.write_text(build(summary, notes=notes, **kwargs))
    return out
