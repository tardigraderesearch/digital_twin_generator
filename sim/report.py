"""Write sim/REPORT.md from a Monte-Carlo summary (python -m sim report RUN_DIR)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .score import THRESHOLDS

HERE = Path(__file__).resolve().parent


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
    if notes:
        lines += [notes.strip(), ""]
    lines += ["## Monte-Carlo run", "",
              f"{n} randomized robots, light level {summary.get('light')}, wall time "
              f"{summary.get('wall_s', 0) / 60:.0f} min (real-time sessions in parallel).", "",
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
    return "\n".join(lines) + "\n"


def write(run_dir: Path, out: Path | None = None, **kwargs) -> Path:
    summary = json.loads((Path(run_dir) / "summary.json").read_text())
    notes_path = HERE / "report_notes.md"
    notes = notes_path.read_text() if notes_path.is_file() else None
    out = out or HERE / "REPORT.md"
    out.write_text(build(summary, notes=notes, **kwargs))
    return out
