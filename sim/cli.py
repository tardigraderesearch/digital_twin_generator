"""python -m sim: the virtual bench.

    python -m sim run --seed 3 [--video]          one randomized robot, end to end
    python -m sim montecarlo --n 50 --workers 16  many robots, pass rates, failing seeds
    python -m sim replay --seed 17                same robot/scene/noise as a Monte-Carlo seed
    python -m sim report runs/sim-mc              write sim/REPORT.md
    python -m sim calibrate --seed 3              virtual camera vs the real OAK-D bench numbers
    python -m sim gui-smoke --seed 3              robo-prober's ROM window driven by events (needs DISPLAY)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LIGHT = {"dim": 25.0, "medium": 55.0, "bright": 120.0}


def _noise(light: str | None):
    if light in (None, "mixed", "random"):
        return None
    return {"full_well": LIGHT.get(light) or float(light)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one randomized robot, end to end")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="runs/sim")
    r.add_argument("--video", action="store_true", help="also write sim-run.mp4/.gif (virtual camera + angles)")
    r.add_argument("--light", default=None, help="dim | medium | bright | <full-well electrons> (default random)")
    rp = sub.add_parser("replay", help="re-run one seed (same robot, scene and noise draws)")
    rp.add_argument("--seed", type=int, required=True)
    rp.add_argument("--out", default="runs/sim-replay")
    rp.add_argument("--light", default="mixed")
    rp.add_argument("--video", action="store_true")
    m = sub.add_parser("montecarlo", help="many randomized robots in parallel")
    m.add_argument("--n", type=int, default=50)
    m.add_argument("--start", type=int, default=0)
    m.add_argument("--workers", type=int, default=16)
    m.add_argument("--out", default="runs/sim-mc")
    m.add_argument("--light", default="mixed", help="mixed (dim..bright per seed) or a fixed level")
    m.add_argument("--report", action="store_true", help="write sim/REPORT.md when done")
    rep = sub.add_parser("report", help="write sim/REPORT.md from a Monte-Carlo run")
    rep.add_argument("run_dir")
    rep.add_argument("--calibration-seeds", type=int, nargs="*", default=[])
    rep.add_argument("--video-dir", help="run folder holding sim-run.mp4/.gif to link")
    c = sub.add_parser("calibrate", help="detection rate, size and jitter per tag on a static scene")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--frames", type=int, default=30)
    c.add_argument("--light", default=None)
    g = sub.add_parser("gui-smoke", help="rom-gui driven by events against the bench (needs DISPLAY)")
    g.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd in ("run", "replay"):
        from .run import run_one
        light = a.light
        noise = _noise(light)
        if light == "mixed":
            import numpy as np
            rng = np.random.default_rng(30_000 + a.seed)
            noise = {"full_well": float(np.exp(rng.uniform(np.log(18), np.log(160))))}
        score = run_one(a.seed, Path(a.out), video=a.video, camera_noise=noise)
        failed = [k for k, v in (score.get("checks") or {}).items() if not v]
        print(json.dumps({"seed": a.seed, "pass": score.get("pass"), "failed": failed,
                          "out": str(Path(a.out) / f"seed-{a.seed:04d}")}, indent=1))
        return 0 if score.get("pass") else 1
    if a.cmd == "montecarlo":
        from .montecarlo import run_batch
        summary = run_batch(range(a.start, a.start + a.n), Path(a.out), workers=a.workers, light=a.light)
        ok = sum(bool(r.get("pass")) for r in summary["results"])
        print(f"{ok}/{len(summary['results'])} passed; summary: {Path(a.out) / 'summary.json'}")
        if a.report:
            from .report import write
            print(f"report: {write(Path(a.out))}")
        return 0
    if a.cmd == "report":
        from .calibrate import format_table, measure
        from .report import write
        cal = "\n\n".join(format_table(measure(s)) for s in a.calibration_seeds) or None
        artifacts = None
        if a.video_dir:
            d = Path(a.video_dir)
            artifacts = {p.name: str(p) for p in sorted(d.glob("sim-*")) if p.suffix in (".mp4", ".gif", ".png")}
        print(f"report: {write(Path(a.run_dir), calibration=cal, artifacts=artifacts)}")
        return 0
    if a.cmd == "calibrate":
        from .calibrate import format_table, measure
        print(format_table(measure(a.seed, a.frames, noise=_noise(a.light))))
        return 0
    if a.cmd == "gui-smoke":
        from .gui_smoke import run_gui_smoke
        out = run_gui_smoke(a.seed)
        print(json.dumps(out, indent=1, default=str))
        return 0 if out["ok"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
