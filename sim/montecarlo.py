"""Monte-Carlo over randomized robots: run N seeds in parallel, aggregate.

Each seed is a full bench session in its own process (its own firmware
bench and pty), in real time. Failing seeds are listed in
``failing_seeds.json`` with their failed checks; ``python -m sim replay
--seed N`` rebuilds the same robot, scene and camera noise (thread timing
differs run to run, so a replay is the same bench, not the same frames).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _worker(seed: int, out_root: str, light: str | None) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    from .run import run_one
    import numpy as np
    noise = None
    if light == "mixed":   # dim (OAK-D bench, ISO 1600) to an ordinary lit room
        rng = np.random.default_rng(30_000 + seed)
        noise = {"full_well": float(np.exp(rng.uniform(np.log(18), np.log(160))))}
    try:
        return run_one(seed, Path(out_root), camera_noise=noise)
    except Exception as error:  # noqa: BLE001 - a harness crash is recorded, not fatal to the batch
        import traceback
        return {"seed": seed, "pass": False, "errors": [f"harness: {error}", traceback.format_exc()],
                "checks": {"no_errors": False}}


def run_batch(seeds, out_root: Path, workers: int = 16, light: str | None = "mixed", progress=print) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, s, str(out_root), light): s for s in seeds}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            failed = [k for k, v in (r.get("checks") or {}).items() if not v]
            progress(f"seed {r['seed']:4d}: {'PASS' if r.get('pass') else 'FAIL'}"
                     f"{'' if not failed else '  ' + ', '.join(failed)}  ({len(results)}/{len(futures)})")
    results.sort(key=lambda r: r["seed"])
    summary = {"seeds": [r["seed"] for r in results], "light": light, "wall_s": round(time.monotonic() - t0, 1),
               "results": results}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    failing = {r["seed"]: [k for k, v in (r.get("checks") or {}).items() if not v] for r in results
               if not r.get("pass")}
    (out_root / "failing_seeds.json").write_text(json.dumps(failing, indent=1) + "\n")
    return summary
