"""One bench run: build a random robot, run the operator session, build the twin, score it."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .bench import build_world
from .operator import OperatorConfig, run_session
from .score import score_run


def run_one(seed: int, out_root: Path, cfg: OperatorConfig | None = None, *, video: bool = False,
            camera_noise: dict | None = None) -> dict:
    out = Path(out_root) / f"seed-{seed:04d}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    world = build_world(seed, camera_noise=camera_noise)
    (out / "truth.json").write_text(json.dumps(world.truth.to_dict(), indent=1) + "\n")
    recorder = None
    if video:
        from .video import FrameRecorder
        recorder = FrameRecorder(world)
        world.camera.tap = recorder.tap
    session = run_session(world, out, cfg).to_dict()
    twin = out / "twin"
    build_error = None
    try:
        from dtg.pipeline import BuildOptions, build_twin
        build_twin(BuildOptions(log=out / "observations.jsonl", out=twin, markers=out / "markers.json",
                                name=f"sim_bot_{seed}"), progress=lambda *_: None)
    except Exception:  # noqa: BLE001 - a pipeline crash is a finding
        build_error = traceback.format_exc()
        session.setdefault("errors", []).append("build_twin: " + build_error.strip().splitlines()[-1])
        twin = None
    score = score_run(world, session, twin)
    score["build_error"] = build_error
    score["timings"] = {**session.get("timings", {}), "run_s": round(time.monotonic() - t0, 1)}
    score["focus"] = session.get("focus", {})
    score["camera"] = {k: round(float(v), 3) for k, v in world.camera.noise.items()}
    score["confirmations"] = session.get("confirmations", {})
    (out / "score.json").write_text(json.dumps(score, indent=2, default=str) + "\n")
    if recorder is not None:
        recorder.save(out)
    return score
