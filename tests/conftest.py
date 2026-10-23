import os
from pathlib import Path

import pytest

# Test data stays out of git: bot1 is referenced by path and tests skip without it.
_CANDIDATES = [Path.home() / p / "assets/source/bot1.obj"
               for p in ("Projects/robo_prober", "Projects/robo-prober", "Projects/robo-prober-v0.1-old")]
BOT1 = Path(os.environ.get("DTG_BOT1") or next((p for p in _CANDIDATES if p.is_file()), _CANDIDATES[0]))


@pytest.fixture(scope="session")
def bot1_scan():
    if not BOT1.is_file():
        pytest.skip(f"bot1 scan not found at {BOT1} (set DTG_BOT1)")
    from dtg.scan import load_scan
    return load_scan(BOT1)


@pytest.fixture(scope="session")
def bot1_report(bot1_scan):
    from dtg.health import check_scan
    return check_scan(bot1_scan)


@pytest.fixture(scope="session")
def synth():
    """A synthetic log plus wiggle and kinematics results."""
    from dtg.kinematics import infer_kinematics
    from dtg.linkage import infer_linkage
    from dtg.markers import parse_markers
    from dtg.synthetic import generate
    log = generate(seed=0)
    markers = parse_markers(log.header["markers"])
    wiggle = infer_linkage(log, markers)
    return log, markers, wiggle, infer_kinematics(log, wiggle)
