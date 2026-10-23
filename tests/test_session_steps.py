"""Session folders, the step functions and the per-step CLI, on a synthetic camera log."""
import json

import pytest

from dtg import steps
from dtg.cli import main
from dtg.session import Session, SessionError, latest


@pytest.fixture()
def maps(tmp_path, monkeypatch):
    monkeypatch.setenv("DTG_MAPS_DIR", str(tmp_path / "maps"))
    return tmp_path / "maps"


@pytest.fixture()
def synth_files(tmp_path):
    from dtg.synthetic import generate
    log = generate(seed=0)
    path = log.save(tmp_path / "syn" / "observations.jsonl")
    markers = tmp_path / "syn" / "markers.json"
    markers.write_text(json.dumps(log.header["markers"]))
    return path, markers


def test_session_new_open_never_overwrites(maps):
    a = Session.new("bot 1")
    b = Session.new("bot 1")
    assert a.root != b.root and a.root.parent == maps and a.root.name.startswith("bot-1-")
    assert latest() == b.root
    s = Session.open("latest")
    s.set_step("scan", "done", "x")
    assert Session.open(b.root).step("scan")["status"] == "done"
    with pytest.raises(SessionError):
        s.set_step("scan", "great")
    with pytest.raises(SessionError):
        Session.open(maps)


def test_steps_on_a_recorded_log(maps, synth_files):
    log, markers = synth_files
    s = Session.new("demo")
    m = steps.load_tags(s, markers)
    assert s.markers_path.is_file() and s.step("tags")["status"] == "done" and m.root_tag == 0
    s.import_log(log)
    res = steps.analyse_wiggle(s)
    assert len([c for c in res.channels if c.joint]) == 3 and s.step("wiggle")["status"] == "done"
    st = steps.analyse_rom(s)
    assert len(st.kin.joints) == 3 and s.step("rom")["status"] == "done"
    s.set_input("servo_models", {"ch0": "MG996R"})
    st = steps.run_parts(s)
    provs = {r["provenance"] for r in steps.parts_table(st)}
    assert {"library-servo", "stand-in"} <= provs
    report = steps.run_export(s)
    assert (s.twin_dir / "robot.urdf").is_file() and (s.twin_dir / "comms_map.json").is_file()
    assert all(j["operator"] == "not asked" for j in report["joints"])
    assert report["verified"] is False and s.step("export")["status"] == "done"


def test_operator_rejection_removes_a_joint_and_flags_stay_unconfirmed(maps, synth_files):
    log, markers = synth_files
    s = Session.new("demo")
    steps.load_tags(s, markers)
    s.import_log(log)
    res = steps.analyse_wiggle(s)
    ch = next(c.channel for c in res.channels if c.joint)
    s.confirm_channel(ch, "rejected")
    res = steps.analyse_wiggle(s)
    c = next(c for c in res.channels if c.channel == ch)
    assert c.status == "rejected" and c.joint is None
    report = steps.run_export(s)
    assert ch not in [j["channel"] for j in report["joints"]]
    assert any("operator rejected" in f for f in report["flags"])
    # a flagged ROM fit with no answer is reported as not confirmed
    s.confirm_channel(ch, None)
    st = steps.analyse_rom(s)
    j = st.kin.joints[0]
    j.status = "flagged"
    from dtg.pipeline import export_stage
    rep = export_stage(st, lambda *_: None)
    assert rep["unconfirmed_flagged_joints"] == [j.name]
    assert rep["joints"][0]["operator"] == "NOT CONFIRMED"
    s.confirm_joint(j.name, "confirmed")
    st.opt.confirmations = s.confirmations
    rep = export_stage(st, lambda *_: None)
    assert rep["unconfirmed_flagged_joints"] == [] and rep["joints"][0]["operator"] == "confirmed"


def test_cli_runs_each_step(maps, synth_files, capsys):
    log, markers = synth_files
    assert main(["session", "new", "cli"]) == 0
    root = capsys.readouterr().out.strip()
    assert main(["tags", str(markers), "--session", root]) == 0
    assert main(["wiggle", str(log), "--session", "latest"]) == 0
    assert "3 joint(s) mapped" in capsys.readouterr().out
    assert main(["rom", "--session", "latest"]) == 0
    assert main(["parts", "--session", "latest", "--servo", "ch0=MG996R"]) == 0
    assert main(["register", "--session", "latest"]) == 0          # no scan: skipped, said so
    assert "no scan" in capsys.readouterr().out
    assert main(["export", "--session", "latest"]) == 0
    assert "robot.urdf" in capsys.readouterr().out
    assert main(["session", "show"]) == 0
    out = capsys.readouterr().out
    assert "8. Export" in out and "done" in out
    assert main(["doctor"]) == 0


def test_cli_live_wiggle_refuses_without_start_or_cutoff(maps):
    main(["session", "new", "x"])
    with pytest.raises(SystemExit, match="bad_start"):
        main(["wiggle", "--session", "latest", "--port", "emulator", "--power-cutoff-confirmed"])
    with pytest.raises(SystemExit, match="cutoff_unconfirmed"):
        main(["wiggle", "--session", "latest", "--port", "emulator", "--start-us", "1500"])
