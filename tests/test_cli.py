import json

from dtg.cli import main


def test_cli_synth_wiggle_build(tmp_path, capsys):
    log = tmp_path / "obs.jsonl"
    assert main(["synth", "--out", str(log)]) == 0
    assert main(["wiggle", str(log)]) == 0
    out = capsys.readouterr().out
    assert "no_response" in out and "j_forearm" in out
    assert main(["build", str(log), "--out", str(tmp_path / "twin"), "--servo", "ch0=MG996R"]) == 0
    report = json.loads((tmp_path / "twin" / "report.json").read_text())
    assert len(report["joints"]) == 3
    assert main(["servos"]) == 0


def test_markers_parse_like_aruco_tag_maker():
    from dtg.markers import parse_markers
    m = parse_markers({"generator": "aruco-tag-maker 0.1.0", "dictionary": "DICT_4X4_50", "tag_size_mm": 30.0,
                       "tile_size_mm": 40.0, "tags": {"base": {"id": 0}, "upper_arm": {"id": 1}}})
    assert m.root_tag == 0 and m.name_for(1) == "upper_arm" and m.tile_size_mm == 40.0
