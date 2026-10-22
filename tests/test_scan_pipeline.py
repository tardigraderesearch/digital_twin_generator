import json

import numpy as np
import pytest
import trimesh

from dtg.registration import RegistrationError, apply, register_tags, similarity_from_points, tag_positions_model
from dtg.segment import segment_by_linkage
from dtg.synthetic import synthetic_scan


def test_similarity_recovers_transform():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(6, 3))
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=2).as_matrix()
    dst = 0.001 * src @ R.T + [0.1, -0.2, 0.3]
    T, scale, rms = similarity_from_points(src, dst)
    assert scale == pytest.approx(0.001) and rms < 1e-12
    with pytest.raises(RegistrationError):
        similarity_from_points(src[:2], dst[:2])


def test_register_and_segment_synthetic_scan(synth):
    _, _, _, kin = synth
    sc = synthetic_scan()
    tags = tag_positions_model(kin)
    reg = register_tags(sc["tag_centres_scan"], tags)
    assert reg.rms_m * 1000 < 2.0 and reg.scale == pytest.approx(0.001, rel=0.01)
    V = apply(reg.T_model_scan, sc["vertices"])
    seg = segment_by_linkage(V, sc["faces"], kin, tags)
    names = np.array(seg.links + ["unassigned"], dtype=object)[seg.vertex_labels]
    accuracy = (names == sc["truth_labels"]).mean()
    assert accuracy > 0.75
    assert 0 < (seg.vertex_labels < 0).mean() < 0.1  # ambiguity is kept, not forced


def test_build_twin_end_to_end(tmp_path):
    from dtg.pipeline import BuildOptions, build_twin
    from dtg.synthetic import generate
    log = generate(seed=0)
    log.save(tmp_path / "obs.jsonl")
    sc = synthetic_scan()
    trimesh.Trimesh(sc["vertices"], sc["faces"], process=False).export(str(tmp_path / "scan.obj"))
    (tmp_path / "tags.json").write_text(json.dumps({str(k): v.tolist() for k, v in sc["tag_centres_scan"].items()}))
    report = build_twin(BuildOptions(log=tmp_path / "obs.jsonl", out=tmp_path / "twin", scan=tmp_path / "scan.obj",
                                     scan_tags=tmp_path / "tags.json", servo_models={"ch4": "MG996R"}, preview=False),
                        progress=lambda m: None)
    provs = {p["provenance"] for l in report["links"].values() for p in l["parts"]}
    assert {"scanned", "library-servo"} <= provs
    assert report["urdf"]["joints"] == 3
    assert (tmp_path / "twin" / "comms_map.json").is_file()


def test_build_without_scan_uses_stand_ins(tmp_path):
    from dtg.pipeline import BuildOptions, build_twin
    from dtg.synthetic import generate
    generate(seed=0).save(tmp_path / "obs.jsonl")
    report = build_twin(BuildOptions(log=tmp_path / "obs.jsonl", out=tmp_path / "twin"), progress=lambda m: None)
    provs = {p["provenance"] for l in report["links"].values() for p in l["parts"]}
    assert provs == {"stand-in"}
