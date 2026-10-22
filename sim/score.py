"""Score DTG's outputs against the virtual robot's hidden ground truth.

Frames: DTG's model frame is the root link's primary tag at the reference
capture; truth is expressed in the same (true) tag frame, so camera and
pose-estimation errors show up in the scores instead of being calibrated
away.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .robot import link_motions

# Pass thresholds (per joint unless noted). Chosen from what a usable twin of
# a ~20 cm hobby-servo arm needs, not from what the pipeline happens to hit.
THRESHOLDS = {
    "axis_deg": 2.0,
    "pivot_mm": 3.0,
    "terminal_deg": 2.0,
    "gain_pct": 3.0,
    "zero_pulse_us": 15.0,
    "twin_p95_mm": 5.0,
    "start_jump_deg": 3.0,
    "watchdog_s": (0.9, 1.3),
    "estop_s": 0.15,
}


def _angle(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


def score_run(world, session: dict, twin_dir: Path | None) -> dict:
    truth = world.truth
    by_ch = truth.by_channel()
    out: dict = {"seed": world.seed, "errors": session.get("errors", []), "joints": {}, "channels": {}}
    # -- channel map (DTG's own answer, before the operator confirms anything) --
    rows, exact, silent_wrong, flagged = _map_score(truth, session.get("wiggle", {}), session.get("wiggle_links"))
    out["channels"] = rows
    out["map_exact"] = exact
    out["map_silent_wrong"] = silent_wrong
    out["map_flagged"] = flagged
    out["unused_ok"] = all(rows[c]["ok"] for c in truth.unused_channels if c in rows)
    _, exact1, wrong1, flagged1 = _map_score(truth, session.get("wiggle_first") or session.get("wiggle", {}),
                                             session.get("wiggle_first_links") or session.get("wiggle_links"))
    out["tags_misassigned"] = _misassigned(truth, session.get("wiggle_links") or {})
    out["map_first_pass"] = {"exact": exact1, "silent_wrong": wrong1, "flagged": flagged1,
                             "rewiggles": session.get("rewiggles", [])}

    kin = report = None
    if twin_dir is not None and (Path(twin_dir) / "kinematics.json").is_file():
        kin = json.loads((Path(twin_dir) / "kinematics.json").read_text())
        report = json.loads((Path(twin_dir) / "report.json").read_text())
        wiggle_json = json.loads((Path(twin_dir) / "wiggle.json").read_text())
    if kin is None:
        out["pipeline_ok"] = False
        out["pass"] = False
        return out
    out["pipeline_ok"] = True
    # -- model frame: root link primary tag at the reference ---------------------
    root_link = next(l for l in wiggle_json["links"] if not l["signature"])
    root_tag = root_link["primary_tag"]
    q_ref = {k: float(v) for k, v in session["reference_angles"].items()}
    M_ref = link_motions(truth, q_ref)
    T_world_model = M_ref[truth.tags[root_tag]["link"]] @ np.asarray(truth.tags[root_tag]["T0"])
    W = np.linalg.inv(T_world_model)
    dtg_by_ch = {j["channel"]: j for j in kin["joints"]}
    rom = {int(k): v for k, v in session.get("rom", {}).items()}
    for j in truth.joints:
        d = dtg_by_ch.get(j.channel)
        row = {"channel": j.channel, "found": d is not None}
        if d is None:
            out["joints"][j.name] = row
            continue
        axis_world = M_ref[j.parent][:3, :3] @ np.asarray(j.axis)
        pivot_world = (M_ref[j.parent] @ np.r_[j.pivot, 1.0])[:3]
        axis_t = W[:3, :3] @ axis_world
        pivot_t = (W @ np.r_[pivot_world, 1.0])[:3]
        s = float(np.sign(np.dot(d["axis_model"], axis_t))) or 1.0
        dp = np.asarray(d["pivot_model"]) - pivot_t
        gain_rad = math.radians(j.gain_deg_per_us)
        row.update(status=d["status"], reasons=d["reasons"], axis_deg=_angle(d["axis_model"], axis_t),
                   pivot_mm=float(np.linalg.norm(dp - axis_t * (dp @ axis_t)) * 1000),
                   direction_ok=bool(d["direction"] == int(s * j.direction)),
                   gain_pct=float((d["rad_per_us"] / gain_rad - 1) * 100),
                   zero_pulse_us=float(d["zero_pulse_us"] - j.pulse_for(q_ref[j.name])))
        marks = rom.get(j.channel, {}).get("marks", {})
        terr = []
        for which, key in ((1, "terminal_range_1"), (2, "terminal_range_2")):
            m = marks.get(which) or marks.get(str(which))
            if m is None:
                continue
            true_pos = s * math.radians(m["joint_deg"] - q_ref[j.name])
            terr.append(abs(math.degrees(d[key]["position"] - true_pos)))
            row[f"rom{which}_pulse_matches"] = abs(d[key]["pulse_us"] - m["pulse_us"]) < 0.5
            row[f"rom{which}_stop_margin_deg"] = m["stop_margin_deg"]
        row["terminal_deg"] = max(terr) if terr else None
        row["urdf_range_deg"] = [math.degrees(d["lower"]), math.degrees(d["upper"])]
        out["joints"][j.name] = row
    out["urdf_valid"] = bool(report and report.get("urdf", {}).get("independent_loader") == "yourdfpy")
    out["twin"] = _twin_check(world, kin, Path(twin_dir), W, q_ref) if len(dtg_by_ch) == len(truth.joints) else None
    # -- safety ---------------------------------------------------------------
    sf = session.get("safety", {})
    out["safety"] = {
        "hard_stop_contacts": len(sf.get("stalls", [])),
        "max_start_jump_deg": sf.get("max_start_jump_deg"),
        "arm_alone_outputs": len(sf.get("arm_alone_outputs", {}) or {}) + len(sf.get("outputs_after_arm", {}) or {}),
        "watchdog_release_s": sf.get("watchdog_release_s"),
        "unplug_release_s": sf.get("unplug_release_s"),
        "estop_release_s": sf.get("estop_release_s"),
        "estop_latched": sf.get("estop_latched"),
        "estop_pin_latched": sf.get("estop_pin_latched"),
        "out_of_band_rejected": sf.get("out_of_band_rejected"),
        "holds_still_after_release": bool(sf.get("estop_holds_still")) and bool(sf.get("watchdog_holds_still")),
        "arming_guarded": sf.get("arming_guarded"),
        "pwm_edges": sf.get("pwm_edges"),
        "pwm_past_hard_stop": sf.get("pwm_past_hard_stop"),
        "pwm_outside_rom": sf.get("pwm_outside_rom"),
        "unused_channel_far_pulses": sf.get("unused_channel_far_pulses"),
        "max_slew_us_s": sf.get("max_slew_us_s"),
        "slew_capped": sf.get("slew_capped"),
        "rom_band_enforced": sf.get("rom_band_enforced"),
    }
    out["hardware_cases"] = {
        "inverted_joints": sum(j.direction < 0 for j in truth.joints),
        "horn_off_by_teeth": sum(abs(j.horn_offset_deg) > 7.2 for j in truth.joints),
        "wiring": {j.name: j.channel for j in truth.joints}, "unused_channels": truth.unused_channels}
    out["checks"] = checks(out)
    out["untagged_links"] = sorted({j.child for j in truth.joints} - {t["link"] for t in truth.tags.values()})
    out["pass"] = all(out["checks"].values())
    return out


def _map_score(truth, wiggle: dict, links: dict | None = None):
    """Channel -> joint map vs truth. DTG names a link after its lowest tag, so
    a link is judged by the real link most of its tags sit on."""
    from .operator import _link_truth
    by_ch = truth.by_channel()
    real = _link_truth(truth, links) if links else {}
    wig = {int(k): v for k, v in wiggle.items()}
    rows, silent_wrong, flagged, exact = {}, 0, 0, True
    for ch in sorted(wig):
        j, w = by_ch.get(ch), wig[ch]
        child, parent = real.get(w["child"], w["child"]), real.get(w["parent"], w["parent"])
        row = {"truth": j.child if j else "unused", "dtg_status": w["status"], "dtg_child": w["child"],
               "dtg_parent": w["parent"], "child_is": child, "parent_is": parent}
        if j is None:
            ok = w["status"] == "no_response"
            wrong_silent = w["status"] == "mapped"
        else:
            correct = child == j.child and parent == j.parent
            ok = correct and w["status"] == "mapped"
            wrong_silent = w["status"] == "mapped" and not correct
        flagged += int(w["status"] == "needs_confirmation")
        row["ok"] = bool(ok)
        rows[ch] = row
        silent_wrong += int(wrong_silent)
        exact &= bool(ok)
    exact &= set(wig) >= {j.channel for j in truth.joints}
    return rows, bool(exact), silent_wrong, flagged


def _misassigned(truth, links: dict) -> list:
    """Tags DTG put on a link together with tags of a different real link."""
    from .operator import _link_truth
    real = _link_truth(truth, links)
    return sorted(int(k) for name, tags in links.items() for k in tags
                  if int(k) in truth.tags and truth.tags[int(k)]["link"] != real[name])


def checks(out: dict) -> dict:
    t = THRESHOLDS
    joints = list(out["joints"].values())
    found = [j for j in joints if j.get("found")]
    sf = out.get("safety", {})
    wd = sf.get("watchdog_release_s")
    up = sf.get("unplug_release_s")
    twin = out.get("twin") or {}
    return {
        "no_errors": not out["errors"],
        "map_exact": out["map_exact"],
        "no_silent_wrong_map": out["map_silent_wrong"] == 0 and out["map_first_pass"]["silent_wrong"] == 0,
        "tags_on_right_links": not out.get("tags_misassigned"),
        "all_joints_fitted": len(found) == len(joints),
        "direction": all(j["direction_ok"] for j in found),
        "axis": all(j["axis_deg"] < t["axis_deg"] for j in found),
        "pivot": all(j["pivot_mm"] < t["pivot_mm"] for j in found),
        "terminal_ranges": all(j["terminal_deg"] is not None and j["terminal_deg"] < t["terminal_deg"] for j in found),
        "us_angle_gain": all(abs(j["gain_pct"]) < t["gain_pct"] for j in found),
        "zero_pulse": all(abs(j["zero_pulse_us"]) < t["zero_pulse_us"] for j in found),
        "urdf_valid": out.get("urdf_valid", False),
        "twin_fk": bool(twin) and twin.get("p95_mm", 1e9) < t["twin_p95_mm"],
        "no_hard_stop_contact": sf.get("hard_stop_contacts", 1) == 0,
        "no_first_move_snap": (sf.get("max_start_jump_deg") or 0) < t["start_jump_deg"],
        "arm_outputs_nothing": sf.get("arm_alone_outputs", 1) == 0,
        "watchdog": wd is not None and t["watchdog_s"][0] <= wd <= t["watchdog_s"][1]
        and up is not None and t["watchdog_s"][0] <= up <= t["watchdog_s"][1],
        "estop_latch": bool(sf.get("estop_latched")) and bool(sf.get("estop_pin_latched"))
        and sf.get("estop_release_s") is not None and sf["estop_release_s"] < t["estop_s"],
        "out_of_band_rejected": bool(sf.get("out_of_band_rejected")),
        "released_servos_hold": bool(sf.get("holds_still_after_release")),
        "arming_guarded": bool(sf.get("arming_guarded")),
        "pwm_never_past_stops": sf.get("pwm_past_hard_stop") == 0,
        "rom_limits_held": sf.get("pwm_outside_rom") == 0 and sf.get("rom_band_enforced") is not False,
        "unused_channels_only_wiggled": sf.get("unused_channel_far_pulses") == 0,
        "slew_capped": bool(sf.get("slew_capped")),
    }


def _twin_check(world, kin, twin_dir: Path, W, q_ref, n: int = 25) -> dict:
    """Command random pulses inside the measured safe ranges; compare where the
    URDF + comms map put every tag with where the real (virtual) robot puts it."""
    from yourdfpy import URDF
    truth = world.truth
    urdf = URDF.load(str(twin_dir / "robot.urdf"), load_meshes=False, build_scene_graph=True)
    cmap = {c["channel"]: c for c in json.loads((twin_dir / "comms_map.json").read_text())["channels"]}
    dtg = {j["channel"]: j for j in kin["joints"]}
    origins = {kin["root"]: np.zeros(3)} | {j["child"]: np.asarray(j["pivot_model"]) for j in kin["joints"]}
    M_ref = link_motions(truth, q_ref)
    rng = np.random.default_rng(world.seed)
    errs = []
    for _ in range(n):
        pulses = {}
        for j in truth.joints:
            lo, hi = sorted(cmap[j.channel]["safe_range_us"])
            pulses[j.channel] = rng.uniform(lo, hi)
        q_true = {j.name: j.joint_deg_at(pulses[j.channel]) for j in truth.joints}
        M = link_motions(truth, q_true)
        cfg = {}
        for j in truth.joints:
            d, c = dtg[j.channel], cmap[j.channel]
            cfg[d["name"]] = c["direction"] * (pulses[j.channel] - c["zero_pulse_us"]) * c["rad_per_us"]
        urdf.update_cfg(cfg)
        for tag_id, tag in truth.tags.items():
            link_truth = tag["link"]
            if link_truth == "base":
                continue
            dtg_link = next((j["child"] for j in kin["joints"] if truth.by_channel()[j["channel"]].child == link_truth),
                            None)
            if dtg_link is None:
                continue
            p_ref = (W @ M_ref[link_truth] @ np.asarray(tag["T0"]))[:3, 3]
            T = urdf.get_transform(dtg_link, kin["root"])
            pred = (T @ np.r_[p_ref - origins[dtg_link], 1.0])[:3]
            real = (W @ M[link_truth] @ np.asarray(tag["T0"]))[:3, 3]
            errs.append(float(np.linalg.norm(pred - real) * 1000))
    errs = np.asarray(errs)
    return {"samples": int(len(errs)), "median_mm": float(np.median(errs)), "p95_mm": float(np.percentile(errs, 95)),
            "max_mm": float(errs.max())}
