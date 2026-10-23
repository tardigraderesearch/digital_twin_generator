"""The DTG window, all eight steps, against the virtual bench.

A scripted operator presses the window's own buttons (the same methods the
buttons call), waits for each job like a person would, answers the flagged
results by looking at the (simulated) robot, and takes one screenshot per
tab. Then the twin is scored against the hidden truth, and the firmware's PWM
log is audited for safety (hard-stop contact, ROM band, slew).

    python -m sim gui-walk --seed 2 [--screenshots docs/screenshots]

Needs a display (DISPLAY). Only this process's own window is opened, and it
is closed at the end.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

TAB_FILES = ["1-scan", "2-tags", "3-wiggle", "4-range-of-motion", "5-register", "6-cut-links", "7-missing-parts",
             "8-export"]


class Walker:
    def __init__(self, app, root, shots: Path | None):
        self.app, self.root, self.shots = app, root, shots
        self.timeline: list = []

    def pump(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.root.update()
            time.sleep(0.02)

    def wait(self, what: str, timeout: float = 900.0) -> None:
        t0 = time.monotonic()
        self.pump(0.15)
        while self.app.busy:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"{what}: still busy after {timeout:.0f} s")
            self.pump(0.1)
        self.pump(0.3)
        status = self.app.status.get()
        self.timeline.append({"step": what, "s": round(time.monotonic() - t0, 1), "status": status})
        if status.startswith("Problem"):
            raise RuntimeError(f"{what}: {status}")

    def tab(self, key: str) -> None:
        self.app.goto(key)
        self.pump(0.4)

    def shot(self, index: int) -> str | None:
        if self.shots is None:
            return None
        self.root.lift()
        self.pump(0.8)
        self.shots.mkdir(parents=True, exist_ok=True)
        path = self.shots / f"{TAB_FILES[index]}.png"
        wid = int(self.root.wm_frame(), 16)
        subprocess.run(["import", "-window", str(wid), str(path)], check=True, capture_output=True)
        return str(path)

    def select(self, table, iid) -> None:
        table.selection_set(str(iid))
        table.focus(str(iid))
        table.see(str(iid))
        self.pump(0.1)


def walk(seed: int = 2, *, shots: Path | None = None, light: float = 60.0, out: Path | None = None) -> dict:
    import tkinter as tk
    from dtg.gui import TwinApp
    from sim.operator import _pwm_audit
    from sim.scan import write_scan
    from sim.score import score_run

    out = Path(out or tempfile.mkdtemp(prefix="dtg-gui-walk-"))
    os.environ["DTG_MAPS_DIR"] = str(out / "maps")
    root = tk.Tk()
    app = TwinApp(root, port=f"sim:{seed}", sim_seed=seed, sim_light=light)
    root.geometry("1500x980")
    w = Walker(app, root, shots)
    result: dict = {"seed": seed, "light": light, "source": "simulation", "session": str(app.session.root), "shots": []}
    try:
        w.pump(1.0)
        # -- connect first: the sim brings the robot, its scan and its tags --------------------------
        app.cutoff_var.set(True)
        app.connect()
        w.wait("connect + focus")
        world = app.bench.world
        scan = write_scan(world, out / "scan")
        result["scan"] = {k: scan[k] for k in ("scan", "faces")}
        # -- 1. scan ----------------------------------------------------------------------------------
        w.tab("scan")
        app.scan_var.set(scan["scan"])
        app.run_health()
        w.wait("health check")
        # the operator looks at the preview: the "wire" runs are arm edges, not cables; keep them
        app.fix_vars["wires"].set(False)
        app.run_health(fix=True)
        w.wait("cleaned copy")
        result["shots"].append(w.shot(0))
        # -- 2. tags ----------------------------------------------------------------------------------
        w.tab("tags")
        app.load_sim_tags()
        app._work("Camera preview", app._preview_job, app._preview_done)
        w.wait("tag check")
        result["shots"].append(w.shot(1))
        # -- 3. wiggle --------------------------------------------------------------------------------
        w.tab("wiggle")
        app.start_var.set("1500")
        app.channels_var.set("0-7")
        world._reference_angles = world.robot.joint_angles()     # at rest, as scanned
        app.wiggle_all()
        w.wait("wiggle 0-7")
        res = app.wiggle_res
        first = {c.channel: c.status for c in res.channels}
        for _ in range(2):                       # DTG says: wiggle the flagged ones again
            if not any(c.status == "needs_confirmation" for c in app.wiggle_res.channels):
                break
            app.rewiggle_flagged()
            w.wait("re-wiggle flagged")
        truth = world.truth
        by_ch = truth.by_channel()
        seen_link = _link_truth(truth, {l.name: l.tags for l in app.wiggle_res.links})
        answers = {}
        for c in app.wiggle_res.channels:
            if c.status != "needs_confirmation":
                continue
            j = by_ch.get(c.channel)
            right = j is not None and seen_link.get(c.child) == j.child and seen_link.get(c.parent) == j.parent
            w.select(app.wiggle_table, c.channel)
            app.answer_channel("confirmed" if right else "rejected")
            answers[c.channel] = "confirmed" if right else "rejected"
            w.pump(0.2)
        result["wiggle"] = {"first_pass": first, "final": {c.channel: c.status for c in app.wiggle_res.channels},
                            "operator": answers}
        result["shots"].append(w.shot(2))
        # -- 4. range of motion -----------------------------------------------------------------------
        w.tab("rom")
        app.start_rom()
        w.wait("ROM armed")
        rng = np.random.default_rng(20_000 + seed)
        rom_marks: dict = {}
        for ch, name, start in app._rom_joints():
            j = by_ch.get(ch)
            if j is None:
                continue
            w.select(app.rom_table, ch)
            app.rom_enable()
            w.wait(f"enable {ch}")
            marks = {}
            margin = float(rng.uniform(4.0, 10.0))
            first_side = int(rng.choice([1, -1]))
            step = int(app.rom_step_var.get())
            for which, sign in ((1, first_side), (2, -first_side)):
                for _ in range(60):
                    step_deg = step * j.gain_deg_per_us
                    if world.robot.stop_margin_deg(j.name, sign) - step_deg < margin:
                        break                                    # the operator sees the stop coming
                    rom = app.bench.rom.channels[ch]
                    pos = app.bench.rom.position(ch)
                    low, high = rom.slider_range()
                    if not low <= pos + sign * step <= high:
                        if not rom.can_widen():
                            break
                        app.rom_widen()
                        w.pump(0.1)
                        continue
                    app.rom_step(sign)
                    w.wait(f"step {ch}")
                app.rom_mark(which)
                w.wait(f"ROM {which} on {ch}")
                pulse = getattr(app.bench.rom.channels[ch], f"rom{which}_us")
                marks[which] = {"pulse_us": pulse,
                                "joint_deg": world.robot.joint_angles()[j.name],
                                "stop_margin_deg": world.robot.stop_margin_deg(j.name, sign)}
            fw = app.bench.controller.status(ch)
            band_matches = [fw["min_us"], fw["max_us"]] == list(app.bench.rom.channels[ch].range)
            rom_marks[ch] = {"joint": j.name, "marks": marks, "rom_band_enforced": band_matches,
                             "range_us": app.bench.rom.channels[ch].range, "rom_complete_t": world.clock()}
        app.fit_rom()
        w.wait("ROM fit")
        # flagged fits: the operator compares the drawn axis with the robot (sim: the truth, within 5 deg)
        joint_answers = {}
        for iid in app.rom_table.get_children():
            if app.rom_table.set(iid, "fit") != "flagged":
                continue
            fit = next(j for j in app.state.kin.joints if j.channel == int(iid))
            ok = _axis_ok(world, fit, app)
            w.select(app.rom_table, iid)
            app.answer_joint("confirmed" if ok else "rejected")
            joint_answers[fit.name] = "confirmed" if ok else "rejected"
        result["rom_answers"] = joint_answers
        result["shots"].append(w.shot(3))
        app.disarm()
        w.pump(0.5)
        # -- 5. register ------------------------------------------------------------------------------
        w.tab("register")
        app.session.set_input("scan_tags", json.loads(Path(scan["scan_tags"]).read_text()))
        app.run_register()
        w.wait("register")
        result["register"] = app.session.step("register")
        result["shots"].append(w.shot(4))
        # -- 6. cut links -----------------------------------------------------------------------------
        w.tab("segment")
        app.run_segment()
        w.wait("segment")
        result["segment"] = app.session.step("segment")
        result["shots"].append(w.shot(5))
        # -- 7. parts ---------------------------------------------------------------------------------
        w.tab("parts")
        for iid in app.servo_table.get_children():
            w.select(app.servo_table, iid)
            app.servo_var.set("towerpro-mg996r")
            app.assign_servo()
        app.run_parts()
        w.wait("parts")
        result["parts"] = app.session.step("parts")
        result["shots"].append(w.shot(6))
        # -- 8. export --------------------------------------------------------------------------------
        w.tab("export")
        app.run_export()
        w.wait("export")
        result["export"] = app.session.step("export")
        result["shots"].append(w.shot(7))
        # -- score against the hidden truth -----------------------------------------------------------
        session = {"wiggle": {c.channel: {"status": c.status, "child": c.child, "parent": c.parent,
                                          "direction": c.direction, "reasons": c.reasons}
                              for c in app.wiggle_res.channels},
                   "wiggle_links": {l.name: l.tags for l in app.wiggle_res.links},
                   "reference_angles": _reference_angles(world), "rom": rom_marks, "errors": []}
        score = score_run(world, session, app.session.twin_dir)
        result["joints"] = score.get("joints")
        result["map_exact"] = score.get("map_exact")
        result["map_silent_wrong"] = score.get("map_silent_wrong")
        result["urdf_valid"] = score.get("urdf_valid")
        result["twin"] = score.get("twin")
        result["safety"] = {"hard_stop_contacts": len(world.robot.stalls),
                            "max_start_jump_deg": max((s["jump_deg"] for s in world.robot.starts), default=0.0),
                            **_pwm_audit(world, rom_marks)}
        result["timeline"] = w.timeline
    finally:
        try:
            app.close()
        except Exception:  # noqa: BLE001
            pass
    (out / "gui-walk.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    result["out"] = str(out)
    return result


def _reference_angles(world) -> dict:
    ref = getattr(world, "_reference_angles", None)
    return ref or {e["event"]: e for e in world.log}.get("reference", {}).get("angles") or world.robot.joint_angles()


def _link_truth(truth, links: dict) -> dict:
    from collections import Counter
    out = {}
    for name, tags in links.items():
        votes = Counter(truth.tags[int(k)]["link"] for k in tags if int(k) in truth.tags)
        out[name] = votes.most_common(1)[0][0] if votes else None
    return out


def _axis_ok(world, fit, app, tol_deg: float = 5.0) -> bool:
    """Does the fitted axis match the real one (what an operator checks on the 3D view)?"""
    from sim.robot import link_motions
    truth = world.truth
    j = next((j for j in truth.joints if j.channel == fit.channel), None)
    if j is None:
        return False
    wig = app.wiggle_res
    root_tag = next(l for l in wig.links if not l.signature).primary_tag
    q = _reference_angles(world)
    M = link_motions(truth, q)
    W = np.linalg.inv(M[truth.tags[root_tag]["link"]] @ np.asarray(truth.tags[root_tag]["T0"]))
    axis_t = W[:3, :3] @ (M[j.parent][:3, :3] @ np.asarray(j.axis))
    c = abs(float(np.dot(fit.axis_model, axis_t)) / np.linalg.norm(fit.axis_model))
    return float(np.degrees(np.arccos(min(1.0, c)))) < tol_deg
