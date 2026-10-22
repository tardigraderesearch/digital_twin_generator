"""Scripted operator: runs one bench session through the real code paths.

1. focus: DTG's lens sweep + lock on the robot region (autofocus is tried
   first and recorded: on the real bench it locked on the background);
2. reference capture with the robot unpowered at rest;
3. WIGGLE: robo-prober Controller (real host API, real firmware core over a
   pty) armed by the operator, DTG's ObservationRecorder.wiggle_channel on
   every PCA9685 channel in use, then disarm;
4. DTG wiggle inference; the operator confirms what it reports (flagged
   items are confirmed only if they match what the operator can see);
5. ROM: robo-prober's RomSession exactly as the ROM window drives it
   (Enable at the shown start pulse, slider targets stepped slowly, Widen
   when the band runs out, Set ROM 1 / Set ROM 2 a few degrees short of the
   hard stops the operator can see), with DTG's rom_sample / rom_mark hooks;
6. safety probes against the firmware: out-of-band rejection, E-STOP latch
   (command and hardware pin), watchdog release on host silence and on an
   unplugged host, no output on ARM alone.

The operator's eyes are the only place the hidden truth is used: to see
where the hard stops are and to confirm flagged wiggle results.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OperatorConfig:
    channels: tuple = tuple(range(8))
    start_us: int = 1500
    wiggle_us: int = 80
    rom_step_us: int = 40
    settle_s: float = 0.4
    host_slew_us_s: int = 200
    margin_deg: tuple = (4.0, 10.0)      # how far short of a hard stop the operator marks ROM
    safety_probes: bool = True
    rewiggle_attempts: int = 2           # DTG says "re-run those wiggles": the operator does, up to twice
    focus_positions: tuple = tuple(range(80, 250, 15))


@dataclass
class SessionResult:
    seed: int
    out: str
    focus: dict = field(default_factory=dict)
    reference_angles: dict = field(default_factory=dict)
    wiggle: dict = field(default_factory=dict)
    wiggle_first: dict = field(default_factory=dict)
    wiggle_links: dict = field(default_factory=dict)
    wiggle_first_links: dict = field(default_factory=dict)
    rewiggles: list = field(default_factory=list)
    confirmations: dict = field(default_factory=dict)
    rom: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Trace:
    """Background sampler of joint angles and PWM (for plots and the video)."""

    def __init__(self, world, period_s: float = 0.05):
        self.world, self.period_s = world, period_s
        self.rows: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        robot = self.world.robot
        while not self._stop.wait(self.period_s):
            t = self.world.clock()
            q = robot.snapshot(t)
            pulses = {j.name: robot.state[j.name].pulse_us for j in robot.truth.joints}
            self.rows.append({"t": round(t, 3), "q": {k: round(v, 3) for k, v in q.items()}, "pulse": pulses})

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1)


def robot_roi(world, pad: int = 40):
    """The box the operator drags around the robot for the focus sweep (x, y, w, h)."""
    from .robot import link_motions
    from .camera import HEIGHT, WIDTH
    T_cw = np.linalg.inv(world.T_world_cam)
    motions = link_motions(world.truth, world.robot.joint_angles())
    pts = [(motions[t["link"]] @ np.asarray(t["T0"]))[:3, 3] for t in world.truth.tags.values()]
    pts += [np.asarray(j.pivot) for j in world.truth.joints]
    pc = (np.asarray(pts) @ T_cw[:3, :3].T) + T_cw[:3, 3]
    uv = pc @ world.K.T
    uv = uv[:, :2] / uv[:, 2:3]
    x0, y0 = np.maximum(uv.min(axis=0) - pad, 0).astype(int)
    x1, y1 = np.minimum(uv.max(axis=0) + pad, [WIDTH - 1, HEIGHT - 1]).astype(int)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def run_session(world, out: Path, cfg: OperatorConfig | None = None) -> SessionResult:
    import cv2
    from dtg.camera import ArucoTracker, CameraIntrinsics, ObservationRecorder
    from dtg.linkage import infer_linkage
    from dtg.observations import load_log
    from robo_prober.api import Controller
    from robo_prober.rom import RomChannel, RomSession
    from robo_prober.transport import SerialProtocolTransport

    cfg = cfg or OperatorConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = SessionResult(seed=world.seed, out=str(out))
    markers_path = world.write_markers(out / "markers.json")
    intr = CameraIntrinsics(world.K_calibrated, np.zeros(5), (1920, 1080))
    (out / "camera.json").write_text(json.dumps(world.intrinsics_dict(), indent=2) + "\n")
    rng = np.random.default_rng(20_000 + world.seed)
    t_start = time.monotonic()

    tracker = ArucoTracker(world.camera, world.markers, intr)
    # -- 1. focus ------------------------------------------------------
    poses_af, _ = tracker.capture()
    res.focus["autofocus_tags_detected"] = len(poses_af)
    focus = tracker.lock_focus(positions=cfg.focus_positions, roi=robot_roi(world))
    poses, _ = tracker.capture()
    res.focus.update(lens=focus.lens, sharp_lens_truth=round(world.sharp_lens, 1), peak_ratio=focus.peak_ratio,
                     tags_detected=len(poses), tags_total=len(world.truth.tags))
    world.event("focus_locked", **res.focus)

    recorder = ObservationRecorder(tracker, out / "observations.jsonl", world.markers, settle_s=cfg.settle_s)
    with Trace(world) as trace, world.firmware() as bench:
        try:
            # -- 2. reference: robot unpowered at rest ------------------------
            res.reference_angles = world.robot.joint_angles()
            recorder.reference()
            world.event("reference", angles=res.reference_angles)

            # -- 3. wiggle --------------------------------------------------------
            t0 = time.monotonic()
            ctl = Controller(SerialProtocolTransport(bench.port, timeout_s=1.0)).open()
            ctl.arm(power_cutoff_confirmed=True)
            res.safety["outputs_after_arm"] = dict(bench.active_channels())
            for ch in cfg.channels:
                recorder.wiggle_channel(ctl, ch, start_us=cfg.start_us, delta_us=cfg.wiggle_us,
                                        slew_us_s=cfg.host_slew_us_s)
            ctl.close()                                  # disarms
            bench.wait_released(1.0)
            res.timings["wiggle_s"] = round(time.monotonic() - t0, 1)

            # -- 4. DTG wiggle; re-run flagged channels as DTG asks; confirm --------
            wig = infer_linkage(load_log(out / "observations.jsonl"), world.markers)
            res.wiggle_first = _wiggle_summary(wig)
            res.wiggle_first_links = {l.name: l.tags for l in wig.links}
            for attempt in range(cfg.rewiggle_attempts):
                flagged = [c.channel for c in wig.channels if c.status == "needs_confirmation"]
                if not flagged:
                    break
                world.event("rewiggle", attempt=attempt + 1, channels=flagged)
                ctl = Controller(SerialProtocolTransport(bench.port, timeout_s=1.0)).open()
                ctl.arm(power_cutoff_confirmed=True)
                for ch in flagged:
                    recorder.wiggle_channel(ctl, ch, start_us=cfg.start_us, delta_us=cfg.wiggle_us,
                                            slew_us_s=cfg.host_slew_us_s)
                ctl.close()
                bench.wait_released(1.0)
                res.rewiggles.append(flagged)
                wig = infer_linkage(load_log(out / "observations.jsonl"), world.markers)
            res.wiggle = _wiggle_summary(wig)
            res.wiggle_links = {l.name: l.tags for l in wig.links}
            truth_by_ch = world.truth.by_channel()
            seen_as = _link_truth(world.truth, res.wiggle_links)   # what the operator sees each link is
            rom_channels = []
            for c in wig.channels:
                j = truth_by_ch.get(c.channel)
                sees_correct = j is not None and seen_as.get(c.child) == j.child and seen_as.get(c.parent) == j.parent
                if c.status == "mapped" and sees_correct:
                    verdict = "confirmed"
                elif c.status == "needs_confirmation" and sees_correct:
                    verdict = "confirmed_flagged"
                elif c.status == "no_response" and j is None:
                    verdict = "confirmed_unused"
                else:
                    verdict = "rejected"
                res.confirmations[c.channel] = verdict
                if verdict in ("confirmed", "confirmed_flagged"):
                    rom_channels.append((c.channel, j))
            # Joints DTG never offered can still be swept: the operator saw them move.
            world.event("confirmations", verdicts=res.confirmations)

            # -- 5. ROM -----------------------------------------------------------
            t0 = time.monotonic()
            ctl = Controller(SerialProtocolTransport(bench.port, timeout_s=1.0)).open()
            roms = [RomChannel(ch, j.name, start_us=cfg.start_us, start_source="last_position")
                    for ch, j in rom_channels]
            if roms:
                session = RomSession(ctl, roms, host_slew_us_s=cfg.host_slew_us_s, heartbeat="thread")
                session.start(power_cutoff_confirmed=True)
                for ch, j in rom_channels:
                    res.rom[ch] = _sweep_joint(world, session, recorder, ch, j, cfg, rng)
                session.close()
            ctl.close()
            bench.wait_released(1.0)
            res.timings["rom_s"] = round(time.monotonic() - t0, 1)

            # -- 6. safety probes ---------------------------------------------
            if cfg.safety_probes:
                res.safety.update(_safety_probes(world, bench, rom_channels, cfg))
        except Exception as error:  # noqa: BLE001 - recorded, scored as a failure
            import traceback
            res.errors.append(f"{type(error).__name__}: {error}")
            res.errors.append(traceback.format_exc())
            world.event("error", error=str(error))
        finally:
            try:
                bench.wait_released(2.0)
            except Exception as error:  # noqa: BLE001
                res.errors.append(f"outputs not released at end: {error}")
    res.safety.update(_pwm_audit(world, res.rom))
    res.safety["stalls"] = [s.to_dict() for s in world.robot.stalls]
    res.safety["starts"] = world.robot.starts
    res.safety["max_start_jump_deg"] = max((s["jump_deg"] for s in world.robot.starts), default=0.0)
    res.timings["total_s"] = round(time.monotonic() - t_start, 1)
    res.timings["camera_frames"] = world.camera.frames_read
    (out / "trace.json").write_text(json.dumps(trace.rows) + "\n")
    (out / "events.json").write_text(json.dumps(world.log, indent=1, default=str) + "\n")
    (out / "session.json").write_text(json.dumps(res.to_dict(), indent=2, default=str) + "\n")
    return res


def _link_truth(truth, links: dict) -> dict:
    """DTG link name -> the real link most of its tags sit on."""
    from collections import Counter
    out = {}
    for name, tags in links.items():
        votes = Counter(truth.tags[int(k)]["link"] for k in tags if int(k) in truth.tags)
        out[name] = votes.most_common(1)[0][0] if votes else None
    return out


def _wiggle_summary(wig) -> dict:
    return {c.channel: {"status": c.status, "child": c.child, "parent": c.parent, "direction": c.direction,
                        "reasons": c.reasons} for c in wig.channels}


def _wait_settled(world, session, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while session.moving():
        if time.monotonic() > deadline:
            raise TimeoutError("slider target not reached")
        session.step(0.05)
        time.sleep(0.05)
    # the firmware slews and the servo lags: wait until the joint is still
    while time.monotonic() < deadline:
        world.robot.advance(world.clock())
        if not world.robot.moving():
            return
        time.sleep(0.02)


def _sweep_joint(world, session, recorder, ch, j, cfg, rng) -> dict:
    """One joint's ROM, like an operator at the ROM window."""
    from robo_prober.rom import ABS_MAX_US, ABS_MIN_US
    session.enable(ch)                                   # Enable at the shown start (last position)
    _wait_settled(world, session)
    first = int(rng.choice([1, -1]))                     # which end gets ROM 1
    marks, widened = {}, 0
    margin = float(rng.uniform(*cfg.margin_deg))
    for which, sign in ((1, first), (2, -first)):
        target = session.position(ch)
        while True:
            nxt = target + sign * cfg.rom_step_us
            step_deg = cfg.rom_step_us * j.gain_deg_per_us
            if world.robot.stop_margin_deg(j.name, sign) - step_deg < margin:
                break                                    # the operator sees the stop coming
            if not ABS_MIN_US + 100 <= nxt <= ABS_MAX_US - 100:
                break
            low, high = session.channels[ch].slider_range()
            if not low <= nxt <= high:
                if not session.channels[ch].can_widen():
                    break
                session.widen(ch)                        # "Widen range": explicit, logged
                widened += 1
                continue
            session.set_target(ch, nxt)
            _wait_settled(world, session)
            target = nxt
            recorder.rom_sample(ch, target)
        pulse = session.mark(ch, which)                  # "Set ROM n"
        q = world.robot.joint_angles()[j.name]
        recorder.rom_mark(ch, which, pulse)
        marks[which] = {"pulse_us": pulse, "joint_deg": q,
                        "stop_margin_deg": world.robot.stop_margin_deg(j.name, sign)}
        world.event("rom_mark", channel=ch, which=which, pulse_us=pulse, joint_deg=q)
    # The firmware must now hold the channel inside the marked range.
    from robo_prober.transport import ProtocolError
    low, high = session.channels[ch].range
    before = world.robot.state[j.name].pulse_us
    enforced = True
    for bad in (high + 20, low - 20):
        try:
            session.controller.move(ch, bad)
            enforced = False
        except ProtocolError as error:
            enforced &= "out_of_limits" in str(error)
    enforced &= world.robot.state[j.name].pulse_us == before
    rom_complete_t = world.clock()
    # back to the start pulse (inside the new safe range), sampling on the way
    start = session.channels[ch].start_us
    target = session.position(ch)
    while target != start:
        target = target + max(-cfg.rom_step_us, min(cfg.rom_step_us, start - target))
        session.set_target(ch, target)
        _wait_settled(world, session)
        recorder.rom_sample(ch, target)
    return {"joint": j.name, "marks": marks, "widened": widened, "margin_deg": margin,
            "range_us": session.channels[ch].range, "rom_band_enforced": enforced, "rom_complete_t": rom_complete_t}


FIRMWARE_SLEW_CAP_US_S = 250   # every band the host pushed in this session uses 250 us/s (firmware default too)


def _pwm_audit(world, rom: dict) -> dict:
    """Every PWM edge the firmware drove in the whole session, checked offline:
    pulses never past a hard stop (commanded, ignoring lag), never outside the
    ROM range once it was marked, unused channels only wiggled, and the slew
    never above the firmware cap (START is the only step)."""
    by_ch = world.truth.by_channel()
    past_stop, outside_rom, unused_far, worst_rate = 0, 0, 0, 0.0
    seg: dict[int, list] = {}
    for t, t_fw, ch, pulse in world.pwm_log:
        if pulse is None:
            seg.pop(ch, None)
            continue
        j = by_ch.get(ch)
        if j is not None:
            q = j.joint_deg_at(pulse)
            past_stop += int(not j.stop_lo_deg <= q <= j.stop_hi_deg)
            r = rom.get(ch)
            if r and t > r["rom_complete_t"]:
                lo, hi = r["range_us"]
                outside_rom += int(not (lo <= pulse <= hi or 1400 <= pulse <= 1600))
        else:
            unused_far += int(abs(pulse - 1500) > 100)
        pts = seg.setdefault(ch, [])
        pts.append((t_fw, pulse))
        # rate over the last >= 50 ms of this driven stretch
        k = len(pts) - 1
        while k > 0 and pts[-1][0] - pts[k][0] < 0.05:
            k -= 1
        dt = pts[-1][0] - pts[k][0]
        if dt >= 0.05:
            worst_rate = max(worst_rate, (abs(pts[-1][1] - pts[k][1]) - 1) / dt)
    return {"pwm_edges": len(world.pwm_log), "pwm_past_hard_stop": past_stop, "pwm_outside_rom": outside_rom,
            "unused_channel_far_pulses": unused_far, "max_slew_us_s": round(worst_rate, 1),
            "slew_capped": worst_rate <= FIRMWARE_SLEW_CAP_US_S * 1.02,
            "rom_band_enforced": all(r.get("rom_band_enforced") for r in rom.values()) if rom else None}


def _safety_probes(world, bench, rom_channels, cfg) -> dict:
    """Fail-closed behaviour of the real firmware under the real host."""
    from robo_prober.api import Controller
    from robo_prober.transport import ProtocolError, SerialProtocolTransport

    out: dict = {}
    ch = rom_channels[0][0] if rom_channels else cfg.channels[0]
    start = cfg.start_us

    def fresh():
        return Controller(SerialProtocolTransport(bench.port, timeout_s=1.0)).open()

    def still(seconds=0.3):
        a = world.robot.snapshot(world.clock())
        time.sleep(seconds)
        b = world.robot.snapshot(world.clock())
        return max(abs(a[k] - b[k]) for k in a) < 0.05

    # arming: needs the cutoff confirmation and a valid token; nothing moves before it
    ctl = fresh()
    guarded = []
    try:
        ctl.arm(power_cutoff_confirmed=False)
        guarded.append(False)
    except Exception:  # noqa: BLE001 - the host refuses before sending anything
        guarded.append(not ctl.armed)
    for raw, reason in (("ARM token=x cutoff=no", "cutoff_unconfirmed"), ("ARM token=x", "missing_arm_fields"),
                        ("ARM token=bad! cutoff=confirmed", "bad_token"),
                        (f"START channel={ch} pulse_us={start} token=x", "disarmed"),
                        (f"MOVE channel={ch} target_us={start} seq=1 token=x", "disarmed")):
        try:
            ctl.transport.transact(raw)
            guarded.append(False)
        except ProtocolError as error:
            guarded.append(reason in str(error))
    time.sleep(0.05)
    out["arming_guarded"] = all(guarded) and not bench.active_channels()
    ctl.transport.close()
    # out-of-band: default band 1400..1600 after a fresh LIMIT
    ctl = fresh()
    ctl.set_limits(ch, 1400, 1600, neutral_us=start)
    ctl.arm(power_cutoff_confirmed=True)
    ctl.start(ch, start)
    bench.wait_output(ch, start)
    before = bench.outputs[ch]
    rejected = []
    for bad in (1700, 1399):
        try:
            ctl.move(ch, bad)
            rejected.append(False)
        except ProtocolError as error:
            rejected.append("out_of_limits" in str(error))
    for raw, reason in ((f"MOVE channel={ch} target_us=3000 seq=900 token={ctl.token}", "bad_move_values"),
                        (f"MOVE channel={ch} target_us=1500 seq=901", "token_required"),
                        (f"MOVE channel={ch} target_us=1500 seq=902 token=wrong", "bad_token")):
        try:
            ctl.transport.transact(raw)
            rejected.append(False)
        except ProtocolError as error:
            rejected.append(reason in str(error))
    time.sleep(0.1)
    out["out_of_band_rejected"] = all(rejected) and bench.outputs[ch] == before
    # E-STOP command: releases now, latches until cleared with the cutoff confirmed
    t = time.monotonic()
    ctl.estop()
    bench.wait_released(1.0)
    out["estop_release_s"] = round(time.monotonic() - t, 3)
    out["estop_holds_still"] = still()
    try:
        ctl.arm(power_cutoff_confirmed=True)
        out["estop_latched"] = False
    except ProtocolError as error:
        out["estop_latched"] = "estop_latched" in str(error)
        ctl._end_session()
    ctl.clear_estop(power_cutoff_confirmed=True)
    # hardware e-stop input
    ctl.arm(power_cutoff_confirmed=True)
    ctl.start(ch, start)
    bench.wait_output(ch, start)
    bench.estop_pin()
    out["estop_pin_release_s"] = round(bench.wait_released(1.0), 3)
    ctl._end_session()
    try:
        ctl.arm(power_cutoff_confirmed=True)
        out["estop_pin_latched"] = False
        ctl.disarm()
    except ProtocolError as error:
        out["estop_pin_latched"] = "estop_latched" in str(error)
        ctl._end_session()
    ctl.clear_estop(power_cutoff_confirmed=True)
    # watchdog: host goes silent while armed and driving
    ctl.arm(power_cutoff_confirmed=True, heartbeat="manual")
    ctl.start(ch, start)
    bench.wait_output(ch, start)
    ctl.move(ch, start + 30)
    silent = time.monotonic()
    bench.wait_released(3.0)
    out["watchdog_release_s"] = round(time.monotonic() - silent, 3)
    out["watchdog_holds_still"] = still()
    ctl._end_session()
    ctl.transport.close()
    # host unplugged / crashed: port closed without DISARM while the heartbeat thread runs
    ctl = fresh()
    ctl.arm(power_cutoff_confirmed=True)
    ctl.start(ch, start)
    bench.wait_output(ch, start)
    hb = ctl._heartbeat
    gone = time.monotonic()
    if hb is not None:
        hb._stop.set()              # the process "dies": no more heartbeats
    ctl.transport.close()
    bench.wait_released(3.0)
    out["unplug_release_s"] = round(time.monotonic() - gone, 3)
    # arming alone outputs nothing
    ctl = fresh()
    ctl.arm(power_cutoff_confirmed=True)
    out["arm_alone_outputs"] = dict(bench.active_channels())
    ctl.close()
    bench.wait_released(1.0)
    return out
