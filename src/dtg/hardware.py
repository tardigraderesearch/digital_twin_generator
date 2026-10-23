"""The bench: controller + camera for the wiggle and range-of-motion steps.

Everything that touches the controller goes through robo-prober (imported as
a library: discovery cascade, firmware protocol, arming, ROM session). DTG
adds the camera and records what it sees. Sources:

    "sim" / "sim:<seed>"  the virtual bench (sim/): robo-prober's real firmware
                          core behind a pseudo terminal, a simulated robot and
                          a rendered camera. No hardware.
    "emulator"            robo-prober's in-process firmware model (no camera, no robot)
    "/dev/ttyACM0", "COM3", ...  a real controller (needs a camera too)

Safety (PROGRAM_STANDARD section 3), enforced here and in the firmware:

1. ``connect`` only reads: HELLO, then the read-only discovery survey.
   A latched e-stop is detected on connect and must be cleared first.
2. ``arm`` needs ``cutoff_confirmed=True`` (the operator ticked "power cutoff
   within reach"); robo-prober makes the session token.
3. Pins something else drives are never targets (robo-prober's survey
   classes them ``driven_*``; the firmware refuses them too: ``pin_driven``).
4. Each wiggle first sends a LIMIT band (start +/- delta, slew cap) so the
   firmware clamps the move even if the host misbehaves; the watchdog
   releases every output if the host goes quiet.
5. Every first move is explicit: the operator types the start pulse; there
   is no silent default.
6. Refusals carry a reason code (``BenchError.reason``).
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WIGGLE_SLEW_US_S = 250       # the firmware's default slew cap; the band keeps the wiggle small


class BenchError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def sim_available() -> tuple[bool, str]:
    """The virtual bench needs the repo's sim/ folder, robo-prober's firmware sources and aruco-tag-maker."""
    if not (REPO / "sim" / "bench.py").is_file():
        return False, "sim/ folder not found (source checkout only)"
    try:
        from robo_prober.firmware_bench import bench_available
        if not bench_available():
            return False, "robo-prober firmware bench unavailable (needs a C++ compiler and a source checkout)"
        import aruco_tag_maker  # noqa: F401
    except ImportError as exc:
        return False, f"missing {exc.name} (run: make sim-env, then use sim/.venv/bin/python)"
    return True, "ok"


def parse_channels(text: str) -> list[int]:
    """'0-3, 5, 104' -> [0, 1, 2, 3, 5, 104]."""
    out: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def format_channels(channels) -> str:
    chans = sorted(channels)
    runs, start = [], None
    for i, c in enumerate(chans):
        if start is None:
            start = c
        if i + 1 == len(chans) or chans[i + 1] != c + 1:
            runs.append(f"{start}" if start == c else f"{start}-{c}")
            start = None
    return ", ".join(runs)


@dataclass
class Target:
    channel: int
    key: str
    cls: str


class Bench:
    """One connection to a controller (+ camera). Use from one worker thread at a time."""

    def __init__(self, source: str, *, camera=None, intrinsics=None, seed: int = 0, sim_light: float | None = 60.0):
        self.source = source
        self.camera = camera            # cv2 index / URL or a capture object; None for the emulator
        self.intrinsics = intrinsics    # dtg.camera.CameraIntrinsics
        self.seed = seed
        self.sim_light = sim_light
        self.world = None
        self._fw = None
        self.controller = None
        self.survey = None
        self.targets: list[Target] = []
        self.tracker = None
        self.recorder = None
        self.rom = None                 # robo_prober.rom.RomSession while the ROM step runs
        self.estop_latched = False
        self.lock = threading.RLock()
        self.log: list[str] = []

    # ------------------------------------------------------------ connect (read-only)
    @property
    def is_sim(self) -> bool:
        return self.source == "sim" or self.source.startswith("sim:")

    @property
    def connected(self) -> bool:
        return self.controller is not None

    @property
    def armed(self) -> bool:
        return bool(self.controller is not None and (self.controller.armed or (self.rom and self.rom.started)))

    def connect(self) -> dict:
        from robo_prober.api import Controller, connect
        from robo_prober.transport import SerialProtocolTransport
        if self.connected:
            return self.controller.info
        if self.is_sim:
            ok, why = sim_available()
            if not ok:
                raise BenchError("sim_unavailable", why)
            if str(REPO) not in sys.path:
                sys.path.insert(0, str(REPO))
            from sim.bench import build_world
            if ":" in self.source:
                self.seed = int(self.source.split(":", 1)[1])
            noise = {"full_well": self.sim_light} if self.sim_light else None
            self.world = build_world(self.seed, camera_noise=noise)
            self._fw = self.world.firmware()
            fw = self._fw.__enter__()
            self.controller = Controller(SerialProtocolTransport(fw.port, timeout_s=1.0)).open()
            if self.camera is None:
                self.camera = self.world.camera
            if self.intrinsics is None:
                from .camera import CameraIntrinsics
                self.intrinsics = CameraIntrinsics(self.world.K_calibrated, np.zeros(5), (1920, 1080))
        elif self.source == "emulator":
            self.controller = connect(emulator=True)
        else:
            self.controller = connect(self.source)
        info = self.controller.info
        self.estop_latched = info.get("mode") == "estop"
        self._note(f"connected to {info.get('id', '?')} fw {info.get('fw', '?')} ({self.source})"
                   + ("; E-STOP IS LATCHED: clear it before anything moves" if self.estop_latched else ""))
        return info

    def discover(self) -> list[Target]:
        """Read-only survey (robo-prober's discovery cascade). Nothing is driven."""
        from robo_prober import find
        ctl = self._require()
        if "survey" not in ctl.caps:
            # older firmware: no survey; the PCA9685 channels are the only outputs it drives
            self.targets = [Target(ch, f"PCA9685 ch {ch}", "pca9685") for ch in range(16)]
            self._note("firmware has no survey: offering PCA9685 channels 0-15")
            return self.targets
        self.survey = find.discover(ctl)
        self.targets = [Target(t.channel, t.key, t.cls) for t in find.targets(self.survey)]
        self._note(self.survey.summary())
        return self.targets

    def markers_manifest(self) -> dict | None:
        """The simulated bench's markers.json (its tags come from aruco-tag-maker)."""
        return self.world.manifest if self.world is not None else None

    def camera_dict(self) -> dict:
        return self.intrinsics.to_dict() if self.intrinsics is not None else {}

    # ------------------------------------------------------------ arming
    def arm(self, *, cutoff_confirmed: bool) -> str:
        ctl = self._require()
        if not cutoff_confirmed:
            raise BenchError("cutoff_unconfirmed", "tick 'Physical power cutoff is within reach' first")
        if self.estop_latched:
            raise BenchError("estop_latched", "the e-stop is latched: clear it first")
        if ctl.armed:
            return ctl.token
        with self.lock:
            token = ctl.arm(power_cutoff_confirmed=True)
        self._note("armed (session token issued; watchdog fed while this program runs)")
        return token

    def disarm(self) -> None:
        with self.lock:
            if self.rom is not None:
                self._close_rom()
            if self.controller is not None and self.controller.armed:
                self.controller.disarm()
        self._note("disarmed: every output released")

    def estop(self) -> None:
        """Latching E-STOP. Never waits for a busy worker: it goes straight to the transport."""
        ctl = self.controller
        if ctl is None:
            return
        try:
            if self.rom is not None:
                self.rom.estop()
            ctl.estop()
        finally:
            self.rom = None
            self.estop_latched = True
            self._note("E-STOP: every output released and latched until Clear")

    def clear_estop(self, *, cutoff_confirmed: bool) -> None:
        ctl = self._require()
        if not cutoff_confirmed:
            raise BenchError("cutoff_unconfirmed", "tick 'Physical power cutoff is within reach' to clear the e-stop")
        ctl.clear_estop(power_cutoff_confirmed=True)
        self.estop_latched = False
        self._note("e-stop cleared (nothing armed)")

    def close(self) -> None:
        try:
            if self.controller is not None:
                try:
                    self.disarm()
                finally:
                    self.controller.close()
        finally:
            self.controller = None
            if self._fw is not None:
                self._fw.__exit__(None, None, None)
                self._fw = None

    # ------------------------------------------------------------ camera
    def open_camera(self, markers, *, focus: bool = True, roi=None):
        from .camera import ArucoTracker
        if self.camera is None:
            raise BenchError("no_camera", "the wiggle and ROM are observed by the camera; pick one")
        if self.intrinsics is None:
            raise BenchError("no_intrinsics", "load the camera calibration (camera.json) first")
        self.tracker = ArucoTracker(self.camera, markers, self.intrinsics)
        if focus:
            if roi is None and self.world is not None:
                from sim.operator import robot_roi
                roi = robot_roi(self.world)          # the box the operator drags around the robot
            self.tracker.lock_focus(roi=roi)
            self._note(f"focus locked at lens {self.tracker.focus.lens} (peak {self.tracker.focus.peak_ratio:.1f}x)")
        return self.tracker

    def start_recording(self, log_path: Path, markers, *, append: bool = True, settle_s: float = 0.4):
        """Observation log for the wiggle/ROM captures. Appends to an existing log by default."""
        from .camera import ObservationRecorder
        if self.tracker is None:
            self.open_camera(markers)
        log_path = Path(log_path)
        previous = log_path.read_text() if append and log_path.is_file() else ""
        self.recorder = ObservationRecorder(self.tracker, log_path, markers, settle_s=settle_s)
        if previous:
            # keep the earlier captures (the first recording's header wins); time keeps increasing
            log_path.write_text(previous if previous.endswith("\n") else previous + "\n")
            self.recorder.t0 -= _last_t(previous)
        return self.recorder

    def preview_image(self, width: int = 720):
        """One camera frame with the detected tags outlined (PIL image), or None without a camera."""
        if self.tracker is None:
            return None
        import cv2
        from PIL import Image
        from .camera import preprocess
        ok, image = self.tracker.cap.read()
        if not ok:
            return None
        gray = preprocess(image, clahe=True)
        corners, ids, _ = self.tracker.detector.detectMarkers(gray)
        rgb = cv2.cvtColor(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                pts = c.reshape(4, 2).astype(int)
                cv2.polylines(rgb, [pts], True, (46, 230, 122), 4)
                cv2.putText(rgb, str(int(i)), tuple(pts[0] + [0, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (46, 230, 122), 3)
        img = Image.fromarray(rgb)
        if self.world is not None:                       # crop to the robot like the operator's box
            from sim.operator import robot_roi
            x, y, w, h = robot_roi(self.world, pad=120)
            img = img.crop((x, y, x + w, y + h))
        return img.resize((width, max(1, int(img.height * width / img.width))))

    def visible_tags(self) -> list[int]:
        if self.tracker is None:
            return []
        poses, _ = self.tracker.capture(frames=4)
        return sorted(poses)

    # ------------------------------------------------------------ wiggle (step 3)
    def reference(self):
        """Reference capture: the robot in the pose it was scanned in (joint zero)."""
        if self.recorder is None:
            raise BenchError("no_recorder", "start the recording first")
        return self.recorder.reference()

    def wiggle(self, channel: int, *, start_us: int, delta_us: int = 80, slew_us_s: float = 200.0):
        """Small back-and-forth on one output, a settled capture at each point.

        The firmware gets a LIMIT band of start +/- delta (plus wherever the
        channel is held) first, so it clamps the move by itself.
        """
        from .camera import MAX_WIGGLE_US
        ctl = self._require()
        if not ctl.armed:
            raise BenchError("disarmed", "arm the controller first (power cutoff within reach)")
        if self.estop_latched:
            raise BenchError("estop_latched", "the e-stop is latched")
        if self.survey is not None:
            info = self.survey.pin(channel - 100) if channel >= 100 else None
            if info is not None and not info.drivable:
                raise BenchError("pin_driven", f"GPIO {info.pin} is {info.cls}: never driven")
        if not isinstance(start_us, int) or not 800 <= start_us <= 2200:
            raise BenchError("bad_start", "type the start pulse (800-2200 us): the pulse the servo is most likely at")
        delta = int(min(abs(delta_us), MAX_WIGGLE_US))
        with self.lock:
            if "limit" in ctl.caps:
                held = ctl.position(channel)
                low, high = start_us - delta, start_us + delta
                if held is not None:
                    low, high = min(low, held), max(high, held)
                ctl.set_limits(channel, low, high, neutral_us=start_us, slew_us_s=WIGGLE_SLEW_US_S)
            frames = self.recorder.wiggle_channel(ctl, channel, start_us=start_us, delta_us=delta,
                                                  slew_us_s=slew_us_s)
        return frames

    # ------------------------------------------------------------ ROM (step 4)
    def start_rom(self, joints: list[tuple[int, str, int]], *, cutoff_confirmed: bool):
        """joints: [(channel, name, start_us)]. Re-arms through robo-prober's RomSession."""
        from robo_prober.rom import RomChannel, RomSession
        ctl = self._require()
        if not cutoff_confirmed:
            raise BenchError("cutoff_unconfirmed", "tick 'Physical power cutoff is within reach' first")
        if self.estop_latched:
            raise BenchError("estop_latched", "the e-stop is latched: clear it first")
        with self.lock:
            if self.rom is not None:
                self._close_rom()
            if ctl.armed:
                ctl.disarm()
            roms = [RomChannel(ch, name, start_us=start, start_source="last_position") for ch, name, start in joints]
            self.rom = RomSession(ctl, roms, heartbeat="thread")
            self.rom.start(power_cutoff_confirmed=True)
        self._note(f"ROM session armed for {len(joints)} joint(s)")
        return self.rom

    def rom_enable(self, channel: int) -> int:
        with self.lock:
            pulse = self._rom().enable(channel)
        self._settle()
        return pulse

    def rom_move(self, channel: int, target_us: int, *, sample: bool = True, timeout_s: float = 20.0) -> int:
        """Slew toward a target (clamped to the allowed band), wait, then a settled capture."""
        rom = self._rom()
        with self.lock:
            pulse = rom.set_target(channel, target_us)
        deadline = time.monotonic() + timeout_s
        while rom.moving():
            if time.monotonic() > deadline:
                raise BenchError("timeout", "target not reached")
            with self.lock:
                rom.step(0.05)
            time.sleep(0.05)
        self._settle()
        if sample and self.recorder is not None:
            self.recorder.rom_sample(channel, rom.position(channel))
        return pulse

    def rom_mark(self, channel: int, which: int):
        with self.lock:
            pulse = self._rom().mark(channel, which)
        frame = self.recorder.rom_mark(channel, which, pulse) if self.recorder is not None else None
        return pulse, frame

    def rom_widen(self, channel: int):
        with self.lock:
            return self._rom().widen(channel)

    def _settle(self):
        if self.world is not None:                  # the sim knows when its servos stop
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                self.world.robot.advance(self.world.clock())
                if not self.world.robot.moving():
                    return
                time.sleep(0.02)

    def _rom(self):
        if self.rom is None or not self.rom.started:
            raise BenchError("disarmed", "start the ROM session first (arm)")
        return self.rom

    def _close_rom(self):
        try:
            self.rom.close()
        finally:
            self.rom = None

    # ------------------------------------------------------------ helpers
    def _require(self):
        if self.controller is None:
            raise BenchError("not_connected", "connect first")
        return self.controller

    def _note(self, text: str) -> None:
        self.log.append(time.strftime("%H:%M:%S ") + text)


def _last_t(text: str) -> float:
    for line in reversed(text.splitlines()):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "t" in d:
            return float(d["t"]) + 1.0
    return 0.0
