"""Virtual robot: a 3-joint desktop arm modelled on bot1, with hidden ground truth.

bot1 (robo-prober/assets/source/bot1.obj, ~180 x 200 x 190 mm with its
turntable) is a cylindrical base with a yaw turret, an upper arm on a
shoulder pitch joint and a hanging forearm on an elbow pitch joint. Every run
draws a new robot from a seed:

* link sizes around bot1's (base radius/height, upper arm and forearm length,
  small assembly tilt of the pitch axes);
* wiring: which PCA9685 channel drives which joint, the rest unused;
* servo: rotation direction, pulse -> angle gain, centre trim, slew speed,
  first-order lag, gear backlash;
* horn mounting offset (whole spline teeth plus a little), so pulse 1500 is
  not the design zero pose;
* mechanical hard stops. Commanding past a stop stalls the servo against it;
  the robot logs that as a "damage risk" event.

Time advances in ``update(dt)`` from the PWM pulses the (real) firmware is
outputting; a released channel leaves the servo limp where it is.
Geometry is in metres, world frame z up; the base never moves.
"""
from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

SPLINE_TEETH = 25           # common hobby-servo output spline
CHANNELS_IN_USE = 8         # the operator wiggles PCA9685 channels 0..7
NEUTRAL_US = 1500


@dataclass
class JointTruth:
    name: str
    parent: str
    child: str
    axis: list                 # unit axis in world at the design-zero pose
    pivot: list                # point on the axis, world, design-zero pose (m)
    channel: int
    direction: int             # +1 / -1: joint angle sign per servo angle
    gain_deg_per_us: float     # servo angle per microsecond
    trim_deg: float            # servo centre error at 1500 us
    horn_offset_deg: float     # horn mounted this far off (teeth + slop)
    speed_deg_s: float         # servo no-load speed
    lag_s: float               # first-order lag of the servo loop
    backlash_deg: float        # gear lash at the joint
    stop_lo_deg: float         # hard stops, joint angle (design-zero = 0)
    stop_hi_deg: float
    rest_offset_deg: float     # where the joint sits at power-up, relative to its 1500 us angle

    def joint_deg_at(self, pulse_us: float) -> float:
        """Joint angle the servo would hold at this pulse (no stops, no lash)."""
        return self.direction * (self.gain_deg_per_us * (pulse_us - NEUTRAL_US) + self.trim_deg) + self.horn_offset_deg

    def pulse_for(self, joint_deg: float) -> float:
        return NEUTRAL_US + ((joint_deg - self.horn_offset_deg) * self.direction - self.trim_deg) / self.gain_deg_per_us


@dataclass
class RobotTruth:
    seed: int
    joints: list[JointTruth]
    unused_channels: list[int]
    dims: dict
    tags: dict                 # tag id -> {"link", "T0" (4x4 world pose at design zero), "name"}

    def joint(self, name: str) -> JointTruth:
        return next(j for j in self.joints if j.name == name)

    def by_channel(self) -> dict[int, JointTruth]:
        return {j.channel: j for j in self.joints}

    def to_dict(self) -> dict:
        return {"seed": self.seed, "joints": [asdict(j) for j in self.joints],
                "unused_channels": self.unused_channels, "dims": self.dims,
                "tags": {str(k): {"link": v["link"], "name": v["name"], "T0": np.asarray(v["T0"]).tolist()}
                         for k, v in self.tags.items()}}


LINKS = ("base", "turret", "upper_arm", "forearm")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _tilted(axis, rng, max_deg) -> list:
    """Axis with a small random assembly tilt."""
    tilt = Rotation.from_rotvec(_unit(rng.normal(size=3)) * math.radians(rng.uniform(0, max_deg)))
    return tilt.apply(_unit(axis)).tolist()


def random_robot(seed: int) -> RobotTruth:
    """Draw a bot1-like arm (tags are placed later, by sim.scene, at the rest pose)."""
    rng = np.random.default_rng(seed)
    dims = {
        "base_r": rng.uniform(0.040, 0.050), "base_h": rng.uniform(0.045, 0.060),
        "turret_h": rng.uniform(0.040, 0.050), "upper_len": rng.uniform(0.085, 0.110),
        "fore_len": rng.uniform(0.075, 0.095), "arm_w": 0.022, "arm_t": 0.020,
        "shoulder_y": -rng.uniform(0.036, 0.044), "elbow_dy": -rng.uniform(0.020, 0.024),
    }
    z_sh = dims["base_h"] + dims["turret_h"] * 0.7
    dims["shoulder"] = [0.0, dims["shoulder_y"], z_sh]
    dims["elbow"] = [dims["upper_len"], dims["shoulder_y"] + dims["elbow_dy"], z_sh]
    channels = rng.permutation(CHANNELS_IN_USE)
    specs = [
        ("j_turret", "base", "turret", _tilted((0, 0, 1), rng, 0.8), [0.0, 0.0, dims["base_h"]]),
        ("j_upper_arm", "turret", "upper_arm", _tilted((0, 1, 0), rng, 1.5), dims["shoulder"]),
        ("j_forearm", "upper_arm", "forearm", _tilted((0, 1, 0), rng, 1.5), dims["elbow"]),
    ]
    joints = []
    for i, (name, parent, child, axis, pivot) in enumerate(specs):
        gain = rng.uniform(0.085, 0.105)
        direction = int(rng.choice([-1, 1]))
        trim = rng.uniform(-2.0, 2.0)
        horn = int(rng.integers(-2, 3)) * 360.0 / SPLINE_TEETH + rng.uniform(-3.0, 3.0)
        j = JointTruth(name=name, parent=parent, child=child, axis=axis, pivot=list(pivot), channel=int(channels[i]),
                       direction=direction, gain_deg_per_us=gain, trim_deg=trim, horn_offset_deg=horn,
                       speed_deg_s=rng.uniform(300, 500), lag_s=rng.uniform(0.02, 0.07),
                       backlash_deg=rng.uniform(0.0, 0.6), stop_lo_deg=0.0, stop_hi_deg=0.0,
                       rest_offset_deg=rng.uniform(-1.5, 1.5))
        # Hard stops: 400..850 us of travel either side of 1500, in joint degrees.
        q0 = j.joint_deg_at(NEUTRAL_US)
        a, b = j.joint_deg_at(NEUTRAL_US - rng.uniform(400, 850)), j.joint_deg_at(NEUTRAL_US + rng.uniform(400, 850))
        j.stop_lo_deg, j.stop_hi_deg = min(a, b), max(a, b)
        assert j.stop_lo_deg < q0 < j.stop_hi_deg
        joints.append(j)
    unused = sorted(int(c) for c in channels[len(specs):])
    return RobotTruth(seed=seed, joints=joints, unused_channels=unused, dims=dims, tags={})


def joint_motion(j: JointTruth, q_deg: float) -> np.ndarray:
    """World motion of the child for joint angle q (design zero = identity)."""
    R = Rotation.from_rotvec(np.asarray(j.axis) * math.radians(q_deg)).as_matrix()
    p = np.asarray(j.pivot, float)
    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = p - R @ p
    return G


def link_motions(truth: RobotTruth, q_deg: dict[str, float]) -> dict[str, np.ndarray]:
    """World motion of every link (maps design-zero world points to current world points)."""
    out = {"base": np.eye(4)}
    for j in truth.joints:  # ordered root to tip
        out[j.child] = out[j.parent] @ joint_motion(j, q_deg[j.name])
    return out


@dataclass
class ServoState:
    servo_deg: float           # servo output angle (internal, before lash)
    joint_deg: float           # joint angle after lash and stops
    pulse_us: int | None = None
    target_deg: float = 0.0
    stalled: bool = False


@dataclass
class StallEvent:
    t: float
    joint: str
    channel: int
    pulse_us: int
    commanded_deg: float
    stop_deg: float

    def to_dict(self):
        return asdict(self)


class VirtualRobot:
    """Servo dynamics driven by PWM pulses. Thread safe: the bench thread
    writes pulses and integrates, the camera reads poses."""

    def __init__(self, truth: RobotTruth):
        self.truth = truth
        self._lock = threading.Lock()
        self.state: dict[str, ServoState] = {}
        for j in truth.joints:
            q = j.joint_deg_at(NEUTRAL_US) + j.rest_offset_deg
            self.state[j.name] = ServoState(servo_deg=(q - j.horn_offset_deg) * j.direction, joint_deg=q)
        self._by_channel = truth.by_channel()
        self.t = 0.0
        self.stalls: list[StallEvent] = []
        self.starts: list[dict] = []        # first pulse after release: how far the joint had to jump
        self.max_speed_deg_s: dict[str, float] = {j.name: 0.0 for j in truth.joints}
        self.unused_pulses: dict[int, int] = {}

    # ------------------------------------------------------------ inputs
    def set_pulse(self, t: float, channel: int, pulse_us: int | None) -> None:
        with self._lock:
            self._advance(t)
            j = self._by_channel.get(channel)
            if j is None:
                if pulse_us is not None:
                    self.unused_pulses[channel] = pulse_us
                return
            s = self.state[j.name]
            if s.pulse_us is None and pulse_us is not None:
                target = j.joint_deg_at(pulse_us)
                self.starts.append({"t": t, "joint": j.name, "channel": channel, "pulse_us": pulse_us,
                                    "jump_deg": abs(target - s.joint_deg)})
            s.pulse_us = pulse_us

    def advance(self, t: float) -> None:
        with self._lock:
            self._advance(t)

    def _advance(self, t: float) -> None:
        dt_total = t - self.t
        if dt_total <= 0:
            return
        steps = max(1, int(math.ceil(dt_total / 0.002)))
        dt = dt_total / steps
        for _ in range(steps):
            self.t += dt
            for j in self.truth.joints:
                self._step_joint(j, self.state[j.name], dt)

    def _step_joint(self, j: JointTruth, s: ServoState, dt: float) -> None:
        if s.pulse_us is None:      # released: limp, holds where it is
            s.stalled = False
            return
        target = j.gain_deg_per_us * (s.pulse_us - NEUTRAL_US) + j.trim_deg
        s.target_deg = target
        # first-order lag, capped at the servo's speed
        rate = (target - s.servo_deg) / max(j.lag_s, dt)
        rate = max(-j.speed_deg_s, min(j.speed_deg_s, rate))
        new_servo = s.servo_deg + rate * dt
        # joint follows the servo through the gear lash
        want = j.direction * new_servo + j.horn_offset_deg
        q = s.joint_deg
        half = j.backlash_deg / 2
        if want > q + half:
            q = want - half
        elif want < q - half:
            q = want + half
        stalled = False
        if q > j.stop_hi_deg or q < j.stop_lo_deg:
            stop = j.stop_hi_deg if q > j.stop_hi_deg else j.stop_lo_deg
            q = stop
            stalled = True
            # the servo cannot turn past the stop either (it stalls against it)
            new_servo = (stop + (half if want < stop else -half) - j.horn_offset_deg) * j.direction
            commanded = j.direction * target + j.horn_offset_deg
            if not s.stalled:
                self.stalls.append(StallEvent(self.t, j.name, j.channel, s.pulse_us, commanded, stop))
        speed = abs(q - s.joint_deg) / dt
        self.max_speed_deg_s[j.name] = max(self.max_speed_deg_s[j.name], speed)
        s.servo_deg, s.joint_deg, s.stalled = new_servo, q, stalled

    # ------------------------------------------------------------ outputs
    def joint_angles(self) -> dict[str, float]:
        with self._lock:
            return {k: v.joint_deg for k, v in self.state.items()}

    def snapshot(self, t: float | None = None) -> dict[str, float]:
        with self._lock:
            if t is not None:
                self._advance(t)
            return {k: v.joint_deg for k, v in self.state.items()}

    def moving(self, tol_deg: float = 0.05) -> bool:
        with self._lock:
            for j in self.truth.joints:
                s = self.state[j.name]
                if s.pulse_us is not None and not s.stalled:
                    if abs(s.target_deg - s.servo_deg) > tol_deg:
                        return True
            return False

    def stop_margin_deg(self, name: str, pulse_direction: int) -> float:
        """Degrees left before the hard stop the joint is heading for when the
        pulse moves in ``pulse_direction``. This is what the operator sees."""
        j = self.truth.joint(name)
        q = self.joint_angles()[name]
        heading = pulse_direction * j.direction
        return (j.stop_hi_deg - q) if heading > 0 else (q - j.stop_lo_deg)
