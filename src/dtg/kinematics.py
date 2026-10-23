"""ROM step: joint axis, origin, terminal ranges and pulse<->angle map.

Runs per joint discovered by the wiggle (dtg.linkage), on the camera frames
captured while the operator moves that joint's slider and presses
Set ROM 1 / Set ROM 2. The scanner is not involved.

For joint c with parent link P and child link C (link pose = averaged tag
poses, see linkage.link_pose):

    Q_t   = inv(P_t) @ C_t                child in parent coordinates
    D_t   = Q_t @ inv(Q_ref)              motion since the reference pose
    axis  = first right-singular vector of the stacked rotation vectors of D_t
            (canonical sign: largest component positive)
    theta = rotvec(D_t) . axis            joint position, 0 at the reference
    pivot = least squares (R_t - I) p = -t_t, the point the child turns about,
            taken where the axis passes closest to the child's origin;
            cross-checked against a circle fit of the child tag CENTRES
            (_circle_fit): when the two disagree by more than
            max(5 mm, 4 sigma of the circle fit) the joint is flagged
    slope = line fit theta vs pulse       -> rad/us, direction, zero pulse
    terminal range 1/2 = theta (and child pose) at the ROM 1 / ROM 2 marks

Reference pose = the "reference" capture (the pose the robot was scanned
in); its joint positions are URDF zero. The model frame is the root link's
frame at the reference pose; metres throughout.

Quality gates (flag, never silently export): axis ratio s1/s0, pivot
residual, linearity of theta vs pulse, ROM mark vs fitted line (catches an
end stop or windup: commanded kept moving but the joint did not), wiggle vs
ROM axis/direction agreement, and enough angular excitation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .linkage import WiggleResult, canonical_axis, link_pose, relative_pose
from .observations import ObservationLog, inv, mean_pose


@dataclass
class RomParams:
    max_axis_ratio: float = 0.10       # s1/s0 of rotvec SVD (plan: start at 0.1)
    max_pivot_rms_mm: float = 4.0
    min_r2: float = 0.995
    max_mark_error_deg: float = 3.0
    max_wiggle_axis_deg: float = 15.0
    min_range_deg: float = 10.0
    prismatic_max_rot_deg: float = 2.0
    prismatic_min_travel_mm: float = 3.0
    pivot_method: str = "rotation"     # rotation (pose pairs) | circle (tag centres)
    max_pivot_disagreement_mm: float = 5.0  # orientation- and centre-based pivots must agree this well
    pivot_parent_frame: str = "per_frame"  # per_frame | mean (parent is static during one joint's ROM)
    refine_axis: bool = True
    lateral_noise_mm: float = 0.5      # camera noise model for weighting (depth is worse than lateral)
    depth_noise_mm: float = 1.2
    min_circle_arc_deg: float = 20.0


@dataclass
class TerminalRange:
    mark: str                    # "ROM 1" | "ROM 2"
    pulse_us: float
    position: float              # rad (revolute) or m (prismatic) from the reference pose
    position_deg: float | None
    predicted_from_pulse: float  # what the fitted pulse map says at this pulse
    child_pose_in_parent: list   # 4x4, metres
    tag_poses_in_parent: dict    # tag id -> 4x4 for every visible tag of the child link


@dataclass
class JointFit:
    name: str
    channel: int
    parent: str
    child: str
    type: str
    axis_parent: list
    pivot_parent: list
    axis_model: list
    pivot_model: list
    direction: int
    rad_per_us: float            # or m/us for prismatic
    zero_pulse_us: float         # pulse at the reference (URDF zero) pose
    lower: float
    upper: float
    terminal_range_1: TerminalRange
    terminal_range_2: TerminalRange
    quality: dict
    status: str = "ok"           # ok | flagged
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KinematicModel:
    root: str
    link_ref_model: dict          # link -> 4x4 pose at the reference, model frame
    link_tags: dict               # link -> [tag ids]
    joints: list[JointFit]
    channels: list[dict]          # wiggle channel results (incl. no_response)
    flags: list[str]
    link_offsets: dict = field(default_factory=dict)  # link -> {tag id: T_linkframe_tag}

    def joint(self, name: str) -> JointFit:
        return next(j for j in self.joints if j.name == name)

    def to_dict(self) -> dict:
        return {"schema": "dtg.kinematics/1", "frame": "root link frame at the reference (scan) pose, metres",
                "root": self.root, "links": {k: {"tags": self.link_tags[k], "ref_pose_model": v,
                                                 "tag_offsets": self.link_offsets.get(k, {})}
                                             for k, v in self.link_ref_model.items()},
                "joints": [j.to_dict() for j in self.joints], "channels": self.channels, "flags": self.flags}


def _reference_frame(log: ObservationLog, flags: list):
    ref = log.select(phase="reference")
    if ref:
        return ref
    flags.append("no reference capture in the log: using the first wiggle_base frame as joint zero; "
                 "the scan may have been taken in a different pose")
    return log.select(event="wiggle_base")[:1]


def _circle_fit(frames, parent, child, axis0, pivot0, p: RomParams):
    """Axis line from the child's tag CENTRES, which are far better measured
    than tag orientations: every child tag centre, in the parent frame,
    travels a circle about the joint axis. Fit one axis line (plus a radius
    and height per tag) to all of them, weighting each residual by the
    camera's noise in that direction (depth noise is larger than lateral).

    Returns (axis, pivot, info) or None when there is too little to fit."""
    from scipy.optimize import least_squares
    pts, rays, tags = [], [], []
    P_mean = None
    if p.pivot_parent_frame == "mean":
        est = [P for f in frames if (P := link_pose(f, parent)) is not None]
        if not est:
            return None
        P_mean = mean_pose(est)
    for f in frames:
        P = P_mean if P_mean is not None else link_pose(f, parent)
        if P is None:
            continue
        Pi = inv(P)
        for k in child.tags:
            T = f.poses.get(k)
            if T is None:
                continue
            c = np.asarray(T)[:3, 3]
            pts.append(Pi[:3, :3] @ c + Pi[:3, 3])
            rays.append(Pi[:3, :3] @ (c / (np.linalg.norm(c) or 1.0)))
            tags.append(k)
    if len(pts) < 6:
        return None
    pts, rays = np.asarray(pts), np.asarray(rays)
    keys = sorted(set(tags))
    idx = np.array([keys.index(k) for k in tags])
    n = len(keys)
    a0 = np.asarray(axis0, float) / np.linalg.norm(axis0)
    e1 = np.cross(a0, [1.0, 0, 0] if abs(a0[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a0, e1)
    sl, sd = p.lateral_noise_mm, p.depth_noise_mm

    def unpack(x):
        a = a0 + x[0] * e1 + x[1] * e2
        a /= np.linalg.norm(a)
        piv = pivot0 + x[2] * e1 + x[3] * e2
        return a, piv, x[4:4 + n], x[4 + n:]

    def resid(x):
        a, piv, r, h = unpack(x)
        d = pts - piv
        hh = d @ a
        radial = d - np.outer(hh, a)
        rho = np.linalg.norm(radial, axis=1)
        nr = radial / np.maximum(rho, 1e-9)[:, None]
        # noise std along the radial and axial directions: sl^2 + (sd^2 - sl^2) (dir . ray)^2
        sr = np.sqrt(sl ** 2 + (sd ** 2 - sl ** 2) * np.einsum("ij,ij->i", nr, rays) ** 2) / 1000
        sa = np.sqrt(sl ** 2 + (sd ** 2 - sl ** 2) * (rays @ a) ** 2) / 1000
        return np.concatenate([(rho - r[idx]) / sr, (hh - h[idx]) / sa])

    d0 = pts - pivot0
    h0 = d0 @ a0
    rho0 = np.linalg.norm(d0 - np.outer(h0, a0), axis=1)
    x0 = np.concatenate([[0, 0, 0, 0], [rho0[idx == i].mean() for i in range(n)],
                         [h0[idx == i].mean() for i in range(n)]])
    if not p.refine_axis:
        sol = least_squares(lambda x: resid(np.r_[0.0, 0.0, x]), x0[2:], loss="soft_l1", f_scale=3.0)
        x = np.r_[0.0, 0.0, sol.x]
    else:
        sol = least_squares(resid, x0, loss="soft_l1", f_scale=3.0)
        x = sol.x
    a, piv, r, _ = unpack(x)
    # angular coverage of the tag centres about the fitted axis (a short arc pins the centre badly)
    d = pts - piv
    rad = d - np.outer(d @ a, a)
    u = np.cross(a, e1 if abs(a @ e1) < 0.9 else e2)
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    ang = np.arctan2(rad @ w, rad @ u)
    spans = []
    for i in range(n):
        s = np.sort(ang[idx == i])
        if len(s) > 1:
            gaps = np.diff(np.r_[s, s[0] + 2 * np.pi])
            spans.append(float(np.degrees(2 * np.pi - gaps.max())))
    rms = float(np.sqrt(np.mean((np.linalg.norm(rad, axis=1) - r[idx]) ** 2)) * 1000)
    if a @ a0 < 0:
        a = -a
    # standard error of the line's position (Gauss-Newton covariance, scaled by the fit's own residual)
    sigma_mm = float("nan")
    try:
        J = sol.jac
        dof = max(1, J.shape[0] - J.shape[1])
        cov = np.linalg.inv(J.T @ J) * (2 * sol.cost / dof)
        k = 2 if p.refine_axis else 0
        sigma_mm = float(np.sqrt(max(0.0, cov[k, k] + cov[k + 1, k + 1])) * 1000)
    except np.linalg.LinAlgError:
        pass
    return a, piv, {"tag_points": len(pts), "arc_deg": max(spans, default=0.0), "circle_rms_mm": rms,
                    "circle_pivot_sigma_mm": sigma_mm}


def fit_joint(log: ObservationLog, wiggle: WiggleResult, ch, ref_frames, link_ref_model,
              p: RomParams) -> JointFit | None:
    parent, child = wiggle.link(ch.parent), wiggle.link(ch.child)
    reasons = []
    refs = [q for f in ref_frames if (q := relative_pose(f, parent, child)) is not None]
    frames = [f for f in log.frames if f.channel == ch.channel and f.phase in ("rom", "wiggle")]
    if not refs:
        reasons.append("parent/child not both visible in the reference capture; using the wiggle start pose")
        refs = [q for f in frames if f.event == "wiggle_base" and (q := relative_pose(f, parent, child)) is not None]
    if not refs:
        return None
    Q_ref = mean_pose(refs)
    rows = []
    for f in frames:
        Q = relative_pose(f, parent, child)
        if Q is None or f.pulse_us is None:
            continue
        D = Q @ inv(Q_ref)
        rows.append((f, Q, D))
    rom_rows = [r for r in rows if r[0].phase == "rom"]
    if len(rom_rows) < 4:
        return None
    pulses = np.array([r[0].pulse_us for r in rows])
    rotvecs = np.array([Rotation.from_matrix(r[2][:3, :3]).as_rotvec() for r in rows])
    trans = np.array([r[2][:3, 3] for r in rows])
    rot_range = np.degrees(np.ptp(np.linalg.norm(rotvecs, axis=1)))
    travel = np.ptp(np.linalg.norm(trans - trans.mean(axis=0), axis=1)) * 1000
    jtype = "revolute"
    if np.degrees(np.linalg.norm(rotvecs, axis=1).max()) < p.prismatic_max_rot_deg and travel > p.prismatic_min_travel_mm:
        jtype = "prismatic"
    quality = {"samples": len(rows), "rom_samples": len(rom_rows)}
    if jtype == "revolute":
        _, s, vt = np.linalg.svd(rotvecs, full_matrices=False)
        axis = canonical_axis(vt[0])
        theta = rotvecs @ axis
        A = np.concatenate([r[2][:3, :3] - np.eye(3) for r in rows])
        b = np.concatenate([-r[2][:3, 3] for r in rows])
        p0 = np.linalg.lstsq(A, b, rcond=None)[0]
        c0 = Q_ref[:3, 3]
        pivot = p0 + axis * ((c0 - p0) @ axis)
        # Second, independent estimate of the axis line from the tag centres
        # alone: used as the pivot (pivot_method="circle") or as a cross-check.
        circ = _circle_fit(frames, parent, child, axis, pivot, p)
        quality["pivot_method"] = "rotation"
        if circ is not None and circ[2]["arc_deg"] >= p.min_circle_arc_deg:
            c_axis, c_piv = circ[0], circ[1]
            dp = (pivot - c_piv) - axis * ((pivot - c_piv) @ axis)
            quality.update(**circ[2], pivot_disagreement_mm=float(np.linalg.norm(dp) * 1000),
                           axis_disagreement_deg=float(np.degrees(np.arccos(min(1.0, abs(c_axis @ axis))))))
            if p.pivot_method == "circle":
                axis, pivot = c_axis, c_piv + c_axis * ((c0 - c_piv) @ c_axis)
                theta = rotvecs @ axis
                quality["pivot_method"] = "circle"
            sig = quality.get("circle_pivot_sigma_mm")
            limit = max(p.max_pivot_disagreement_mm, 4.0 * sig if sig == sig else 0.0)
            if quality["pivot_disagreement_mm"] > limit:
                reasons.append(f"pivot from tag orientations and from tag centres differ by "
                               f"{quality['pivot_disagreement_mm']:.1f} mm (allowed {limit:.1f}): pivot uncertain")
        resid = np.array([np.linalg.norm(r[2][:3, :3] @ pivot + r[2][:3, 3] - pivot) for r in rows])
        quality.update(axis_ratio=float(s[1] / s[0]) if s[0] > 0 else 1.0,
                       pivot_rms_mm=float(np.sqrt(np.mean(resid ** 2)) * 1000),
                       range_deg=float(np.degrees(np.ptp(theta))))
        if quality["axis_ratio"] > p.max_axis_ratio:
            reasons.append(f"axis not a clean hinge (ratio {quality['axis_ratio']:.3f} > {p.max_axis_ratio}): "
                           "backlash, loose tag, coupled joint or bad detections")
        if quality["pivot_rms_mm"] > p.max_pivot_rms_mm:
            reasons.append(f"child does not turn about one fixed point ({quality['pivot_rms_mm']:.1f} mm rms)")
        if quality["range_deg"] < p.min_range_deg:
            reasons.append(f"only {quality['range_deg']:.1f} deg of motion seen: axis poorly determined")
    else:
        u = trans - trans.mean(axis=0)
        _, s, vt = np.linalg.svd(u, full_matrices=False)
        axis = canonical_axis(vt[0])
        theta = trans @ axis
        pivot = Q_ref[:3, 3].copy()
        quality.update(axis_ratio=float(s[1] / s[0]) if s[0] > 0 else 1.0, range_mm=float(np.ptp(theta) * 1000))
    slope, intercept = np.polyfit(pulses, theta, 1)
    pred = slope * pulses + intercept
    ss_res = float(((theta - pred) ** 2).sum())
    ss_tot = float(((theta - theta.mean()) ** 2).sum()) or 1e-12
    quality.update(r2=1 - ss_res / ss_tot, fit_rms=float(np.sqrt(ss_res / len(theta))))
    if quality["r2"] < p.min_r2:
        reasons.append(f"position is not linear in pulse (R^2 {quality['r2']:.4f}): slipping horn, backlash or end stop")
    direction = 1 if slope > 0 else -1
    zero_pulse = float(-intercept / slope) if slope != 0 else float("nan")

    def terminal(event, label):
        hits = [(r, th) for r, th in zip(rows, theta) if r[0].event == event]
        if not hits:
            # Tags hidden at the instant of the mark: use settled captures at the
            # same commanded pulse (the slider sample taken just before the mark).
            marks = [f for f in frames if f.event == event and f.pulse_us is not None]
            if marks:
                pulse = marks[-1].pulse_us
                hits = [(r, th) for r, th in zip(rows, theta)
                        if r[0].phase == "rom" and abs(r[0].pulse_us - pulse) < 0.5]
                if hits:
                    reasons.append(f"{label}: tags hidden at the mark; used {len(hits)} capture(s) at the same pulse")
        if not hits:
            return None
        pos = float(np.mean([th for _, th in hits]))
        pulse = float(np.mean([r[0].pulse_us for r, _ in hits]))
        tag_poses = {}
        for r, _ in hits[:1]:
            P = link_pose(r[0], parent)
            for k in child.tags:
                if k in r[0].poses:
                    tag_poses[str(k)] = (inv(P) @ r[0].poses[k]).round(9).tolist()
        return TerminalRange(mark=label, pulse_us=pulse, position=pos,
                             position_deg=float(np.degrees(pos)) if jtype == "revolute" else None,
                             predicted_from_pulse=float(slope * pulse + intercept),
                             child_pose_in_parent=mean_pose([r[1] for r, _ in hits]).round(9).tolist(),
                             tag_poses_in_parent=tag_poses)
    def extrapolated(event, label):
        # Tags not visible at the mark (or at that pulse at all): a tag that faces
        # the camera at rest turns away or hides behind another link near the
        # ends of a large range. Use the pulse fit, flagged, rather than drop the joint.
        marks = [f for f in frames if f.event == event and f.pulse_us is not None]
        if not marks or jtype != "revolute":
            return None
        pulse = float(marks[-1].pulse_us)
        pos = float(slope * pulse + intercept)
        span = (pulses.min(), pulses.max())
        reasons.append(f"{label}: tags not visible at {pulse:.0f} us; position extrapolated from the pulse fit "
                       f"(seen {span[0]:.0f}..{span[1]:.0f} us)")
        G = np.eye(4)
        G[:3, :3] = Rotation.from_rotvec(axis * pos).as_matrix()
        G[:3, 3] = pivot - G[:3, :3] @ pivot
        return TerminalRange(mark=label, pulse_us=pulse, position=pos, position_deg=float(np.degrees(pos)),
                             predicted_from_pulse=pos, child_pose_in_parent=(G @ Q_ref).round(9).tolist(),
                             tag_poses_in_parent={})
    t1 = terminal("rom1", "ROM 1") or extrapolated("rom1", "ROM 1")
    t2 = terminal("rom2", "ROM 2") or extrapolated("rom2", "ROM 2")
    if t1 is None or t2 is None:
        reasons.append("ROM 1 and/or ROM 2 was not marked (or the tags were hidden at the mark)")
        return None
    for t in (t1, t2):
        err = abs(t.position - t.predicted_from_pulse)
        err_deg = np.degrees(err) if jtype == "revolute" else err * 1000
        if err_deg > p.max_mark_error_deg:
            reasons.append(f"{t.mark}: measured position lags the commanded pulse by {err_deg:.1f}"
                           f"{' deg' if jtype == 'revolute' else ' mm'}: end stop, stall or windup")
    if ch.rough_axis_parent is not None and jtype == "revolute":
        agree = np.degrees(np.arccos(np.clip(abs(np.dot(ch.rough_axis_parent, axis)), -1, 1)))
        quality["wiggle_axis_agreement_deg"] = float(agree)
        same_sign = np.dot(ch.rough_axis_parent, axis) > 0
        wiggle_dir = ch.direction if same_sign else -ch.direction
        quality["wiggle_direction_agrees"] = bool(wiggle_dir == direction)
        if agree > p.max_wiggle_axis_deg:
            reasons.append(f"wiggle and ROM axes differ by {agree:.0f} deg")
        if wiggle_dir != direction:
            reasons.append("wiggle and ROM disagree on direction")
    P_ref = np.asarray(link_ref_model[parent.name])
    axis_model = P_ref[:3, :3] @ axis
    pivot_model = (P_ref @ np.r_[pivot, 1.0])[:3]
    return JointFit(name=ch.joint, channel=ch.channel, parent=parent.name, child=child.name, type=jtype,
                    axis_parent=axis.tolist(), pivot_parent=pivot.tolist(), axis_model=axis_model.tolist(),
                    pivot_model=pivot_model.tolist(), direction=direction, rad_per_us=float(abs(slope)),
                    zero_pulse_us=zero_pulse, lower=min(t1.position, t2.position), upper=max(t1.position, t2.position),
                    terminal_range_1=t1, terminal_range_2=t2, quality=quality,
                    status="flagged" if reasons or ch.status != "mapped" else "ok",
                    reasons=([f"wiggle: {r}" for r in ch.reasons] if ch.status != "mapped" else []) + reasons)


def infer_kinematics(log: ObservationLog, wiggle: WiggleResult, params: RomParams | None = None) -> KinematicModel:
    p = params or RomParams()
    flags = list(wiggle.flags)
    ref_frames = _reference_frame(log, flags)
    root = wiggle.link(next(l.name for l in wiggle.links if not l.signature))
    link_ref_model = {}
    for l in wiggle.links:
        est = []
        for f in ref_frames:
            R, L = link_pose(f, root), link_pose(f, l)
            if R is not None and L is not None:
                est.append(inv(R) @ L)
        if est:
            link_ref_model[l.name] = mean_pose(est).tolist()
        else:
            flags.append(f"link {l.name} not visible in the reference capture")
    joints = []
    for ch in wiggle.channels:
        if not ch.joint:
            continue
        if ch.parent not in link_ref_model:
            flags.append(f"{ch.joint}: parent {ch.parent} has no reference pose; joint skipped")
            continue
        fit = fit_joint(log, wiggle, ch, ref_frames, link_ref_model, p)
        if fit is None:
            flags.append(f"{ch.joint} (channel {ch.channel}): not enough ROM data (sweep + ROM 1 + ROM 2 needed)")
            continue
        joints.append(fit)
    return KinematicModel(root=root.name, link_ref_model=link_ref_model,
                          link_tags={l.name: l.tags for l in wiggle.links}, joints=joints,
                          channels=[c.to_dict() for c in wiggle.channels], flags=flags,
                          link_offsets={l.name: l.offsets for l in wiggle.links})
