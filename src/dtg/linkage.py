"""WIGGLE step: which controller channel drives which joint.

Wiring is unknown. Each channel gets a tiny pulse change (wiggle_plus /
wiggle_minus around wiggle_base, one channel at a time) while the camera
tracks every tag. From the log:

1. motion of every tag relative to the root (base) tag, per channel
   (relative poses cancel camera shake and base wobble);
2. a tag's signature = the set of channels that move it;
3. tags with the same signature and constant relative pose = one rigid link;
4. channel c's child link = the link with the smallest signature containing
   c; its parent = the link whose signature is that minus {c};
5. direction: sign of the rotation about the (canonicalised) rough axis when
   the pulse goes up.

A tag counts as moved only if it moves on both sides of the wiggle, in
opposite directions and by similar amounts; any other large move is flagged. Never guesses: channels where nothing moves are
"no_response"; two channels
claiming one child, a missing intermediate tag, occluded tags or a
non-rigid tag group are flagged "needs_confirmation" with a reason.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .markers import Markers
from .observations import ObservationLog, inv, mean_pose, rot_angle


@dataclass
class WiggleParams:
    min_rot_deg: float = 1.5        # a tag "moved" if it turned more than this ...
    min_trans_mm: float = 3.0       # ... or shifted more than this (relative to root)
    noise_factor: float = 3.0       # ... and more than noise_factor x measured jitter
    rigid_rot_deg: float = 3.0      # tags of one link must hold relative pose within this
    rigid_trans_mm: float = 4.0
    prismatic_rot_deg: float = 1.0  # less rotation than this but clear translation = prismatic


@dataclass
class Link:
    name: str
    tags: list[int]
    signature: list[int]           # channels that move this link
    primary_tag: int
    offsets: dict[int, list] = field(default_factory=dict)  # T_primary_tag per tag
    unknown_channels: list[int] = field(default_factory=list)  # wiggles where this link was hidden and
                                                               # could not be placed (signature incomplete)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChannelResult:
    channel: int
    status: str                    # mapped | no_response | needs_confirmation
    joint: str | None = None
    parent: str | None = None
    child: str | None = None
    joint_type: str | None = None  # revolute | prismatic
    direction: int | None = None   # +1: pulse up turns +axis; -1 the other way
    rough_axis_parent: list | None = None  # unit axis in the parent link frame
    moved_tags: list[int] = field(default_factory=list)
    occluded_tags: list[int] = field(default_factory=list)
    rotation_deg: float | None = None
    translation_mm: float | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WiggleResult:
    root_tag: int
    links: list[Link]
    channels: list[ChannelResult]
    noise: dict
    flags: list[str]

    def link(self, name: str) -> Link:
        return next(l for l in self.links if l.name == name)

    @property
    def joints(self) -> list[ChannelResult]:
        return [c for c in self.channels if c.joint]

    def to_dict(self) -> dict:
        return {"schema": "dtg.wiggle/1", "root_tag": self.root_tag, "noise": self.noise, "flags": self.flags,
                "links": [l.to_dict() for l in self.links], "channels": [c.to_dict() for c in self.channels]}


def canonical_axis(axis: np.ndarray) -> np.ndarray:
    """Sign convention: largest-magnitude component positive. Mirrored joints
    then share an axis and differ only in the direction flag."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return -axis if axis[np.argmax(np.abs(axis))] < 0 else axis


def choose_root(log: ObservationLog, markers: Markers | None) -> int:
    seen = log.tag_ids
    if markers and markers.root_tag is not None and markers.root_tag in seen:
        return markers.root_tag
    # fallback: the tag whose camera-frame position varies least over the log
    spread = {}
    for k in seen:
        pts = np.array([f.poses[k][:3, 3] for f in log.frames if k in f.poses])
        spread[k] = float(np.linalg.norm(pts.std(axis=0))) if len(pts) > 1 else np.inf
    return min(spread, key=spread.get)


def _rel(frame, root, tag, anchors=None):
    """Pose of ``tag`` relative to the root tag. If the root tag itself is
    hidden in this capture, a tag proven rigid with it (``anchors``: tag ->
    T_root_tag) stands in for it."""
    if tag not in frame.poses:
        return None
    if root in frame.poses:
        return inv(frame.poses[root]) @ frame.poses[tag]
    est = [frame.poses[k] @ inv(T) for k, T in (anchors or {}).items() if k in frame.poses and k != tag]
    return inv(mean_pose(est)) @ frame.poses[tag] if est else None


def root_anchors(wiggle, root, tags, channels, p) -> dict:
    """Tags that provably sit on the root link: seen together with the root
    tag in at least one moving capture (plus or minus) of EVERY channel, and
    never moving relative to it. Only then could any motion have shown."""
    anchors = {}
    for k in tags:
        rel, covered = [], set()
        for f in wiggle:
            if root in f.poses and k in f.poses:
                rel.append(inv(f.poses[root]) @ f.poses[k])
                if f.event in ("wiggle_plus", "wiggle_minus"):
                    covered.add(f.channel)
        if covered != set(channels) or len(rel) < 3:
            continue
        M = mean_pose(rel)
        dev = np.array([_delta(M, r) for r in rel])
        rot_lim = max(p.rigid_rot_deg, 5 * float(np.median(dev[:, 0])))
        trans_lim = max(p.rigid_trans_mm, 5 * float(np.median(dev[:, 1])))
        if dev[:, 0].max() <= rot_lim and dev[:, 1].max() <= trans_lim:
            anchors[k] = M
    return anchors


def _delta(A, B):
    """rotation angle (deg) and translation (mm) between two relative poses."""
    D = inv(A) @ B
    return np.degrees(rot_angle(D[:3, :3])), float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000)


def infer_linkage(log: ObservationLog, markers: Markers | None = None,
                  params: WiggleParams | None = None) -> WiggleResult:
    p = params or WiggleParams()
    root = choose_root(log, markers)
    names = (lambda k: markers.name_for(k)) if markers else (lambda k: f"tag{k}")
    tags = [k for k in log.tag_ids if k != root]
    flags: list[str] = []
    wiggle = log.select(phase="wiggle")
    if not wiggle:
        raise ValueError("log has no wiggle frames: run the wiggle step before ROM")
    channels = sorted({f.channel for f in wiggle})
    anchors = root_anchors(wiggle, root, tags, channels, p)

    # jitter: spread of each tag's relative pose over all base captures where
    # its own channel was not being moved is a fair noise estimate
    base_rel: dict[int, list] = {k: [] for k in tags}
    for f in wiggle:
        if f.event == "wiggle_base":
            for k in tags:
                r = _rel(f, root, k, anchors)
                if r is not None:
                    base_rel[k].append((f.channel, r))
    rot_noise, trans_noise = [], []
    for c in channels:
        for k in tags:
            rs = [r for ch, r in base_rel[k] if ch == c]
            for a, b in zip(rs, rs[1:]):
                dr, dt = _delta(a, b)
                rot_noise.append(dr)
                trans_noise.append(dt)
    noise = {"rot_deg_median": float(np.median(rot_noise)) if rot_noise else 0.0,
             "trans_mm_median": float(np.median(trans_noise)) if trans_noise else 0.0}
    rot_thr = max(p.min_rot_deg, p.noise_factor * noise["rot_deg_median"])
    trans_thr = max(p.min_trans_mm, p.noise_factor * noise["trans_mm_median"])
    noise.update(rot_threshold_deg=rot_thr, trans_threshold_mm=trans_thr)

    # per channel: which tags moved
    moved: dict[int, set] = {}
    occluded: dict[int, set] = {}
    one_sided: dict[int, set] = {}
    motion: dict[tuple, tuple] = {}
    for c in channels:
        fr = [f for f in wiggle if f.channel == c]
        moved[c], occluded[c], one_sided[c] = set(), set(), set()
        for k in tags:
            base = [_rel(f, root, k, anchors) for f in fr if f.event == "wiggle_base"]
            base = [b for b in base if b is not None]
            ends = {e: [_rel(f, root, k, anchors) for f in fr if f.event == e] for e in ("wiggle_plus", "wiggle_minus")}
            ends = {e: [x for x in v if x is not None] for e, v in ends.items()}
            if not base or not ends["wiggle_plus"] or not ends["wiggle_minus"]:
                occluded[c].add(k)
                continue
            B = mean_pose(base)
            ends_mean = [mean_pose(v) for v in ends.values()]
            sides = [_delta(B, E) for E in ends_mean]
            dr = max(r for r, _ in sides)
            dt = max(t for _, t in sides)
            motion[(c, k)] = (dr, dt)
            # A joint wiggle moves its child on BOTH sides of the base pulse, in
            # opposite directions and by similar amounts. Anything else is noise,
            # a glint or a partial occlusion (ignored when small), or a hard stop
            # or bad detection (flagged when large).
            score = [max(r / rot_thr, t / trans_thr) for r, t in sides]
            rot_led = sum(r / rot_thr for r, _ in sides) >= sum(t / trans_thr for _, t in sides)
            if rot_led:
                vec = [Rotation.from_matrix((E @ inv(B))[:3, :3]).as_rotvec() for E in ends_mean]
            else:
                vec = [E[:3, 3] - B[:3, 3] for E in ends_mean]
            opposite = float(np.dot(vec[0], vec[1])) < 0
            balanced = max(score) < 3 * min(score)
            if min(score) > 1 and opposite and balanced:
                moved[c].add(k)
            elif max(score) > 2:
                one_sided[c].add(k)
        if root not in set().union(*[set(f.poses) for f in fr]):
            flags.append(f"channel {c}: root tag {root} never seen during its wiggle")

    # signatures -> links. A tag hidden during some wiggles has those channels
    # unknown; it joins the fully-observed group it agrees with on every known
    # channel (and moves rigidly with), else it becomes its own link, flagged.
    known = {k: {c for c in channels if k not in occluded[c]} for k in tags}
    signature = {k: frozenset(c for c in channels if k in moved[c]) for k in tags}
    groups: dict[frozenset, list[int]] = {frozenset(): [root]}
    partial = []
    for k in tags:
        if known[k] == set(channels):
            groups.setdefault(signature[k], []).append(k)
        else:
            partial.append(k)
    rigid_r = max(p.rigid_rot_deg, 3 * noise["rot_deg_median"])
    rigid_t = max(p.rigid_trans_mm, 3 * noise["trans_mm_median"])

    def rigid_with(k, members):
        devs = [_rigidity(log, m, k) for m in members]
        devs = [d for d in devs if d is not None]
        return bool(devs) and all(r <= rigid_r and t <= rigid_t for r, t in devs)

    unknown: dict[frozenset, set] = {}       # placeholder group -> channels its motion is unknown for
    for k in partial:
        fits = [sig for sig in groups if {c for c in sig if c in known[k]} == set(signature[k])
                and rigid_with(k, groups[sig])]
        if len(fits) == 1:
            groups[fits[0]].append(k)
            if fits[0] in unknown:
                unknown[fits[0]] |= set(channels) - known[k]
            flags.append(f"tag {k} was hidden during channel(s) {sorted(set(channels) - known[k])}; "
                         f"grouped by its other channels")
        else:
            key = signature[k] | {-1 - k}       # unique placeholder signature
            groups.setdefault(key, []).append(k)
            unknown[key] = set(channels) - known[k]
            flags.append(f"tag {k} was hidden during channel(s) {sorted(set(channels) - known[k])} and "
                         "cannot be placed on a link: re-run those wiggles with it visible")
    links: list[Link] = []
    seen_counts = {k: sum(k in f.poses for f in log.frames) for k in log.tag_ids}
    for sig, members in sorted(groups.items(), key=lambda kv: (len(kv[0]), sorted(kv[0]))):
        members = sorted(members, key=lambda k: (k != root, -seen_counts[k], k))
        primary = members[0]
        name = names(root) if not sig else names(min(members))
        offsets = {}
        for k in members:
            rel = [inv(f.poses[primary]) @ f.poses[k] for f in log.frames if primary in f.poses and k in f.poses]
            if not rel:
                flags.append(f"tag {k} never seen together with tag {primary}; cannot join it to link {name}")
                continue
            M = mean_pose(rel)
            dev = np.array([_delta(M, r) for r in rel])
            r90, t90 = np.percentile(dev[:, 0], 90), np.percentile(dev[:, 1], 90)
            if r90 > rigid_r or t90 > rigid_t:
                flags.append(f"tags {primary} and {k} share a signature but do not move rigidly together "
                             f"(90th pct {r90:.1f} deg / {t90:.1f} mm): coupled joint or loose tag? confirm")
            offsets[k] = M.tolist()
        links.append(Link(name=_unique(name, links), tags=members, signature=sorted(c for c in sig if c >= 0),
                          primary_tag=primary, offsets=offsets, unknown_channels=sorted(unknown.get(sig, ()))))
    # Only fully placed links can be looked up by signature (an unplaced link's
    # visible signature is incomplete and could collide with, e.g., the base).
    by_sig = {frozenset(l.signature): l for l in links if not l.unknown_channels}
    unplaced = [l for l in links if l.unknown_channels]
    for c in channels:  # occlusion only matters when a whole link was hidden
        hidden_links = [l.name for l in links if l.tags and all(k in occluded[c] for k in l.tags if k != root)
                        and any(k != root for k in l.tags)]
        occluded[c] = {k for l in links if l.name in hidden_links for k in l.tags}

    results: list[ChannelResult] = []
    child_claims: dict[str, list[int]] = {}
    for c in channels:
        res = ChannelResult(channel=c, status="no_response", moved_tags=sorted(moved[c]),
                            occluded_tags=sorted(occluded[c]))
        if res.occluded_tags:
            res.reasons.append(f"every tag of a link was hidden during this wiggle: {res.occluded_tags}")
        if one_sided[c]:
            res.reasons.append(f"tags {sorted(one_sided[c])} moved inconsistently (one side only, or not back and "
                               "forth): end stop, slipping horn, partial occlusion or a bad detection; check by eye")
        if not moved[c]:
            if res.occluded_tags or one_sided[c]:
                res.status = "needs_confirmation"
                res.reasons.append("nothing moved on both sides, but some tags were hidden or moved one way: "
                                   "re-run with a clear view")
            else:
                res.reasons.append("no tag moved: channel unused, unpowered or driving an untagged part")
            results.append(res)
            continue
        candidates = [l for l in links if c in l.signature]
        smallest = min(len(l.signature) for l in candidates)
        child = [l for l in candidates if len(l.signature) == smallest]
        if len(child) != 1:
            res.status = "needs_confirmation"
            res.reasons.append(f"several links could be the child: {[l.name for l in child]}")
            results.append(res)
            continue
        child = child[0]
        doubt = [l.name for l in unplaced if l is child or c in l.unknown_channels
                 or set(l.signature) <= set(child.signature) - {c}]
        if doubt:
            res.reasons.append(f"links {doubt} were hidden during some wiggles and could be this joint's child "
                               "or parent: re-run those wiggles with them visible")
        parent = by_sig.get(frozenset(child.signature) - {c})
        res.status = "mapped" if not one_sided[c] and not doubt else "needs_confirmation"
        if parent is None:
            subset = [l for l in links if set(l.signature) < set(child.signature) and c not in l.signature]
            parent = max(subset, key=lambda l: len(l.signature)) if subset else by_sig[frozenset()]
            res.status = "needs_confirmation"
            res.reasons.append(f"no tagged link moves exactly with channels {sorted(set(child.signature) - {c})}: "
                               f"a link between {parent.name} and {child.name} has no tag; parent is a guess")
        res.parent, res.child = parent.name, child.name
        res.joint = f"j_{child.name}"
        child_claims.setdefault(child.name, []).append(c)
        # rough axis + direction from the child's motion relative to the parent
        axis, rot_deg, trans_mm, direction, jtype = _rough_axis(log, c, parent, child, p)
        res.rough_axis_parent = None if axis is None else axis.tolist()
        res.rotation_deg, res.translation_mm = rot_deg, trans_mm
        res.direction, res.joint_type = direction, jtype
        if axis is None:
            res.status = "needs_confirmation"
            res.reasons.append("parent or child tags hidden during the wiggle: no direction")
        results.append(res)
    for child, chans in child_claims.items():
        if len(chans) > 1:
            for r in results:
                if r.channel in chans:
                    r.status = "needs_confirmation"
                    r.reasons.append(f"channels {chans} all move {child} first: coupled servos, "
                                     "or a missing tag between them")
    return WiggleResult(root_tag=root, links=links, channels=results, noise=noise, flags=flags)


def _rigidity(log, a, b):
    """Worst deviation (deg, mm) of tag b's pose relative to tag a over every
    frame where both are visible; None if never together.

    The worst frame, not a percentile: a tag hidden during some wiggles is
    placed on a link by this test, and the one or two frames where it was
    seen moving relative to that link are exactly the evidence that matters
    (a 90th percentile let a moving tag join the static base)."""
    rel = [inv(f.poses[a]) @ f.poses[b] for f in log.frames if a in f.poses and b in f.poses]
    if len(rel) < 3:
        return None
    M = mean_pose(rel)
    dev = np.array([_delta(M, r) for r in rel])
    return float(dev[:, 0].max()), float(dev[:, 1].max())


def _unique(name, links):
    existing = {l.name for l in links}
    out, n = name, 2
    while out in existing:
        out, n = f"{name}_{n}", n + 1
    return out


def link_pose(frame, link: Link) -> np.ndarray | None:
    """Pose of a link's frame (= its primary tag) in camera coordinates from
    whichever of its tags are visible, averaged."""
    est = []
    for k, off in link.offsets.items():
        if k in frame.poses:
            est.append(frame.poses[k] @ inv(np.asarray(off)))
    return mean_pose(est) if est else None


def relative_pose(frame, parent: Link, child: Link) -> np.ndarray | None:
    P, C = link_pose(frame, parent), link_pose(frame, child)
    return None if P is None or C is None else inv(P) @ C


def _rough_axis(log, c, parent, child, p: WiggleParams):
    fr = log.select(phase="wiggle", channel=c)
    rel = {e: [r for f in fr if f.event == e and (r := relative_pose(f, parent, child)) is not None]
           for e in ("wiggle_base", "wiggle_plus", "wiggle_minus")}
    if not all(rel.values()):
        return None, None, None, None, None
    B = mean_pose(rel["wiggle_base"])
    Dp = mean_pose(rel["wiggle_plus"]) @ inv(B)
    Dm = mean_pose(rel["wiggle_minus"]) @ inv(B)
    rv = Rotation.from_matrix(Dp[:3, :3]).as_rotvec() - Rotation.from_matrix(Dm[:3, :3]).as_rotvec()
    tv = (mean_pose(rel["wiggle_plus"])[:3, 3] - mean_pose(rel["wiggle_minus"])[:3, 3])
    rot_deg = float(np.degrees(np.linalg.norm(rv)) / 2)
    trans_mm = float(np.linalg.norm(tv) * 1000 / 2)
    if rot_deg < p.prismatic_rot_deg and trans_mm > p.min_trans_mm:
        axis = canonical_axis(tv)
        return axis, rot_deg, trans_mm, int(np.sign(tv @ axis)), "prismatic"
    axis = canonical_axis(rv)
    return axis, rot_deg, trans_mm, int(np.sign(rv @ axis)) or 1, "revolute"
