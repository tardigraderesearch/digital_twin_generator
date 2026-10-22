"""Scan health check: find the usual 3D-scan problems on a robot scan and
offer non-destructive fixes.

Detectors (all heuristics; every finding is a suggestion the operator reviews
in the preview before any fix is applied):

* support planes  - turntable / floor / bench under the robot, plus anything
                    below it and low features (turntable feet, star ridges)
                    outside the robot's own footprint.
* wires           - thin (< wire_max_mm), locally tube-like geometry forming
                    a run longer than wire_min_len_mm: servo leads captured as
                    loops. Thin straight rails can also match; check preview.
* holes           - boundary loops. Small ones are fillable; large open
                    regions (unscanned bottoms) are reported only.
* fragments       - disconnected islands that are small next to the robot.
* low confidence  - rough / noisy patches (high local surface variation):
                    "mush" where a servo or joint was poorly captured.

Lengths are physical millimetres; they are converted to scan units with the
scan's units guess (Scan.mm_per_unit). Nothing here mutates the Scan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from .scan import Scan


@dataclass
class HealthParams:
    plane_tol_mm: float = 1.5
    plane_min_area_frac: float = 0.08
    plane_one_sided_frac: float = 0.85
    plane_max_count: int = 2
    plane_min_footprint_ratio: float = 1.5  # a support is wider than the robot standing on it
    low_feature_mm: float = 10.0       # features lower than this outside the footprint are support
    footprint_band_mm: float = 10.0    # slab above low_feature_mm that defines the robot footprint
    footprint_margin_mm: float = 2.0
    wire_max_mm: float = 4.5           # max wire/tube diameter
    wire_min_len_mm: float = 30.0
    hole_fill_max_mm: float = 40.0     # perimeter of holes that may be auto-filled
    fragment_max_frac: float = 0.02    # islands smaller than this share of area
    blob_variation: float = 0.12       # local surface variation threshold
    blob_min_area_mm2: float = 40.0
    sample_count: int = 80000
    seed: int = 0


@dataclass
class Finding:
    kind: str            # support_plane | wire | hole | fragment | low_confidence
    label: str
    severity: str        # info | warn | problem
    message: str         # plain-language explanation
    fix: str | None      # name of the fix toggle that addresses it
    faces: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)
    points: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)
    loop: np.ndarray | None = field(default=None, repr=False)  # ordered vertex loop for holes
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("faces", "points", "loop")}
        d["face_count"] = int(len(self.faces))
        d["point_count"] = int(len(self.points))
        return d


@dataclass
class HealthReport:
    scan: dict
    findings: list[Finding]
    params: HealthParams
    skipped: list[str] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def checklist(self) -> list[str]:
        """Plain-language checklist, one line per problem class."""
        lines = []
        def line(ok: bool, text: str) -> None:
            lines.append(("[ok]   " if ok else "[fix]  ") + text)
        planes = [f for f in self.by_kind("support_plane") if f.fix]
        line(not planes, f"Turntable / floor: {len(planes)} support plane(s) welded to the scan"
             + (" (toggle 'Remove planes')" if planes else ""))
        wires = self.by_kind("wire")
        total = sum(f.metrics.get("length_mm", 0) for f in wires)
        line(not wires, f"Wires: {len(wires)} thin tube-like run(s), about {total:.0f} mm in total"
             + (" (toggle 'Strip wires'; check the preview first)" if wires else ""))
        holes = self.by_kind("hole")
        small = [h for h in holes if h.fix]
        line(not small, f"Holes: {len(holes)} boundary loop(s); {len(small)} small enough to fill"
             + (" (toggle 'Fill small holes')" if small else ""))
        big = [h for h in holes if not h.fix]
        if big:
            lines.append(f"[info] {len(big)} large open region(s) (e.g. unscanned underside) left as-is")
        frags = self.by_kind("fragment")
        line(not frags, f"Fragments: {len(frags)} small disconnected island(s)"
             + (" (toggle 'Drop fragments')" if frags else ""))
        blobs = self.by_kind("low_confidence")
        line(not blobs, f"Low-confidence blobs: {len(blobs)} rough/noisy patch(es); servo bodies there may need a library part")
        for s in self.skipped:
            lines.append(f"[skip] {s}")
        return lines

    def to_dict(self) -> dict:
        return {"schema": "dtg.health-report/1", "scan": self.scan, "params": asdict(self.params),
                "checklist": self.checklist(), "skipped": self.skipped,
                "findings": [f.to_dict() for f in self.findings]}


# --------------------------------------------------------------------------- helpers

def _elements(scan: Scan):
    """Per-element centre, unit normal and weight (area for faces, 1 for points)."""
    if scan.is_mesh:
        tri = scan.vertices[scan.faces]
        centres = tri.mean(axis=1)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(cross, axis=1)
        normals = cross / np.maximum(norm, 1e-15)[:, None]
        return centres, normals, norm / 2.0
    pts = scan.vertices
    normals = _pca_normals(pts)
    spacing = _median_spacing(pts)
    return pts, normals, np.full(len(pts), spacing ** 2)


def _median_spacing(pts: np.ndarray) -> float:
    sample = pts[:: max(1, len(pts) // 5000)]
    d, _ = cKDTree(pts).query(sample, k=2)
    return float(np.median(d[:, 1])) or 1.0


def _pca_normals(pts: np.ndarray, k: int = 16) -> np.ndarray:
    _, idx = cKDTree(pts).query(pts, k=min(k, len(pts)))
    nb = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


def _local_pca(points: np.ndarray, idx: np.ndarray, valid: np.ndarray):
    """Eigenvalues (ascending) of neighbourhoods given padded index arrays."""
    nb = points[np.where(valid, idx, 0)]
    w = valid[..., None].astype(float)
    cnt = np.maximum(w.sum(axis=1), 1.0)
    mean = (nb * w).sum(axis=1) / cnt
    centred = (nb - mean[:, None, :]) * w
    cov = np.einsum("nki,nkj->nij", centred, centred) / cnt[:, None]
    vals, vecs = np.linalg.eigh(cov)
    return np.maximum(vals, 0.0), vecs


def _clusters(points: np.ndarray, radius: float) -> np.ndarray:
    """Connected components of a radius graph. Returns labels."""
    if not len(points):
        return np.zeros(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
                       shape=(len(points), len(points)))
    return connected_components(graph, directed=False)[1]


def _sample(weights: np.ndarray, count: int, seed: int) -> np.ndarray:
    nonzero = np.flatnonzero(weights > 0)
    if len(nonzero) <= count:
        return nonzero
    rng = np.random.default_rng(seed)
    p = weights / weights.sum()
    return np.unique(rng.choice(len(weights), size=count, replace=False, p=p))


def _ids(scan: Scan, mask: np.ndarray) -> dict:
    ids = np.flatnonzero(mask)
    return {"faces": ids} if scan.is_mesh else {"points": ids}


# --------------------------------------------------------------------------- detectors

def detect_support_planes(scan: Scan, centres, normals, weights, p: HealthParams) -> list[Finding]:
    tol = scan.mm(p.plane_tol_mm)
    total = weights.sum()
    available = np.ones(len(centres), dtype=bool)
    rng = np.random.default_rng(p.seed)
    findings = []
    for number in range(p.plane_max_count):
        cand = np.flatnonzero(available)
        if len(cand) < 50:
            break
        prob = weights[cand] / weights[cand].sum()
        best = None
        for _ in range(300):
            i = cand[rng.choice(len(cand), p=prob)]
            n = normals[i]
            dist = (centres[cand] - centres[i]) @ n
            inl = (np.abs(dist) < tol) & (np.abs(normals[cand] @ n) > 0.9)
            score = weights[cand][inl].sum()
            if best is None or score > best[0]:
                best = (score, i, n)
        score, i, n = best
        if score < p.plane_min_area_frac * total:
            break
        # refine with a least-squares plane through the inliers
        dist = (centres - centres[i]) @ n
        inl = available & (np.abs(dist) < tol) & (np.abs(normals @ n) > 0.9)
        pts = centres[inl]
        centre = pts.mean(axis=0)
        normal = np.linalg.svd(pts - centre, full_matrices=False)[2][2]
        h = (centres - centre) @ normal
        # robot side = the side holding most of the remaining area
        above = weights[available & (h > tol)].sum()
        below = weights[available & (h < -tol)].sum()
        if below > above:
            normal, h, above, below = -normal, -h, below, above
        one_sided = above / max(above + below, 1e-12)
        if one_sided < p.plane_one_sided_frac:
            available &= ~inl  # a wall of the robot itself, not a support
            continue
        on_plane = available & (np.abs(h) <= tol)
        under = available & (h < -tol)
        low = available & (h > tol) & (h < scan.mm(p.low_feature_mm))
        # robot footprint: what stands in a slab just above the low features
        e1 = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        uv = np.c_[(centres - centre) @ e1, (centres - centre) @ e2]
        band = available & (h >= scan.mm(p.low_feature_mm)) & (h < scan.mm(p.low_feature_mm + p.footprint_band_mm))
        low_outside = low
        footprint_area = 0.0
        if band.sum() >= 3:
            try:
                hull = ConvexHull(uv[band])
                verts = uv[band][hull.vertices]
                c0 = verts.mean(axis=0)
                grow = verts + (verts - c0) / np.maximum(np.linalg.norm(verts - c0, axis=1, keepdims=True), 1e-9) * scan.mm(p.footprint_margin_mm)
                low_outside = low & (Delaunay(grow).find_simplex(uv) < 0)
                footprint_area = float(hull.volume)
            except Exception:  # noqa: BLE001 - degenerate footprint: treat all low features as support
                low_outside = low
        plane_area = float(weights[on_plane].sum())
        if footprint_area > 0 and plane_area < p.plane_min_footprint_ratio * footprint_area:
            # flat underside of the robot itself (e.g. a base plate), not a turntable or floor
            findings.append(Finding(
                kind="support_plane", label="flat underside (kept)", severity="info", fix=None,
                message="A flat face under the robot no wider than its footprint: treated as the robot's own base.",
                metrics={"normal": normal.tolist(), "point": centre.tolist(), "plane_area_units2": plane_area,
                         "footprint_area_units2": footprint_area}, **_ids(scan, on_plane)))
            available &= ~on_plane
            continue
        mask = on_plane | under | low_outside
        label = "turntable / floor" if number == 0 else f"support plane {number + 1}"
        frac = weights[mask].sum() / total
        findings.append(Finding(
            kind="support_plane", label=label, severity="problem", fix="remove_planes",
            message=(f"A flat {label} is welded under the robot ({frac:.0%} of the surface), "
                     f"plus {int(under.sum())} elements below it and {int(low_outside.sum())} low features "
                     f"(feet/ridges) outside the robot's footprint."),
            metrics={"normal": normal.tolist(), "point": centre.tolist(), "area_fraction": float(frac),
                     "plane_elements": int(on_plane.sum()), "below_elements": int(under.sum()),
                     "low_feature_elements": int(low_outside.sum()), "footprint_area_units2": footprint_area,
                     "one_sided_fraction": float(one_sided)},
            **_ids(scan, mask)))
        available &= ~mask
    return findings


def detect_fragments(scan: Scan, weights, p: HealthParams) -> list[Finding]:
    if scan.is_mesh:
        f = scan.faces
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(scan.vertices),) * 2)
        vlabels = connected_components(graph, directed=False)[1]
        labels = vlabels[f[:, 0]]
    else:
        labels = _clusters(scan.vertices, 3 * _median_spacing(scan.vertices))
    area = np.bincount(labels, weights=weights)
    main = int(np.argmax(area))
    total = area.sum()
    findings = []
    small = [i for i in np.argsort(-area) if i != main and area[i] < p.fragment_max_frac * total]
    if small:
        mask = np.isin(labels, small)
        findings.append(Finding(
            kind="fragment", label=f"{len(small)} fragment(s)", severity="warn", fix="drop_fragments",
            message=(f"{len(small)} small piece(s) float free of the robot ({mask.sum()} elements, "
                     f"{area[small].sum() / total:.2%} of the surface): scan debris or broken-off thin parts."),
            metrics={"count": len(small), "area_fraction": float(area[small].sum() / total),
                     "sizes": [int((labels == i).sum()) for i in small[:50]]},
            **_ids(scan, mask)))
    big_islands = [i for i in range(len(area)) if i != main and area[i] >= p.fragment_max_frac * total]
    for i in big_islands:
        findings.append(Finding(
            kind="fragment", label=f"large island {i}", severity="info", fix=None,
            message=f"A large separate piece ({area[i] / total:.1%} of the surface). Kept: it may be a real part.",
            metrics={"area_fraction": float(area[i] / total)}, **_ids(scan, labels == i)))
    return findings


def _boundary_loops(faces: np.ndarray):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    b = uniq[counts == 1]
    if not len(b):
        return []
    nodes, inv = np.unique(b.ravel(), return_inverse=True)
    be = inv.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(len(nodes),) * 2).tocsr()
    graph = graph + graph.T
    _, comp = connected_components(graph, directed=False)
    degree = np.asarray((graph > 0).sum(axis=1)).ravel()
    loops = []
    order = np.argsort(comp, kind="stable")
    splits = np.cumsum(np.bincount(comp))[:-1]
    for members in np.split(order, splits):
        simple = bool((degree[members] == 2).all())
        loop = None
        if simple:
            # walk the cycle
            start = members[0]
            loop = [start]
            prev, cur = -1, start
            while True:
                nbrs = graph.indices[graph.indptr[cur]:graph.indptr[cur + 1]]
                nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
                if nxt == start:
                    break
                loop.append(nxt)
                prev, cur = cur, nxt
                if len(loop) > len(members):
                    loop, simple = None, False
                    break
        loops.append((nodes[members], None if loop is None else nodes[np.array(loop)], simple))
    return loops


def detect_holes(scan: Scan, p: HealthParams) -> list[Finding]:
    findings = []
    v = scan.vertices
    for k, (members, loop, simple) in enumerate(_boundary_loops(scan.faces)):
        pts = v[members]
        if loop is not None:
            perim = float(np.linalg.norm(v[np.roll(loop, -1)] - v[loop], axis=1).sum())
        else:
            perim = float(len(members) * np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        span = float(np.ptp(pts, axis=0).max())
        perim_mm = perim * scan.mm_per_unit
        fillable = simple and loop is not None and perim_mm <= p.hole_fill_max_mm
        findings.append(Finding(
            kind="hole", label=f"hole {k}", severity="warn" if fillable else "info",
            fix="fill_holes" if fillable else None,
            message=(f"Gap in the surface, {perim_mm:.1f} mm around"
                     + ("; small enough to patch." if fillable else
                        ("; too big to patch automatically (unscanned region)." if simple else
                         "; tangled edge, not patched automatically."))),
            metrics={"perimeter_mm": perim_mm, "span_mm": span * scan.mm_per_unit, "edges": int(len(members)),
                     "simple": simple, "centre": pts.mean(axis=0).tolist()},
            points=members, loop=loop))
    return findings


def detect_wires_and_blobs(scan: Scan, centres, normals, weights, exclude: np.ndarray, p: HealthParams):
    """Thin free-hanging runs (wires) and rough noisy patches (low confidence).

    Wire test (tuned on bot1): sample the surface, measure local wall thickness
    (distance to the nearest opposite-facing surface behind each sample), keep
    samples thinner than wire_max_mm, link them into connected runs and keep
    runs whose extent is at least wire_min_len_mm. Thin plates on a link only
    give short speckles at this threshold; long thin runs are cables (or a
    long thin rail: the preview is for the operator to judge).
    """
    findings: list[Finding] = []
    idx_s = _sample(np.where(exclude, 0.0, weights + 1e-15), p.sample_count, p.seed)
    if len(idx_s) < 100:
        return findings, None
    S, Nrm = centres[idx_s], normals[idx_s] * _outward_sign(scan)
    tree = cKDTree(S)
    spacing = float(np.median(tree.query(S[:: max(1, len(S) // 3000)], k=2)[0][:, 1]))
    sample_area = weights[idx_s].sum() / len(S) if len(S) else 0.0
    _, nearest_sample = tree.query(centres)
    elem_ok = ~exclude
    in_wire = np.zeros(len(S), dtype=bool)
    thickness = np.full(len(S), np.inf)

    if scan.is_mesh:
        R = scan.mm(p.wire_max_mm)
        d, idx = tree.query(S, k=48, distance_upper_bound=R * 1.6)
        valid = np.isfinite(d)
        safe = np.where(valid, idx, 0)
        off = S[safe] - S[:, None, :]
        depth = -(off * Nrm[:, None, :]).sum(axis=2)  # distance behind the surface
        lateral = np.linalg.norm(off + depth[..., None] * Nrm[:, None, :], axis=2)
        opposite = (Nrm[safe] * Nrm[:, None, :]).sum(axis=2) < -0.5
        ok = valid & opposite & (depth > 0.2 * spacing) & (lateral < np.maximum(0.8 * depth, 2 * spacing))
        thickness = np.where(ok, depth, np.inf).min(axis=1)
        thin_ids = np.flatnonzero(thickness < R)
        if len(thin_ids) > 10:
            lab = _clusters(S[thin_ids], max(4 * spacing, scan.mm(2.0)))
            runs = []
            for c in np.unique(lab):
                members = thin_ids[lab == c]
                if len(members) < 20:
                    continue
                extent_mm = float(np.ptp(S[members], axis=0).max()) * scan.mm_per_unit
                if extent_mm < p.wire_min_len_mm:
                    continue
                runs.append((extent_mm, members))
            runs.sort(key=lambda x: -x[0])
            for n, (extent_mm, members) in enumerate(runs):
                in_wire[members] = True
                flag = np.zeros(len(S), dtype=bool)
                flag[members] = True
                elems = np.flatnonzero(flag[nearest_sample] & elem_ok)
                area_mm2 = len(members) * sample_area * scan.mm_per_unit ** 2
                findings.append(Finding(
                    kind="wire", label=f"wire {n}", severity="problem", fix="strip_wires",
                    message=(f"Thin run about {extent_mm:.0f} mm long, under {p.wire_max_mm:g} mm thick: "
                             "looks like a servo lead or cable captured by the scanner."),
                    metrics={"length_mm": extent_mm, "area_mm2": float(area_mm2),
                             "median_thickness_mm": float(np.median(thickness[members]) * scan.mm_per_unit),
                             "centre": S[members].mean(axis=0).tolist()},
                    faces=elems))

    # ---- low confidence: local surface variation at a small scale (outside wires)
    d, idx = tree.query(S, k=16)
    vals, _ = _local_pca(S, idx, np.isfinite(d))
    variation = vals[:, 0] / np.maximum(vals.sum(axis=1), 1e-15)
    rough = (variation > p.blob_variation) & ~in_wire
    if rough.any():
        rough_ids = np.flatnonzero(rough)
        lab = _clusters(S[rough_ids], 3 * spacing)
        keep = []
        for c in np.unique(lab):
            members = rough_ids[lab == c]
            area_mm2 = len(members) * sample_area * scan.mm_per_unit ** 2
            if area_mm2 >= p.blob_min_area_mm2:
                keep.append((area_mm2, members))
        keep.sort(key=lambda x: -x[0])
        for n, (area_mm2, members) in enumerate(keep[:30]):
            flag = np.zeros(len(S), dtype=bool)
            flag[members] = True
            mask = flag[nearest_sample] & elem_ok
            findings.append(Finding(
                kind="low_confidence", label=f"rough patch {n}", severity="warn", fix=None,
                message=(f"Noisy, lumpy surface (~{area_mm2:.0f} mm^2): the scanner was unsure here. "
                         "If this is a servo or joint, use a library servo or stand-in for it."),
                metrics={"area_mm2": float(area_mm2), "centre": S[members].mean(axis=0).tolist(),
                         "mean_variation": float(variation[members].mean())},
                **_ids(scan, mask)))
    return findings, {"thickness": thickness, "variation": variation, "sample_ids": idx_s}


def _outward_sign(scan: Scan) -> float:
    """+1 if face normals point out of the solid, -1 if the scanner wrote them
    inward (signed volume < 0 on a consistently wound mesh)."""
    if not scan.is_mesh:
        return 1.0
    tri = scan.vertices[scan.faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    return -1.0 if signed < 0 else 1.0


def _mst_length(points: np.ndarray) -> float:
    from scipy.sparse.csgraph import minimum_spanning_tree
    if len(points) < 2:
        return 0.0
    if len(points) > 3000:
        points = points[np.random.default_rng(0).choice(len(points), 3000, replace=False)]
    k = min(8, len(points) - 1)
    d, idx = cKDTree(points).query(points, k=k + 1)
    rows = np.repeat(np.arange(len(points)), k)
    g = coo_matrix((d[:, 1:].ravel() + 1e-12, (rows, idx[:, 1:].ravel())), shape=(len(points),) * 2)
    mst = minimum_spanning_tree(g)
    # disconnected kNN graph pieces are fine: MST forest length
    return float(mst.sum())


# --------------------------------------------------------------------------- entry point

def check_scan(scan: Scan, params: HealthParams | None = None) -> HealthReport:
    p = params or HealthParams()
    centres, normals, weights = _elements(scan)
    findings: list[Finding] = []
    skipped: list[str] = []
    findings += detect_support_planes(scan, centres, normals, weights, p)
    findings += detect_fragments(scan, weights, p)
    if scan.is_mesh:
        findings += detect_holes(scan, p)
    else:
        skipped.append("Holes and wires need a mesh; this scan is a point cloud (mesh it in the scanner software first).")
    exclude = np.zeros(len(centres), dtype=bool)
    for f in findings:
        if f.kind in ("support_plane", "fragment") and f.fix:
            exclude[f.faces if scan.is_mesh else f.points] = True
    extra, _ = detect_wires_and_blobs(scan, centres, normals, weights, exclude, p)
    findings += extra
    if scan.units == "unknown":
        skipped.append("Scan units unknown: millimetre thresholds may be wrong; set units before trusting results.")
    return HealthReport(scan=scan.summary(), findings=findings, params=p, skipped=skipped)


# --------------------------------------------------------------------------- fixes

@dataclass
class FixOptions:
    """Every fix is opt-in: the operator reviews the preview, then toggles."""
    remove_planes: bool = False
    strip_wires: bool = False
    fill_holes: bool = False
    drop_fragments: bool = False


@dataclass
class FixResult:
    vertices: np.ndarray
    faces: np.ndarray | None
    provenance: np.ndarray       # per face (or point): "scanned" | "repaired"
    removed: dict                # fix name -> count of elements removed
    added: dict                  # fix name -> count of elements added
    kept_from_original: np.ndarray  # original element ids kept, in output order

    def summary(self) -> dict:
        return {"removed": self.removed, "added": self.added,
                "elements_out": int(len(self.faces) if self.faces is not None else len(self.vertices))}


def apply_fixes(scan: Scan, report: HealthReport, options: FixOptions) -> FixResult:
    """Return a cleaned copy. The original Scan and file are untouched."""
    n = len(scan.faces) if scan.is_mesh else len(scan.vertices)
    drop = np.zeros(n, dtype=bool)
    removed: dict[str, int] = {}
    for f in report.findings:
        if f.fix and f.fix != "fill_holes" and getattr(options, f.fix, False):
            ids = f.faces if scan.is_mesh else f.points
            before = drop.sum()
            drop[ids] = True
            removed[f.fix] = removed.get(f.fix, 0) + int(drop.sum() - before)
    keep = np.flatnonzero(~drop)
    if not scan.is_mesh:
        pts = scan.vertices[keep].copy()
        return FixResult(pts, None, np.full(len(pts), "scanned", dtype=object), removed, {}, keep)
    faces = scan.faces[keep]
    new_faces = []
    new_vertices = []
    added = 0
    if options.fill_holes:
        base = len(scan.vertices)
        live = np.zeros(len(scan.vertices), dtype=bool)
        live[faces.ravel()] = True
        for f in report.by_kind("hole"):
            if f.fix != "fill_holes" or f.loop is None or not live[f.loop].all():
                continue
            loop = f.loop
            centre = scan.vertices[loop].mean(axis=0)
            c_idx = base + len(new_vertices)
            new_vertices.append(centre)
            # orient the fan to match the neighbouring faces: boundary edge (a, b)
            # appears in its face as a->b, the patch must use b->a.
            a, b = loop[0], loop[1]
            owner = faces[(faces == a).any(axis=1) & (faces == b).any(axis=1)]
            forward = True
            if len(owner):
                row = list(owner[0])
                forward = row[(row.index(a) + 1) % 3] != b
            ring = loop if forward else loop[::-1]
            for i in range(len(ring)):
                new_faces.append([ring[i], ring[(i + 1) % len(ring)], c_idx])
            added += len(ring)
    vertices = np.vstack([scan.vertices, np.asarray(new_vertices).reshape(-1, 3)])
    all_faces = np.vstack([faces, np.asarray(new_faces, dtype=np.int64).reshape(-1, 3)])
    provenance = np.array(["scanned"] * len(faces) + ["repaired"] * len(new_faces), dtype=object)
    # drop unreferenced vertices
    used, inverse = np.unique(all_faces.ravel(), return_inverse=True)
    return FixResult(vertices[used], inverse.reshape(-1, 3), provenance, removed,
                     {"fill_holes": added} if added else {}, keep)


def save_fixed(result: FixResult, path) -> None:
    import trimesh
    if result.faces is None:
        trimesh.PointCloud(result.vertices).export(str(path))
    else:
        trimesh.Trimesh(result.vertices, result.faces, process=False).export(str(path))
