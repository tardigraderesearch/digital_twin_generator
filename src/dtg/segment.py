"""Cut the scan into links using the tag -> link assignment.

Input: the (cleaned) scan already registered into the model frame, the
kinematic model (which tags belong to which link, joint pivots and axes) and
the tag poses in the model frame.

1. Seeds: surface within half a tile of each tag centre belongs to that
   tag's link (the tags are physical tiles on the link).
2. Geodesic labelling: one Dijkstra per link over the surface graph; every
   vertex goes to the link it is geodesically closest to. Links fused by the
   scanner at a joint still split near the middle of the bridge, and wires
   or the turntable never pull a vertex across a gap.
3. Joint cut guides (EXPERIMENTAL, off by default): near each joint a plane
   through the fitted pivot perpendicular to the axis decides parent vs
   child when their seeds sit on opposite sides of it. The pivot's position
   ALONG the axis is not observable from tag motion, so on the synthetic
   arm this lowers accuracy (82% -> 76%); kept only for experiments.
4. Ambiguity is kept, not forced (review gate 4): vertices whose two best
   links are within ambiguity_margin, or unreachable from any seed, are
   labelled -1 ("unassigned") and exported as their own part.

Motion-consistency voting with a second scan (Scan B) is a TODO.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass
class SegmentParams:
    seed_radius: float = 0.012        # m, about half a 25 mm tag tile
    ambiguity_margin: float = 0.002   # m of geodesic distance
    guide_radius: float = 0.06        # m around each pivot where joint guides apply
    use_joint_guides: bool = False   # experimental, see module docstring


@dataclass
class Segmentation:
    links: list[str]
    vertex_labels: np.ndarray         # index into links, -1 = unassigned
    face_labels: np.ndarray
    guides: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def link_faces(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.face_labels == self.links.index(name))


def _graph(vertices, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1) + 1e-12
    n = len(vertices)
    return coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()


def segment_by_linkage(vertices, faces, kin, tag_poses_model: dict, params: SegmentParams | None = None) -> Segmentation:
    p = params or SegmentParams()
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    links = sorted(kin.link_tags, key=lambda l: (l != kin.root, l))
    tree = cKDTree(vertices)
    graph = _graph(vertices, faces)
    dist = np.full((len(links), len(vertices)), np.inf)
    seeds_per_link = {}
    for i, link in enumerate(links):
        seeds = set()
        for tag in kin.link_tags[link]:
            if tag in tag_poses_model:
                seeds.update(tree.query_ball_point(np.asarray(tag_poses_model[tag])[:3, 3], r=p.seed_radius))
        seeds_per_link[link] = np.array(sorted(seeds), dtype=np.int64)
        if len(seeds):
            dist[i] = dijkstra(graph, directed=False, indices=seeds_per_link[link], min_only=True)
    order = np.argsort(dist, axis=0)
    best = order[0]
    d1 = dist[best, np.arange(len(vertices))]
    d2 = dist[order[1], np.arange(len(vertices))] if len(links) > 1 else np.full(len(vertices), np.inf)
    labels = best.copy()
    labels[~np.isfinite(d1)] = -1
    ambiguous = np.isfinite(d1) & (d2 - d1 < p.ambiguity_margin)
    guides = []
    if p.use_joint_guides:
        for j in kin.joints:
            if j.parent not in links or j.child not in links:
                continue
            pi, ci = links.index(j.parent), links.index(j.child)
            pivot, axis = np.asarray(j.pivot_model), np.asarray(j.axis_model)
            ps, cs = seeds_per_link[j.parent], seeds_per_link[j.child]
            if not len(ps) or not len(cs):
                continue
            side_p = np.sign(np.median((vertices[ps] - pivot) @ axis))
            side_c = np.sign(np.median((vertices[cs] - pivot) @ axis))
            applied = bool(side_p and side_c and side_p != side_c)
            guides.append({"joint": j.name, "applied": applied,
                           "reason": "parent and child seeds on opposite sides of the pivot plane" if applied
                           else "seeds on the same side: geodesic split kept"})
            if not applied:
                continue
            near = np.linalg.norm(vertices - pivot, axis=1) < p.guide_radius
            pair = near & np.isin(labels, [pi, ci])
            side = np.sign((vertices - pivot) @ axis)
            labels[pair & (side == side_c)] = ci
            labels[pair & (side == side_p)] = pi
            ambiguous[pair] = False
    labels[ambiguous] = -1
    fl = labels[faces]
    face_labels = np.where((fl[:, 0] == fl[:, 1]) | (fl[:, 0] == fl[:, 2]), fl[:, 0], fl[:, 1])
    face_labels = np.where(fl[:, 1] == fl[:, 2], fl[:, 1], face_labels)
    stats = {"vertices": int(len(vertices)), "unassigned_vertices": int((labels < 0).sum()),
             "per_link_faces": {l: int((face_labels == i).sum()) for i, l in enumerate(links)},
             "unassigned_faces": int((face_labels < 0).sum()),
             "seeds": {l: int(len(s)) for l, s in seeds_per_link.items()},
             "links_without_seeds": [l for l, s in seeds_per_link.items() if not len(s)]}
    return Segmentation(links=links, vertex_labels=labels, face_labels=face_labels, guides=guides, stats=stats)


def link_meshes(vertices, faces, seg: Segmentation):
    """{link or 'unassigned': trimesh} in the same frame as the input."""
    import trimesh
    out = {}
    names = seg.links + ["unassigned"]
    for i, name in enumerate(names):
        idx = np.flatnonzero(seg.face_labels == (i if name != "unassigned" else -1))
        if not len(idx):
            continue
        used, inv = np.unique(faces[idx].ravel(), return_inverse=True)
        out[name] = trimesh.Trimesh(vertices[used], inv.reshape(-1, 3), process=False)
    return out
