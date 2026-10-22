"""Register the camera/tag model frame to the scan.

The tags are physical relief tiles, so they are in the scan. Given matching
points (tag centres seen by the camera vs the same tags located in the scan,
or any 3+ points picked by hand: the manual fallback), a similarity transform
(scale, rotation, translation) maps scan coordinates into the model frame.
Scale is solved too, so a scan in unknown units comes out in metres.

    T, scale, rms = similarity_from_points(scan_pts, model_pts)
    model_pt = T @ [scan_pt, 1]     # T includes the scale

Locating tag tiles in the scan automatically (relief pattern / texture) is
NOT implemented yet: refine_tag_centre() snaps a rough pick to the tile's
flat face, which is enough for click-assisted registration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


class RegistrationError(ValueError):
    pass


def similarity_from_points(src, dst, *, with_scale: bool = True):
    """Umeyama: T (4x4, scale folded into the rotation block), scale, rms (dst units)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise RegistrationError("need at least 3 matching 3D points")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    if np.linalg.matrix_rank(xs, tol=1e-9 * max(1.0, np.abs(xs).max())) < 2:
        raise RegistrationError("points are collinear: pick points that span a triangle")
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    S = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    R = u @ S @ vt
    var = (xs ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ S) / var) if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = mu_d - scale * R @ mu_s
    resid = (src @ T[:3, :3].T + T[:3, 3]) - dst
    return T, scale, float(np.sqrt((resid ** 2).sum(axis=1).mean()))


def apply(T, points) -> np.ndarray:
    return np.asarray(points, float) @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def tag_positions_model(kin) -> dict[int, np.ndarray]:
    """Tag poses in the model frame at the reference pose (from the camera)."""
    out = {}
    for link, info in kin.link_tags.items():
        if link not in kin.link_ref_model:
            continue
        L = np.asarray(kin.link_ref_model[link])
        for k, off in kin.link_offsets.get(link, {}).items():
            out[int(k)] = L @ np.asarray(off)
    return out


@dataclass
class Registration:
    T_model_scan: np.ndarray
    scale: float
    rms_m: float
    method: str
    pairs: int

    def to_dict(self) -> dict:
        return {"T_model_scan": self.T_model_scan.tolist(), "scale_model_per_scan_unit": self.scale,
                "rms_mm": self.rms_m * 1000, "method": self.method, "pairs": self.pairs}


def register_tags(scan_centres: dict[int, np.ndarray], model_tag_poses: dict[int, np.ndarray],
                  *, max_rms_mm: float = 5.0) -> Registration:
    """Register from tag centres located in the scan (>= 3 tags, not collinear)."""
    ids = sorted(set(scan_centres) & set(model_tag_poses))
    if len(ids) < 3:
        raise RegistrationError(f"need >= 3 tags found in both the scan and the camera log, have {ids}")
    src = np.array([scan_centres[i] for i in ids])
    dst = np.array([np.asarray(model_tag_poses[i])[:3, 3] for i in ids])
    T, s, rms = similarity_from_points(src, dst)
    if rms * 1000 > max_rms_mm:
        raise RegistrationError(f"tag registration residual {rms * 1000:.1f} mm > {max_rms_mm} mm: "
                                "a tag was mislocated or a link moved between scan and reference capture")
    return Registration(T, s, rms, "tag centres", len(ids))


def register_points(scan_points, model_points) -> Registration:
    """Manual fallback: 3+ points clicked in the scan and the same points in the model frame."""
    T, s, rms = similarity_from_points(scan_points, model_points)
    return Registration(T, s, rms, "manual points", len(scan_points))


def refine_tag_centre(points: np.ndarray, approx, tile_size: float, tree: cKDTree | None = None):
    """Snap a rough tag pick to the centroid and normal of the flat tile face around it."""
    tree = tree or cKDTree(points)
    idx = tree.query_ball_point(np.asarray(approx, float), r=tile_size * 0.5)
    if len(idx) < 10:
        raise RegistrationError("no surface near the picked tag position")
    P = points[idx]
    c = P.mean(axis=0)
    normal = np.linalg.svd(P - c, full_matrices=False)[2][2]
    flatness = float(np.abs((P - c) @ normal).std())
    return c, normal, flatness


def locate_tags_in_scan(scan, markers):
    # TODO: find the relief tag tiles in the scan automatically: candidate flat squares of
    # tile_size_mm from the health-check plane finder, then decode the raised/sunken module
    # pattern (or vertex colour when the scan is textured) against the ArUco dictionary.
    raise NotImplementedError("automatic tag finding in the scan is not implemented; "
                              "pick tag centres (refine_tag_centre) or use 3-point alignment")
