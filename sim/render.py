"""A small software z-buffer renderer (numpy + OpenCV), enough for a camera bench.

Triangles are flat-shaded (ambient + one directional light); textured
triangles (ArUco tiles, the wall, the table) look up an albedo image through
a planar frame. Static items (table, wall, the fixed base and its tags) are
rasterised once; each frame only redraws the moving links inside their
screen bounding box, with supersampling for anti-aliased edges. Output is
linear radiance (float32 BGR in 0..1) plus a depth map; the camera model in
sim.camera turns that into a noisy, defocused 8-bit image.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Texture:
    image: np.ndarray            # float32 albedo, HxW (grey) or HxWx3 (BGR), 0..1
    frame: np.ndarray            # 4x4: texture plane frame at design zero (x right, y up, z out)
    size: tuple                  # (width, height) in metres, centred on the frame origin


@dataclass
class Item:
    """A triangle mesh rigidly attached to one link (vertices at design zero, world frame)."""
    link: str
    verts: np.ndarray            # (N, 3)
    tris: np.ndarray             # (M, 3) int, counter-clockwise seen from outside
    albedo: np.ndarray           # (M, 3) BGR 0..1
    texture: Texture | None = None
    name: str = ""


@dataclass
class Scene:
    items: list[Item] = field(default_factory=list)
    light_dir: np.ndarray = field(default_factory=lambda: np.array([0.3, -0.5, 0.8]))
    ambient: float = 0.45

    def static_items(self):
        return [i for i in self.items if i.link in ("world", "base")]

    def moving_items(self):
        return [i for i in self.items if i.link not in ("world", "base")]


# ---------------------------------------------------------------- meshes

def box(lo, hi, T=None):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    # vertex index = 4*ix + 2*iy + iz
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    tris = np.array(tris)
    if T is not None:
        v = v @ T[:3, :3].T + T[:3, 3]
    return v, _fix_winding(v, tris)


def cylinder(cx, cy, r, z0, z1, n=24):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]
    v = np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)], [[cx, cy, z0], [cx, cy, z1]]])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    return v, _fix_winding(v, np.array(tris))


def quad(T, w, h):
    """Rectangle w x h centred on frame T, facing +z of T."""
    c = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    v = c @ T[:3, :3].T + T[:3, 3]
    return v, np.array([(0, 1, 2), (0, 2, 3)])


def _fix_winding(v, tris):
    """Orient every triangle outward (away from the mesh centroid); fine for convex parts."""
    centre = v.mean(axis=0)
    out = tris.copy()
    for k, (a, b, c) in enumerate(tris):
        n = np.cross(v[b] - v[a], v[c] - v[a])
        if np.dot(n, (v[a] + v[b] + v[c]) / 3 - centre) < 0:
            out[k] = (a, c, b)
    return out


def item(link, mesh, color, texture=None, name=""):
    v, t = mesh
    return Item(link, np.asarray(v, float), np.asarray(t, int), np.tile(np.asarray(color, float), (len(t), 1)),
                texture, name)


# ---------------------------------------------------------------- rasteriser

class Renderer:
    def __init__(self, scene: Scene, K: np.ndarray, size: tuple[int, int], T_cam_world: np.ndarray,
                 supersample: int = 2):
        self.scene, self.K, self.size, self.T_cw, self.ss = scene, np.asarray(K, float), size, T_cam_world, supersample
        W, H = size
        self.Kss = self.K.copy()
        self.Kss[:2] *= supersample
        self.Kss[0, 2] = (self.K[0, 2] + 0.5) * supersample - 0.5
        self.Kss[1, 2] = (self.K[1, 2] + 0.5) * supersample - 0.5
        self.Kinv = np.linalg.inv(self.Kss)
        self.light = np.asarray(scene.light_dir, float) / np.linalg.norm(scene.light_dir)
        Ws, Hs = W * supersample, H * supersample
        self.z_static = np.full((Hs, Ws), np.inf, np.float32)
        self.c_static = np.zeros((Hs, Ws, 3), np.float32)
        for it in scene.static_items():
            self._draw(it, np.eye(4), self.z_static, self.c_static, (0, 0, Ws, Hs))
        self.c_static_1x = cv2.resize(self.c_static, size, interpolation=cv2.INTER_AREA)
        self.z_static_1x = cv2.resize(self.z_static, size, interpolation=cv2.INTER_NEAREST)
        self.c_static = self.c_static.astype(np.float16)   # memory: only ROI copies are used later

    def render(self, motions: dict[str, np.ndarray]):
        """(radiance BGR float32 HxWx3, depth HxW in metres) for the given link motions."""
        W, H = self.size
        ss = self.ss
        moving = self.scene.moving_items()
        # screen bbox of all moving geometry
        pts = np.vstack([(it.verts @ motions[it.link][:3, :3].T + motions[it.link][:3, 3]) for it in moving])
        pc = pts @ self.T_cw[:3, :3].T + self.T_cw[:3, 3]
        pc = pc[pc[:, 2] > 0.02]
        uv = (pc @ self.K.T)[:, :2] / pc[:, 2:3]
        x0, y0 = np.floor(uv.min(axis=0)).astype(int) - 4
        x1, y1 = np.ceil(uv.max(axis=0)).astype(int) + 4
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        color = self.c_static_1x.copy()
        depth = self.z_static_1x.copy()
        if x1 <= x0 or y1 <= y0:
            return color, depth
        roi = (x0 * ss, y0 * ss, x1 * ss, y1 * ss)
        z = self.z_static[roi[1]:roi[3], roi[0]:roi[2]].copy()
        c = self.c_static[roi[1]:roi[3], roi[0]:roi[2]].astype(np.float32)
        for it in moving:
            self._draw(it, motions[it.link], z, c, roi)
        color[y0:y1, x0:x1] = cv2.resize(c, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        depth[y0:y1, x0:x1] = z[::ss, ::ss][: y1 - y0, : x1 - x0]
        return color, depth

    def _draw(self, it: Item, motion: np.ndarray, zbuf, cbuf, roi) -> None:
        M = self.T_cw @ motion
        v = it.verts @ M[:3, :3].T + M[:3, 3]
        if (v[:, 2] <= 0.02).any():
            return
        p = v @ self.Kss.T
        uv = p[:, :2] / p[:, 2:3]
        rx0, ry0, rx1, ry1 = roi
        tex = it.texture
        if tex is not None:
            T_tex = M @ tex.frame
            R_tex, t_tex = T_tex[:3, :3], T_tex[:3, 3]
            timg = tex.image
            th, tw = timg.shape[:2]
        for k, (a, b, cc) in enumerate(it.tris):
            A, B, C = v[a], v[b], v[cc]
            n = np.cross(B - A, C - A)
            nn = np.linalg.norm(n)
            if nn == 0 or np.dot(n, A) >= 0:   # back face (camera at the origin)
                continue
            n = n / nn
            P = uv[[a, b, cc]]
            bx0 = max(int(np.floor(P[:, 0].min())), rx0)
            bx1 = min(int(np.ceil(P[:, 0].max())) + 1, rx1)
            by0 = max(int(np.floor(P[:, 1].min())), ry0)
            by1 = min(int(np.ceil(P[:, 1].max())) + 1, ry1)
            if bx1 <= bx0 or by1 <= by0:
                continue
            xs = np.arange(bx0, bx1, dtype=np.float32)
            ys = np.arange(by0, by1, dtype=np.float32)[:, None]
            (ax, ay), (bxp, byp), (cx_, cy_) = P
            area = (bxp - ax) * (cy_ - ay) - (byp - ay) * (cx_ - ax)
            if abs(area) < 1e-9:
                continue
            s = 1.0 if area > 0 else -1.0
            w0 = s * ((bxp - ax) * (ys - ay) - (byp - ay) * (xs - ax))
            w1 = s * ((cx_ - bxp) * (ys - byp) - (cy_ - byp) * (xs - bxp))
            w2 = s * ((ax - cx_) * (ys - cy_) - (ay - cy_) * (xs - cx_))
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            # exact depth: intersect the pixel ray with the triangle plane
            d = float(np.dot(n, A))
            rx = self.Kinv[0, 0] * xs + self.Kinv[0, 2]
            ry = self.Kinv[1, 1] * ys + self.Kinv[1, 2]
            denom = n[0] * rx + n[1] * ry + n[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                zz = d / denom
            sub_z = zbuf[by0 - ry0:by1 - ry0, bx0 - rx0:bx1 - rx0]
            win = inside & (zz > 0) & (zz < sub_z)
            if not win.any():
                continue
            sub_z[win] = zz[win]
            n_world = motion[:3, :3] @ np.cross(it.verts[b] - it.verts[a], it.verts[cc] - it.verts[a])
            n_world /= np.linalg.norm(n_world)
            shade = self.scene.ambient + (1 - self.scene.ambient) * max(0.0, float(n_world @ self.light))
            sub_c = cbuf[by0 - ry0:by1 - ry0, bx0 - rx0:bx1 - rx0]
            if tex is None:
                sub_c[win] = it.albedo[k] * shade
                continue
            yy, xx = np.nonzero(win)
            zw = zz[yy, xx]
            X = np.c_[rx[xx] * zw, ry[yy, 0] * zw, zw]
            local = (X - t_tex) @ R_tex
            u = np.clip(((local[:, 0] / tex.size[0]) + 0.5) * tw, 0, tw - 1).astype(int)
            vv = np.clip((0.5 - local[:, 1] / tex.size[1]) * th, 0, th - 1).astype(int)
            alb = timg[vv, u]
            if alb.ndim == 1:
                alb = alb[:, None] * np.ones(3)
            sub_c[yy, xx] = alb * shade
