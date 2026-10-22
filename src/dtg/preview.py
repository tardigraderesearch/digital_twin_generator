"""Static preview images (matplotlib, headless, cross-platform).

Three orthographic views plus one oblique view; every face is painted
(depth-sorted). Problem regions from a HealthReport are coloured, holes are
outlined. Large meshes are drawn in full: 300k faces take ~20-40 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KIND_COLOURS = {
    "support_plane": (0.95, 0.55, 0.15),
    "wire": (0.90, 0.12, 0.15),
    "fragment": (0.65, 0.25, 0.85),
    "low_confidence": (0.95, 0.85, 0.15),
    "hole": (1.0, 0.1, 0.9),
}
KIND_NAMES = {
    "support_plane": "turntable / floor / feet",
    "wire": "wire (thin tube)",
    "fragment": "fragment",
    "low_confidence": "low-confidence blob",
    "hole": "hole outline",
}
BASE = (0.62, 0.70, 0.80)
LABEL_PALETTE = [(0.91, 0.47, 0.26), (0.26, 0.66, 0.90), (0.44, 0.76, 0.36), (0.78, 0.41, 0.76),
                 (0.86, 0.74, 0.29), (0.43, 0.47, 0.86), (0.31, 0.76, 0.72), (0.60, 0.60, 0.60)]


def _views(vertices):
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    # oblique: rotate 35 deg about z then 30 deg about x
    a, b = np.radians(35), np.radians(-60)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return centre, [("X / Y (top)", np.eye(3)),
                    ("X / Z (front)", np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)),
                    ("Y / Z (side)", np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)),
                    ("oblique", rx @ rz)]


def render_mesh(vertices, faces, face_colours, path, title, legend=(), outlines=(), points_mode=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.patches import Patch

    centre, views = _views(vertices)
    v0 = vertices - centre
    span = float(np.ptp(v0, axis=0).max()) * 0.56 or 1.0
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.4), facecolor="#111827")
    for ax, (name, rot) in zip(axes, views):
        v = v0 @ rot.T
        if points_mode:
            order = np.argsort(v[:, 2])
            ax.scatter(v[order, 0], v[order, 1], s=0.3, c=np.asarray(face_colours)[order], linewidths=0)
        else:
            tri = v[faces]
            order = np.argsort(tri[:, :, 2].mean(axis=1), kind="stable")
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            ln = np.linalg.norm(normal, axis=1)
            shade = 0.35 + 0.65 * np.divide(np.abs(normal[:, 2]), ln, out=np.zeros_like(ln), where=ln > 0)
            cols = np.clip(np.asarray(face_colours) * shade[:, None], 0, 1)
            ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=cols[order], edgecolors="none",
                                             antialiased=False))
        for loop_pts, colour in outlines:
            p = (loop_pts - centre) @ rot.T
            segs = np.stack([p[:, :2], np.roll(p[:, :2], -1, axis=0)], axis=1)
            ax.add_collection(LineCollection(segs, colors=[colour], linewidths=1.2, zorder=5))
        ax.set(xlim=(-span, span), ylim=(-span, span), aspect="equal")
        ax.set_title(name, color="white")
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9ca3af", labelsize=7)
    if legend:
        fig.legend(handles=[Patch(color=c, label=l) for l, c in legend], loc="lower center",
                   ncol=min(6, len(legend)), facecolor="#1f2937", labelcolor="white", frameon=False)
    fig.suptitle(title, color="white")
    fig.tight_layout(rect=(0, 0.06 if legend else 0, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return Path(path)


def render_health(scan, report, path) -> Path:
    n = len(scan.faces) if scan.is_mesh else len(scan.vertices)
    colours = np.tile(np.asarray(BASE), (n, 1))
    present = []
    for kind in ("support_plane", "fragment", "low_confidence", "wire"):
        findings = [f for f in report.by_kind(kind) if (f.fix or kind == "low_confidence")]
        for f in findings:
            colours[f.faces if scan.is_mesh else f.points] = KIND_COLOURS[kind]
        if findings:
            present.append((KIND_NAMES[kind], KIND_COLOURS[kind]))
    outlines = []
    if scan.is_mesh:
        holes = [h for h in report.by_kind("hole") if h.loop is not None]
        for h in holes:
            outlines.append((scan.vertices[h.loop], KIND_COLOURS["hole"] if h.fix else (0.3, 0.9, 1.0)))
        if holes:
            present.append(("small hole (fillable)", KIND_COLOURS["hole"]))
            present.append(("large opening", (0.3, 0.9, 1.0)))
    title = (f"{Path(report.scan['path']).name}: scan health check  |  units {scan.units}  |  "
             f"{n:,} {'faces' if scan.is_mesh else 'points'}  |  grey = OK")
    if scan.is_mesh:
        return render_mesh(scan.vertices, scan.faces, colours, path, title, present, outlines)
    return render_mesh(scan.vertices, None, colours, path, title, present, points_mode=True)


def render_labels(vertices, faces, labels, names, path, title) -> Path:
    labels = np.asarray(labels)
    colours = np.array([LABEL_PALETTE[i % len(LABEL_PALETTE)] if i >= 0 else (0.25, 0.25, 0.25) for i in labels])
    legend = [(n, LABEL_PALETTE[i % len(LABEL_PALETTE)]) for i, n in enumerate(names)]
    if (labels < 0).any():
        legend.append(("unassigned / ambiguous", (0.25, 0.25, 0.25)))
    return render_mesh(vertices, faces, colours, path, title, legend)
