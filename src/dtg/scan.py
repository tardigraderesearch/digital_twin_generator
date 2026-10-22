"""Load a scan (mesh or point cloud) without changing the original.

Supported: OBJ, PLY, STL, GLB/GLTF, OFF via trimesh. A file that has no faces
(e.g. a scanner's PLY point cloud) loads as a point cloud. The loaded arrays
are treated as read-only; every repair works on a copy (see dtg.health).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

SCAN_SUFFIXES = (".obj", ".ply", ".stl", ".glb", ".gltf", ".off")


class ScanError(ValueError):
    pass


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guess_units(extent: float) -> tuple[str, float]:
    """Guess scan units from the largest bounding-box extent.

    Returns (name, millimetres per scan unit). Desk/bench robots are 20 mm to
    2 m across, so 20..5000 reads as mm and 0.02..5 as metres. The guess is
    always reported as a guess, never as a measurement.
    """
    if 20.0 <= extent <= 5000.0:
        return "mm (guessed)", 1.0
    if 2.0 <= extent < 20.0:
        return "cm (guessed)", 10.0
    if 0.02 <= extent < 2.0:
        return "m (guessed)", 1000.0
    return "unknown", 1.0


@dataclass
class Scan:
    path: Path
    sha256: str
    kind: str  # "mesh" or "points"
    vertices: np.ndarray
    faces: np.ndarray | None = None
    colors: np.ndarray | None = None
    units: str = "unknown"
    mm_per_unit: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def is_mesh(self) -> bool:
        return self.faces is not None and len(self.faces) > 0

    def mesh(self) -> trimesh.Trimesh:
        if not self.is_mesh:
            raise ScanError("this scan is a point cloud; no faces")
        return trimesh.Trimesh(self.vertices, self.faces, process=False, validate=False)

    @property
    def extent(self) -> np.ndarray:
        return np.ptp(self.vertices, axis=0)

    def mm(self, value_mm: float) -> float:
        """Convert a physical length in millimetres to scan units."""
        return value_mm / self.mm_per_unit

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "kind": self.kind,
            "vertices": int(len(self.vertices)),
            "faces": int(len(self.faces)) if self.is_mesh else 0,
            "units": self.units,
            "mm_per_unit": self.mm_per_unit,
            "bounds": [self.vertices.min(axis=0).tolist(), self.vertices.max(axis=0).tolist()],
        }


def load_scan(path: str | Path, *, units: str | None = None, mm_per_unit: float | None = None) -> Scan:
    """Load a scan file. Vertices are merged by trimesh so that topology
    (holes, components) is measured on the welded surface, as scanners often
    write duplicated seam vertices; the file on disk is never touched."""
    path = Path(path)
    if not path.is_file():
        raise ScanError(f"scan not found: {path}")
    if path.suffix.lower() not in SCAN_SUFFIXES:
        raise ScanError(f"unsupported scan type {path.suffix}; use one of {', '.join(SCAN_SUFFIXES)}")
    try:
        loaded = trimesh.load(str(path), force=None)
    except Exception as exc:  # noqa: BLE001 - surface any loader failure plainly
        raise ScanError(f"could not read {path.name}: {exc}") from exc
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump(concatenate=False) if hasattr(g, "vertices")] if hasattr(loaded, "dump") else list(loaded.geometry.values())
        meshes = [g for g in geoms if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if meshes:
            loaded = trimesh.util.concatenate(meshes)
        else:
            clouds = [np.asarray(g.vertices) for g in geoms if len(getattr(g, "vertices", []))]
            if not clouds:
                raise ScanError(f"{path.name} contains no geometry")
            loaded = trimesh.PointCloud(np.vstack(clouds))
    colors = None
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
        vertices = np.asarray(loaded.vertices, dtype=float)
        faces = np.asarray(loaded.faces, dtype=np.int64)
        kind = "mesh"
        try:
            vc = loaded.visual.vertex_colors if loaded.visual.kind == "vertex" else None
            colors = None if vc is None else np.asarray(vc)[:, :3]
        except Exception:  # noqa: BLE001
            colors = None
    else:
        vertices = np.asarray(loaded.vertices, dtype=float)
        faces = None
        kind = "points"
        c = getattr(loaded, "colors", None)
        colors = None if c is None or not len(c) else np.asarray(c)[:, :3]
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ScanError(f"{path.name} has no finite vertices")
    vertices.setflags(write=False)
    if faces is not None:
        faces.setflags(write=False)
    if mm_per_unit is None:
        units, mm_per_unit = guess_units(float(np.ptp(vertices, axis=0).max()))
    elif units is None:
        units = "given"
    return Scan(path=path, sha256=sha256(path), kind=kind, vertices=vertices, faces=faces,
                colors=colors, units=units, mm_per_unit=float(mm_per_unit))
