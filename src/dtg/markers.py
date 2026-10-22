"""Read the markers.json that aruco-tag-maker writes.

    {"dictionary": "DICT_4X4_50", "tag_size_mm": 30.0, "tile_size_mm": 40.0,
     "tags": {"base": {"id": 0}, "upper_arm": {"id": 1}}}

Plain JSON parsing (same rules as robo-prober's loader), so neither
dependency has to be installed. Optional DTG-only hint: a tag entry may carry
{"role": "root"} to name the fixed base tag; otherwise a tag called "base",
"body" or "root" is used, else the tag that moves least.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DICT_RE = re.compile(r"^DICT_[A-Z0-9_]+$")
ROOT_NAMES = ("base", "body", "root", "chassis")


class MarkersError(ValueError):
    pass


@dataclass(frozen=True)
class Markers:
    dictionary: str
    tags: dict[str, int] = field(default_factory=dict)
    tag_size_mm: float | None = None
    tile_size_mm: float | None = None
    root_tag: int | None = None

    def name_for(self, marker_id: int) -> str:
        for name, value in self.tags.items():
            if value == marker_id:
                return name
        return f"tag{marker_id}"

    @property
    def ids(self) -> list[int]:
        return sorted(self.tags.values())

    def to_dict(self) -> dict:
        return {"dictionary": self.dictionary, "tag_size_mm": self.tag_size_mm, "tile_size_mm": self.tile_size_mm,
                "tags": {n: {"id": i} for n, i in self.tags.items()}, "root_tag": self.root_tag}


def parse_markers(data: Any) -> Markers:
    if not isinstance(data, dict):
        raise MarkersError("markers.json must contain a JSON object")
    dictionary = data.get("dictionary")
    if not isinstance(dictionary, str) or not _DICT_RE.match(dictionary):
        raise MarkersError("markers.json needs a dictionary such as DICT_4X4_50")
    raw_tags = data.get("tags")
    if not isinstance(raw_tags, dict) or not raw_tags:
        raise MarkersError('markers.json needs a non-empty tags object {name: {"id": n}}')
    tags: dict[str, int] = {}
    root = None
    for name, entry in raw_tags.items():
        marker_id = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name or isinstance(marker_id, bool) or not isinstance(marker_id, int) or marker_id < 0:
            raise MarkersError(f"tag {name!r} needs a non-negative integer id")
        tags[name] = marker_id
        if isinstance(entry, dict) and entry.get("role") == "root":
            root = marker_id
    if len(set(tags.values())) != len(tags):
        raise MarkersError("marker ids in markers.json must be unique")
    if root is None:
        for name in ROOT_NAMES:
            if name in tags:
                root = tags[name]
                break
    sizes = []
    for key in ("tag_size_mm", "tile_size_mm"):
        size = data.get(key)
        if size is not None and (isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0):
            raise MarkersError(f"{key} must be a positive number")
        sizes.append(float(size) if size is not None else None)
    return Markers(dictionary=dictionary, tags=tags, tag_size_mm=sizes[0], tile_size_mm=sizes[1], root_tag=root)


def load_markers(path: str | Path) -> Markers:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarkersError(f"could not read {path}: {exc}") from exc
    return parse_markers(data)
