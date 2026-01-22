#!/usr/bin/env python3
"""Compute average distance of the last local_path point across ORAD-3D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

DEFAULT_SPLITS = ("training", "validation", "testing")


def _iter_sequence_dirs(split_dir: Path) -> Iterable[Path]:
    for child in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and not child.name.endswith(".zip"):
            yield child


def _iter_local_path_jsons(root_dir: Path, splits: List[str]) -> Iterable[Path]:
    for split in splits:
        split_dir = root_dir / split
        if not split_dir.is_dir():
            continue
        for seq_dir in _iter_sequence_dirs(split_dir):
            local_dir = seq_dir / "local_path"
            if not local_dir.is_dir():
                continue
            for p in sorted(local_dir.glob("*.json"), key=lambda q: q.name):
                yield p


def _point_from_value(value: object) -> Optional[Tuple[float, float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x = float(value[0])
            y = float(value[1])
            z = float(value[2]) if len(value) > 2 else 0.0
        except (TypeError, ValueError):
            return None
        return (x, y, z)
    if isinstance(value, dict):
        if "position" in value and isinstance(value["position"], dict):
            value = value["position"]
        if "x" in value and "y" in value:
            try:
                x = float(value["x"])
                y = float(value["y"])
                z = float(value.get("z", 0.0))
            except (TypeError, ValueError):
                return None
            return (x, y, z)
    return None


def _extract_last_point(
    obj: dict,
    key: str,
    relative_to_first: bool,
) -> Tuple[Optional[Tuple[float, float, float]], str]:
    if key not in obj:
        return None, "missing"
    pts = obj.get(key)
    if not isinstance(pts, list) or not pts:
        return None, "missing"

    last = _point_from_value(pts[-1])
    if last is None:
        return None, "parse"

    if not relative_to_first:
        return last, "ok"

    first = _point_from_value(pts[0])
    if first is None:
        return None, "parse"

    return (last[0] - first[0], last[1] - first[1], last[2] - first[2]), "ok"


def _distance(point: Tuple[float, float, float]) -> float:
    return math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)


def _resolve_splits(root_dir: Path, raw_splits: Optional[str]) -> List[str]:
    if raw_splits:
        return [s.strip() for s in raw_splits.split(",") if s.strip()]

    existing = [s for s in DEFAULT_SPLITS if (root_dir / s).is_dir()]
    if existing:
        return existing

    return [p.name for p in root_dir.iterdir() if p.is_dir() and not p.name.endswith(".zip")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average distance of the last local_path point across ORAD-3D",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="ORAD-3D root containing splits (training/validation/testing).",
    )
    parser.add_argument(
        "--key",
        default="trajectory_ins",
        help="Trajectory key in local_path JSON (default: trajectory_ins).",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated split names (default: auto-detect).",
    )
    parser.add_argument(
        "--relative-to-first",
        action="store_true",
        help="Measure distance from the first point instead of the origin.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root)
    splits = _resolve_splits(root_dir, args.splits)

    distances: List[float] = []
    files_seen = 0
    skipped_missing = 0
    skipped_parse = 0

    for path in _iter_local_path_jsons(root_dir, splits):
        files_seen += 1
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            skipped_parse += 1
            continue
        if not isinstance(obj, dict):
            skipped_parse += 1
            continue

        point, status = _extract_last_point(obj, key=args.key, relative_to_first=args.relative_to_first)
        if point is None:
            if status == "parse":
                skipped_parse += 1
            else:
                skipped_missing += 1
            continue

        distances.append(_distance(point))

    avg = sum(distances) / len(distances) if distances else 0.0

    print(f"splits: {', '.join(splits)}")
    print(f"key: {args.key}")
    print(f"relative_to_first: {bool(args.relative_to_first)}")
    print(f"average_distance: {avg:.6f}")
    print(f"count_used: {len(distances)}")
    print(f"count_files: {files_seen}")
    print(f"skipped_missing: {skipped_missing}")
    print(f"skipped_parse: {skipped_parse}")


if __name__ == "__main__":
    main()
