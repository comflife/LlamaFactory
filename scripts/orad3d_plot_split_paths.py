#!/usr/bin/env python3
"""Plot many ORAD-3D local_path trajectories onto one canvas.

Goal: quick inspection of how much trajectories bend left/right.

- Scans split directory like /data3/ORAD-3D/validation
- Reads local_path/*.json (default trajectory_ins)
- Converts each trajectory to ego-relative forward/lateral (subtract first point)
- Draws all polylines on a single white canvas with distinct colors
- Prints simple left/right/straight stats based on final lateral offset

This is NOT a metric-accurate camera projection.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Trajectory2D:
    seq: str
    ts: str
    forward: np.ndarray  # meters, >=0 typically
    lateral: np.ndarray  # meters, left/right


def _iter_sequence_dirs(split_dir: Path) -> Iterator[Path]:
    for child in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and not child.name.endswith(".zip"):
            yield child


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _extract_ts(name: str) -> str:
    # local_path files are like 1620463040283.json
    return Path(name).stem


def _load_xy_from_local_path(path: Path, key: str) -> List[Tuple[float, float]]:
    obj = _read_json(path)
    pts = obj.get(key)
    if not isinstance(pts, list) or not pts:
        return []

    out: List[Tuple[float, float]] = []
    for p in pts:
        if not isinstance(p, list) or len(p) < 2:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _to_forward_lateral(
    xy: List[Tuple[float, float]],
    forward_axis: str,
    flip_lateral: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not xy:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    x0, y0 = xy[0]
    xs = np.asarray([x - x0 for x, _ in xy], dtype=np.float64)
    ys = np.asarray([y - y0 for _, y in xy], dtype=np.float64)

    # Empirically for ORAD-3D, forward often matches the second value (y).
    if forward_axis == "y":
        forward = ys
        lateral = xs
    else:
        forward = xs
        lateral = ys

    if flip_lateral:
        lateral = -lateral

    return forward, lateral


def iter_trajectories(
    split_dir: Path,
    key: str,
    forward_axis: str,
    flip_lateral: bool,
    max_total: Optional[int],
    max_per_sequence: Optional[int],
) -> Iterator[Trajectory2D]:
    emitted = 0
    for seq_dir in _iter_sequence_dirs(split_dir):
        local_dir = seq_dir / "local_path"
        if not local_dir.is_dir():
            continue

        per_seq = 0
        for p in sorted(local_dir.glob("*.json"), key=lambda q: q.name):
            xy = _load_xy_from_local_path(p, key=key)
            if len(xy) < 2:
                continue

            forward, lateral = _to_forward_lateral(xy, forward_axis=forward_axis, flip_lateral=flip_lateral)
            if forward.size < 2:
                continue

            yield Trajectory2D(seq=seq_dir.name, ts=_extract_ts(p.name), forward=forward, lateral=lateral)

            emitted += 1
            per_seq += 1

            if max_per_sequence is not None and per_seq >= max_per_sequence:
                break
            if max_total is not None and emitted >= max_total:
                return


def _auto_scale(
    trajectories: List[Trajectory2D],
    canvas_size: Tuple[int, int],
    margin: int,
) -> float:
    if not trajectories:
        return 10.0

    w, h = canvas_size
    max_forward = 0.0
    max_abs_lat = 0.0
    for t in trajectories:
        if t.forward.size:
            max_forward = max(max_forward, float(np.max(t.forward)))
        if t.lateral.size:
            max_abs_lat = max(max_abs_lat, float(np.max(np.abs(t.lateral))))

    usable_h = max(1, h - 2 * margin)
    usable_w = max(1, w - 2 * margin)

    scale_h = (0.95 * usable_h / max_forward) if max_forward > 1e-6 else 10.0
    scale_w = (0.95 * (usable_w / 2.0) / max_abs_lat) if max_abs_lat > 1e-6 else scale_h

    return max(0.5, min(scale_h, scale_w))


def _color_for_index(i: int, n: int, alpha: int) -> Tuple[int, int, int, int]:
    # HSV rainbow
    h = (i / max(1, n)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.95, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), int(alpha))


def _draw_trajectory(
    draw: ImageDraw.ImageDraw,
    t: Trajectory2D,
    canvas_size: Tuple[int, int],
    margin: int,
    scale: float,
    color: Tuple[int, int, int, int],
    line_width: int,
) -> None:
    w, h = canvas_size
    cx = w / 2.0
    bottom = h - margin

    # Map: forward -> up, lateral -> left/right
    u = cx + (t.lateral * scale)
    v = bottom - (t.forward * scale)

    pts = list(zip(u.tolist(), v.tolist()))
    if len(pts) < 2:
        return

    def clamp(p: Tuple[float, float]) -> Tuple[float, float]:
        return (min(max(p[0], 0.0), w - 1.0), min(max(p[1], 0.0), h - 1.0))

    prev = pts[0]
    for cur in pts[1:]:
        draw.line([clamp(prev), clamp(cur)], fill=color, width=line_width)
        prev = cur


def _classify_turn(final_lateral_m: float, threshold_m: float) -> str:
    if final_lateral_m > threshold_m:
        return "right"
    if final_lateral_m < -threshold_m:
        return "left"
    return "straight"


def build_plot(
    trajectories: List[Trajectory2D],
    out_path: Path,
    canvas_size: Tuple[int, int],
    margin: int,
    scale_px_per_meter: Optional[float],
    alpha: int,
    line_width: int,
    turn_threshold_m: float,
) -> None:
    img = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    scale = scale_px_per_meter if scale_px_per_meter is not None else _auto_scale(trajectories, canvas_size, margin)

    left = right = straight = 0
    n = len(trajectories)

    for i, t in enumerate(trajectories):
        color = _color_for_index(i, n, alpha=alpha)
        _draw_trajectory(
            draw,
            t,
            canvas_size=canvas_size,
            margin=margin,
            scale=scale,
            color=color,
            line_width=line_width,
        )

        final_lat = float(t.lateral[-1]) if t.lateral.size else 0.0
        cls = _classify_turn(final_lat, threshold_m=turn_threshold_m)
        if cls == "left":
            left += 1
        elif cls == "right":
            right += 1
        else:
            straight += 1

    # Annotation
    font = ImageFont.load_default()
    summary = (
        f"N={n}  left={left}  right={right}  straight={straight}  "
        f"thr={turn_threshold_m:.2f}m  scale={scale:.2f}px/m"
    )
    draw.rectangle([margin - 6, margin - 6, canvas_size[0] - margin + 6, margin + 16], fill=(255, 255, 255, 220))
    draw.text((margin, margin), summary, fill=(0, 0, 0, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)

    print(summary)
    print(f"[OK] wrote image -> {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Overlay ORAD-3D split trajectories on one canvas.")
    ap.add_argument("--split-dir", type=Path, default=Path("/data3/ORAD-3D/validation"))
    ap.add_argument(
        "--trajectory-key",
        type=str,
        default="trajectory_ins",
        choices=["trajectory_ins", "trajectory_hmi", "trajectory_ins_past", "trajectory_hmi_past"],
    )
    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true", help="Flip left/right sign if desired")
    ap.add_argument("--max-total", type=int, default=None, help="Optional cap for total trajectories")
    ap.add_argument("--max-per-seq", type=int, default=None, help="Optional cap per sequence")
    ap.add_argument("--out", type=Path, default=Path("/home/byounggun/LlamaFactory/orad3d_validation_paths.png"))
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--height", type=int, default=2000)
    ap.add_argument("--margin", type=int, default=60)
    ap.add_argument(
        "--scale-px-per-meter",
        type=float,
        default=None,
        help="If omitted, auto-scales to fit all trajectories.",
    )
    ap.add_argument("--alpha", type=int, default=70, help="Line alpha (0-255)")
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument(
        "--turn-threshold-m",
        type=float,
        default=1.0,
        help="Classify as left/right if final lateral offset exceeds this.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    trajs = list(
        iter_trajectories(
            split_dir=args.split_dir,
            key=args.trajectory_key,
            forward_axis=args.forward_axis,
            flip_lateral=bool(args.flip_lateral),
            max_total=args.max_total,
            max_per_sequence=args.max_per_seq,
        )
    )

    if not trajs:
        print(f"[WARN] no trajectories found under {args.split_dir}")
        return 0

    build_plot(
        trajectories=trajs,
        out_path=args.out,
        canvas_size=(int(args.width), int(args.height)),
        margin=int(args.margin),
        scale_px_per_meter=args.scale_px_per_meter,
        alpha=int(args.alpha),
        line_width=int(args.line_width),
        turn_threshold_m=float(args.turn_threshold_m),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
