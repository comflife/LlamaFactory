#!/usr/bin/env python3
"""Plot ORAD-3D trajectories with Z on a single composite image.

Creates a 2-panel visualization on a white background:
- Left: Top-down view (lateral vs forward)
- Right: Elevation profile (z vs forward)

All trajectories from a split (e.g. /data3/ORAD-3D/validation) are overlaid.
Each trajectory gets a distinct color; the same color is used in both panels.

This is for dataset inspection (not a camera projection).
"""

from __future__ import annotations

import argparse
import colorsys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Trajectory3D:
    seq: str
    ts: str
    forward: np.ndarray
    lateral: np.ndarray
    z: np.ndarray


def _iter_sequence_dirs(split_dir: Path) -> Iterator[Path]:
    for child in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and not child.name.endswith(".zip"):
            yield child


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _extract_ts(name: str) -> str:
    return Path(name).stem


def _load_xyz(path: Path, key: str) -> List[Tuple[float, float, float]]:
    obj = _read_json(path)
    pts = obj.get(key)
    if not isinstance(pts, list) or not pts:
        return []

    out: List[Tuple[float, float, float]] = []
    for p in pts:
        if not isinstance(p, list) or len(p) < 3:
            continue
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def _to_forward_lateral_z(
    xyz: List[Tuple[float, float, float]],
    forward_axis: str,
    flip_lateral: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not xyz:
        z = np.zeros((0,), dtype=np.float64)
        return z, z, z

    x0, y0, z0 = xyz[0]
    xs = np.asarray([x - x0 for x, _, _ in xyz], dtype=np.float64)
    ys = np.asarray([y - y0 for _, y, _ in xyz], dtype=np.float64)
    zs = np.asarray([z - z0 for _, _, z in xyz], dtype=np.float64)

    if forward_axis == "y":
        forward = ys
        lateral = xs
    else:
        forward = xs
        lateral = ys

    if flip_lateral:
        lateral = -lateral

    return forward, lateral, zs


def iter_trajectories(
    split_dir: Path,
    key: str,
    forward_axis: str,
    flip_lateral: bool,
    max_total: Optional[int],
    max_per_sequence: Optional[int],
) -> Iterator[Trajectory3D]:
    emitted = 0
    for seq_dir in _iter_sequence_dirs(split_dir):
        local_dir = seq_dir / "local_path"
        if not local_dir.is_dir():
            continue

        per_seq = 0
        for p in sorted(local_dir.glob("*.json"), key=lambda q: q.name):
            xyz = _load_xyz(p, key=key)
            if len(xyz) < 2:
                continue

            forward, lateral, z = _to_forward_lateral_z(xyz, forward_axis=forward_axis, flip_lateral=flip_lateral)
            if forward.size < 2:
                continue

            yield Trajectory3D(seq=seq_dir.name, ts=_extract_ts(p.name), forward=forward, lateral=lateral, z=z)

            emitted += 1
            per_seq += 1

            if max_per_sequence is not None and per_seq >= max_per_sequence:
                break
            if max_total is not None and emitted >= max_total:
                return


def _color_for_index(i: int, n: int, alpha: int) -> Tuple[int, int, int, int]:
    h = (i / max(1, n)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.95, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), int(alpha))


def _auto_scale_xy(trajs: List[Trajectory3D], size: Tuple[int, int], margin: int) -> float:
    w, h = size
    max_forward = 0.0
    max_abs_lat = 0.0
    for t in trajs:
        if t.forward.size:
            max_forward = max(max_forward, float(np.max(t.forward)))
        if t.lateral.size:
            max_abs_lat = max(max_abs_lat, float(np.max(np.abs(t.lateral))))

    usable_h = max(1, h - 2 * margin)
    usable_w = max(1, w - 2 * margin)

    scale_h = (0.95 * usable_h / max_forward) if max_forward > 1e-6 else 10.0
    scale_w = (0.95 * (usable_w / 2.0) / max_abs_lat) if max_abs_lat > 1e-6 else scale_h
    return max(0.5, min(scale_h, scale_w))


def _auto_scale_fz(trajs: List[Trajectory3D], size: Tuple[int, int], margin: int) -> Tuple[float, float, float, float]:
    """Return (scale_f, scale_z, z_min, z_max)."""
    w, h = size
    max_forward = 0.0
    z_min = 0.0
    z_max = 0.0
    first = True

    for t in trajs:
        if t.forward.size:
            max_forward = max(max_forward, float(np.max(t.forward)))
        if t.z.size:
            if first:
                z_min = float(np.min(t.z))
                z_max = float(np.max(t.z))
                first = False
            else:
                z_min = min(z_min, float(np.min(t.z)))
                z_max = max(z_max, float(np.max(t.z)))

    usable_h = max(1, h - 2 * margin)
    usable_w = max(1, w - 2 * margin)

    scale_f = (0.95 * usable_w / max_forward) if max_forward > 1e-6 else 10.0
    z_range = max(1e-6, z_max - z_min)
    scale_z = (0.95 * usable_h / z_range)

    return max(0.5, scale_f), max(0.5, scale_z), z_min, z_max


def _draw_polyline(draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], size: Tuple[int, int], color, width: int) -> None:
    w, h = size

    def clamp(p: Tuple[float, float]) -> Tuple[float, float]:
        return (min(max(p[0], 0.0), w - 1.0), min(max(p[1], 0.0), h - 1.0))

    if len(pts) < 2:
        return

    prev = pts[0]
    for cur in pts[1:]:
        draw.line([clamp(prev), clamp(cur)], fill=color, width=width)
        prev = cur


def build_plot(
    trajs: List[Trajectory3D],
    out_path: Path,
    panel_size: Tuple[int, int],
    margin: int,
    scale_xy: Optional[float],
    scale_f: Optional[float],
    scale_z: Optional[float],
    alpha: int,
    line_width: int,
) -> None:
    pw, ph = panel_size

    canvas = Image.new("RGBA", (pw * 2, ph), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Left panel (XY)
    s_xy = scale_xy if scale_xy is not None else _auto_scale_xy(trajs, (pw, ph), margin)

    # Right panel (FZ)
    if scale_f is None or scale_z is None:
        s_f_auto, s_z_auto, z_min, z_max = _auto_scale_fz(trajs, (pw, ph), margin)
        s_f = s_f_auto if scale_f is None else scale_f
        s_z = s_z_auto if scale_z is None else scale_z
    else:
        s_f = scale_f
        s_z = scale_z
        # Need z_min/z_max for centering; recompute.
        _, _, z_min, z_max = _auto_scale_fz(trajs, (pw, ph), margin)

    # XY mapping: forward->up, lateral->left/right
    cx = pw / 2.0
    bottom = ph - margin

    # FZ mapping: forward->right, z->up (centered by z_min/z_max)
    left_x = pw + margin
    z_center = 0.5 * (z_min + z_max)
    z_mid_y = ph / 2.0

    n = len(trajs)
    for i, t in enumerate(trajs):
        color = _color_for_index(i, n, alpha=alpha)

        # Left: XY
        u = cx + (t.lateral * s_xy)
        v = bottom - (t.forward * s_xy)
        _draw_polyline(draw, list(zip(u.tolist(), v.tolist())), (pw, ph), color=color, width=line_width)

        # Right: FZ
        fx = left_x + (t.forward * s_f)
        zy = z_mid_y - ((t.z - z_center) * s_z)
        pts_fz = [(float(x), float(y)) for x, y in zip(fx.tolist(), zy.tolist())]
        # Offset panel clamp by shifting x back for clamp; draw directly (clamp uses full canvas).
        _draw_polyline(draw, pts_fz, (pw * 2, ph), color=color, width=line_width)

    # Panel separators + labels
    draw.line([(pw, 0), (pw, ph)], fill=(0, 0, 0, 80), width=2)

    font = ImageFont.load_default()
    draw.text((margin, margin), f"XY (lateral vs forward)  scale={s_xy:.2f}px/m", fill=(0, 0, 0, 255), font=font)
    draw.text(
        (pw + margin, margin),
        f"FZ (forward vs z)  scale_f={s_f:.2f}px/m  z_range=[{z_min:.2f},{z_max:.2f}]m",
        fill=(0, 0, 0, 255),
        font=font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)
    print(f"[OK] wrote image -> {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Overlay ORAD-3D split trajectories with Z on one canvas.")
    ap.add_argument("--split-dir", type=Path, default=Path("/data3/ORAD-3D/validation"))
    ap.add_argument(
        "--trajectory-key",
        type=str,
        default="trajectory_ins",
        choices=["trajectory_ins", "trajectory_hmi", "trajectory_ins_past", "trajectory_hmi_past"],
    )
    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true")
    ap.add_argument("--max-total", type=int, default=None)
    ap.add_argument("--max-per-seq", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("/home/byounggun/LlamaFactory/orad3d_validation_paths_xy_fz.png"))

    ap.add_argument("--panel-width", type=int, default=1400)
    ap.add_argument("--panel-height", type=int, default=900)
    ap.add_argument("--margin", type=int, default=60)

    ap.add_argument("--scale-xy", type=float, default=None, help="Optional fixed px/m for XY panel")
    ap.add_argument("--scale-f", type=float, default=None, help="Optional fixed px/m for forward axis in FZ panel")
    ap.add_argument("--scale-z", type=float, default=None, help="Optional fixed px/m for z axis in FZ panel")

    ap.add_argument("--alpha", type=int, default=70)
    ap.add_argument("--line-width", type=int, default=2)
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
        trajs=trajs,
        out_path=args.out,
        panel_size=(int(args.panel_width), int(args.panel_height)),
        margin=int(args.margin),
        scale_xy=args.scale_xy,
        scale_f=args.scale_f,
        scale_z=args.scale_z,
        alpha=int(args.alpha),
        line_width=int(args.line_width),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
