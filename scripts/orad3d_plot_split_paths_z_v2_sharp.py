#!/usr/bin/env python3
"""Plot only "sharp" ORAD-3D trajectories (XY turn + FZ pitch change).

This is a v2 of scripts/orad3d_plot_split_paths_z.py.

What it does
- Scans a split directory (e.g. /data3/ORAD-3D/validation)
- Reads local_path/*.json trajectories (default: trajectory_ins)
- Converts to ego-relative forward/lateral/z (subtract the first point)
- Detects "sharp" trajectories by:
  - XY: max change of heading angle between consecutive segments
  - FZ: max change of pitch angle between consecutive segments (z vs forward)
- Overlays ONLY the sharp trajectories on a single composite image:
  - Left panel: XY (lateral vs forward)
  - Right panel: FZ (forward vs z)

This is for dataset inspection (not camera projection).

python /home/byounggun/LlamaFactory/scripts/orad3d_plot_split_paths_z_v2_sharp.py --split-dir /data3/ORAD-3D/validation --out /home/byounggun/LlamaFactory/orad3d_validation_paths_xy_fz_sharp.png --out-list /home/byounggun/LlamaFactory/orad3d_validation_sharp_list.txt --xy-dtheta-deg 40 --fz-dtheta-deg 20
"""

from __future__ import annotations

import argparse
import colorsys
import json
import shutil
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

    # Empirically for ORAD-3D local_path, forward often aligns with the 2nd value.
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
            if len(xyz) < 3:
                continue

            forward, lateral, z = _to_forward_lateral_z(xyz, forward_axis=forward_axis, flip_lateral=flip_lateral)
            if forward.size < 3:
                continue

            yield Trajectory3D(seq=seq_dir.name, ts=_extract_ts(p.name), forward=forward, lateral=lateral, z=z)

            emitted += 1
            per_seq += 1

            if max_per_sequence is not None and per_seq >= max_per_sequence:
                break
            if max_total is not None and emitted >= max_total:
                return


def _unwrap_angle(a: np.ndarray) -> np.ndarray:
    return np.unwrap(a)


def _max_delta_angle_deg(angle_rad: np.ndarray) -> float:
    if angle_rad.size < 2:
        return 0.0
    a = _unwrap_angle(angle_rad)
    d = np.diff(a)
    return float(np.max(np.abs(np.degrees(d)))) if d.size else 0.0


def score_sharpness(
    t: Trajectory3D,
    min_step_forward_m: float,
) -> Tuple[float, float]:
    """Returns (max_dheading_deg, max_dpitch_deg)."""

    df = np.diff(t.forward)
    dl = np.diff(t.lateral)
    dz = np.diff(t.z)

    # Filter segments with almost no forward progress (unstable angles).
    ok = np.abs(df) >= float(min_step_forward_m)
    if not np.any(ok):
        return 0.0, 0.0

    df = df[ok]
    dl = dl[ok]
    dz = dz[ok]

    heading = np.arctan2(dl, df)  # radians
    pitch = np.arctan2(dz, df)  # radians

    return _max_delta_angle_deg(heading), _max_delta_angle_deg(pitch)


def filter_sharp(
    trajs: List[Trajectory3D],
    xy_dtheta_deg: float,
    fz_dtheta_deg: float,
    min_step_forward_m: float,
) -> Tuple[List[Trajectory3D], List[Tuple[str, str, float, float]]]:
    kept: List[Trajectory3D] = []
    meta: List[Tuple[str, str, float, float]] = []

    for t in trajs:
        dh, dp = score_sharpness(t, min_step_forward_m=min_step_forward_m)
        if dh >= float(xy_dtheta_deg) or dp >= float(fz_dtheta_deg):
            kept.append(t)
            meta.append((t.seq, t.ts, dh, dp))

    # Sort by most sharp first (use max of two)
    meta_sorted = sorted(meta, key=lambda x: max(x[2], x[3]), reverse=True)
    kept_sorted: List[Trajectory3D] = []
    order = {(seq, ts): i for i, (seq, ts, _, _) in enumerate(meta_sorted)}
    kept_sorted = sorted(kept, key=lambda t: order.get((t.seq, t.ts), 10**9))

    return kept_sorted, meta_sorted


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
    title: str,
) -> None:
    pw, ph = panel_size

    canvas = Image.new("RGBA", (pw * 2, ph), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    s_xy = scale_xy if scale_xy is not None else _auto_scale_xy(trajs, (pw, ph), margin)

    if scale_f is None or scale_z is None:
        s_f_auto, s_z_auto, z_min, z_max = _auto_scale_fz(trajs, (pw, ph), margin)
        s_f = s_f_auto if scale_f is None else scale_f
        s_z = s_z_auto if scale_z is None else scale_z
    else:
        s_f = scale_f
        s_z = scale_z
        _, _, z_min, z_max = _auto_scale_fz(trajs, (pw, ph), margin)

    cx = pw / 2.0
    bottom = ph - margin

    left_x = pw + margin
    z_center = 0.5 * (z_min + z_max)
    z_mid_y = ph / 2.0

    n = len(trajs)
    for i, t in enumerate(trajs):
        color = _color_for_index(i, n, alpha=alpha)

        u = cx + (t.lateral * s_xy)
        v = bottom - (t.forward * s_xy)
        _draw_polyline(draw, list(zip(u.tolist(), v.tolist())), (pw, ph), color=color, width=line_width)

        fx = left_x + (t.forward * s_f)
        zy = z_mid_y - ((t.z - z_center) * s_z)
        pts_fz = [(float(x), float(y)) for x, y in zip(fx.tolist(), zy.tolist())]
        _draw_polyline(draw, pts_fz, (pw * 2, ph), color=color, width=line_width)

    draw.line([(pw, 0), (pw, ph)], fill=(0, 0, 0, 80), width=2)

    font = ImageFont.load_default()
    draw.text((margin, margin), f"XY (lat vs fwd) scale={s_xy:.2f}px/m", fill=(0, 0, 0, 255), font=font)
    draw.text(
        (pw + margin, margin),
        f"FZ (fwd vs z) scale_f={s_f:.2f}px/m  z_range=[{z_min:.2f},{z_max:.2f}]m",
        fill=(0, 0, 0, 255),
        font=font,
    )
    draw.text((margin, margin + 16), title, fill=(0, 0, 0, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)


def _resolve_image_path(split_dir: Path, seq: str, ts: str, image_folder: str) -> Optional[Path]:
    seq_dir = split_dir / seq
    if image_folder == "image_data":
        p = seq_dir / "image_data" / f"{ts}.png"
        if p.exists():
            return p
    elif image_folder == "gt_image":
        p = seq_dir / "gt_image" / f"{ts}_fillcolor.png"
        if p.exists():
            return p

    # Fallbacks: try both conventions.
    p = seq_dir / "image_data" / f"{ts}.png"
    if p.exists():
        return p
    p = seq_dir / "gt_image" / f"{ts}_fillcolor.png"
    if p.exists():
        return p

    # Last resort: any file starting with timestamp in image folders.
    for folder in ("image_data", "gt_image"):
        d = seq_dir / folder
        if not d.is_dir():
            continue
        matches = sorted(d.glob(f"{ts}*"), key=lambda x: x.name)
        if matches:
            return matches[0]
    return None


def export_sharp_images(
    split_dir: Path,
    meta: List[Tuple[str, str, float, float]],
    xy_dtheta_deg: float,
    fz_dtheta_deg: float,
    export_root: Path,
    image_folder: str,
) -> Tuple[int, int, int]:
    """Export images into xy_sharp/ and z_sharp/.

    Returns (xy_count, z_count, missing_count).
    If a frame is both xy-sharp and z-sharp, it is copied into both folders.
    """

    xy_dir = export_root / "xy_sharp"
    z_dir = export_root / "z_sharp"
    xy_dir.mkdir(parents=True, exist_ok=True)
    z_dir.mkdir(parents=True, exist_ok=True)

    xy_count = 0
    z_count = 0
    missing = 0

    xy_list = []
    z_list = []

    for seq, ts, dh, dp in meta:
        is_xy = dh >= float(xy_dtheta_deg)
        is_z = dp >= float(fz_dtheta_deg)
        if not (is_xy or is_z):
            continue

        src = _resolve_image_path(split_dir, seq=seq, ts=ts, image_folder=image_folder)
        if src is None or not src.exists():
            missing += 1
            continue

        ext = src.suffix.lower() if src.suffix else ".png"
        dst_name = f"{seq}_{ts}{ext}"

        if is_xy:
            shutil.copy2(src, xy_dir / dst_name)
            xy_count += 1
            xy_list.append((seq, ts, dh, dp))
        if is_z:
            shutil.copy2(src, z_dir / dst_name)
            z_count += 1
            z_list.append((seq, ts, dh, dp))

    # Write per-folder lists
    def write_list(path: Path, rows: List[Tuple[str, str, float, float]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write("seq\ttimestamp\tmax_dheading_deg\tmax_dpitch_deg\n")
            for seq, ts, dh, dp in rows:
                f.write(f"{seq}\t{ts}\t{dh:.2f}\t{dp:.2f}\n")

    write_list(xy_dir / "sharp_xy_list.txt", xy_list)
    write_list(z_dir / "sharp_z_list.txt", z_list)

    return xy_count, z_count, missing


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Overlay only sharp ORAD-3D trajectories (XY+FZ).")
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

    ap.add_argument("--xy-dtheta-deg", type=float, default=30.0, help="Sharp XY if max heading change >= this")
    ap.add_argument("--fz-dtheta-deg", type=float, default=15.0, help="Sharp FZ if max pitch change >= this")
    ap.add_argument(
        "--min-step-forward-m",
        type=float,
        default=0.05,
        help="Ignore segments with tiny forward step when computing angles",
    )

    ap.add_argument(
        "--image-folder",
        type=str,
        default="image_data",
        choices=["image_data", "gt_image"],
        help="Which image folder to export frames from",
    )
    ap.add_argument(
        "--export-root",
        type=Path,
        default=Path("/home/byounggun/LlamaFactory/orad3d_validation_sharp_frames"),
        help="Root folder to export sharp frame images into (xy_sharp/ and z_sharp/)",
    )
    ap.add_argument(
        "--export-images",
        action="store_true",
        help="If set, copy selected frames' images into --export-root",
    )

    ap.add_argument("--out", type=Path, default=Path("/home/byounggun/LlamaFactory/orad3d_validation_paths_xy_fz_sharp.png"))
    ap.add_argument("--out-list", type=Path, default=Path("/home/byounggun/LlamaFactory/orad3d_validation_sharp_list.txt"))

    ap.add_argument("--panel-width", type=int, default=1400)
    ap.add_argument("--panel-height", type=int, default=900)
    ap.add_argument("--margin", type=int, default=60)

    ap.add_argument("--scale-xy", type=float, default=None)
    ap.add_argument("--scale-f", type=float, default=None)
    ap.add_argument("--scale-z", type=float, default=None)

    ap.add_argument("--alpha", type=int, default=110)
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=None, help="If set, only plot the top-K sharpest")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    all_trajs = list(
        iter_trajectories(
            split_dir=args.split_dir,
            key=args.trajectory_key,
            forward_axis=args.forward_axis,
            flip_lateral=bool(args.flip_lateral),
            max_total=args.max_total,
            max_per_sequence=args.max_per_seq,
        )
    )

    if not all_trajs:
        print(f"[WARN] no trajectories found under {args.split_dir}")
        return 0

    kept, meta = filter_sharp(
        all_trajs,
        xy_dtheta_deg=float(args.xy_dtheta_deg),
        fz_dtheta_deg=float(args.fz_dtheta_deg),
        min_step_forward_m=float(args.min_step_forward_m),
    )

    if args.top_k is not None:
        topk = max(0, int(args.top_k))
        meta = meta[:topk]
        kept = kept[:topk]

    print(
        f"total={len(all_trajs)}  sharp={len(kept)}  "
        f"xy_dtheta>={args.xy_dtheta_deg:.1f}deg or fz_dtheta>={args.fz_dtheta_deg:.1f}deg"
    )

    # Write list
    args.out_list.parent.mkdir(parents=True, exist_ok=True)
    with args.out_list.open("w", encoding="utf-8") as f:
        f.write("seq\ttimestamp\tmax_dheading_deg\tmax_dpitch_deg\n")
        for seq, ts, dh, dp in meta:
            f.write(f"{seq}\t{ts}\t{dh:.2f}\t{dp:.2f}\n")

    if args.export_images:
        xy_count, z_count, missing = export_sharp_images(
            split_dir=args.split_dir,
            meta=meta,
            xy_dtheta_deg=float(args.xy_dtheta_deg),
            fz_dtheta_deg=float(args.fz_dtheta_deg),
            export_root=args.export_root,
            image_folder=str(args.image_folder),
        )
        print(
            f"[OK] exported frames -> {args.export_root}  "
            f"xy_sharp={xy_count}  z_sharp={z_count}  missing_images={missing}"
        )

    if not kept:
        print("[WARN] no sharp trajectories matched thresholds; try lowering thresholds.")
        return 0

    title = (
        f"sharp only: N={len(kept)}  xy>= {args.xy_dtheta_deg:.0f}deg  fz>= {args.fz_dtheta_deg:.0f}deg  "
        f"(forward_axis={args.forward_axis}, flip_lateral={bool(args.flip_lateral)})"
    )

    build_plot(
        trajs=kept,
        out_path=args.out,
        panel_size=(int(args.panel_width), int(args.panel_height)),
        margin=int(args.margin),
        scale_xy=args.scale_xy,
        scale_f=args.scale_f,
        scale_z=args.scale_z,
        alpha=int(args.alpha),
        line_width=int(args.line_width),
        title=title,
    )

    print(f"[OK] wrote image -> {args.out}")
    print(f"[OK] wrote list  -> {args.out_list}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
