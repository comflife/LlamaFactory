#!/usr/bin/env python3
"""Visualize ORPO chosen/rejected pairs with XY + Z plots (offline JSONL).

Reads ORPO JSONL produced by scripts/orad3d_build_vlm_orpo.py and renders
composites showing:
  - image with chosen (green) + rejected (red) overlays
  - side panel with XY plots for chosen/rejected
  - bottom Z trend panel for chosen/rejected


python3 scripts/orad3d_visualize_orpo_pairs_withz.py \
  --jsonl /home/work/datasets/bg/orad3d_orpo2/orad3d_train_orpo.jsonl \
  --media-root /home/work/datasets/bg/ORAD-3D \
  --out-dir /home/work/byounggun/LlamaFactory/orad3d_orpo_pair_viz_z \
  --num-samples 20 \
  --seed 0

"""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


_TRAJ_TOKEN_RE = re.compile(r"<\s*trajectory\s*>", re.IGNORECASE)
_POINT_RE = re.compile(r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]")


@dataclass(frozen=True)
class OrpoSample:
    index: int
    image_path: Path
    scene_text: str
    chosen_text: str
    rejected_text: str
    chosen_points: List[List[float]]
    rejected_points: List[List[float]]
    meta: dict


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _extract_message_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join([p for p in parts if p])
    return str(content)


def _extract_assistant_text(value: object) -> str:
    if isinstance(value, dict):
        return _extract_message_text(value.get("content"))
    return _extract_message_text(value)


def _extract_trajectory_section(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TRAJ_TOKEN_RE.search(text)
    if not m:
        return None
    return text[m.end() :].strip() or ""


def _extract_trajectory_points(text: str) -> List[List[float]]:
    pts: List[List[float]] = []
    for m in _POINT_RE.finditer(text):
        try:
            x = float(m.group(1))
            y = float(m.group(2))
            z = float(m.group(3))
            pts.append([x, y, z])
        except Exception:
            continue
    return pts


def _split_text_and_traj(text: str) -> Tuple[str, List[List[float]]]:
    if not isinstance(text, str):
        return "", []
    m = _TRAJ_TOKEN_RE.search(text)
    if not m:
        return text.strip(), []
    scene_text = text[: m.start()].strip()
    traj_section = text[m.end() :].strip()
    points = _extract_trajectory_points(traj_section)
    return scene_text, points


def _resolve_image_path(image_value: str, media_root: Optional[Path]) -> Path:
    p = Path(image_value)
    if p.is_absolute() or media_root is None:
        return p
    return media_root / p


def _wrap_lines(text: str, width_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return ["(empty)"]
    lines: List[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=width_chars, break_long_words=True, replace_whitespace=False))
    return lines


def _traj_xyz_to_pixels(
    points_xyz: List[List[float]],
    image_size: Tuple[int, int],
    *,
    forward_axis: str,
    flip_lateral: bool,
    scale_px_per_meter: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], float]:
    if not points_xyz or len(points_xyz) < 2:
        return [], 0.0

    width, height = image_size
    xs = [float(p[0]) for p in points_xyz if isinstance(p, list) and len(p) >= 2]
    ys = [float(p[1]) for p in points_xyz if isinstance(p, list) and len(p) >= 2]
    if len(xs) < 2 or len(ys) < 2:
        return [], 0.0

    x0, y0 = xs[0], ys[0]
    dx = [x - x0 for x in xs]
    dy = [y - y0 for y in ys]

    if forward_axis == "y":
        forward = dy
        lateral = dx
    else:
        forward = dx
        lateral = dy

    if flip_lateral:
        lateral = [-v for v in lateral]

    if scale_px_per_meter is None:
        max_forward = max(forward) if forward else 0.0
        max_lateral = max((abs(v) for v in lateral), default=0.0)
        scale_h = (0.80 * height / max_forward) if max_forward > 1e-6 else 10.0
        scale_w = (0.45 * width / max_lateral) if max_lateral > 1e-6 else 10.0
        scale_px_per_meter = max(1.0, min(scale_h, scale_w))

    uvs: List[Tuple[float, float]] = []
    for f, lat in zip(forward, lateral):
        u = (width / 2.0) + (lat * scale_px_per_meter)
        v = (height - 1.0) - (f * scale_px_per_meter)
        uvs.append((float(u), float(v)))

    return uvs, float(scale_px_per_meter)


def _shared_traj_scale(
    image_size: Tuple[int, int],
    *,
    forward_axis: str,
    flip_lateral: bool,
    point_sets: Sequence[Optional[Sequence[Sequence[float]]]],
) -> Optional[float]:
    scales: List[float] = []
    for pts in point_sets:
        if not pts or len(pts) < 2:
            continue
        _, scale = _traj_xyz_to_pixels(
            list(pts),
            image_size,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
            scale_px_per_meter=None,
        )
        if scale > 0:
            scales.append(scale)
    return min(scales) if scales else None


def _draw_polyline(
    image: Image.Image,
    uv: List[Tuple[float, float]],
    *,
    color: Tuple[int, int, int],
    width: int,
) -> None:
    if len(uv) < 2:
        return
    draw = ImageDraw.Draw(image)
    w, h = image.size

    def clamp(p: Tuple[float, float]) -> Tuple[float, float]:
        return (min(max(p[0], 0.0), w - 1.0), min(max(p[1], 0.0), h - 1.0))

    prev = uv[0]
    for cur in uv[1:]:
        draw.line([clamp(prev), clamp(cur)], fill=color, width=width)
        prev = cur


def _draw_trajectory_plot(
    *,
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    points_xyz: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
    color: Tuple[int, int, int],
    fixed_scale: Optional[float] = None,
) -> None:
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)

    if len(points_xyz) < 2:
        draw.text((x0 + 8, y0 + 8), "(no trajectory)", fill=(0, 0, 0))
        return

    xs = [p[0] for p in points_xyz]
    ys = [p[1] for p in points_xyz]
    x_base, y_base = xs[0], ys[0]
    xs = [x - x_base for x in xs]
    ys = [y - y_base for y in ys]

    if forward_axis == "y":
        forward = ys
        lateral = xs
    else:
        forward = xs
        lateral = ys

    if flip_lateral:
        lateral = [-v for v in lateral]

    margin = 10
    usable_w = max(1, w - 2 * margin)
    usable_h = max(1, h - 2 * margin)

    if fixed_scale is None:
        max_fwd = max(forward) if forward else 1.0
        max_lat = max(abs(v) for v in lateral) if lateral else 1.0
        max_fwd = max(max_fwd, 1e-3)
        max_lat = max(max_lat, 1e-3)
        scale = min(usable_h / max_fwd, (usable_w / 2.0) / max_lat)
        scale = max(0.5, min(scale, 200.0))
    else:
        scale = float(fixed_scale)

    cx = x0 + w / 2.0
    bottom = y0 + h - margin
    pts: List[Tuple[float, float]] = []
    for f, lat in zip(forward, lateral):
        px = cx + lat * scale
        py = bottom - f * scale
        pts.append((px, py))

    draw.line([(cx, y0 + margin), (cx, y0 + h - margin)], fill=(220, 220, 220), width=1)
    draw.line([(x0 + margin, bottom), (x0 + w - margin, bottom)], fill=(220, 220, 220), width=1)

    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill=color, width=3)
    for p in pts:
        r = 3
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)

    draw.text((x0 + 8, y0 + 6), f"N={len(points_xyz)}  scale={scale:.1f}px/m", fill=(0, 0, 0))


def _max_forward_lateral(
    points_xyz: List[List[float]],
    *,
    forward_axis: str,
    flip_lateral: bool,
) -> Tuple[float, float]:
    if len(points_xyz) < 2:
        return 0.0, 0.0

    xs = [p[0] for p in points_xyz]
    ys = [p[1] for p in points_xyz]
    x_base, y_base = xs[0], ys[0]
    xs = [x - x_base for x in xs]
    ys = [y - y_base for y in ys]

    if forward_axis == "y":
        forward = ys
        lateral = xs
    else:
        forward = xs
        lateral = ys

    if flip_lateral:
        lateral = [-v for v in lateral]

    max_fwd = max(forward) if forward else 0.0
    max_lat = max(abs(v) for v in lateral) if lateral else 0.0
    return max_fwd, max_lat


def _shared_plot_scale(
    box: Tuple[int, int, int, int],
    chosen_points: List[List[float]],
    rejected_points: List[List[float]],
    *,
    forward_axis: str,
    flip_lateral: bool,
) -> Optional[float]:
    max_fwd_c, max_lat_c = _max_forward_lateral(
        chosen_points, forward_axis=forward_axis, flip_lateral=flip_lateral
    )
    max_fwd_r, max_lat_r = _max_forward_lateral(
        rejected_points, forward_axis=forward_axis, flip_lateral=flip_lateral
    )
    max_fwd = max(max_fwd_c, max_fwd_r, 1e-3)
    max_lat = max(max_lat_c, max_lat_r, 1e-3)

    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    margin = 10
    usable_w = max(1, w - 2 * margin)
    usable_h = max(1, h - 2 * margin)
    scale = min(usable_h / max_fwd, (usable_w / 2.0) / max_lat)
    return max(0.5, min(scale, 200.0))


def _trajectory_z_diff(
    points_a: List[List[float]],
    points_b: List[List[float]],
) -> Optional[float]:
    if not points_a or not points_b:
        return None
    n = min(len(points_a), len(points_b))
    if n < 2:
        return None
    diffs: List[float] = []
    for pa, pb in zip(points_a[:n], points_b[:n]):
        if len(pa) < 3 or len(pb) < 3:
            continue
        diffs.append(abs(float(pa[2]) - float(pb[2])))
    if not diffs:
        return None
    return float(sum(diffs) / len(diffs))


def _render_overlay(
    *,
    image: Image.Image,
    header: str,
    chosen_points: List[List[float]],
    rejected_points: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
    line_width: int,
) -> Image.Image:
    base = image.convert("RGB")
    overlay = base.copy()

    shared_scale = _shared_traj_scale(
        overlay.size,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        point_sets=[chosen_points, rejected_points],
    )

    uv_rej, scale = _traj_xyz_to_pixels(
        rejected_points,
        overlay.size,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        scale_px_per_meter=shared_scale,
    )
    _draw_polyline(overlay, uv_rej, color=(220, 20, 60), width=line_width)

    uv_ch, scale = _traj_xyz_to_pixels(
        chosen_points,
        overlay.size,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        scale_px_per_meter=shared_scale,
    )
    _draw_polyline(overlay, uv_ch, color=(0, 180, 0), width=line_width)

    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    bar_h = 24
    draw.rectangle([(0, 0), (overlay.size[0], bar_h)], fill=(20, 20, 20))
    draw.text(
        (10, 6),
        f"{header}  chosen:N={len(chosen_points)}  rejected:N={len(rejected_points)}  scale={scale:.1f}px/m",
        fill=(255, 255, 255),
        font=font,
    )
    return overlay


def _render_orpo_panel(
    *,
    size: Tuple[int, int],
    header: str,
    scene_text: str,
    meta_lines: Sequence[str],
    chosen_points: List[List[float]],
    rejected_points: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    pad = 12
    header_h = 26
    draw.rectangle([(0, 0), (w, header_h)], fill=(245, 245, 245))
    draw.text((pad, 6), header, fill=(0, 0, 0), font=font)

    y = header_h + 8
    if meta_lines:
        for line in meta_lines:
            draw.text((pad, y), line, fill=(70, 70, 70), font=font)
            y += 14
        y += 4

    text_max_h = max(70, int(h * 0.18))
    max_chars = max(30, (w - 2 * pad) // 6)
    for line in _wrap_lines(scene_text, width_chars=max_chars):
        if y > header_h + text_max_h:
            draw.text((pad, y), "...", fill=(0, 0, 0), font=font)
            y += 16
            break
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += 14

    plot_top = y + pad
    plot_h = h - plot_top - pad
    plot_w = (w - 3 * pad) // 2
    left_box = (pad, plot_top, pad + plot_w, plot_top + plot_h)
    right_box = (pad * 2 + plot_w, plot_top, pad * 2 + plot_w * 2, plot_top + plot_h)

    shared_scale = _shared_plot_scale(
        left_box,
        chosen_points,
        rejected_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )

    draw.text((left_box[0], plot_top - 16), "CHOSEN", fill=(0, 120, 0), font=font)
    draw.text((right_box[0], plot_top - 16), "REJECTED", fill=(220, 20, 60), font=font)

    _draw_trajectory_plot(
        draw=draw,
        box=left_box,
        points_xyz=chosen_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        color=(0, 160, 0),
        fixed_scale=shared_scale,
    )
    _draw_trajectory_plot(
        draw=draw,
        box=right_box,
        points_xyz=rejected_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        color=(220, 20, 60),
        fixed_scale=shared_scale,
    )

    return img


def _render_z_trend_panel(
    *,
    size: Tuple[int, int],
    chosen_points: List[List[float]],
    rejected_points: List[List[float]],
) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    pad_left = 46
    pad_right = 16
    pad_top = 22
    pad_bottom = 26

    plot_box = (pad_left, pad_top, w - pad_right, h - pad_bottom)
    x0, y0, x1, y1 = plot_box
    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)
    draw.text((12, 4), "Z trend (relative)", fill=(0, 0, 0), font=font)

    def _z_series(points: List[List[float]]) -> List[float]:
        if not points or len(points) < 2:
            return []
        z0 = float(points[0][2]) if len(points[0]) > 2 else 0.0
        out: List[float] = []
        for p in points:
            if len(p) < 3:
                out.append(0.0)
            else:
                out.append(float(p[2]) - z0)
        return out

    chosen_z = _z_series(chosen_points)
    rejected_z = _z_series(rejected_points)
    all_z = chosen_z + rejected_z
    if all_z:
        z_min = min(all_z)
        z_max = max(all_z)
    else:
        z_min, z_max = -1.0, 1.0

    if abs(z_max - z_min) < 1e-6:
        z_min -= 1.0
        z_max += 1.0

    usable_w = max(1, x1 - x0)
    usable_h = max(1, y1 - y0)

    def _to_xy(series: List[float]) -> List[Tuple[float, float]]:
        if not series:
            return []
        n = len(series)
        if n == 1:
            xs = [x0 + usable_w / 2.0]
        else:
            xs = [x0 + (i * usable_w / (n - 1)) for i in range(n)]
        ys = [y1 - ((z - z_min) / (z_max - z_min)) * usable_h for z in series]
        return list(zip(xs, ys))

    if z_min <= 0.0 <= z_max:
        y_zero = y1 - ((0.0 - z_min) / (z_max - z_min)) * usable_h
        draw.line([(x0, y_zero), (x1, y_zero)], fill=(220, 220, 220), width=1)

    def _draw_series(points: List[Tuple[float, float]], color: Tuple[int, int, int]) -> None:
        if len(points) < 2:
            return
        for a, b in zip(points[:-1], points[1:]):
            draw.line([a, b], fill=color, width=2)
        for p in points:
            r = 2
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)

    _draw_series(_to_xy(chosen_z), color=(0, 160, 0))
    _draw_series(_to_xy(rejected_z), color=(220, 20, 60))

    draw.text((x0, y1 + 4), f"CHOSEN N={len(chosen_z)}", fill=(0, 120, 0), font=font)
    draw.text((x0 + 140, y1 + 4), f"REJECTED N={len(rejected_z)}", fill=(220, 20, 60), font=font)

    return img


def _make_composite(
    *,
    image: Image.Image,
    header: str,
    scene_text: str,
    meta_lines: Sequence[str],
    chosen_points: List[List[float]],
    rejected_points: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
    panel_width: int,
    line_width: int,
) -> Image.Image:
    overlay = _render_overlay(
        image=image,
        header=header,
        chosen_points=chosen_points,
        rejected_points=rejected_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        line_width=line_width,
    )

    panel = _render_orpo_panel(
        size=(panel_width, overlay.size[1]),
        header="ORPO pair",
        scene_text=scene_text,
        meta_lines=meta_lines,
        chosen_points=chosen_points,
        rejected_points=rejected_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )

    top = Image.new("RGB", (overlay.size[0] + panel.size[0], overlay.size[1]), (255, 255, 255))
    top.paste(overlay, (0, 0))
    top.paste(panel, (overlay.size[0], 0))

    z_panel_h = max(140, overlay.size[1] // 4)
    z_panel = _render_z_trend_panel(
        size=(top.size[0], z_panel_h),
        chosen_points=chosen_points,
        rejected_points=rejected_points,
    )

    out = Image.new("RGB", (top.size[0], top.size[1] + z_panel.size[1]), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(z_panel, (0, top.size[1]))
    return out


def _parse_orpo_record(
    *,
    index: int,
    obj: dict,
    media_root: Optional[Path],
) -> Optional[OrpoSample]:
    images = obj.get("images")
    if not isinstance(images, list) or not images:
        return None

    chosen_text = _extract_assistant_text(obj.get("chosen"))
    rejected_text = _extract_assistant_text(obj.get("rejected"))
    if not chosen_text or not rejected_text:
        return None

    scene_text, chosen_points = _split_text_and_traj(chosen_text)
    _scene_rej, rejected_points = _split_text_and_traj(rejected_text)
    if len(chosen_points) < 2 or len(rejected_points) < 2:
        return None

    image_path = _resolve_image_path(str(images[0]), media_root)
    if not image_path.is_file():
        return None

    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    return OrpoSample(
        index=index,
        image_path=image_path,
        scene_text=scene_text,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
        chosen_points=chosen_points,
        rejected_points=rejected_points,
        meta=meta,
    )


def _load_samples(
    *,
    jsonl_path: Path,
    media_root: Optional[Path],
    num_samples: int,
    seed: int,
) -> List[OrpoSample]:
    random.seed(seed)
    if num_samples <= 0:
        out: List[OrpoSample] = []
        for idx, obj in enumerate(_iter_jsonl(jsonl_path)):
            sample = _parse_orpo_record(index=idx, obj=obj, media_root=media_root)
            if sample is not None:
                out.append(sample)
        return out

    reservoir: List[OrpoSample] = []
    seen = 0
    for idx, obj in enumerate(_iter_jsonl(jsonl_path)):
        sample = _parse_orpo_record(index=idx, obj=obj, media_root=media_root)
        if sample is None:
            continue
        seen += 1
        if len(reservoir) < num_samples:
            reservoir.append(sample)
            continue
        j = random.randint(0, seen - 1)
        if j < num_samples:
            reservoir[j] = sample
    return reservoir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize ORPO chosen/rejected pairs with Z plots.")
    ap.add_argument("--jsonl", type=Path, required=True, help="ORPO training JSONL with chosen/rejected")
    ap.add_argument("--media-root", type=Path, default=None, help="Base directory for relative image paths")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=20, help="How many samples to render (<=0 means all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true")
    ap.add_argument("--panel-width", type=int, default=560)
    ap.add_argument("--line-width", type=int, default=4)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    samples = _load_samples(
        jsonl_path=args.jsonl,
        media_root=args.media_root,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
    )
    if not samples:
        raise SystemExit(f"No valid ORPO pairs found in: {args.jsonl}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, s in enumerate(samples, start=1):
        try:
            image = Image.open(s.image_path)
        except Exception:
            continue

        seq = str(s.meta.get("sequence", "")).strip()
        ts = str(s.meta.get("timestamp", "")).strip()
        header = f"{seq} / {ts}".strip(" /") or f"index={s.index}"

        meta_lines: List[str] = []
        if "neg_z_diff" in s.meta and s.meta.get("neg_z_diff") is not None:
            meta_lines.append(f"neg_z_diff={s.meta.get('neg_z_diff')}")
        else:
            z_diff = _trajectory_z_diff(s.chosen_points, s.rejected_points)
            if z_diff is not None:
                meta_lines.append(f"neg_z_diff={z_diff:.6f}")
        if "neg_traj_sim" in s.meta and s.meta.get("neg_traj_sim") is not None:
            meta_lines.append(f"neg_traj_sim={s.meta.get('neg_traj_sim')}")

        comp = _make_composite(
            image=image,
            header=header,
            scene_text=s.scene_text,
            meta_lines=meta_lines,
            chosen_points=s.chosen_points,
            rejected_points=s.rejected_points,
            forward_axis=str(args.forward_axis),
            flip_lateral=bool(args.flip_lateral),
            panel_width=int(args.panel_width),
            line_width=int(args.line_width),
        )

        stem = f"{i:04d}"
        if seq or ts:
            stem = f"{stem}_{seq}_{ts}".strip("_")
        out_path = args.out_dir / f"{stem}.png"
        comp.save(out_path)
        saved += 1

    print(f"[OK] wrote {saved} visualizations -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
