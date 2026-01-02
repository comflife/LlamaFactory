#!/usr/bin/env python3

"""Create quick visualization samples for ORAD-3D.

For each sampled timestamp, this script:
- Loads the camera image from image_data/*.png.
- Draws the local_path trajectory as an ego-centric XY overlay:
    - The first trajectory point is treated as (0, 0).
    - X is forward (mapped upward from the bottom of the image).
    - Y is lateral (mapped left/right from the center of the image).
    - Z is ignored.
- Renders the corresponding scene_data text in a right-side panel.
- Saves a single composite image.

This is intended for quick dataset structure/quality inspection (not metric-accurate camera projection).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Sample:
    seq_name: str
    timestamp: str
    image_path: Path
    local_path_path: Path
    scene_path: Path


def load_trajectory_xy(local_path_path: Path) -> List[Tuple[float, float]]:
    """Loads ego trajectory and returns (x, y) pairs.

    The dataset stores trajectory as xyz; this function intentionally ignores z.
    """

    obj = json.loads(local_path_path.read_text(encoding="utf-8", errors="ignore"))
    pts = obj.get("trajectory_ins", [])
    out: List[Tuple[float, float]] = []
    for p in pts:
        if not isinstance(p, list) or len(p) < 2:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def ego_xy_to_pixels(
    xy: List[Tuple[float, float]],
    image_size: Tuple[int, int],
    scale_px_per_meter: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Maps ego-centric (x forward, y lateral) to image pixels.

    - Origin is bottom-center of the image.
    - The first trajectory point is treated as the origin (0,0).
    """

    if not xy:
        return []

    width, height = image_size
    # Empirically for ORAD-3D local_path trajectory_ins, the second value tends to
    # grow more consistently with motion; treat it as forward.
    x0, y0 = xy[0]
    lateral = np.asarray([x - x0 for x, _ in xy], dtype=np.float64)
    forward = np.asarray([y - y0 for _, y in xy], dtype=np.float64)

    if scale_px_per_meter is None:
        max_forward = float(np.max(forward)) if forward.size else 0.0
        max_lateral = float(np.max(np.abs(lateral))) if lateral.size else 0.0

        scale_h = (0.80 * height / max_forward) if max_forward > 1e-6 else 10.0
        scale_w = (0.45 * width / max_lateral) if max_lateral > 1e-6 else 10.0
        scale_px_per_meter = max(1.0, min(scale_h, scale_w))

    u = (width / 2.0) + (lateral * scale_px_per_meter)
    v = (height - 1.0) - (forward * scale_px_per_meter)
    return list(zip(u.tolist(), v.tolist()))


def draw_polyline(
    image: Image.Image,
    uv: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
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


def render_text_panel(
    text: str,
    panel_size: Tuple[int, int],
    title: str,
) -> Image.Image:
    panel_w, panel_h = panel_size
    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    # Use default font to avoid system font dependency.
    font = ImageFont.load_default()
    x0, y0 = 14, 12

    draw.text((x0, y0), title, fill=(0, 0, 0), font=font)
    y = y0 + 18

    # Wrap text by characters; good enough for quick inspection.
    wrapped = textwrap.fill(text.strip().replace("\n", " "), width=80)
    draw.text((x0, y), wrapped, fill=(0, 0, 0), font=font)
    return panel


def gather_samples(seq_dir: Path) -> List[Sample]:
    seq_name = seq_dir.name

    image_dir = seq_dir / "image_data"
    local_path_dir = seq_dir / "local_path"
    scene_dir = seq_dir / "scene_data"

    if not (image_dir.is_dir() and local_path_dir.is_dir() and scene_dir.is_dir()):
        return []

    # Keys are timestamps without extension (gt_image uses *_fillcolor.png).
    image_keys = {p.stem for p in image_dir.glob("*.png")}
    local_keys = {p.stem for p in local_path_dir.glob("*.json")}
    scene_keys = {p.stem for p in scene_dir.glob("*.txt")}

    keys = sorted(image_keys & local_keys & scene_keys)
    if not keys:
        return []

    samples: List[Sample] = []
    for ts in keys:
        samples.append(
            Sample(
                seq_name=seq_name,
                timestamp=ts,
                image_path=image_dir / f"{ts}.png",
                local_path_path=local_path_dir / f"{ts}.json",
                scene_path=scene_dir / f"{ts}.txt",
            )
        )

    return samples


def build_composite(sample: Sample, panel_width: int, scale_px_per_meter: Optional[float]) -> Image.Image:
    base = Image.open(sample.image_path).convert("RGB")
    w, h = base.size

    xy = load_trajectory_xy(sample.local_path_path)
    uv = ego_xy_to_pixels(xy, (w, h), scale_px_per_meter=scale_px_per_meter)

    overlay = base.copy()
    draw_polyline(overlay, uv)

    scene_text = sample.scene_path.read_text(encoding="utf-8", errors="ignore")
    title = f"{sample.seq_name} / {sample.timestamp}"
    panel = render_text_panel(scene_text, (panel_width, h), title=title)

    canvas = Image.new("RGB", (w + panel_width, h), (255, 255, 255))
    canvas.paste(overlay, (0, 0))
    canvas.paste(panel, (w, 0))
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/data3/ORAD-3D/validation"),
        help="Path containing ORAD-3D split folders or sequences.",
    )
    p.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Optional specific sequence folder name (e.g., x2021_0222_1745).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/byounggun/LlamaFactory/orad3d_samples"),
        help="Output directory for composite images.",
    )
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--panel_width", type=int, default=820)
    p.add_argument(
        "--scale_px_per_meter",
        type=float,
        default=None,
        help="Optional fixed scale for XY overlay. If omitted, auto-scales per sample.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root = args.dataset_root
    if args.sequence:
        seq_dirs = [root / args.sequence]
    else:
        seq_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.endswith(".zip")]

    all_samples: List[Sample] = []
    for sd in sorted(seq_dirs, key=lambda p: p.name):
        all_samples.extend(gather_samples(sd))

    if not all_samples:
        raise SystemExit(f"No samples found under {root} (sequence={args.sequence!r}).")

    k = min(args.num_samples, len(all_samples))
    picked = random.sample(all_samples, k=k)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(sorted(picked, key=lambda x: (x.seq_name, x.timestamp)), start=1):
        comp = build_composite(s, panel_width=args.panel_width, scale_px_per_meter=args.scale_px_per_meter)
        out = args.out_dir / f"{i:02d}_{s.seq_name}_{s.timestamp}.png"
        comp.save(out)

    print(f"Wrote {k} samples to: {args.out_dir}")


if __name__ == "__main__":
    main()
