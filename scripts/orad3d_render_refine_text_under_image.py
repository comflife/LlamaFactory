#!/usr/bin/env python3
"""Render ORAD-3D raw/refined scene text under each image and save the result.

Expected data layout (per split/sequence):
  <split>/<sequence>/{image_data,scene_data,scene_data_refine}/<timestamp>.*

Example:
  python scripts/orad3d_render_refine_text_under_image.py \
    --orad-root /home/work/datasets/bg/ORAD-3D \
    --split validation \
    --out-dir orad3d_text_compare
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SamplePaths:
    split: str
    sequence: str
    timestamp: str
    image_path: Path
    scene_path: Path
    refine_path: Optional[Path]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _iter_sequence_dirs(split_dir: Path) -> Iterable[Path]:
    for child in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and not child.name.endswith(".zip"):
            yield child


def _iter_samples(
    *,
    orad_root: Path,
    split: str,
    image_folder: str,
    scene_folder: str,
    refine_folder: str,
    require_refine: bool,
) -> Iterable[SamplePaths]:
    split_dir = orad_root / split
    if not split_dir.is_dir():
        raise SystemExit(f"Split dir not found: {split_dir}")

    for seq_dir in _iter_sequence_dirs(split_dir):
        image_dir = seq_dir / image_folder
        scene_dir = seq_dir / scene_folder
        refine_dir = seq_dir / refine_folder
        if not image_dir.is_dir() or not scene_dir.is_dir():
            continue
        for img_path in sorted(image_dir.glob("*.png"), key=lambda p: p.name):
            ts = img_path.stem
            scene_path = scene_dir / f"{ts}.txt"
            refine_path = refine_dir / f"{ts}.txt"
            if not scene_path.exists():
                continue
            if require_refine and not refine_path.exists():
                continue
            yield SamplePaths(
                split=split,
                sequence=seq_dir.name,
                timestamp=ts,
                image_path=img_path,
                scene_path=scene_path,
                refine_path=refine_path if refine_path.exists() else None,
            )


def _select_samples(items: List[SamplePaths], num_samples: int, seed: int) -> List[SamplePaths]:
    if num_samples <= 0 or len(items) <= num_samples:
        return items
    rng = random.Random(seed)
    return rng.sample(items, k=num_samples)


def _load_font(font_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if hasattr(draw, "textlength"):
        return float(draw.textlength(text, font=font))
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    return float(font.getsize(text)[0])


def _line_height(font: ImageFont.ImageFont) -> int:
    if hasattr(font, "getbbox"):
        bbox = font.getbbox("Ag")
        return int(bbox[3] - bbox[1])
    return int(font.getsize("Ag")[1])


def _wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if not text:
        return [""]
    lines: List[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            lines.append("")
            continue
        words = raw.split()
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if _text_width(draw, trial, font) <= max_width:
                current = trial
                continue
            if current:
                lines.append(current)
            if _text_width(draw, word, font) <= max_width:
                current = word
                continue
            chunk = ""
            for ch in word:
                trial_chunk = f"{chunk}{ch}"
                if _text_width(draw, trial_chunk, font) <= max_width or not chunk:
                    chunk = trial_chunk
                else:
                    lines.append(chunk)
                    chunk = ch
            current = chunk
        if current:
            lines.append(current)
    return lines


def _render_text_below_image(
    *,
    image: Image.Image,
    raw_text: str,
    refined_text: Optional[str],
    font: ImageFont.ImageFont,
    padding: int,
    line_spacing: int,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
) -> Image.Image:
    base = image.convert("RGB")
    draw = ImageDraw.Draw(base)
    max_width = max(1, base.width - (2 * padding))

    lines: List[str] = []
    lines.extend(_wrap_text("Raw text:", draw, font, max_width))
    lines.extend(_wrap_text(raw_text, draw, font, max_width))
    lines.append("")
    lines.extend(_wrap_text("Refined text:", draw, font, max_width))
    if refined_text:
        lines.extend(_wrap_text(refined_text, draw, font, max_width))
    else:
        lines.extend(_wrap_text("(missing)", draw, font, max_width))

    lh = _line_height(font)
    step = lh + max(0, line_spacing)
    text_height = (len(lines) * step - max(0, line_spacing)) if lines else lh
    text_height += 2 * padding

    canvas = Image.new("RGB", (base.width, base.height + text_height), bg_color)
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = base.height + padding
    x = padding
    for line in lines:
        draw.text((x, y), line, fill=text_color, font=font)
        y += step
    return canvas


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render raw/refined ORAD-3D scene text under each image.")
    ap.add_argument("--orad-root", type=Path, required=True, help="Root of ORAD-3D dataset.")
    ap.add_argument("--split", default="validation", help="Split name (train/validation/test).")
    ap.add_argument("--image-folder", default="image_data", help="Image folder name.")
    ap.add_argument("--scene-folder", default="scene_data", help="Raw scene text folder name.")
    ap.add_argument("--refine-folder", default="scene_data_refine", help="Refined scene text folder name.")
    ap.add_argument("--require-refine", action="store_true", help="Skip samples without refined text.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for rendered images.")
    ap.add_argument("--num-samples", type=int, default=0, help="Randomly sample N frames (0 = all).")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for sampling.")
    ap.add_argument("--font-path", default=None, help="Optional .ttf font path.")
    ap.add_argument("--font-size", type=int, default=16, help="Font size when --font-path is set.")
    ap.add_argument("--padding", type=int, default=12, help="Padding around text block.")
    ap.add_argument("--line-spacing", type=int, default=4, help="Extra spacing between lines.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = list(
        _iter_samples(
            orad_root=args.orad_root,
            split=args.split,
            image_folder=args.image_folder,
            scene_folder=args.scene_folder,
            refine_folder=args.refine_folder,
            require_refine=bool(args.require_refine),
        )
    )
    if not items:
        raise SystemExit("No samples found; check --orad-root and folder names.")

    samples = _select_samples(items, num_samples=int(args.num_samples), seed=int(args.seed))
    font = _load_font(args.font_path, int(args.font_size))

    for item in samples:
        image = Image.open(item.image_path)
        raw_text = _read_text(item.scene_path)
        refined_text = _read_text(item.refine_path) if item.refine_path else None
        rendered = _render_text_below_image(
            image=image,
            raw_text=raw_text,
            refined_text=refined_text,
            font=font,
            padding=int(args.padding),
            line_spacing=int(args.line_spacing),
            text_color=(0, 0, 0),
            bg_color=(255, 255, 255),
        )
        out_name = f"{item.sequence}_{item.timestamp}.png"
        rendered.save(args.out_dir / out_name)

    print(f"Saved {len(samples)} images to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
