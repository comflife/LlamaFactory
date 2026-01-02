#!/usr/bin/env python3
"""Export ORAD-3D frames whose scene_data text contains certain keywords.

Use-case: Quickly gather potentially risky scenes (e.g., people/rock/danger) and save
corresponding images for manual inspection.

Default behavior
- Scans split directory like /data3/ORAD-3D/validation
- Reads scene_data/<ts>.txt
- If any keyword matches (case-insensitive substring), exports image_data/<ts>.png
  (or optionally gt_image/<ts>_fillcolor.png) into an output folder.
- Writes a TSV manifest with matched keywords and short text snippet.

Example
python3 scripts/orad3d_export_keyword_scenes.py \
  --split-dir /data3/ORAD-3D/validation \
  --image-folder image_data \
  --keywords people person rock danger \
  --out-dir /home/byounggun/LlamaFactory/orad3d_val_keyword_scenes
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Match:
    seq: str
    ts: str
    keywords: Tuple[str, ...]
    scene_path: Path
    image_path: Path
    snippet: str


def _iter_sequence_dirs(split_dir: Path) -> Iterator[Path]:
    for child in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and not child.name.endswith(".zip"):
            yield child


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_keywords(keywords: Sequence[str]) -> Tuple[str, ...]:
    # Lowercase; keep unique stable order
    out: List[str] = []
    seen = set()
    for k in keywords:
        k2 = k.strip().lower()
        if not k2 or k2 in seen:
            continue
        seen.add(k2)
        out.append(k2)
    return tuple(out)


def _resolve_image(seq_dir: Path, ts: str, image_folder: str) -> Optional[Path]:
    if image_folder == "image_data":
        p = seq_dir / "image_data" / f"{ts}.png"
        return p if p.exists() else None
    if image_folder == "gt_image":
        p = seq_dir / "gt_image" / f"{ts}_fillcolor.png"
        return p if p.exists() else None

    # fallback
    p = seq_dir / "image_data" / f"{ts}.png"
    if p.exists():
        return p
    p = seq_dir / "gt_image" / f"{ts}_fillcolor.png"
    if p.exists():
        return p
    return None


def _match_keywords(text: str, keywords: Tuple[str, ...]) -> Tuple[str, ...]:
    t = text.lower()
    hits = [k for k in keywords if k in t]
    return tuple(hits)


def _make_snippet(text: str, hits: Tuple[str, ...], max_len: int) -> str:
    compact = " ".join(text.replace("\n", " ").split())
    if not compact:
        return ""

    # Try to center around first hit.
    if hits:
        idx = compact.lower().find(hits[0])
        if idx >= 0:
            start = max(0, idx - max_len // 2)
            end = min(len(compact), start + max_len)
            snippet = compact[start:end]
            return snippet

    return compact[:max_len]


def find_matches(
    split_dir: Path,
    image_folder: str,
    keywords: Tuple[str, ...],
    max_total: Optional[int],
    max_per_seq: Optional[int],
    snippet_len: int,
) -> List[Match]:
    matches: List[Match] = []

    for seq_dir in _iter_sequence_dirs(split_dir):
        scene_dir = seq_dir / "scene_data"
        if not scene_dir.is_dir():
            continue

        per = 0
        for scene_path in sorted(scene_dir.glob("*.txt"), key=lambda p: p.name):
            ts = scene_path.stem
            text = _read_text(scene_path)
            hits = _match_keywords(text, keywords)
            if not hits:
                continue

            img = _resolve_image(seq_dir, ts=ts, image_folder=image_folder)
            if img is None:
                continue

            snippet = _make_snippet(text, hits, max_len=snippet_len)
            matches.append(
                Match(
                    seq=seq_dir.name,
                    ts=ts,
                    keywords=hits,
                    scene_path=scene_path,
                    image_path=img,
                    snippet=snippet,
                )
            )

            per += 1
            if max_per_seq is not None and per >= max_per_seq:
                break
            if max_total is not None and len(matches) >= max_total:
                return matches

    return matches


def export_matches(matches: List[Match], out_dir: Path, copy_images: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "matches.tsv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("seq\ttimestamp\tkeywords\timage\tscene\tsnippet\n")
        for m in matches:
            rel_img = f"images/{m.seq}_{m.ts}{m.image_path.suffix.lower()}"
            f.write(
                f"{m.seq}\t{m.ts}\t{','.join(m.keywords)}\t{rel_img}\t{m.scene_path}\t{m.snippet.replace(chr(9),' ')}\n"
            )

            dst = img_dir / f"{m.seq}_{m.ts}{m.image_path.suffix.lower()}"
            if copy_images:
                shutil.copy2(m.image_path, dst)


def _annotate_and_save(src: Path, dst: Path, text: str) -> None:
    img = Image.open(src)

    # Use RGBA for semi-transparent banner.
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()
    # Best-effort text metrics across PIL versions.
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = (right - left), (bottom - top)
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)  # type: ignore[attr-defined]

    pad_x = 6
    pad_y = 4
    banner_h = min(base.size[1], text_h + pad_y * 2)

    # Banner across full width for readability.
    draw.rectangle([(0, 0), (base.size[0], banner_h)], fill=(0, 0, 0, 160))
    draw.text((pad_x, pad_y), text, font=font, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(base, overlay)
    # Preserve original mode as much as possible.
    if img.mode in {"RGB", "L"}:
        out = out.convert(img.mode)
    out.save(dst)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export ORAD-3D images whose scene_data contains keywords.")
    ap.add_argument("--split-dir", type=Path, default=Path("/data3/ORAD-3D/validation"))
    ap.add_argument(
        "--image-folder",
        type=str,
        default="image_data",
        choices=["image_data", "gt_image"],
        help="Which image folder to export from",
    )
    ap.add_argument(
        "--keywords",
        nargs="+",
        default=["people", "person", "rock", "danger"],
        help="Keywords to search (case-insensitive substring match)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/byounggun/LlamaFactory/orad3d_val_keyword_scenes"),
    )
    ap.add_argument("--max-total", type=int, default=None)
    ap.add_argument("--max-per-seq", type=int, default=None)
    ap.add_argument("--snippet-len", type=int, default=220)
    ap.add_argument("--no-copy", action="store_true", help="Only write manifest; do not copy images")
    ap.add_argument(
        "--annotate",
        action="store_true",
        help="If set, write matched keyword(s) at the top of exported images",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    keywords = _normalize_keywords(args.keywords)

    matches = find_matches(
        split_dir=args.split_dir,
        image_folder=str(args.image_folder),
        keywords=keywords,
        max_total=args.max_total,
        max_per_seq=args.max_per_seq,
        snippet_len=int(args.snippet_len),
    )

    export_matches(matches, out_dir=args.out_dir, copy_images=False if args.annotate else not bool(args.no_copy))

    if args.annotate and not bool(args.no_copy):
        img_dir = args.out_dir / "images"
        for m in matches:
            dst = img_dir / f"{m.seq}_{m.ts}{m.image_path.suffix.lower()}"
            text = f"matched: {', '.join(m.keywords)}"
            _annotate_and_save(m.image_path, dst, text=text)

    # Summary
    counts: Dict[str, int] = {k: 0 for k in keywords}
    for m in matches:
        for k in m.keywords:
            counts[k] += 1

    print(f"[OK] matched frames: {len(matches)} -> {args.out_dir}")
    for k in keywords:
        print(f"  {k}: {counts[k]}")

    if matches:
        print(f"[OK] manifest: {args.out_dir / 'matches.tsv'}")
        print(f"[OK] images  : {args.out_dir / 'images'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
