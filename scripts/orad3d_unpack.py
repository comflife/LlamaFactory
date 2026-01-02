#!/usr/bin/env python3
"""Unzip and organize ORAD-3D sequences into split/sequence directories.

Expected input layout (source root):
  ORAD-3D/
    training/*.zip
    validation/*.zip
    testing/*.zip

Output layout (destination root):
  <dst>/
    training/<sequence>/(calib, sparse_depth, dense_depth, ...)
    validation/<sequence>/...
    testing/<sequence>/...

Notes:
- Idempotent by default: if a sequence already has a `.extracted_ok` marker, it is skipped.
- Safe extraction: prevents Zip Slip path traversal.
- Handles common zip structures:
  - zip contains top-level folder <sequence>/...
  - zip contains files/folders directly (calib/..., poses.txt, ...)
  - zip contains an extra wrapper folder (e.g., ORAD-3D/<sequence>/...)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


EXPECTED_TOP_LEVEL = [
    "calib",
    "sparse_depth",
    "dense_depth",
    "lidar_data",
    "local_path",
    "image_data",
    "occupancy",
    "gt_image",
    "gt_image_multi_seg",
    "scene_data",
    "poses.txt",
]


@dataclass(frozen=True)
class Options:
    src_root: Path
    dst_root: Path
    splits: tuple[str, ...]
    dry_run: bool
    force: bool
    keep_temp: bool
    limit: int | None


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _is_within_directory(base_dir: Path, target: Path) -> bool:
    try:
        base_dir_resolved = base_dir.resolve(strict=False)
        target_resolved = target.resolve(strict=False)
    except Exception:
        return False
    return os.path.commonpath([str(base_dir_resolved)]) == os.path.commonpath([str(base_dir_resolved), str(target_resolved)])


def _safe_extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Safely extract zip into extract_to, preventing path traversal."""
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = extract_to / member.filename
            if not _is_within_directory(extract_to, member_path):
                raise RuntimeError(f"Unsafe path in zip (Zip Slip): {member.filename}")
        zf.extractall(extract_to)


def _list_top_level_entries(dir_path: Path) -> list[Path]:
    entries = [p for p in dir_path.iterdir() if p.name not in ("__MACOSX",) and not p.name.startswith(".")]
    return sorted(entries, key=lambda p: p.name)


def _choose_payload_root(extracted_root: Path, sequence_name: str) -> Path:
    """Choose the directory that actually contains the sequence payload."""
    entries = _list_top_level_entries(extracted_root)

    # If a single folder exists, drill down (common wrapper folders).
    cur = extracted_root
    for _ in range(3):
        entries = _list_top_level_entries(cur)
        if len(entries) == 1 and entries[0].is_dir():
            cur = entries[0]
            continue
        break

    # If there's a directory matching the sequence name, prefer it.
    entries = _list_top_level_entries(cur)
    for p in entries:
        if p.is_dir() and p.name == sequence_name:
            return p

    # If there's a nested split wrapper like ORAD-3D/<seq>, pick the best candidate.
    # Heuristic: find a dir that contains at least one expected top-level item.
    best = cur
    best_score = -1
    candidates = [cur] + [p for p in entries if p.is_dir()]
    for cand in candidates:
        score = 0
        cand_entries = {p.name for p in _list_top_level_entries(cand)}
        for expected in EXPECTED_TOP_LEVEL:
            if expected in cand_entries:
                score += 1
        if score > best_score:
            best = cand
            best_score = score

    return best


def _atomic_move_tree(src_dir: Path, dst_dir: Path) -> None:
    """Move all children of src_dir into dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        target = dst_dir / child.name
        if target.exists():
            # Merge directories; for files, overwrite only if same size? Keep conservative.
            if child.is_dir() and target.is_dir():
                for sub in child.rglob("*"):
                    rel = sub.relative_to(child)
                    t = target / rel
                    if sub.is_dir():
                        t.mkdir(parents=True, exist_ok=True)
                    else:
                        t.parent.mkdir(parents=True, exist_ok=True)
                        if not t.exists():
                            shutil.move(str(sub), str(t))
                # After moving contents, remove empty dirs.
                shutil.rmtree(child, ignore_errors=True)
            else:
                # If file exists, leave it as-is.
                continue
        else:
            shutil.move(str(child), str(target))


def _validate_sequence_dir(seq_dir: Path) -> list[str]:
    missing: list[str] = []
    present = {p.name for p in seq_dir.iterdir()} if seq_dir.exists() else set()
    # We don't require everything (dataset variants differ), but warn if nothing familiar.
    familiar = [name for name in EXPECTED_TOP_LEVEL if name in present]
    if not familiar:
        missing.append("no_expected_items_found")
    return missing


def process_zip(zip_path: Path, dst_split_dir: Path, opt: Options) -> None:
    sequence_name = zip_path.stem
    seq_dir = dst_split_dir / sequence_name
    marker = seq_dir / ".extracted_ok"

    if marker.exists() and not opt.force:
        print(f"[SKIP] {zip_path.name} -> already extracted ({seq_dir})")
        return

    if opt.dry_run:
        print(f"[DRY] would extract {zip_path} -> {seq_dir}")
        return

    dst_split_dir.mkdir(parents=True, exist_ok=True)

    tmp_parent = dst_split_dir
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{sequence_name}.tmp_", dir=str(tmp_parent)))

    try:
        _safe_extract_zip(zip_path, tmp_dir)
        payload_root = _choose_payload_root(tmp_dir, sequence_name)

        # Ensure clean target if force
        if seq_dir.exists() and opt.force:
            shutil.rmtree(seq_dir)

        seq_dir.mkdir(parents=True, exist_ok=True)
        _atomic_move_tree(payload_root, seq_dir)

        # Cleanup tmp dir
        if not opt.keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        issues = _validate_sequence_dir(seq_dir)
        marker.write_text("ok\n", encoding="utf-8")

        if issues:
            print(f"[WARN] {sequence_name}: {issues} ({seq_dir})")
        else:
            print(f"[OK] {sequence_name} -> {seq_dir}")

    except Exception as exc:
        _eprint(f"[ERR] failed on {zip_path}: {exc}")
        if opt.keep_temp:
            _eprint(f"      temp kept at: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Unzip/organize ORAD-3D sequence zips into split/sequence folders.")
    ap.add_argument("--src", dest="src_root", default="/data3/ORAD-3D", help="Source root containing split folders")
    ap.add_argument(
        "--dst",
        dest="dst_root",
        default="/data3/datasets/ORAD-3D",
        help="Destination root to write organized sequences",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Set dst_root = src_root (extract alongside zips under the same split folders)",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation", "testing"],
        choices=["training", "validation", "testing"],
        help="Which splits to process",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    ap.add_argument("--force", action="store_true", help="Re-extract even if already extracted")
    ap.add_argument("--keep-temp", action="store_true", help="Keep temporary extraction folder on success")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N zip files per split")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    if args.in_place:
        dst_root = src_root

    opt = Options(
        src_root=src_root,
        dst_root=dst_root,
        splits=tuple(args.splits),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        keep_temp=bool(args.keep_temp),
        limit=args.limit,
    )

    for split in opt.splits:
        src_split_dir = opt.src_root / split
        if not src_split_dir.exists():
            _eprint(f"[WARN] missing split dir: {src_split_dir}")
            continue

        zip_files = sorted(src_split_dir.glob("*.zip"))
        if opt.limit is not None:
            zip_files = zip_files[: opt.limit]

        if not zip_files:
            print(f"[INFO] no zip files in {src_split_dir}")
            continue

        dst_split_dir = opt.dst_root / split
        print(f"[INFO] split={split} zips={len(zip_files)} src={src_split_dir} dst={dst_split_dir}")

        for zp in zip_files:
            process_zip(zp, dst_split_dir, opt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
