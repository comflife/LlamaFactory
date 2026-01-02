#!/usr/bin/env python3
"""Build a ShareGPT-style multimodal JSONL for LLaMAFactory VLM fine-tuning from ORAD-3D (v2).

v2 change
- Instead of a single <trajectory> block, we split the trajectory points into two halves:
    - <close_traj>: first half (near)
    - <far_traj>: second half (far)

Notes
- This script reads the same ORAD-3D folder layout as v1.
- By default it uses the FULL trajectory points from the dataset and then splits into halves.
- Optional downsampling is available via --num-points; the split is applied after downsampling.

Example
python3 scripts/orad3d_build_vlm_sharegpt_v2_close_far.py \
  --orad-root /data3/ORAD-3D \
  --splits training validation testing \
  --image-folder image_data \
  --trajectory-key trajectory_ins \
  --relative-media \
  --media-root /data3/ORAD-3D \
  --out /data3/orad3d_vlm/orad3d_all_close_far.jsonl \
  --write-dataset-info \
  --dataset-name orad3d_vlm_close_far
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


_TS_RE = re.compile(r"^(?P<ts>\d+)")


@dataclass(frozen=True)
class BuildOptions:
    orad_root: Path
    splits: tuple[str, ...]
    out_path: Path
    image_folder: str
    prompt_text: str
    system_text: str
    trajectory_key: str
    num_points: int | None
    relative_media: bool
    media_root: Path
    max_samples: int | None
    max_per_sequence: int | None
    write_dataset_info: bool
    dataset_name: str


def _extract_timestamp(name: str) -> str | None:
    match = _TS_RE.match(name)
    if not match:
        return None
    return match.group("ts")


def _iter_sequences(split_dir: Path) -> Iterator[Path]:
    if not split_dir.exists():
        return
    for child in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            yield child


def _choose_points(points: list[list[float]], num_points: int | None) -> list[list[float]]:
    if not points:
        return []
    # None or <=0 means "use all points".
    if num_points is None or num_points <= 0:
        return points
    if len(points) <= num_points:
        return points

    if num_points == 1:
        return [points[0]]

    last_idx = len(points) - 1
    indices = [round(i * last_idx / (num_points - 1)) for i in range(num_points)]

    seen = set()
    selected: list[list[float]] = []
    for idx in indices:
        idx = max(0, min(last_idx, int(idx)))
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(points[idx])

    while len(selected) < num_points:
        selected.append(points[last_idx])

    return selected


def _format_points(points: list[list[float]]) -> str:
    formatted = []
    for p in points:
        if not isinstance(p, list) or len(p) < 3:
            continue
        x, y, z = p[0], p[1], p[2]
        formatted.append(f"[{x:.3f},{y:.3f},{z:.3f}]")
    return ",".join(formatted)


def _split_close_far(points: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    if not points:
        return [], []
    if len(points) == 1:
        return points, []

    split = (len(points) + 1) // 2  # first half gets the extra
    return points[:split], points[split:]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_media_path(image_path: Path, opt: BuildOptions) -> str:
    if not opt.relative_media:
        return str(image_path)
    try:
        return str(image_path.relative_to(opt.media_root))
    except Exception:
        return str(image_path)


def _iter_frame_samples(seq_dir: Path, split: str, opt: BuildOptions) -> Iterator[dict]:
    image_dir = seq_dir / opt.image_folder
    scene_dir = seq_dir / "scene_data"
    local_dir = seq_dir / "local_path"

    if not image_dir.exists() or not scene_dir.exists() or not local_dir.exists():
        return

    image_files = [p for p in sorted(image_dir.iterdir(), key=lambda p: p.name) if p.is_file()]
    produced = 0

    for img_path in image_files:
        ts = _extract_timestamp(img_path.name)
        if ts is None:
            continue

        scene_path = scene_dir / f"{ts}.txt"
        local_path = local_dir / f"{ts}.json"

        if not scene_path.exists() or not local_path.exists():
            continue

        try:
            scene_text = _read_text(scene_path)
            local_obj = _read_json(local_path)
        except Exception:
            continue

        traj_points = local_obj.get(opt.trajectory_key)
        if not isinstance(traj_points, list) or len(traj_points) == 0:
            continue

        try:
            traj_points = [list(map(float, p[:3])) for p in traj_points if isinstance(p, list) and len(p) >= 3]
        except Exception:
            continue

        chosen = _choose_points(traj_points, opt.num_points)
        close_pts, far_pts = _split_close_far(chosen)

        close_text = _format_points(close_pts)
        far_text = _format_points(far_pts)
        if not close_text:
            continue

        assistant_text = f"{scene_text}\n<close_traj>\n{close_text}"
        if far_text:
            assistant_text += f"\n<far_traj>\n{far_text}"
        else:
            assistant_text += "\n<far_traj>\n"

        user_text = f"<image>\n{opt.prompt_text}"

        sample = {
            "messages": [
                {"role": "system", "content": opt.system_text},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "images": [_to_media_path(img_path, opt)],
            "meta": {"split": split, "sequence": seq_dir.name, "timestamp": ts},
        }
        yield sample

        produced += 1
        if opt.max_per_sequence is not None and produced >= opt.max_per_sequence:
            return


def build_samples(opt: BuildOptions) -> Iterable[dict]:
    emitted = 0
    for split in opt.splits:
        split_dir = opt.orad_root / split
        for seq_dir in _iter_sequences(split_dir):
            for sample in _iter_frame_samples(seq_dir, split, opt):
                yield sample
                emitted += 1
                if opt.max_samples is not None and emitted >= opt.max_samples:
                    return


def _write_dataset_info(out_dir: Path, opt: BuildOptions) -> None:
    info = {
        opt.dataset_name: {
            "file_name": opt.out_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
                "observation_tag": "observation",
                "function_tag": "function_call",
            },
        }
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ORAD-3D VLM ShareGPT JSONL for LLaMAFactory (v2 close/far).")
    ap.add_argument("--orad-root", default="/data3/ORAD-3D", help="Root folder containing training/validation/testing")
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["training"],
        choices=["training", "validation", "testing"],
        help="Splits to scan",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--image-folder",
        default="image_data",
        choices=["gt_image", "image_data"],
        help="Which image folder to use as the VLM input",
    )
    ap.add_argument(
        "--prompt",
        default="I am seeing an off-road driving image. Please generate a safe drivable trajectory for my vehicle to follow.",
        help="User prompt text (English)",
    )
    ap.add_argument(
        "--system",
        default=(
            "You are an off-road autonomous driving agent. "
            "Given an input camera image, describe the scene and provide a safe drivable trajectory. "
            "Output the near part of the trajectory after a <close_traj> token, and the far part after a <far_traj> token, "
            "each as a comma-separated list of [x,y,z] points."
        ),
        help="System message",
    )
    ap.add_argument(
        "--trajectory-key",
        default="trajectory_ins",
        choices=["trajectory_ins", "trajectory_hmi", "trajectory_ins_past", "trajectory_hmi_past"],
        help="Which trajectory field to read from local_path JSON",
    )
    ap.add_argument(
        "--num-points",
        type=int,
        default=0,
        help="Optional downsampling. 0 means use ALL original points, then split into halves.",
    )
    ap.add_argument(
        "--relative-media",
        action="store_true",
        help="Store image paths relative to --media-root (recommended if you use --media_dir in LLaMAFactory)",
    )
    ap.add_argument("--media-root", default="/data3/ORAD-3D", help="Base path for relative media paths")
    ap.add_argument("--max-samples", type=int, default=None, help="Stop after emitting N samples total")
    ap.add_argument("--max-per-seq", type=int, default=None, help="Stop after emitting N samples per sequence")
    ap.add_argument(
        "--write-dataset-info",
        action="store_true",
        help="Write dataset_info.json next to output for LLaMAFactory local loading",
    )
    ap.add_argument("--dataset-name", default="orad3d_vlm_close_far", help="Dataset name key used in dataset_info.json")
    args = ap.parse_args()

    opt = BuildOptions(
        orad_root=Path(args.orad_root),
        splits=tuple(args.splits),
        out_path=Path(args.out),
        image_folder=args.image_folder,
        prompt_text=args.prompt,
        system_text=args.system,
        trajectory_key=args.trajectory_key,
        num_points=int(args.num_points),
        relative_media=bool(args.relative_media),
        media_root=Path(args.media_root),
        max_samples=args.max_samples,
        max_per_sequence=args.max_per_seq,
        write_dataset_info=bool(args.write_dataset_info),
        dataset_name=args.dataset_name,
    )

    opt.out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with opt.out_path.open("w", encoding="utf-8") as f:
        for sample in build_samples(opt):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    if opt.write_dataset_info:
        _write_dataset_info(opt.out_path.parent, opt)

    print(f"[OK] wrote {count} samples -> {opt.out_path}")
    if opt.write_dataset_info:
        print(f"[OK] wrote dataset_info.json -> {opt.out_path.parent / 'dataset_info.json'}")

    if count == 0:
        print(
            "[WARN] no samples were emitted. Check that sequences are extracted and folders exist: "
            "image_data/scene_data/local_path (and gt_image if selected)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
