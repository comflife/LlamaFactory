#!/usr/bin/env python3
"""Compare ORAD-3D raw vs refined scene text on future-trajectory generation.

This script mirrors the data layout used by scripts/orad3d_xai_refine_scene_text_all_splits.py:
  <split>/<sequence>/{image_data,scene_data,scene_data_refine,local_path}/<timestamp>.*

For each frame, it:
  - loads the image, original scene text, refined scene text, and past ego path
  - runs the VLM twice (raw text vs refined text) to generate a future trajectory
  - saves a JSONL manifest plus optional overlay PNGs for both runs

Ground-truth future trajectories are taken from local_path/<ts>.json (--future-key).
Past paths come from --past-key and are embedded into the prompt.


python scripts/orad3d_compare_refine_text_trajectory.py --base-model Qwen/Qwen3-VL-2B-Instruct --orad-root /home/work/datasets/bg/ORAD-3D --split validation --image-folder image_data --past-key trajectory_ins_past --future-key trajectory_ins --use-sharegpt-format --save-overlays --require-refine --num-samples 8 --min-points 4 --default-num-points 12 --out-dir /home/work/byounggun/LlamaFactory/orad3d_compare_refine_base
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    # Reuse VLM helpers (model loading, prompt preparation, trajectory parsing, overlays).
    from scripts import orad3d_infer_vlm_trajectory_samples as infer  # type: ignore
except Exception as exc:  # pragma: no cover - defensive import guard
    raise SystemExit(f"Failed to import inference helpers: {exc}")


@dataclass(frozen=True)
class SamplePaths:
    split: str
    sequence: str
    timestamp: str
    image_path: Path
    scene_path: Path
    refine_path: Optional[Path]
    local_json: Path


@dataclass
class RunOutcome:
    prompt: str
    output_text: str
    trajectory_points: List[List[float]]
    composite_path: Optional[str]
    mean_l2_to_gt: Optional[float]
    final_l2_to_gt: Optional[float]


@dataclass
class PairResult:
    key: str
    split: str
    sequence: str
    timestamp: str
    image_path: str
    scene_text: str
    scene_text_refine: Optional[str]
    past_points: List[List[float]]
    gt_points: Optional[List[List[float]]]
    raw: RunOutcome
    refine: RunOutcome
    comparison_path: Optional[str]
    combined_path: Optional[str]


@dataclass(frozen=True)
class Calib:
    fx: float
    fy: float
    cx: float
    cy: float
    R: List[List[float]]  # 3x3
    t: List[float]  # 3x1


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _parse_floats_from_line(prefix: str, line: str) -> List[float]:
    if not line.startswith(prefix):
        return []
    return [float(x) for x in line[len(prefix) :].strip().split() if x]


def _load_calib(seq_dir: Path, ts: str) -> Optional[Calib]:
    path = seq_dir / "calib" / f"{ts}.txt"
    if not path.exists():
        return None

    k_vals: List[float] = []
    rt_vals: List[float] = []
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if line.startswith("cam_K:"):
                k_vals = _parse_floats_from_line("cam_K:", line)
            elif line.startswith("cam_RT:"):
                rt_vals = _parse_floats_from_line("cam_RT:", line)
        if len(k_vals) != 9 or len(rt_vals) != 16:
            return None
        fx, _, cx, _, fy, cy, _, _, _ = k_vals
        R = [
            [rt_vals[0], rt_vals[1], rt_vals[2]],
            [rt_vals[4], rt_vals[5], rt_vals[6]],
            [rt_vals[8], rt_vals[9], rt_vals[10]],
        ]
        t = [rt_vals[3], rt_vals[7], rt_vals[11]]
        return Calib(fx=fx, fy=fy, cx=cx, cy=cy, R=R, t=t)
    except Exception:
        return None


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
        local_dir = seq_dir / "local_path"

        if not image_dir.is_dir() or not scene_dir.is_dir() or not local_dir.is_dir():
            continue

        for img_path in sorted(image_dir.glob("*.png"), key=lambda p: p.name):
            ts = img_path.stem
            scene_path = scene_dir / f"{ts}.txt"
            refine_path = refine_dir / f"{ts}.txt"
            local_json = local_dir / f"{ts}.json"

            if not scene_path.exists() or not local_json.exists():
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
                local_json=local_json,
            )


def _downsample_points(points: Sequence[Sequence[float]], max_points: int) -> List[List[float]]:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return [list(map(float, p[:2])) for p in points if len(p) >= 2]

    last_idx = len(points) - 1
    if max_points == 1:
        return [list(map(float, points[0][:2]))]

    indices = [round(i * last_idx / (max_points - 1)) for i in range(max_points)]
    seen = set()
    selected: List[List[float]] = []
    for idx in indices:
        idx = max(0, min(last_idx, int(idx)))
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(list(map(float, points[idx][:2])))

    while len(selected) < max_points:
        selected.append(list(map(float, points[last_idx][:2])))
    return selected


def _format_points_for_prompt(points: Sequence[Sequence[float]], max_points: int, decimals: int = 2) -> str:
    pts = _downsample_points(points, max_points=max_points)
    return ",".join(f"[{p[0]:.{decimals}f},{p[1]:.{decimals}f}]" for p in pts if len(p) >= 2)


def _make_coord_hint(forward_axis: str, flip_lateral: bool) -> str:
    if forward_axis == "x":
        lat_axis = "y"
    else:
        lat_axis = "x"
    lat_positive = "left" if flip_lateral else "right"
    return (
        f"Coordinate convention: forward = +{forward_axis} (straight driving increases {forward_axis}); "
        f"lateral = {lat_axis} (positive {lat_axis} is toward the {lat_positive}); z is ignored for visualization."
    )


def _extract_points_xy(text: Optional[str]) -> List[List[float]]:
    if not text:
        return []
    pts: List[List[float]] = []
    point_re = re.compile(
        r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?\s*\]"
    )
    for m in point_re.finditer(text):
        try:
            x = float(m.group(1))
            y = float(m.group(2))
            pts.append([x, y])
        except Exception:
            continue
    return pts


def _project_points_with_calib(points: List[List[float]], calib: Calib) -> List[Tuple[float, float]]:
    if not points:
        return []

    def project(R: List[List[float]], t: List[float]) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for p in points:
            if len(p) < 3:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            xc = R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0]
            yc = R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1]
            zc = R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2]
            if zc <= 1e-6:
                continue
            u = calib.fx * (xc / zc) + calib.cx
            v = calib.fy * (yc / zc) + calib.cy
            out.append((u, v))
        return out

    # Try direct extrinsic; if nothing is in front of camera, also try inverse.
    direct = project(calib.R, calib.t)
    if len(direct) >= 2:
        return direct

    # Invert extrinsic: camera->ego assumed; inverse to ego->camera.
    Rt = [
        [calib.R[0][0], calib.R[1][0], calib.R[2][0]],
        [calib.R[0][1], calib.R[1][1], calib.R[2][1]],
        [calib.R[0][2], calib.R[1][2], calib.R[2][2]],
    ]
    t_inv = [
        -(Rt[0][0] * calib.t[0] + Rt[0][1] * calib.t[1] + Rt[0][2] * calib.t[2]),
        -(Rt[1][0] * calib.t[0] + Rt[1][1] * calib.t[1] + Rt[1][2] * calib.t[2]),
        -(Rt[2][0] * calib.t[0] + Rt[2][1] * calib.t[1] + Rt[2][2] * calib.t[2]),
    ]
    inverse = project(Rt, t_inv)
    return inverse if len(inverse) > len(direct) else direct


def _load_paths_from_local(
    path: Path, *, past_key: str, future_key: str, max_past_points: int
) -> Tuple[List[List[float]], Optional[List[List[float]]]]:
    obj = _read_json(path)
    past_raw = obj.get(past_key, [])
    future_raw = obj.get(future_key, [])

    def _normalize(seq: Any) -> List[List[float]]:
        pts: List[List[float]] = []
        if isinstance(seq, list):
            for p in seq:
                if isinstance(p, list) and len(p) >= 2:
                    try:
                        pts.append([float(p[0]), float(p[1])])
                    except Exception:
                        continue
        return pts

    past = _downsample_points(_normalize(past_raw), max_points=max_past_points)
    future = _normalize(future_raw) if isinstance(future_raw, list) else []
    return past, (future if future else None)


def _mean_l2(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Optional[float]:
    n = min(len(a), len(b))
    if n == 0:
        return None
    total = 0.0
    for i in range(n):
        dx = float(a[i][0]) - float(b[i][0])
        dy = float(a[i][1]) - float(b[i][1])
        total += math.sqrt(dx * dx + dy * dy)
    return total / n


def _final_l2(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Optional[float]:
    if not a or not b:
        return None
    pa = a[-1]
    pb = b[-1]
    if len(pa) < 2 or len(pb) < 2:
        return None
    dx = float(pa[0]) - float(pb[0])
    dy = float(pa[1]) - float(pb[1])
    return math.sqrt(dx * dx + dy * dy)


def _run_one_prompt(
    *,
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    system_text: str,
    prompt_text: str,
    use_sharegpt_format: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    skip_special_tokens: bool,
) -> Tuple[str, List[List[float]], str]:
    inputs, input_len = infer._prepare_inputs(
        processor,
        image=image,
        system_text=system_text,
        prompt_text=prompt_text,
        use_sharegpt_format=bool(use_sharegpt_format),
    )
    if torch.cuda.is_available():
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if float(temperature) > 0:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)

    try:
        full_text = processor.batch_decode(out_ids, skip_special_tokens=bool(skip_special_tokens))[0]
    except Exception:
        full_text = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=bool(skip_special_tokens))[0]  # type: ignore

    gen_ids = out_ids
    if input_len > 0 and isinstance(out_ids, torch.Tensor) and out_ids.ndim == 2 and out_ids.shape[1] > input_len:
        gen_ids = out_ids[:, input_len:]

    try:
        out_text = processor.batch_decode(gen_ids, skip_special_tokens=bool(skip_special_tokens))[0]
    except Exception:
        out_text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=bool(skip_special_tokens))[0]  # type: ignore

    out_text = infer._clean_output_text((out_text or "").strip())
    full_text = infer._clean_output_text((full_text or "").strip())

    traj_points = _extract_points_xy(out_text)
    return out_text, traj_points, full_text


def _build_prompt(
    template: str,
    *,
    scene_text: str,
    past_points: List[List[float]],
    max_past_points: int,
    target_points: Optional[int],
    min_points: int,
    coord_hint: str,
) -> str:
    past_str = _format_points_for_prompt(past_points, max_points=max_past_points)
    extra_lines: List[str] = [
        f"Predict at least {min_points} future points; do NOT stop at a single point.",
    ]
    if target_points is not None:
        extra_lines.append(f"Target number of future points: {target_points}.")
    extra_lines.append(
        "Output format: [x0,y0],[x1,y1],...[xN,yN] in meters, comma-separated, no extra text (z is omitted)."
    )

    extra_lines.append(coord_hint)
    extra_constraints = "\n".join(extra_lines)
    return template.format(
        scene_text=scene_text.strip(),
        past_points=past_str,
        target_points=target_points if target_points is not None else "",
        min_points=min_points,
        extra_constraints=extra_constraints,
        coord_hint=coord_hint,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare raw vs refined ORAD-3D scene text for trajectory generation.")

    ap.add_argument("--base-model", type=str, required=True, help="HF model id or local path")
    ap.add_argument("--adapter", type=str, default=None, help="Optional LoRA checkpoint dir (e.g., .../checkpoint-xxx)")
    ap.add_argument("--cache-dir", type=str, default="/home/work/byounggun/.cache/hf")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument(
        "--system",
        type=str,
        default=(
            "You are an off-road autonomous driving agent. "
            "Given an input image, a short scene description, and the past ego trajectory, "
            "produce a safe future trajectory. "
        ),
        help="System message. Set to empty string to disable.",
    )
    ap.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "You are an off-road autonomous driving labeller.\n"
            "1) Review the front-view image and the scene summary.\n"
            "2) From the past ego path, infer speed trend and turning intent (left/right/straight).\n"
            "3) Propose the ego's intent for the next few seconds (accelerate/maintain/decelerate, turn left/right/how much).\n"
            "4) Emit a smooth, physically plausible OFF-ROAD trajectory that follows that intent and avoids obstacles/ruts.\n"
            "Scene description: {scene_text}\n"
            "{coord_hint}\n"
            "Past ego path (oldest->latest, meters, [x,y]): {past_points}\n"
            "{extra_constraints}\n"
            "Return ONLY the trajectory as comma-separated [x,y] points (no extra text)."
        ),
        help="User prompt template; supports {scene_text}, {past_points}, {extra_constraints}, {target_points}, {min_points}.",
    )
    ap.add_argument("--use-sharegpt-format", action="store_true", help="Prefix user text with <image> like training.")

    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--skip-special-tokens", action="store_true", help="Decode with skip_special_tokens=True.")

    ap.add_argument("--orad-root", type=Path, default=Path("/home/work/datasets/bg/ORAD-3D"))
    ap.add_argument("--split", type=str, default="validation", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])
    ap.add_argument("--scene-folder", type=str, default="scene_data")
    ap.add_argument("--refine-folder", type=str, default="scene_data_refine")
    ap.add_argument("--past-key", type=str, default="trajectory_ins_past")
    ap.add_argument("--future-key", type=str, default="trajectory_ins")
    ap.add_argument("--max-past-points", type=int, default=12)
    ap.add_argument("--default-num-points", type=int, default=12, help="Fallback target points when GT is absent.")
    ap.add_argument("--min-points", type=int, default=4, help="Skip samples whose outputs have fewer points.")
    ap.add_argument("--require-refine", action="store_true", help="Skip samples where scene_data_refine is missing.")
    ap.add_argument("--num-samples", type=int, default=8, help="How many frames to run (after filtering).")
    ap.add_argument("--max-scan", type=int, default=None, help="Optional cap on frames scanned before sampling.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--use-calib-projection",
        action="store_true",
        help="Use per-frame calib (cam_K + cam_RT) to project XYZ onto the image; falls back to ego-plane scaling if missing.",
    )

    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true")
    ap.add_argument("--save-overlays", action="store_true", help="Write overlay PNGs for raw/refine outputs.")

    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def _load_model_and_processor_optional(
    *,
    base_model: str,
    adapter: Optional[str],
    cache_dir: Optional[str],
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
) -> Tuple[torch.nn.Module, Any]:
    if adapter:
        return infer._load_model_and_processor(
            base_model=base_model,
            adapter=adapter,
            cache_dir=cache_dir,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    infer._maybe_set_cache_env(cache_dir)
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
        if hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer  # type: ignore[attr-defined]
    except Exception:
        pass

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
    torch_dtype: Any = "auto" if dtype.strip().lower() == "auto" else infer._parse_dtype(dtype)

    model = None
    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            try:
                model = cls.from_pretrained(
                    base_model,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_dir,
                    dtype=torch_dtype,
                    device_map=device_map,
                )
            except TypeError:
                model = cls.from_pretrained(
                    base_model,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(f"Failed to load base model: {base_model} (model_type={getattr(cfg, 'model_type', None)})")

    model.eval()
    return model, processor


def _select_samples(all_items: List[SamplePaths], num: int, seed: int, max_scan: Optional[int]) -> List[SamplePaths]:
    if max_scan is not None:
        all_items = all_items[:max_scan]
    if len(all_items) <= num:
        return all_items
    rng = random.Random(seed)
    return rng.sample(all_items, k=num)


def _save_overlay(
    *,
    image: Image.Image,
    header: str,
    traj_points: List[List[float]],
    gt_points: Optional[List[List[float]]],
    forward_axis: str,
    flip_lateral: bool,
    out_path: Path,
) -> Image.Image:
    overlay = infer._render_overlay(
        image=image,
        header=header,
        points_xyz=traj_points,
        gt_points_xyz=gt_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )
    overlay.save(out_path)
    return overlay


def _save_comparison_strip(raw_img: Image.Image, ref_img: Image.Image, out_path: Path) -> None:
    margin = 12
    h = max(raw_img.height, ref_img.height)
    w = raw_img.width + ref_img.width + (3 * margin)
    canvas = Image.new("RGB", (w, h + 2 * margin), (255, 255, 255))
    canvas.paste(raw_img, (margin, margin))
    canvas.paste(ref_img, (raw_img.width + (2 * margin), margin))
    canvas.save(out_path)


def _save_combined_overlay(
    *,
    image: Image.Image,
    header: str,
    gt_points: Optional[List[List[float]]],
    raw_points: List[List[float]],
    refine_points: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
    out_path: Path,
    calib: Optional[Calib],
) -> None:
    base = image.convert("RGB")

    # We intentionally ignore Z for visualization; use ego-plane scaling only.
    scale_holder = {"value": None}
    ref_set = gt_points if gt_points else (raw_points if raw_points else refine_points)

    def _to_uv(points: List[List[float]]) -> List[Tuple[float, float]]:
        if not points:
            return []
        if scale_holder["value"] is None:
            if ref_set:
                _, scale_holder["value"] = infer._traj_xyz_to_pixels(
                    ref_set,
                    base.size,
                    forward_axis=forward_axis,
                    flip_lateral=flip_lateral,
                    scale_px_per_meter=None,
                )
            else:
                scale_holder["value"] = None
        uv, _ = infer._traj_xyz_to_pixels(
            points,
            base.size,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
            scale_px_per_meter=scale_holder["value"],
        )
        return uv

    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)

    gt_uv = _to_uv(gt_points or [])
    raw_uv = _to_uv(raw_points)
    refine_uv = _to_uv(refine_points)

    infer._draw_polyline(overlay, gt_uv, color=(0, 200, 0), width=3)
    infer._draw_polyline(overlay, raw_uv, color=(220, 20, 60), width=3)
    infer._draw_polyline(overlay, refine_uv, color=(30, 144, 255), width=3)

    font = ImageFont.load_default()
    legend = [
        f"{header}",
        f"GT   (green): {len(gt_points or [])} pts",
        f"Raw  (red):   {len(raw_points)} pts",
        f"Ref  (blue):  {len(refine_points)} pts",
    ]
    y = 10
    for line in legend:
        draw.text((12, y), line, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        y += 14

    overlay.save(out_path)


def main() -> int:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    model, processor = _load_model_and_processor_optional(
        base_model=args.base_model,
        adapter=args.adapter,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )

    all_items = list(
        _iter_samples(
            orad_root=args.orad_root,
            split=args.split,
            image_folder=args.image_folder,
            scene_folder=args.scene_folder,
            refine_folder=args.refine_folder,
            require_refine=bool(args.require_refine),
        )
    )
    if not all_items:
        raise SystemExit("No samples found; check --orad-root and folder names.")

    samples = _select_samples(all_items, num=int(args.num_samples), seed=int(args.seed), max_scan=args.max_scan)

    results: List[PairResult] = []
    saved = 0
    coord_hint = _make_coord_hint(args.forward_axis, bool(args.flip_lateral))

    for idx, item in enumerate(samples, start=1):
        image = Image.open(item.image_path).convert("RGB")
        scene_text = _read_text(item.scene_path)
        scene_text_refine = _read_text(item.refine_path) if item.refine_path else None

        past_pts, gt_pts = _load_paths_from_local(
            item.local_json,
            past_key=args.past_key,
            future_key=args.future_key,
            max_past_points=int(args.max_past_points),
        )
        if not past_pts:
            print(f"[SKIP] {item.sequence}/{item.timestamp}: missing past path ({args.past_key})")
            continue

        target_points = len(gt_pts) if gt_pts else int(args.default_num_points)

        prompt_raw = _build_prompt(
            args.prompt_template,
            scene_text=scene_text,
            past_points=past_pts,
            max_past_points=int(args.max_past_points),
            target_points=target_points,
            min_points=int(args.min_points),
            coord_hint=coord_hint,
        )
        prompt_refine = _build_prompt(
            args.prompt_template,
            scene_text=scene_text_refine or scene_text,
            past_points=past_pts,
            max_past_points=int(args.max_past_points),
            target_points=target_points,
            min_points=int(args.min_points),
            coord_hint=coord_hint,
        )

        out_raw_text, out_raw_traj, _ = _run_one_prompt(
            model=model,
            processor=processor,
            image=image,
            system_text=args.system,
            prompt_text=prompt_raw,
            use_sharegpt_format=bool(args.use_sharegpt_format),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            skip_special_tokens=bool(args.skip_special_tokens),
        )
        out_ref_text, out_ref_traj, _ = _run_one_prompt(
            model=model,
            processor=processor,
            image=image,
            system_text=args.system,
            prompt_text=prompt_refine,
            use_sharegpt_format=bool(args.use_sharegpt_format),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            skip_special_tokens=bool(args.skip_special_tokens),
        )

        if len(out_raw_traj) < int(args.min_points) or len(out_ref_traj) < int(args.min_points):
            print(
                f"[SKIP] {item.sequence}/{item.timestamp}: too few points raw={len(out_raw_traj)} refine={len(out_ref_traj)} (min={args.min_points})"
            )
            continue

        header_base = f"{args.split}_{item.sequence}_{item.timestamp}"
        raw_overlay_path = None
        ref_overlay_path = None
        compare_overlay_path = None
        combined_overlay_path = None
        if bool(args.save_overlays):
            combined_overlay_path = args.out_dir / f"{saved:03d}_{header_base}.png"
            _save_combined_overlay(
                image=image,
                header=header_base,
                gt_points=gt_pts,
                raw_points=out_raw_traj,
                refine_points=out_ref_traj,
                forward_axis=args.forward_axis,
                flip_lateral=bool(args.flip_lateral),
                out_path=combined_overlay_path,
                calib=_load_calib(item.image_path.parent.parent, item.timestamp) if args.use_calib_projection else None,
            )

        raw_mean = _mean_l2(out_raw_traj, gt_pts) if gt_pts else None
        ref_mean = _mean_l2(out_ref_traj, gt_pts) if gt_pts else None
        raw_final = _final_l2(out_raw_traj, gt_pts) if gt_pts else None
        ref_final = _final_l2(out_ref_traj, gt_pts) if gt_pts else None

        pair = PairResult(
            key=header_base,
            split=item.split,
            sequence=item.sequence,
            timestamp=item.timestamp,
            image_path=str(item.image_path),
            scene_text=scene_text,
            scene_text_refine=scene_text_refine,
            past_points=past_pts,
            gt_points=gt_pts,
            raw=RunOutcome(
                prompt=prompt_raw,
                output_text=out_raw_text,
                trajectory_points=out_raw_traj,
                composite_path=str(combined_overlay_path) if combined_overlay_path else None,
                mean_l2_to_gt=raw_mean,
                final_l2_to_gt=raw_final,
            ),
            refine=RunOutcome(
                prompt=prompt_refine,
                output_text=out_ref_text,
                trajectory_points=out_ref_traj,
                composite_path=str(combined_overlay_path) if combined_overlay_path else None,
                mean_l2_to_gt=ref_mean,
                final_l2_to_gt=ref_final,
            ),
            comparison_path=str(compare_overlay_path) if compare_overlay_path else None,
            combined_path=str(combined_overlay_path) if combined_overlay_path else None,
        )
        results.append(pair)
        saved += 1
        print(
            f"[OK] {idx}/{len(samples)} {header_base}: traj_raw={len(out_raw_traj)} traj_refine={len(out_ref_traj)}"
            + (f" gt={len(gt_pts)}" if gt_pts else "")
        )

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {len(results)} pairs -> {manifest_path}")
    if args.save_overlays:
        print(f"Overlays: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
