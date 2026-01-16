#!/usr/bin/env python3
"""Compare multiple LoRA checkpoints on ORAD-3D with XY + Z visualizations.

This script runs inference for multiple adapters, then renders a composite per sample:
  - image overlay with GT + all model trajectories
  - side panel with per-model XY plots (GT vs prediction)
  - bottom Z trend panel with GT + all models

Only samples with valid GT trajectories (>=2 points) are kept by default, similar to
scripts/orad3d_visualize_orpo_pairs_withz.py.

Example:
python scripts/orad3d_compare_vlm_models_withz.py \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --adapter sft_refine=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_refine/checkpoint-208 \
  --adapter sft=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2/checkpoint-182 \
  --adapter orpo=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/orpo2/checkpoint-182 \
  --orad-root /home/work/datasets/bg/ORAD-3D \
  --split training --image-folder image_data --num-samples 20 \
  --out-dir /home/work/byounggun/LlamaFactory/orad3d_compare_models_withz \
  --cache-dir /home/work/byounggun/.cache/hf \
  --use-sharegpt-format --temperature 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)


_TRAJ_TOKEN_RE = re.compile(r"<\s*trajectory\s*>", re.IGNORECASE)
_POINT_RE = re.compile(r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]")

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (220, 20, 60),  # crimson
    (30, 144, 255),  # dodger blue
    (255, 140, 0),  # dark orange
    (138, 43, 226),  # blue violet
    (0, 206, 209),  # dark turquoise
    (160, 82, 45),  # sienna
    (128, 0, 0),  # maroon
]


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    path: str
    color: Tuple[int, int, int]


@dataclass(frozen=True)
class SampleItem:
    key: str
    image_path: Path
    gt_points: List[List[float]]
    meta: Dict[str, Any]


@dataclass
class ModelOutput:
    name: str
    adapter_path: str
    output_text: str
    trajectory_points: List[List[float]]
    valid: bool


@dataclass
class SampleResult:
    key: str
    image_path: str
    gt_trajectory_points: List[List[float]]
    outputs: List[ModelOutput]
    composite_path: str
    meta: Dict[str, Any]


def _normalize_path_str(p: str) -> str:
    return str(p).replace("\\", "/").lstrip("./")


def _extract_text_from_message_content(content: Any) -> str:
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


def _clean_output_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("<tool_call>", "").replace("</tool_call>", "").replace("<tool_call/>", "")
    return cleaned.strip()


def _candidate_image_keys(img_path: Path, *, orad_root: Optional[Path], meta: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    p = img_path
    keys.append(_normalize_path_str(str(p)))
    keys.append(_normalize_path_str(str(p.resolve())))

    if orad_root is not None:
        try:
            rel = p.resolve().relative_to(orad_root.resolve())
            keys.append(_normalize_path_str(str(rel)))
        except Exception:
            pass

    split = str(meta.get("split") or "").strip()
    seq = str(meta.get("sequence") or "").strip()
    ts = str(meta.get("timestamp") or "").strip()
    if split and seq and ts:
        keys.append(_normalize_path_str(f"{split}/{seq}/image_data/{ts}.png"))
        keys.append(_normalize_path_str(f"{split}/{seq}/gt_image/{ts}.png"))

    keys.append(_normalize_path_str(p.name))
    out: List[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _load_gt_trajectories_for_items(
    gt_jsonl: Path,
    *,
    wanted_keys: set[str],
) -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}
    if not gt_jsonl.is_file() or not wanted_keys:
        return out

    with gt_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            images = obj.get("images")
            if not isinstance(images, list) or not images:
                continue

            matched_img_keys: List[str] = []
            for img in images:
                if not isinstance(img, str):
                    continue
                k = _normalize_path_str(img)
                if k in wanted_keys:
                    matched_img_keys.append(k)
                bn = _normalize_path_str(Path(k).name)
                if bn in wanted_keys:
                    matched_img_keys.append(bn)

            if not matched_img_keys:
                continue

            messages = obj.get("messages")
            if not isinstance(messages, list) or not messages:
                continue

            assistant_text = ""
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    assistant_text = _extract_text_from_message_content(m.get("content"))
                    break

            traj_section = _extract_trajectory_section(assistant_text)
            if traj_section is None:
                continue
            pts = _extract_trajectory_points(traj_section)
            if len(pts) < 2:
                continue

            for k in matched_img_keys:
                out.setdefault(k, pts)

    return out


def _infer_local_path_from_image(img_path: Path) -> Optional[Path]:
    parent = img_path.parent
    if parent.name in ("image_data", "gt_image"):
        local_dir = parent.parent / "local_path"
        return local_dir / f"{img_path.stem}.json"
    return None


def _infer_local_path_from_orad_root(img_path: Path, orad_root: Optional[Path]) -> Optional[Path]:
    if orad_root is None:
        return None
    try:
        rel = img_path.resolve().relative_to(orad_root.resolve())
    except Exception:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    split, seq = parts[0], parts[1]
    return orad_root / split / seq / "local_path" / f"{img_path.stem}.json"


def _load_gt_from_local_path(
    *,
    img_path: Path,
    orad_root: Optional[Path],
    meta: Dict[str, Any],
    gt_key: str,
) -> Optional[List[List[float]]]:
    candidates: List[Path] = []

    direct = _infer_local_path_from_image(img_path)
    if direct is not None:
        candidates.append(direct)

    via_root = _infer_local_path_from_orad_root(img_path, orad_root)
    if via_root is not None:
        candidates.append(via_root)

    split = str(meta.get("split") or "").strip()
    seq = str(meta.get("sequence") or "").strip()
    ts = str(meta.get("timestamp") or "").strip()
    if orad_root is not None and split and seq and ts:
        candidates.append(orad_root / split / seq / "local_path" / f"{ts}.json")

    local_json = next((p for p in candidates if p.is_file()), None)
    if local_json is None:
        return None

    try:
        obj = json.loads(local_json.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    pts = obj.get(gt_key)
    if not isinstance(pts, list):
        return None

    out: List[List[float]] = []
    for p in pts:
        if not isinstance(p, list) or len(p) < 2:
            continue
        try:
            out.append([float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0])
        except Exception:
            continue

    if len(out) < 2:
        return None
    return out


def _maybe_set_cache_env(cache_dir: Optional[str]) -> None:
    if not cache_dir:
        return

    try:
        from llamafactory.model.loader import _maybe_set_hf_cache_env  # type: ignore

        _maybe_set_hf_cache_env(cache_dir)
        return
    except Exception:
        pass

    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))


def _parse_dtype(value: str) -> torch.dtype:
    v = value.strip().lower()
    if v in ("auto", ""):
        return torch.float16
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16"):
        return torch.float16
    if v in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _load_model_and_processor(
    *,
    base_model: str,
    adapter: str,
    cache_dir: Optional[str],
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
) -> Tuple[torch.nn.Module, Any]:
    _maybe_set_cache_env(cache_dir)

    tokenizer = None
    processor = None

    adapter_is_dir = False
    try:
        adapter_is_dir = os.path.isdir(adapter)
    except Exception:
        adapter_is_dir = False

    if adapter_is_dir:
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
        except Exception:
            tokenizer = None
        try:
            processor = AutoProcessor.from_pretrained(adapter, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
        except Exception:
            processor = None

    if processor is None:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
        except Exception:
            tokenizer = None

    if tokenizer is not None and hasattr(processor, "tokenizer"):
        try:
            processor.tokenizer = tokenizer  # type: ignore[attr-defined]
        except Exception:
            pass

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)

    model = None
    torch_dtype: Any = "auto" if dtype.strip().lower() == "auto" else _parse_dtype(dtype)

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

    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def _build_messages(prompt_text: str, system_text: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if (system_text or "").strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text.strip()}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (prompt_text or "").strip()},
            ],
        }
    )
    return messages


def _maybe_prefix_image_token(prompt_text: str, *, use_sharegpt_format: bool) -> str:
    txt = (prompt_text or "").strip()
    if not use_sharegpt_format:
        return txt
    if txt.lower().startswith("<image>"):
        return txt
    return f"<image>\n{txt}".strip()


def _prepare_inputs(
    processor: Any,
    *,
    image: Image.Image,
    system_text: str,
    prompt_text: str,
    use_sharegpt_format: bool,
) -> Tuple[Dict[str, torch.Tensor], int]:
    prompt_text = _maybe_prefix_image_token(prompt_text, use_sharegpt_format=use_sharegpt_format)
    messages = _build_messages(prompt_text, system_text)

    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif hasattr(getattr(processor, "tokenizer", None), "apply_chat_template"):
        chat_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        chat_text = f"<image>\n{prompt_text}"

    inputs = processor(text=[chat_text], images=[image], return_tensors="pt", padding=True)
    input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    return inputs, input_len


def _iter_orad_pairs(
    *,
    orad_root: Path,
    split: str,
    image_folder: str,
    max_scan: Optional[int],
) -> List[Tuple[str, str, Path]]:
    split_dir = orad_root / split
    if not split_dir.is_dir():
        raise SystemExit(f"Split dir not found: {split_dir}")

    pairs: List[Tuple[str, str, Path]] = []
    seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and not p.name.endswith(".zip")], key=lambda p: p.name)

    for seq_dir in seq_dirs:
        img_dir = seq_dir / image_folder
        if not img_dir.is_dir():
            continue

        for img_path in sorted(img_dir.glob("*.png"), key=lambda p: p.name):
            ts = img_path.stem
            pairs.append((seq_dir.name, ts, img_path))
            if max_scan is not None and len(pairs) >= max_scan:
                return pairs

    return pairs


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


def _points_forward_lateral(
    points_xyz: List[List[float]],
    *,
    forward_axis: str,
    flip_lateral: bool,
) -> Tuple[List[float], List[float]]:
    if len(points_xyz) < 2:
        return [], []
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

    return forward, lateral


def _shared_plot_scale(
    box: Tuple[int, int, int, int],
    gt_points: List[List[float]],
    pred_points: List[List[float]],
    *,
    forward_axis: str,
    flip_lateral: bool,
) -> Optional[float]:
    fwd_gt, lat_gt = _points_forward_lateral(gt_points, forward_axis=forward_axis, flip_lateral=flip_lateral)
    fwd_pr, lat_pr = _points_forward_lateral(pred_points, forward_axis=forward_axis, flip_lateral=flip_lateral)

    max_fwd = max(max(fwd_gt, default=0.0), max(fwd_pr, default=0.0), 1e-3)
    max_lat = max(max((abs(v) for v in lat_gt), default=0.0), max((abs(v) for v in lat_pr), default=0.0), 1e-3)

    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    margin = 10
    usable_w = max(1, w - 2 * margin)
    usable_h = max(1, h - 2 * margin)
    scale = min(usable_h / max_fwd, (usable_w / 2.0) / max_lat)
    return max(0.5, min(scale, 200.0))


def _plot_points_in_box(
    forward: List[float],
    lateral: List[float],
    *,
    box: Tuple[int, int, int, int],
    scale: float,
) -> List[Tuple[float, float]]:
    if not forward or not lateral:
        return []
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    margin = 10
    cx = x0 + w / 2.0
    bottom = y0 + h - margin
    pts: List[Tuple[float, float]] = []
    for f, lat in zip(forward, lateral):
        px = cx + lat * scale
        py = bottom - f * scale
        pts.append((px, py))
    return pts


def _draw_traj_compare_plot(
    *,
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    gt_points: List[List[float]],
    pred_points: List[List[float]],
    pred_color: Tuple[int, int, int],
    forward_axis: str,
    flip_lateral: bool,
    draw_gt: bool = True,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)

    if len(gt_points) < 2 and len(pred_points) < 2:
        draw.text((x0 + 8, y0 + 8), "(no trajectory)", fill=(0, 0, 0))
        return

    scale = _shared_plot_scale(
        box,
        gt_points,
        pred_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )
    if scale is None:
        draw.text((x0 + 8, y0 + 8), "(no trajectory)", fill=(0, 0, 0))
        return

    margin = 10
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    cx = x0 + w / 2.0
    bottom = y0 + h - margin
    draw.line([(cx, y0 + margin), (cx, y0 + h - margin)], fill=(220, 220, 220), width=1)
    draw.line([(x0 + margin, bottom), (x0 + w - margin, bottom)], fill=(220, 220, 220), width=1)

    if draw_gt and len(gt_points) >= 2:
        fwd_gt, lat_gt = _points_forward_lateral(gt_points, forward_axis=forward_axis, flip_lateral=flip_lateral)
        pts_gt = _plot_points_in_box(fwd_gt, lat_gt, box=box, scale=scale)
        for a, b in zip(pts_gt[:-1], pts_gt[1:]):
            draw.line([a, b], fill=(0, 160, 0), width=3)
        for p in pts_gt:
            r = 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(0, 160, 0))

    if len(pred_points) >= 2:
        fwd_pr, lat_pr = _points_forward_lateral(pred_points, forward_axis=forward_axis, flip_lateral=flip_lateral)
        pts_pr = _plot_points_in_box(fwd_pr, lat_pr, box=box, scale=scale)
        for a, b in zip(pts_pr[:-1], pts_pr[1:]):
            draw.line([a, b], fill=pred_color, width=3)
        for p in pts_pr:
            r = 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=pred_color)


def _mean_l2(gt_points: List[List[float]], pred_points: List[List[float]]) -> Optional[float]:
    if len(gt_points) < 2 or len(pred_points) < 2:
        return None
    n = min(len(gt_points), len(pred_points))
    if n < 2:
        return None
    total = 0.0
    for g, p in zip(gt_points[:n], pred_points[:n]):
        dx = float(g[0]) - float(p[0])
        dy = float(g[1]) - float(p[1])
        dz = float(g[2]) - float(p[2])
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total / n


def _final_l2(gt_points: List[List[float]], pred_points: List[List[float]]) -> Optional[float]:
    if len(gt_points) < 2 or len(pred_points) < 2:
        return None
    n = min(len(gt_points), len(pred_points))
    if n < 1:
        return None
    g = gt_points[n - 1]
    p = pred_points[n - 1]
    dx = float(g[0]) - float(p[0])
    dy = float(g[1]) - float(p[1])
    dz = float(g[2]) - float(p[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _render_overlay_multimodel(
    *,
    image: Image.Image,
    header: str,
    gt_points: List[List[float]],
    model_points: Sequence[Tuple[str, List[List[float]], Tuple[int, int, int]]],
    forward_axis: str,
    flip_lateral: bool,
    line_width: int,
) -> Image.Image:
    base = image.convert("RGB")
    overlay = base.copy()

    point_sets: List[Optional[Sequence[Sequence[float]]]] = [gt_points]
    point_sets.extend([pts for _, pts, _ in model_points])
    shared_scale = _shared_traj_scale(
        overlay.size,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        point_sets=point_sets,
    )

    if len(gt_points) >= 2:
        uv_gt, _ = _traj_xyz_to_pixels(
            gt_points,
            overlay.size,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
            scale_px_per_meter=shared_scale,
        )
        _draw_polyline(overlay, uv_gt, color=(0, 200, 0), width=line_width)

    for _, pts, color in model_points:
        if len(pts) < 2:
            continue
        uv, _ = _traj_xyz_to_pixels(
            pts,
            overlay.size,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
            scale_px_per_meter=shared_scale,
        )
        _draw_polyline(overlay, uv, color=color, width=line_width)

    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    bar_h = 40
    draw.rectangle([(0, 0), (overlay.size[0], bar_h)], fill=(20, 20, 20))
    draw.text((10, 6), header, fill=(255, 255, 255), font=font)

    legend_items: List[Tuple[str, Tuple[int, int, int]]] = [("GT", (0, 200, 0))]
    legend_items.extend([(name, color) for name, _, color in model_points])
    x = 10
    y = 22
    for label, color in legend_items:
        draw.text((x, y), label, fill=color, font=font)
        if hasattr(draw, "textlength"):
            w = int(draw.textlength(label, font=font))
        else:
            w = len(label) * 6
        x += w + 14
        if x > overlay.size[0] - 60:
            x = 10
            y += 12

    return overlay


def _render_model_panel(
    *,
    size: Tuple[int, int],
    header: str,
    gt_points: List[List[float]],
    model_outputs: Sequence[Tuple[ModelOutput, Tuple[int, int, int]]],
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

    if not model_outputs:
        draw.text((pad, header_h + 10), "(no models)", fill=(0, 0, 0), font=font)
        return img

    y = header_h + pad
    gap = 12
    rows = len(model_outputs)
    row_h = (h - y - pad - gap * (rows - 1)) / rows

    def _wrap_lines(text: str, width_chars: int, max_lines: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        lines: List[str] = []
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            for line in textwrap.wrap(raw, width=width_chars, break_long_words=True, replace_whitespace=False):
                if len(lines) >= max_lines:
                    return lines
                lines.append(line)
        return lines

    def _language_only(text: str) -> str:
        if not text:
            return ""
        m = _TRAJ_TOKEN_RE.search(text)
        if not m:
            return text.strip()
        return text[: m.start()].strip()

    for output, color in model_outputs:
        name = output.name
        pts = output.trajectory_points
        row_top = int(y)
        row_bottom = int(y + row_h)
        label_h = 16

        mean_l2 = _mean_l2(gt_points, pts)
        final_l2 = _final_l2(gt_points, pts)
        metrics: List[str] = [f"N={len(pts)}"]
        if mean_l2 is not None:
            metrics.append(f"mean_L2={mean_l2:.3f}")
        if final_l2 is not None:
            metrics.append(f"final_L2={final_l2:.3f}")

        label = f"{name} ({', '.join(metrics)})"
        draw.text((pad, row_top + 2), label, fill=color, font=font)

        available_h = row_h - label_h - 6
        max_text_lines = 0
        if available_h >= 36:
            max_text_lines = min(3, int(available_h // 12) - 1)
        elif available_h >= 24:
            max_text_lines = 1

        text_lines: List[str] = []
        if max_text_lines > 0:
            max_chars = max(30, (w - 2 * pad) // 6)
            lang_text = _language_only(output.output_text)
            text_lines = _wrap_lines(lang_text, width_chars=max_chars, max_lines=max_text_lines)

        if text_lines:
            y_text = row_top + label_h
            for line in text_lines:
                draw.text((pad, y_text), line, fill=(60, 60, 60), font=font)
                y_text += 12
            plot_top = y_text + 4
        else:
            plot_top = row_top + label_h + 2

        if plot_top > row_bottom - 2:
            plot_top = row_bottom - 2

        plot_box = (pad, plot_top, w - pad, row_bottom)
        _draw_traj_compare_plot(
            draw=draw,
            box=plot_box,
            gt_points=gt_points,
            pred_points=pts,
            pred_color=color,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
        )
        y = row_bottom + gap

    return img


def _render_z_trend_panel(
    *,
    size: Tuple[int, int],
    series: Sequence[Tuple[str, List[List[float]], Tuple[int, int, int]]],
) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    pad_left = 46
    pad_right = 16
    pad_top = 22
    pad_bottom = 20

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

    series_data: List[Tuple[str, List[float], Tuple[int, int, int]]] = []
    for name, pts, color in series:
        series_data.append((name, _z_series(pts), color))

    all_z: List[float] = []
    for _, zs, _ in series_data:
        all_z.extend(zs)

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

    def _to_xy(series_vals: List[float]) -> List[Tuple[float, float]]:
        if not series_vals:
            return []
        n = len(series_vals)
        if n == 1:
            xs = [x0 + usable_w / 2.0]
        else:
            xs = [x0 + (i * usable_w / (n - 1)) for i in range(n)]
        ys = [y1 - ((z - z_min) / (z_max - z_min)) * usable_h for z in series_vals]
        return list(zip(xs, ys))

    if z_min <= 0.0 <= z_max:
        y_zero = y1 - ((0.0 - z_min) / (z_max - z_min)) * usable_h
        draw.line([(x0, y_zero), (x1, y_zero)], fill=(220, 220, 220), width=1)

    for name, zs, color in series_data:
        pts = _to_xy(zs)
        if len(pts) < 2:
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            draw.line([a, b], fill=color, width=2)
        for p in pts:
            r = 2
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)

    legend_x = x0 + 6
    legend_y = y0 + 6
    for name, zs, color in series_data:
        label = f"{name} N={len(zs)}"
        draw.text((legend_x, legend_y), label, fill=color, font=font)
        legend_y += 12

    return img


def _make_composite(
    *,
    image: Image.Image,
    header: str,
    gt_points: List[List[float]],
    model_outputs: Sequence[Tuple[ModelOutput, Tuple[int, int, int]]],
    forward_axis: str,
    flip_lateral: bool,
    panel_width: int,
    line_width: int,
) -> Image.Image:
    model_points = [(out.name, out.trajectory_points, color) for out, color in model_outputs]
    overlay = _render_overlay_multimodel(
        image=image,
        header=header,
        gt_points=gt_points,
        model_points=model_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        line_width=line_width,
    )

    panel = _render_model_panel(
        size=(panel_width, overlay.size[1]),
        header="Model comparison",
        gt_points=gt_points,
        model_outputs=model_outputs,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )

    top = Image.new("RGB", (overlay.size[0] + panel.size[0], overlay.size[1]), (255, 255, 255))
    top.paste(overlay, (0, 0))
    top.paste(panel, (overlay.size[0], 0))

    z_panel_h = max(140, overlay.size[1] // 4)
    z_series = [("GT", gt_points, (0, 200, 0))]
    z_series.extend([(name, pts, color) for name, pts, color in model_points])
    z_panel = _render_z_trend_panel(size=(top.size[0], z_panel_h), series=z_series)

    out = Image.new("RGB", (top.size[0], top.size[1] + z_panel.size[1]), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(z_panel, (0, top.size[1]))
    return out


def _parse_adapter_specs(values: Sequence[str]) -> List[AdapterSpec]:
    if not values:
        raise SystemExit("Provide --adapter name=path (repeatable) for at least 2 models.")
    specs: List[AdapterSpec] = []
    seen: set[str] = set()
    for idx, raw in enumerate(values):
        name = raw
        path = raw
        if "=" in raw:
            name, path = raw.split("=", 1)
        name = (name or "").strip() or f"model{idx + 1}"
        path = (path or "").strip()
        if not path:
            raise SystemExit(f"Invalid adapter spec: {raw}")
        if not Path(path).exists():
            raise SystemExit(f"Adapter not found: {path}")
        base_name = name
        suffix = 2
        while name in seen:
            name = f"{base_name}_{suffix}"
            suffix += 1
        seen.add(name)
        color = _COLOR_PALETTE[len(specs) % len(_COLOR_PALETTE)]
        specs.append(AdapterSpec(name=name, path=path, color=color))
    return specs


def _sample_items(
    *,
    args: argparse.Namespace,
    gt_map: Dict[str, List[List[float]]],
) -> List[SampleItem]:
    items: List[SampleItem] = []

    if args.image:
        for p in args.image:
            img_path = Path(p)
            meta = {"source": "image"}
            key = img_path.stem
            gt_points = _load_gt_from_local_path(
                img_path=img_path, orad_root=args.orad_root, meta=meta, gt_key=args.gt_key
            )
            if gt_points is None:
                for cand in _candidate_image_keys(img_path, orad_root=args.orad_root, meta=meta):
                    if cand in gt_map:
                        gt_points = gt_map[cand]
                        break
            if gt_points is None:
                continue
            items.append(SampleItem(key=key, image_path=img_path, gt_points=gt_points, meta=meta))
        return items

    if args.orad_root is None:
        raise SystemExit("Provide either --image (repeatable) or --orad-root for ORAD-3D sampling.")

    pairs = _iter_orad_pairs(
        orad_root=args.orad_root,
        split=args.split,
        image_folder=args.image_folder,
        max_scan=args.max_scan,
    )
    if not pairs:
        raise SystemExit("No ORAD-3D images found for the given split/image-folder.")

    rng = random.Random(args.seed)
    if int(args.num_samples) > 0:
        rng.shuffle(pairs)

    for seq, ts, img_path in pairs:
        key = f"{args.split}_{seq}_{ts}"
        meta = {"source": "orad3d", "split": args.split, "sequence": seq, "timestamp": ts}
        gt_points = _load_gt_from_local_path(
            img_path=img_path, orad_root=args.orad_root, meta=meta, gt_key=args.gt_key
        )
        if gt_points is None:
            for cand in _candidate_image_keys(img_path, orad_root=args.orad_root, meta=meta):
                if cand in gt_map:
                    gt_points = gt_map[cand]
                    break
        if gt_points is None:
            continue
        items.append(SampleItem(key=key, image_path=img_path, gt_points=gt_points, meta=meta))
        if int(args.num_samples) > 0 and len(items) >= int(args.num_samples):
            break

    return items


def _run_inference_for_adapter(
    *,
    adapter: AdapterSpec,
    items: Sequence[SampleItem],
    args: argparse.Namespace,
) -> Dict[str, ModelOutput]:
    model, processor = _load_model_and_processor(
        base_model=args.base_model,
        adapter=adapter.path,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )

    results: Dict[str, ModelOutput] = {}
    skipped_f = None
    if bool(args.debug_save_skipped):
        skipped_path = args.out_dir / f"skipped_outputs_{adapter.name}.jsonl"
        skipped_f = skipped_path.open("w", encoding="utf-8")

    for idx, item in enumerate(items, start=1):
        try:
            image = Image.open(item.image_path).convert("RGB")
        except Exception:
            results[item.key] = ModelOutput(
                name=adapter.name,
                adapter_path=adapter.path,
                output_text="",
                trajectory_points=[],
                valid=False,
            )
            continue

        inputs, input_len = _prepare_inputs(
            processor,
            image=image,
            system_text=args.system,
            prompt_text=args.prompt,
            use_sharegpt_format=bool(args.use_sharegpt_format),
        )
        if torch.cuda.is_available():
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(args.max_new_tokens)}
        if float(args.temperature) > 0:
            gen_kwargs.update({"do_sample": True, "temperature": float(args.temperature), "top_p": float(args.top_p)})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)

        skip_special = bool(args.skip_special_tokens)
        try:
            full_text = processor.batch_decode(out_ids, skip_special_tokens=skip_special)[0]
        except Exception:
            full_text = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=skip_special)[0]  # type: ignore

        gen_ids = out_ids
        if input_len > 0 and isinstance(out_ids, torch.Tensor) and out_ids.ndim == 2 and out_ids.shape[1] > input_len:
            gen_ids = out_ids[:, input_len:]

        try:
            out_text = processor.batch_decode(gen_ids, skip_special_tokens=skip_special)[0]
        except Exception:
            out_text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=skip_special)[0]  # type: ignore

        out_text = _clean_output_text((out_text or "").strip())
        full_text = _clean_output_text((full_text or "").strip())
        traj_section = _extract_trajectory_section(out_text)

        traj_points: List[List[float]] = []
        valid = False
        if traj_section is not None:
            traj_points = _extract_trajectory_points(traj_section)
            valid = len(traj_points) >= 2

        if not valid and bool(args.debug_print_skipped):
            head = (out_text[:220] + ("..." if len(out_text) > 220 else "")).replace("\n", "\\n")
            print(f"[SKIP] {adapter.name} {idx}/{len(items)} {item.key}: {head}")

        if not valid and skipped_f is not None:
            skipped_f.write(
                json.dumps(
                    {
                        "key": item.key,
                        "image_path": str(item.image_path),
                        "output_text": out_text,
                        "full_text": full_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        results[item.key] = ModelOutput(
            name=adapter.name,
            adapter_path=adapter.path,
            output_text=out_text,
            trajectory_points=traj_points,
            valid=valid,
        )

    if skipped_f is not None:
        skipped_f.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare multiple ORAD-3D VLM LoRA adapters with Z plots.")
    ap.add_argument("--base-model", type=str, default=_DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="Adapter spec as name=path (repeatable).",
    )
    ap.add_argument("--cache-dir", type=str, default="/home/work/byounggun/.cache/hf")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument(
        "--system",
        type=str,
        default=(
            "You are an off-road autonomous driving agent. "
            "Given an input camera image, describe the scene and provide a safe drivable trajectory. "
            "Output the trajectory after a <trajectory> token as a comma-separated list of [x,y,z] points."
        ),
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="I am seeing an off-road driving image. Please generate a safe drivable trajectory for my vehicle to follow.",
    )
    ap.add_argument("--use-sharegpt-format", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--skip-special-tokens", action="store_true")

    ap.add_argument("--debug-save-skipped", action="store_true")
    ap.add_argument("--debug-print-skipped", action="store_true")

    ap.add_argument("--image", action="append", default=None)
    ap.add_argument("--orad-root", type=Path, default=None)
    ap.add_argument("--gt-jsonl", type=Path, default=None)
    ap.add_argument("--gt-key", type=str, default="trajectory_ins")
    ap.add_argument("--split", type=str, default="training", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scan", type=int, default=None)

    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true")
    ap.add_argument("--panel-width", type=int, default=620)
    ap.add_argument("--line-width", type=int, default=4)
    ap.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Keep samples even if some models have no parsed trajectory.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adapters = _parse_adapter_specs(args.adapter)
    if len(adapters) < 2:
        raise SystemExit("Provide at least two adapters to compare.")

    gt_map: Dict[str, List[List[float]]] = {}
    if args.gt_jsonl is not None:
        wanted: set[str] = set()
        if args.image:
            for p in args.image:
                img_path = Path(p)
                wanted.update(_candidate_image_keys(img_path, orad_root=args.orad_root, meta={"source": "image"}))
        elif args.orad_root is not None:
            pairs = _iter_orad_pairs(
                orad_root=args.orad_root,
                split=args.split,
                image_folder=args.image_folder,
                max_scan=args.max_scan,
            )
            for seq, ts, img_path in pairs:
                meta = {"source": "orad3d", "split": args.split, "sequence": seq, "timestamp": ts}
                wanted.update(_candidate_image_keys(img_path, orad_root=args.orad_root, meta=meta))
        gt_map = _load_gt_trajectories_for_items(args.gt_jsonl, wanted_keys=wanted)

    items = _sample_items(args=args, gt_map=gt_map)
    if not items:
        raise SystemExit("No samples with valid GT trajectories found.")

    results_by_model: Dict[str, Dict[str, ModelOutput]] = {}
    for adapter in adapters:
        print(f"[LOAD] {adapter.name} -> {adapter.path}")
        results_by_model[adapter.name] = _run_inference_for_adapter(adapter=adapter, items=items, args=args)

    manifest_path = args.out_dir / "manifest.jsonl"
    saved = 0
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in items:
            model_outputs: List[Tuple[ModelOutput, Tuple[int, int, int]]] = []
            outputs: List[ModelOutput] = []
            missing = False
            for adapter in adapters:
                out = results_by_model.get(adapter.name, {}).get(item.key)
                if out is None:
                    out = ModelOutput(
                        name=adapter.name,
                        adapter_path=adapter.path,
                        output_text="",
                        trajectory_points=[],
                        valid=False,
                    )
                outputs.append(out)
                if not out.valid:
                    missing = True
                model_outputs.append((out, adapter.color))

            if missing and not bool(args.allow_missing_models):
                continue

            header = item.key
            comp = _make_composite(
                image=Image.open(item.image_path).convert("RGB"),
                header=header,
                gt_points=item.gt_points,
                model_outputs=model_outputs,
                forward_axis=str(args.forward_axis),
                flip_lateral=bool(args.flip_lateral),
                panel_width=int(args.panel_width),
                line_width=int(args.line_width),
            )

            saved += 1
            out_png = args.out_dir / f"{saved:04d}_{item.key}.png"
            comp.save(out_png)

            result = SampleResult(
                key=item.key,
                image_path=str(item.image_path),
                gt_trajectory_points=item.gt_points,
                outputs=outputs,
                composite_path=str(out_png),
                meta=item.meta,
            )
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {saved} comparisons -> {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
