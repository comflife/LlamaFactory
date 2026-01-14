#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run VLM inference for ORAD-3D and visualize generated trajectory.

This script:
- Loads a base VLM (e.g., Qwen/Qwen3-VL-2B-Instruct) + a LoRA checkpoint directory.
- Samples ORAD-3D frames (or uses explicit --image paths).
- Runs chat-style multimodal generation.
- Parses <trajectory> output into 3D points.
- Saves a composite PNG only when <trajectory> is present: raw image + GT vs prediction trajectory plots.
- Writes a JSONL manifest with raw outputs and parsed points.
- GT trajectories are loaded from local_path/<ts>.json when available (fallback to --gt-jsonl).
- Defaults to base model Qwen/Qwen3-VL-2B-Instruct with LoRA adapter checkpoint-910 at
  /home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/orpo2/checkpoint-910.

Example (ORAD-3D sampling):
CUDA_VISIBLE_DEVICES=0 python scripts/orad3d_infer_vlm_trajectory_samples_withz.py \
  --orad-root /home/work/datasets/bg/ORAD-3D \
  --split training --image-folder image_data --num-samples 30 \
  --out-dir /home/work/byounggun/LlamaFactory/orad3d_infer_samples_910 \
  --cache-dir /home/work/byounggun/.cache/hf \
  --use-sharegpt-format --temperature 0



python scripts/orad3d_infer_vlm_trajectory_samples.py \
  --orad-root /home/work/datasets/bg/ORAD-3D \
  --split training --image-folder image_data --num-samples 30 \
  --out-dir /home/work/byounggun/LlamaFactory/orad3d_infer_samples_910 \
  --cache-dir /home/work/byounggun/.cache/hf \
  --use-sharegpt-format --temperature 0


"""

from __future__ import annotations

import argparse
import json
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


_TRAJ_TOKEN = "<trajectory>"
_TRAJ_TOKEN_RE = re.compile(r"<\s*trajectory\s*>", re.IGNORECASE)
_POINT_RE = re.compile(r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]")

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
_DEFAULT_ADAPTER_PATH = "/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/orpo2/checkpoint-910"


@dataclass(frozen=True)
class InferenceSample:
    key: str
    image_path: str
    prompt: str
    output_text: str
    trajectory_points: List[List[float]]
    gt_trajectory_points: Optional[List[List[float]]]
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

    # Common ORAD-3D structure: <split>/<sequence>/<image_folder>/<timestamp>.png
    split = str(meta.get("split") or "").strip()
    seq = str(meta.get("sequence") or "").strip()
    ts = str(meta.get("timestamp") or "").strip()
    if split and seq and ts:
        keys.append(_normalize_path_str(f"{split}/{seq}/image_data/{ts}.png"))
        keys.append(_normalize_path_str(f"{split}/{seq}/gt_image/{ts}.png"))

    # Fallback basename match.
    keys.append(_normalize_path_str(p.name))
    # De-dup preserving order.
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
    """Load GT trajectories for only the requested images.

    Expects ShareGPT-like JSONL where each record has:
    - images: ["validation/.../image_data/xxx.png", ...]
    - messages: [..., {"role":"assistant","content":"...<trajectory>..."}]
    """

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
                # Also allow basename-based matching.
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
                # First write wins; these should be unique per image.
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

    # Prefer LLaMAFactory's internal helper if available (it also patches module constants).
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
        return torch.float16  # fallback; from_pretrained(torch_dtype="auto") handles better for large models
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

    # Prefer tokenizer/processor saved under the LoRA checkpoint directory.
    # This avoids special-token mismatches (e.g., added <trajectory>) between training and inference.
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

    # If we had to fall back to a base processor but the checkpoint tokenizer exists, inject it.
    if tokenizer is not None and hasattr(processor, "tokenizer"):
        try:
            processor.tokenizer = tokenizer  # type: ignore[attr-defined]
        except Exception:
            pass

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)

    # Pick a reasonable base class for VLM.
    model = None
    torch_dtype: Any = "auto" if dtype.strip().lower() == "auto" else _parse_dtype(dtype)

    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            # transformers>=4.49 started deprecating `torch_dtype` in favor of `dtype`.
            # Try `dtype` first, then fall back to `torch_dtype` for older versions.
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

    # Apply LoRA adapter.
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def _iter_orad_pairs(
    *,
    orad_root: Path,
    split: str,
    image_folder: str,
    max_scan: Optional[int],
) -> List[Tuple[str, str, Path]]:
    # Returns: (seq, ts, img_path)
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


def _build_messages(prompt_text: str, system_text: str) -> List[Dict[str, Any]]:
    # Chat-style format commonly used by Qwen VL processors.
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
    # In training JSONL, the user content was "<image>\n...".
    # For Qwen3-VL, we MUST still provide the image via structured content so the processor
    # can insert the correct image placeholder tokens. Here we only mirror the textual prefix.
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
    # IMPORTANT: For Qwen3-VL, image placeholder tokens are inserted by the processor/chat template.
    # If we bypass it and pass plain text, we can hit: tokens: 0, features: N.
    # So even in ShareGPT-alignment mode, we keep structured multimodal messages and only
    # mirror the textual "<image>" prefix.
    prompt_text = _maybe_prefix_image_token(prompt_text, use_sharegpt_format=use_sharegpt_format)
    messages = _build_messages(prompt_text, system_text)

    chat_text = None
    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif hasattr(getattr(processor, "tokenizer", None), "apply_chat_template"):
        chat_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: include an <image> hint that matches training.
        chat_text = f"<image>\n{prompt_text}"

    inputs = processor(text=[chat_text], images=[image], return_tensors="pt", padding=True)
    input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    return inputs, input_len


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


def _extract_trajectory_section(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TRAJ_TOKEN_RE.search(text)
    if not m:
        return None
    return text[m.end() :].strip() or ""


def _clean_output_text(text: str) -> str:
    # Some chat/VLM models may emit tool-calling markers; they are not useful for our visualization.
    # Keep this conservative to avoid removing legitimate content.
    if not text:
        return ""
    cleaned = text.replace("<tool_call>", "").replace("</tool_call>", "").replace("<tool_call/>", "")
    return cleaned.strip()


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
    """Map ego-centric XY to image pixels (like scripts/orad3d_make_samples.py).

    - Origin is bottom-center of the image.
    - First point is treated as (0,0).
    - Uses only XY; ignores Z.
    """

    if not points_xyz:
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


def _draw_polyline(
    image: Image.Image,
    uv: List[Tuple[float, float]],
    *,
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


def _shared_traj_scale(
    image_size: Tuple[int, int],
    *,
    forward_axis: str,
    flip_lateral: bool,
    point_sets: Sequence[Optional[Sequence[Sequence[float]]]],
) -> Optional[float]:
    """Pick a common px/m scale so GT + prediction overlays are comparable."""

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

    if not scales:
        return None
    return min(scales)


def _render_overlay_panel(
    *,
    image: Image.Image,
    header: str,
    label: str,
    points_xyz: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
    color: Tuple[int, int, int],
    scale_px_per_meter: Optional[float],
) -> Image.Image:
    base = image.convert("RGB")
    overlay = base.copy()

    uv, scale = _traj_xyz_to_pixels(
        points_xyz,
        overlay.size,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        scale_px_per_meter=scale_px_per_meter,
    )
    _draw_polyline(overlay, uv, color=color, width=3)

    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    bar_h = 24
    draw.rectangle([(0, 0), (overlay.size[0], bar_h)], fill=(20, 20, 20))
    if points_xyz:
        text = f"{header} | {label}  N={len(points_xyz)}  scale={scale:.1f}px/m"
    else:
        text = f"{header} | {label}  N=0 (no points)"
    draw.text((10, 6), text, fill=(255, 255, 255), font=font)
    return overlay


def _render_side_by_side_overlays(
    *,
    image: Image.Image,
    header: str,
    pred_points_xyz: List[List[float]],
    gt_points_xyz: Optional[List[List[float]]],
    forward_axis: str,
    flip_lateral: bool,
) -> Image.Image:
    base = image.convert("RGB")
    shared_scale = _shared_traj_scale(
        base.size,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        point_sets=[gt_points_xyz, pred_points_xyz],
    )

    panels: List[Image.Image] = []
    if gt_points_xyz is not None:
        panels.append(
            _render_overlay_panel(
                image=base,
                header=header,
                label="GT",
                points_xyz=gt_points_xyz,
                forward_axis=forward_axis,
                flip_lateral=flip_lateral,
                color=(0, 200, 0),
                scale_px_per_meter=shared_scale,
            )
        )

    panels.append(
        _render_overlay_panel(
            image=base,
            header=header,
            label="Inference",
            points_xyz=pred_points_xyz,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
            color=(220, 20, 60),
            scale_px_per_meter=shared_scale,
        )
    )

    gap = 12 if len(panels) > 1 else 0
    out_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    out_h = panels[0].height if panels else base.height
    out = Image.new("RGB", (out_w, out_h), (255, 255, 255))

    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + gap

    if len(panels) > 1:
        draw = ImageDraw.Draw(out)
        for i in range(1, len(panels)):
            line_x = panels[0].width + (i - 1) * (gap + panels[0].width) + (gap // 2)
            draw.line([(line_x, 0), (line_x, out_h)], fill=(220, 220, 220), width=1)

    return out


def _render_overlay(
    *,
    image: Image.Image,
    header: str,
    points_xyz: List[List[float]],
    gt_points_xyz: Optional[List[List[float]]],
    forward_axis: str,
    flip_lateral: bool,
) -> Image.Image:
    base = image.convert("RGB")
    overlay = base.copy()

    # Draw GT first (green), then prediction (red) on top.
    gt_n = 0
    if gt_points_xyz:
        gt_uv, _ = _traj_xyz_to_pixels(
            gt_points_xyz,
            overlay.size,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
            scale_px_per_meter=None,
        )
        _draw_polyline(overlay, gt_uv, color=(0, 255, 0), width=3)
        gt_n = len(gt_points_xyz)

    uv, scale = _traj_xyz_to_pixels(
        points_xyz,
        overlay.size,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        scale_px_per_meter=None,
    )
    _draw_polyline(overlay, uv, color=(255, 0, 0), width=3)

    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    draw.text(
        (12, 10),
        f"{header}  pred:N={len(points_xyz)}  gt:N={gt_n}  scale={scale:.1f}px/m",
        fill=(255, 255, 255),
        font=font,
    )
    return overlay


def _draw_trajectory_plot(
    *,
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    points_xyz: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
    color: Tuple[int, int, int] = (220, 20, 60),
) -> None:
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)

    if len(points_xyz) < 2:
        draw.text((x0 + 8, y0 + 8), "(no trajectory parsed)", fill=(0, 0, 0))
        return

    # Convert to ego-relative forward/lateral (subtract first point)
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

    max_fwd = max(forward) if forward else 1.0
    max_lat = max(abs(v) for v in lateral) if lateral else 1.0
    max_fwd = max(max_fwd, 1e-3)
    max_lat = max(max_lat, 1e-3)

    margin = 10
    usable_w = max(1, w - 2 * margin)
    usable_h = max(1, h - 2 * margin)

    scale = min(usable_h / max_fwd, (usable_w / 2.0) / max_lat)
    scale = max(0.5, min(scale, 200.0))

    cx = x0 + w / 2.0
    bottom = y0 + h - margin

    pts: List[Tuple[float, float]] = []
    for f, lat in zip(forward, lateral):
        px = cx + lat * scale
        py = bottom - f * scale
        pts.append((px, py))

    # Axis line
    draw.line([(cx, y0 + margin), (cx, y0 + h - margin)], fill=(220, 220, 220), width=1)
    draw.line([(x0 + margin, bottom), (x0 + w - margin, bottom)], fill=(220, 220, 220), width=1)

    # Polyline
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill=color, width=3)

    # Points
    for p in pts:
        r = 3
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)

    draw.text((x0 + 8, y0 + 6), f"traj: N={len(points_xyz)}  scale={scale:.1f}px/m", fill=(0, 0, 0))


def _render_panel(
    *,
    size: Tuple[int, int],
    header: str,
    prompt: str,
    output_text: str,
    traj_points: List[List[float]],
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

    # Text area
    plot_h = int(h * 0.42)
    text_h = h - plot_h - y - pad

    def draw_block(title: str, text: str, y0: int) -> int:
        draw.text((pad, y0), title, fill=(0, 0, 0), font=font)
        y0 += 14
        max_chars = max(30, (w - 2 * pad) // 6)
        lines = _wrap_lines(text, width_chars=max_chars)
        for line in lines:
            if y0 >= header_h + 8 + text_h:
                draw.text((pad, y0), "...", fill=(0, 0, 0), font=font)
                return y0 + 16
            draw.text((pad, y0), line, fill=(0, 0, 0), font=font)
            y0 += 14
        return y0 + 6

    y = draw_block("Prompt:", prompt, y)
    y = draw_block("Output:", output_text, y)

    # Trajectory plot box
    plot_top = h - plot_h - pad
    plot_box = (pad, plot_top, w - pad, h - pad)
    _draw_trajectory_plot(draw=draw, box=plot_box, points_xyz=traj_points, forward_axis=forward_axis, flip_lateral=flip_lateral)

    return img


def _render_traj_compare_panel(
    *,
    size: Tuple[int, int],
    header: str,
    gt_points: Optional[List[List[float]]],
    pred_points: List[List[float]],
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

    plot_top = header_h + pad
    plot_h = h - plot_top - pad
    plot_w = (w - 3 * pad) // 2

    left_box = (pad, plot_top, pad + plot_w, plot_top + plot_h)
    right_box = (pad * 2 + plot_w, plot_top, pad * 2 + plot_w * 2, plot_top + plot_h)

    gt_count = len(gt_points) if gt_points else 0
    pred_count = len(pred_points)

    draw.text((left_box[0], plot_top - 16), f"GT (N={gt_count})", fill=(0, 120, 0), font=font)
    draw.text((right_box[0], plot_top - 16), f"Pred (N={pred_count})", fill=(220, 20, 60), font=font)

    _draw_trajectory_plot(
        draw=draw,
        box=left_box,
        points_xyz=gt_points or [],
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        color=(0, 160, 0),
    )
    _draw_trajectory_plot(
        draw=draw,
        box=right_box,
        points_xyz=pred_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        color=(220, 20, 60),
    )

    return img


def _render_z_trend_panel(
    *,
    size: Tuple[int, int],
    gt_points: Optional[List[List[float]]],
    pred_points: List[List[float]],
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

    def _z_series(points: Optional[List[List[float]]]) -> List[float]:
        if not points or len(points) < 2:
            return []
        z0 = float(points[0][2]) if len(points[0]) > 2 else 0.0
        out: List[float] = []
        for p in points:
            if not isinstance(p, list) or len(p) < 3:
                out.append(0.0)
            else:
                out.append(float(p[2]) - z0)
        return out

    gt_z = _z_series(gt_points)
    pred_z = _z_series(pred_points)

    all_z = gt_z + pred_z
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

    # Zero line
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

    _draw_series(_to_xy(gt_z), color=(0, 160, 0))
    _draw_series(_to_xy(pred_z), color=(220, 20, 60))

    draw.text((x0, y1 + 4), f"GT N={len(gt_z)}", fill=(0, 120, 0), font=font)
    draw.text((x0 + 120, y1 + 4), f"Pred N={len(pred_z)}", fill=(220, 20, 60), font=font)

    return img


def _make_image_with_traj_comparison(
    *,
    image: Image.Image,
    header: str,
    gt_points: Optional[List[List[float]]],
    pred_points: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
) -> Image.Image:
    base = image.convert("RGB")
    panel_w = max(560, base.size[0] // 2)
    panel = _render_traj_compare_panel(
        size=(panel_w, base.size[1]),
        header=header,
        gt_points=gt_points,
        pred_points=pred_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )
    top = Image.new("RGB", (base.size[0] + panel.size[0], base.size[1]), (255, 255, 255))
    top.paste(base, (0, 0))
    top.paste(panel, (base.size[0], 0))

    z_panel_h = max(140, base.size[1] // 4)
    z_panel = _render_z_trend_panel(
        size=(top.size[0], z_panel_h),
        gt_points=gt_points,
        pred_points=pred_points,
    )

    out = Image.new("RGB", (top.size[0], top.size[1] + z_panel.size[1]), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(z_panel, (0, top.size[1]))
    return out


def _make_composite(
    *,
    image: Image.Image,
    header: str,
    prompt: str,
    output_text: str,
    traj_points: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
) -> Image.Image:
    base = image.convert("RGB")
    panel_w = max(560, base.size[0] // 2)
    panel = _render_panel(
        size=(panel_w, base.size[1]),
        header=header,
        prompt=prompt,
        output_text=output_text,
        traj_points=traj_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )

    out = Image.new("RGB", (base.size[0] + panel.size[0], base.size[1]), (255, 255, 255))
    out.paste(base, (0, 0))
    out.paste(panel, (base.size[0], 0))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Infer ORAD-3D VLM and visualize trajectory output.")

    ap.add_argument("--base-model", type=str, default=_DEFAULT_BASE_MODEL, help=f"HF model id or local path (default: {_DEFAULT_BASE_MODEL})")
    ap.add_argument(
        "--adapter",
        type=str,
        default=_DEFAULT_ADAPTER_PATH,
        help=f"LoRA checkpoint dir (default: {_DEFAULT_ADAPTER_PATH})",
    )
    ap.add_argument("--cache-dir", type=str, default="/home/work/byounggun/.cache/hf")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", type=str, default="auto", help="Transformers device_map (default: auto)")
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument(
        "--system",
        type=str,
        default=(
            "You are an off-road autonomous driving agent. "
            "Given an input camera image, describe the scene and provide a safe drivable trajectory. "
            "Output the trajectory after a <trajectory> token as a comma-separated list of [x,y,z] points."
        ),
        help="System message. Set to empty string to disable.",
    )

    ap.add_argument(
        "--prompt",
        type=str,
        default="I am seeing an off-road driving image. Please generate a safe drivable trajectory for my vehicle to follow.",
        help="User prompt text.",
    )

    ap.add_argument(
        "--use-sharegpt-format",
        action="store_true",
        help="Align with training text by prefixing user text with '<image>\\n' (keeps multimodal chat template for Qwen3-VL).",
    )

    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)

    ap.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Decode with skip_special_tokens=True (cleaner text; recommended).",
    )

    ap.add_argument(
        "--debug-save-skipped",
        action="store_true",
        help="Write skipped model outputs to out_dir/skipped_outputs.jsonl for debugging.",
    )

    ap.add_argument(
        "--debug-print-skipped",
        action="store_true",
        help="Print the first 220 chars of output when a sample is skipped.",
    )

    ap.add_argument("--image", action="append", default=None, help="Explicit image path (repeatable)")

    ap.add_argument("--orad-root", type=Path, default=None)
    ap.add_argument(
        "--gt-jsonl",
        type=Path,
        default=None,
        help="Optional ShareGPT-style JSONL containing GT trajectories (e.g., /data3/orad3d_vlm/orad3d_all.jsonl).",
    )
    ap.add_argument(
        "--gt-key",
        type=str,
        default="trajectory_ins",
        help="Key in local_path/*.json to use as GT trajectory when available.",
    )
    ap.add_argument("--split", type=str, default="validation", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scan", type=int, default=None)

    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true")

    ap.add_argument("--out-dir", type=Path, required=True)

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    skipped_path = args.out_dir / "skipped_outputs.jsonl"

    model, processor = _load_model_and_processor(
        base_model=args.base_model,
        adapter=args.adapter,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )

    # Build input list
    items: List[Tuple[str, Path, Dict[str, Any]]] = []
    if args.image:
        for p in args.image:
            img_path = Path(p)
            items.append((img_path.stem, img_path, {"source": "image"}))
    else:
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
        n = min(int(args.num_samples), len(pairs))
        chosen = rng.sample(pairs, k=n) if len(pairs) > n else pairs
        for seq, ts, img_path in chosen:
            key = f"{args.split}_{seq}_{ts}"
            items.append((key, img_path, {"source": "orad3d", "split": args.split, "sequence": seq, "timestamp": ts}))

    results: List[InferenceSample] = []
    saved = 0

    # Preload GT trajectories only for selected items when requested.
    gt_map: Dict[str, List[List[float]]] = {}
    if args.gt_jsonl is not None:
        wanted: set[str] = set()
        for _, p, meta in items:
            wanted.update(_candidate_image_keys(p, orad_root=args.orad_root, meta=meta))
        gt_map = _load_gt_trajectories_for_items(args.gt_jsonl, wanted_keys=wanted)

    # Keep a debug log of skipped outputs when requested.
    skipped_f = None
    if bool(args.debug_save_skipped):
        skipped_f = skipped_path.open("w", encoding="utf-8")

    for idx, (key, img_path, meta) in enumerate(items, start=1):
        image = Image.open(img_path).convert("RGB")

        inputs, input_len = _prepare_inputs(
            processor,
            image=image,
            system_text=args.system,
            prompt_text=args.prompt,
            use_sharegpt_format=bool(args.use_sharegpt_format),
        )
        # Move inputs to the same device as model inputs (device_map=auto spreads params; tensors can stay on cuda:0)
        if torch.cuda.is_available():
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(args.max_new_tokens),
        }
        if float(args.temperature) > 0:
            gen_kwargs.update({"do_sample": True, "temperature": float(args.temperature), "top_p": float(args.top_p)})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)

        skip_special = bool(args.skip_special_tokens)
        try:
            # Decode both full text (debug) and generated-only text (for parsing).
            full_text = processor.batch_decode(out_ids, skip_special_tokens=skip_special)[0]
        except Exception:
            full_text = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=skip_special)[0]  # type: ignore

        # IMPORTANT: Parse only the generated continuation.
        # If we parse the full decoded sequence, `<trajectory>` inside the system instruction can cause false positives.
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
        if traj_section is None:
            print(f"[SKIP] {idx}/{len(items)} {key}: no <trajectory> token")
            if bool(args.debug_print_skipped):
                print("  output(head):", (out_text[:220] + ("..." if len(out_text) > 220 else "")).replace("\n", "\\n"))
            if skipped_f is not None:
                skipped_f.write(
                    json.dumps(
                        {
                            "key": key,
                            "image_path": str(img_path),
                            "output_text": out_text,
                            "full_text": full_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            continue

        traj_points = _extract_trajectory_points(traj_section)
        if len(traj_points) < 2:
            print(f"[SKIP] {idx}/{len(items)} {key}: <trajectory> present but no points parsed")
            if bool(args.debug_print_skipped):
                print("  traj_section(head):", (traj_section[:220] + ("..." if len(traj_section) > 220 else "")).replace("\n", "\\n"))
            if skipped_f is not None:
                skipped_f.write(
                    json.dumps(
                        {"key": key, "image_path": str(img_path), "output_text": out_text, "traj_section": traj_section},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            continue

        gt_points: Optional[List[List[float]]] = _load_gt_from_local_path(
            img_path=img_path,
            orad_root=args.orad_root,
            meta=meta,
            gt_key=args.gt_key,
        )
        if gt_points is None:
            for cand in _candidate_image_keys(img_path, orad_root=args.orad_root, meta=meta):
                if cand in gt_map:
                    gt_points = gt_map[cand]
                    break

        header = key
        overlay = _make_image_with_traj_comparison(
            image=image,
            header=header,
            gt_points=gt_points,
            pred_points=traj_points,
            forward_axis=args.forward_axis,
            flip_lateral=bool(args.flip_lateral),
        )

        saved += 1
        out_png = args.out_dir / f"{saved:03d}_{key}.png"
        overlay.save(out_png)

        sample = InferenceSample(
            key=key,
            image_path=str(img_path),
            prompt=args.prompt,
            output_text=out_text,
            trajectory_points=traj_points,
            gt_trajectory_points=gt_points,
            composite_path=str(out_png),
            meta={**meta, "system": args.system},
        )
        results.append(sample)

        print(f"[OK] {idx}/{len(items)} -> {out_png.name} (traj_points={len(traj_points)})")

    with manifest_path.open("w", encoding="utf-8") as f:
        for s in results:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    if skipped_f is not None:
        skipped_f.close()
        print(f"Skipped outputs: {skipped_path}")

    print(f"[DONE] wrote {len(results)} samples")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
