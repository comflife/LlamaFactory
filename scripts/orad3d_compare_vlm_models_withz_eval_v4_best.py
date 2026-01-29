#!/usr/bin/env python3
"""Run ORAD-3D inference for multiple LoRA adapters and save GT + predictions.

This version skips all composite rendering/metrics and writes per-adapter JSONL outputs.
Each adapter gets its own folder (based on adapter name), and results are appended
incrementally so the job can be resumed safely.

Example:
python scripts/orad3d_compare_vlm_models_withz_eval_v4.py \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --adapter sft_refine=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_refine/checkpoint-3132 \
  --adapter sft=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_8/checkpoint-3132 \
  --adapter orpo=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/orpo2/checkpoint-3132 \
  --orad-root /home/work/datasets/bg/ORAD-3D \
  --split testing --image-folder image_data --num-samples 5 \
  --out-dir /home/work/byounggun/LlamaFactory/orad3d_compare_models_withz_final \
  --cache-dir /home/work/byounggun/.cache/hf \
  --use-sharegpt-format --temperature 1e-6 \
  --hit-threshold 2.0 --frames-per-point 7 --auto-frames-per-point --failure-threshold 2.0

python scripts/orad3d_compare_vlm_models_withz_eval_v4_save.py \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --adapter sft_refine=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_refine/checkpoint-3132 \
  --adapter sft=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_8/checkpoint-3132 \
  --adapter orpo=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/orpo2/checkpoint-3132 \
  --orad-root /home/work/datasets/bg/ORAD-3D \
  --split testing --image-folder image_data --num-samples all \
  --out-dir /home/work/byounggun/LlamaFactory/orad3d_adapter_save \
  --cache-dir /home/work/byounggun/.cache/hf \
  --use-sharegpt-format --temperature 1e-6 \
  --hit-threshold 2.0 --frames-per-point 7 --auto-frames-per-point --failure-threshold 2.0

"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import math
import os
import random
import re
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


def _safe_dir_name(name: str) -> str:
    raw = (name or "").strip()
    raw = raw.replace(os.sep, "_").replace("/", "_")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return raw.strip("._-") or "adapter"


def _load_manifest_keys(manifest_path: Path) -> Set[str]:
    if not manifest_path.exists():
        return set()
    done_keys: Set[str] = set()
    bad_lines = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            key = record.get("key")
            if isinstance(key, str) and key:
                done_keys.add(key)
    if bad_lines:
        print(f"[WARN] {manifest_path} had {bad_lines} invalid lines; ignoring them.")
    return done_keys


@dataclass(frozen=True)
class Calib:
    fx: float
    fy: float
    cx: float
    cy: float
    R: List[List[float]]  # 3x3 camera extrinsic rotation
    t: List[float]  # 3x1 camera extrinsic translation
    lidar_R: Optional[List[List[float]]] = None  # 3x3 lidar->vehicle rotation
    lidar_t: Optional[List[float]] = None  # 3x1 lidar translation (optional)


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
    lidar_r_vals: List[float] = []
    lidar_t_vals: List[float] = []

    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if line.startswith("cam_K:"):
                k_vals = _parse_floats_from_line("cam_K:", line)
            elif line.startswith("cam_RT:"):
                rt_vals = _parse_floats_from_line("cam_RT:", line)
            elif line.startswith("lidar_R:"):
                lidar_r_vals = _parse_floats_from_line("lidar_R:", line)
            elif line.startswith("lidar_T:"):
                lidar_t_vals = _parse_floats_from_line("lidar_T:", line)
        if len(k_vals) != 9 or len(rt_vals) != 16:
            return None
        fx, _, cx, _, fy, cy, _, _, _ = k_vals
        R = [
            [rt_vals[0], rt_vals[1], rt_vals[2]],
            [rt_vals[4], rt_vals[5], rt_vals[6]],
            [rt_vals[8], rt_vals[9], rt_vals[10]],
        ]
        t = [rt_vals[3], rt_vals[7], rt_vals[11]]
        lidar_R = None
        lidar_t = None
        if len(lidar_r_vals) == 9:
            lidar_R = [
                [lidar_r_vals[0], lidar_r_vals[1], lidar_r_vals[2]],
                [lidar_r_vals[3], lidar_r_vals[4], lidar_r_vals[5]],
                [lidar_r_vals[6], lidar_r_vals[7], lidar_r_vals[8]],
            ]
        if len(lidar_t_vals) >= 3:
            lidar_t = [lidar_t_vals[0], lidar_t_vals[1], lidar_t_vals[2]]
        return Calib(fx=fx, fy=fy, cx=cx, cy=cy, R=R, t=t, lidar_R=lidar_R, lidar_t=lidar_t)
    except Exception:
        return None


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

def _infer_seq_dir_for_item(item: SampleItem, *, orad_root: Optional[Path]) -> Optional[Path]:
    parent = item.image_path.parent
    if parent.name in ("image_data", "gt_image"):
        return parent.parent
    split = str(item.meta.get("split") or "").strip()
    seq = str(item.meta.get("sequence") or "").strip()
    if orad_root is not None and split and seq:
        return orad_root / split / seq
    return None


def _apply_rigid_transform(
    points_xyz: List[List[float]],
    R: List[List[float]],
    t: Optional[List[float]] = None,
) -> List[List[float]]:
    if not points_xyz:
        return []
    t = t or [0.0, 0.0, 0.0]
    out: List[List[float]] = []
    for p in points_xyz:
        if len(p) < 3:
            continue
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        out.append(
            [
                R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0],
                R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1],
                R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2],
            ]
        )
    return out


def _project_points_with_calib(points: List[List[float]], calib: Calib) -> List[Tuple[float, float]]:
    if not points:
        return []

    def project(points_xyz: List[List[float]], R: List[List[float]], t: List[float]) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for p in points_xyz:
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

    def best_projection(points_xyz: List[List[float]]) -> List[Tuple[float, float]]:
        direct = project(points_xyz, calib.R, calib.t)
        inverse = project(points_xyz, Rt, t_inv)
        return inverse if len(inverse) > len(direct) else direct

    candidates: List[List[List[float]]] = [points]
    if calib.lidar_R is not None:
        candidates.append(_apply_rigid_transform(points, calib.lidar_R, calib.lidar_t))

    best: List[Tuple[float, float]] = []
    for pts in candidates:
        proj = best_projection(pts)
        if len(proj) > len(best):
            best = proj
    return best



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
    font = ImageFont.load_default()
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

    unit_label = "m"
    if hasattr(draw, "textlength"):
        unit_w = int(draw.textlength(unit_label, font=font))
    else:
        unit_w = len(unit_label) * 6
    draw.text((x1 - unit_w - 6, y0 + 4), unit_label, fill=(110, 110, 110), font=font)

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



def _l2_point_xy(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _l2_point_xyz(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    az = float(a[2]) if len(a) > 2 else 0.0
    bz = float(b[2]) if len(b) > 2 else 0.0
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)




def _cumulative_distances_xy(points: List[List[float]]) -> List[float]:
    if not points:
        return []
    out = [0.0]
    for i in range(1, len(points)):
        out.append(out[-1] + _l2_point_xy(points[i - 1], points[i]))
    return out


def _interp_at_s(points: List[List[float]], s_query: float, s_vals: Optional[List[float]] = None) -> List[float]:
    """Return interpolated point at arc-length s (meters) along XY distance."""
    n = len(points)
    if n == 0:
        return [0.0, 0.0, 0.0]
    if n == 1:
        p = points[0]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]

    if s_vals is None or len(s_vals) != n:
        s_vals = _cumulative_distances_xy(points)
    if not s_vals:
        p = points[0]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]

    s_total = s_vals[-1]
    if s_total <= 0.0:
        p = points[0]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]

    s = float(s_query)
    if s <= 0.0:
        p = points[0]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]
    if s >= s_total:
        p = points[-1]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]

    idx = bisect.bisect_left(s_vals, s)
    if idx <= 0:
        p = points[0]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]
    if idx >= n:
        p = points[-1]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]

    s0 = s_vals[idx - 1]
    s1 = s_vals[idx]
    if s1 <= s0 + 1e-9:
        p = points[idx]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]

    a = (s - s0) / (s1 - s0)
    p0 = points[idx - 1]
    p1 = points[idx]

    def get(p: List[float], k: int) -> float:
        return float(p[k]) if len(p) > k else 0.0

    return [
        (1 - a) * get(p0, 0) + a * get(p1, 0),
        (1 - a) * get(p0, 1) + a * get(p1, 1),
        (1 - a) * get(p0, 2) + a * get(p1, 2),
    ]


def _interp_at_frac(points: List[List[float]], frac: float) -> List[float]:
    """Return interpolated point at progress frac in [0,1] along point index axis."""
    n = len(points)
    if n == 0:
        return [0.0, 0.0, 0.0]
    if n == 1:
        p = points[0]
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]) if len(p) > 2 else 0.0,
        ]

    frac = max(0.0, min(1.0, float(frac)))
    x = frac * (n - 1)
    i0 = int(math.floor(x))
    i1 = min(i0 + 1, n - 1)
    a = x - i0

    p0 = points[i0]
    p1 = points[i1]

    def get(p: List[float], k: int) -> float:
        return float(p[k]) if len(p) > k else 0.0

    return [
        (1 - a) * get(p0, 0) + a * get(p1, 0),
        (1 - a) * get(p0, 1) + a * get(p1, 1),
        (1 - a) * get(p0, 2) + a * get(p1, 2),
    ]



def _resample_points_by_index(points: List[List[float]], target_len: int) -> List[List[float]]:
    if target_len <= 0 or not points:
        return []
    if target_len == 1:
        return [_interp_at_frac(points, 0.0)]
    if len(points) == 1:
        p0 = points[0]
        out = [
            float(p0[0]),
            float(p0[1]),
            float(p0[2]) if len(p0) > 2 else 0.0,
        ]
        return [list(out) for _ in range(target_len)]
    return [_interp_at_frac(points, i / (target_len - 1)) for i in range(target_len)]



def _resample_points(points: List[List[float]], target_len: int) -> List[List[float]]:
    if target_len <= 0 or not points:
        return []
    if target_len == 1:
        return [_interp_at_s(points, 0.0)]
    s_vals = _cumulative_distances_xy(points)
    if not s_vals:
        return []
    s_total = s_vals[-1]
    if s_total <= 1e-6:
        p0 = _interp_at_s(points, 0.0, s_vals)
        return [list(p0) for _ in range(target_len)]
    return [
        _interp_at_s(points, (i / (target_len - 1)) * s_total, s_vals)
        for i in range(target_len)
    ]


def _align_pred_to_gt(pred_points: List[List[float]], gt_points: List[List[float]]) -> List[List[float]]:
    if not pred_points or len(gt_points) < 2:
        return pred_points
    s_gt = _cumulative_distances_xy(gt_points)
    if not s_gt or s_gt[-1] <= 1e-6:
        return pred_points
    s_pred = _cumulative_distances_xy(pred_points)
    return [_interp_at_s(pred_points, s, s_pred) for s in s_gt]


def _parse_horizon_seconds(raw: str) -> List[Tuple[str, float]]:
    if not raw:
        return []
    parts = re.split(r"[\s,]+", raw.strip())
    values: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            v = float(p)
        except Exception:
            continue
        if v > 0:
            values.append(v)
    if not values:
        return []

    out: List[Tuple[str, float]] = []
    seen: set[str] = set()
    for v in values:
        label = f"{v:g}"
        if label in seen:
            continue
        seen.add(label)
        out.append((label, v))
    return out


def _compute_pointwise_metrics(
    pred_points: List[List[float]],
    gt_points: List[List[float]],
    *,
    horizons: Sequence[Tuple[str, float]],
    step_sec: float,
    dist_fn: Any,
    failure_threshold: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    if len(pred_points) < 2 or len(gt_points) < 2:
        return {}
    n = min(len(pred_points), len(gt_points))
    if n < 2:
        return {}

    pred = pred_points[:n]
    gt = gt_points[:n]
    dists = [dist_fn(g, p) for g, p in zip(gt, pred)]
    if not dists:
        return {}

    metrics: Dict[str, Optional[float]] = {}

    step_sec = max(float(step_sec), 1e-6)
    steps_per_sec = max(1, int(round(1.0 / step_sec)))

    if failure_threshold is not None:
        metrics["failure_1s"] = 1.0 if dists[-1] > failure_threshold else 0.0

    if horizons:
        for label, seconds in horizons:
            horizon_steps = max(1, int(round(float(seconds) / step_sec)))
            start = max(0, horizon_steps - steps_per_sec)
            end = horizon_steps
            key = f"ADE_{label}s"
            if end <= n and end > start:
                seg = dists[start:end]
                metrics[key] = sum(seg) / len(seg)
            else:
                metrics[key] = None

    return metrics


def _average_horizon_metrics(
    horizon_metrics: Dict[str, Optional[float]],
    horizon_keys: Sequence[str],
) -> Optional[float]:
    if not horizon_keys:
        return None
    values = [horizon_metrics.get(key) for key in horizon_keys]
    if any(v is None for v in values):
        return None
    return float(sum(v for v in values if v is not None)) / len(values)


def _load_pose_timestamps(seq_dir: Path) -> List[int]:
    pose_path = seq_dir / "poses.txt"
    if not pose_path.is_file():
        return []
    timestamps: List[int] = []
    for raw_line in pose_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if not parts:
            continue
        try:
            ts = int(parts[0])
        except Exception:
            continue
        timestamps.append(ts)
    timestamps.sort()
    return timestamps


def _median_frame_dt_sec(timestamps: Sequence[int]) -> Optional[float]:
    if len(timestamps) < 2:
        return None
    diffs = [(b - a) / 1000.0 for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
    if not diffs:
        return None
    return float(statistics.median(diffs))


def _get_seq_point_step_sec(
    seq_dir: Path,
    *,
    frames_per_point: int,
    fallback: float,
    cache: Dict[str, float],
) -> float:
    if frames_per_point <= 0:
        return fallback
    key = str(seq_dir)
    cached = cache.get(key)
    if cached is not None:
        return cached
    timestamps = _load_pose_timestamps(seq_dir)
    dt_sec = _median_frame_dt_sec(timestamps)
    if dt_sec is None or dt_sec <= 0:
        cache[key] = fallback
    else:
        cache[key] = dt_sec * frames_per_point
    return cache[key]


def _load_pose_xy(seq_dir: Path) -> List[Tuple[int, float, float]]:
    pose_path = seq_dir / "poses.txt"
    if not pose_path.is_file():
        return []
    out: List[Tuple[int, float, float]] = []
    for raw_line in pose_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            ts = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
        except Exception:
            continue
        out.append((ts, x, y))
    out.sort(key=lambda v: v[0])
    return out


def _median_frame_displacement(poses: Sequence[Tuple[int, float, float]]) -> Optional[float]:
    if len(poses) < 2:
        return None
    diffs: List[float] = []
    for (t0, x0, y0), (t1, x1, y1) in zip(poses[:-1], poses[1:]):
        if t1 <= t0:
            continue
        diffs.append(math.hypot(x1 - x0, y1 - y0))
    if not diffs:
        return None
    return float(statistics.median(diffs))


def _collect_traj_step_distances(seq_dir: Path, sample_limit: int) -> List[float]:
    local_dir = seq_dir / "local_path"
    if not local_dir.is_dir():
        return []
    files = sorted(local_dir.glob("*.json"))
    if not files:
        return []
    if sample_limit > 0 and len(files) > sample_limit:
        stride = max(1, len(files) // sample_limit)
        files = files[::stride][:sample_limit]

    dists: List[float] = []
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        traj = obj.get("trajectory_ins") or []
        if len(traj) < 2:
            continue
        for a, b in zip(traj[:-1], traj[1:]):
            try:
                dists.append(_l2_point_xy(a, b))
            except Exception:
                continue
    return dists


def _estimate_frames_per_point_for_seq(
    seq_dir: Path,
    *,
    sample_limit: int,
    max_k: int,
) -> Optional[Dict[str, Any]]:
    poses = _load_pose_xy(seq_dir)
    frame_disp = _median_frame_displacement(poses)
    if frame_disp is None or frame_disp <= 0:
        return None

    step_dists = _collect_traj_step_distances(seq_dir, sample_limit)
    if not step_dists:
        return None
    traj_step = float(statistics.median(step_dists))

    max_k = max(1, int(max_k))
    best_k = 1
    best_err = abs(traj_step - frame_disp)
    for k in range(2, max_k + 1):
        err = abs(traj_step - (frame_disp * k))
        if err < best_err:
            best_err = err
            best_k = k

    ratio = traj_step / frame_disp if frame_disp > 0 else None
    return {
        "seq": seq_dir.name,
        "frames_per_point": best_k,
        "ratio": ratio,
        "frame_disp": frame_disp,
        "traj_step": traj_step,
    }


def _estimate_frames_per_point_from_items(
    items: Sequence[SampleItem],
    *,
    orad_root: Optional[Path],
    sample_limit: int,
    max_k: int,
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    seq_dirs: List[Path] = []
    seen: set[str] = set()
    for item in items:
        seq_dir = _infer_seq_dir_for_item(item, orad_root=orad_root)
        if seq_dir is None:
            continue
        key = str(seq_dir)
        if key in seen:
            continue
        seen.add(key)
        seq_dirs.append(seq_dir)

    rows: List[Dict[str, Any]] = []
    for seq_dir in sorted(seq_dirs, key=lambda p: p.name):
        row = _estimate_frames_per_point_for_seq(seq_dir, sample_limit=sample_limit, max_k=max_k)
        if row is None:
            continue
        rows.append(row)

    if not rows:
        return None, rows

    frames = [row["frames_per_point"] for row in rows if row.get("frames_per_point")]
    if not frames:
        return None, rows

    median_k = int(statistics.median(frames))
    return median_k, rows


def _compute_adapter_metrics(
    *,
    items: Sequence[SampleItem],
    adapters: Sequence[AdapterSpec],
    results_by_model: Dict[str, Dict[str, ModelOutput]],
    hit_threshold: float,
    horizons: Sequence[Tuple[str, float]],
    item_step_sec: Dict[str, float],
    traj_step_sec: float,
    frames_per_point: int,
    failure_threshold: float,
    collect_samples: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    if not items:
        return [], {}

    metrics: List[Dict[str, Any]] = []
    sample_rows: Dict[str, List[Dict[str, Any]]] = {}
    horizon_keys = [f"ADE_{label}s" for label, _ in horizons]
    for adapter in adapters:
        rows: List[Dict[str, Any]] = []

        horizon_sums_xy: Dict[str, float] = {key: 0.0 for key in horizon_keys}
        horizon_counts_xy: Dict[str, int] = {key: 0 for key in horizon_keys}
        horizon_sums_xyz: Dict[str, float] = {key: 0.0 for key in horizon_keys}
        horizon_counts_xyz: Dict[str, int] = {key: 0 for key in horizon_keys}
        point_valid_xy = 0
        point_valid_xyz = 0
        failure_count_xy = 0
        failure_count_xyz = 0

        for item in items:
            out = results_by_model.get(adapter.name, {}).get(item.key)
            if out is None or not out.valid or len(out.trajectory_points) < 1:
                continue
            pred = out.trajectory_points
            gt = item.gt_points
            if len(gt) < 2 or len(pred) < 1:
                continue

            s_gt = _cumulative_distances_xy(gt)
            if not s_gt:
                continue
            s_gt_end = s_gt[-1]
            if s_gt_end <= 1e-6:
                continue

            step_sec = float(item_step_sec.get(item.key, traj_step_sec))
            aligned_pred = pred
            if len(pred) >= 2 and len(pred) != len(gt):
                aligned_pred = _resample_points_by_index(pred, len(gt))
            point_metrics_xy = _compute_pointwise_metrics(
                aligned_pred,
                gt,
                horizons=horizons,
                step_sec=step_sec,
                dist_fn=_l2_point_xy,
                failure_threshold=failure_threshold,
            )
            point_metrics_xyz = _compute_pointwise_metrics(
                aligned_pred,
                gt,
                horizons=horizons,
                step_sec=step_sec,
                dist_fn=_l2_point_xyz,
                failure_threshold=failure_threshold,
            )
            if point_metrics_xy:
                failure_val = point_metrics_xy.get("failure_1s")
                if failure_val is not None:
                    point_valid_xy += 1
                    if failure_val > 0.0:
                        failure_count_xy += 1
                for key in horizon_keys:
                    val = point_metrics_xy.get(key)
                    if val is not None:
                        horizon_sums_xy[key] += float(val)
                        horizon_counts_xy[key] += 1

            if point_metrics_xyz:
                failure_val = point_metrics_xyz.get("failure_1s")
                if failure_val is not None:
                    point_valid_xyz += 1
                    if failure_val > 0.0:
                        failure_count_xyz += 1
                for key in horizon_keys:
                    val = point_metrics_xyz.get(key)
                    if val is not None:
                        horizon_sums_xyz[key] += float(val)
                        horizon_counts_xyz[key] += 1

            if collect_samples:
                s_pred = _cumulative_distances_xy(pred)
                s_pred_end = s_pred[-1] if s_pred else 0.0
                coverage_ratio = s_pred_end / s_gt_end

                gt_len = len(gt)
                total_xy = 0.0
                total_xyz = 0.0
                max_xy = 0.0
                max_xyz = 0.0
                for gt_p, s in zip(gt, s_gt):
                    pred_p = _interp_at_s(pred, s, s_pred)
                    d_xy = _l2_point_xy(gt_p, pred_p)
                    d_xyz = _l2_point_xyz(gt_p, pred_p)
                    total_xy += d_xy
                    total_xyz += d_xyz
                    if d_xy > max_xy:
                        max_xy = d_xy
                    if d_xyz > max_xyz:
                        max_xyz = d_xyz

                ade_xy = total_xy / gt_len
                ade_xyz = total_xyz / gt_len
                pred_last = _interp_at_s(pred, s_gt_end, s_pred)
                fde_xy = _l2_point_xy(gt[-1], pred_last)
                fde_xyz = _l2_point_xyz(gt[-1], pred_last)
                hit_xy = max_xy < hit_threshold
                hit_xyz = max_xyz < hit_threshold
                row = {
                    "key": item.key,
                    "gt_len": gt_len,
                    "pred_len": len(pred),
                    "s_gt": s_gt_end,
                    "s_pred": s_pred_end,
                    "coverage_ratio": coverage_ratio,
                    "ade_xy": ade_xy,
                    "fde_xy": fde_xy,
                    "max_xy": max_xy,
                    "hit_xy": hit_xy,
                    "ade_xyz": ade_xyz,
                    "fde_xyz": fde_xyz,
                    "max_xyz": max_xyz,
                    "hit_xyz": hit_xyz,
                    "point_step_sec": step_sec,
                }
                if point_metrics_xy:
                    row.update({f"{key}_xy": val for key, val in point_metrics_xy.items()})
                if point_metrics_xyz:
                    row.update({f"{key}_xyz": val for key, val in point_metrics_xyz.items()})
                rows.append(row)

        horizon_metrics_xy: Dict[str, Optional[float]] = {}
        for key, total_val in horizon_sums_xy.items():
            count = horizon_counts_xy[key]
            horizon_metrics_xy[key] = (total_val / count) if count else None

        horizon_metrics_xyz: Dict[str, Optional[float]] = {}
        for key, total_val in horizon_sums_xyz.items():
            count = horizon_counts_xyz[key]
            horizon_metrics_xyz[key] = (total_val / count) if count else None

        failure_rate_xy = (failure_count_xy / point_valid_xy) if point_valid_xy else None
        failure_rate_xyz = (failure_count_xyz / point_valid_xyz) if point_valid_xyz else None
        ade_avg_xy = _average_horizon_metrics(horizon_metrics_xy, horizon_keys)
        ade_avg_xyz = _average_horizon_metrics(horizon_metrics_xyz, horizon_keys)

        entry: Dict[str, Any] = {"adapter": adapter.name}
        for key in horizon_keys:
            entry[f"{key}_xy"] = horizon_metrics_xy.get(key)
        entry["ADE_avg_xy"] = ade_avg_xy
        entry["failure_rate_xy"] = failure_rate_xy
        for key in horizon_keys:
            entry[f"{key}_xyz"] = horizon_metrics_xyz.get(key)
        entry["ADE_avg_xyz"] = ade_avg_xyz
        entry["failure_rate_xyz"] = failure_rate_xyz
        metrics.append(entry)

        if collect_samples:
            sample_rows[adapter.name] = rows

    return metrics, sample_rows


def _render_overlay_multimodel(
    *,
    image: Image.Image,
    gt_points: List[List[float]],
    model_points: Sequence[Tuple[str, List[List[float]], Tuple[int, int, int]]],
    forward_axis: str,
    flip_lateral: bool,
    line_width: int,
    calib: Optional[Calib] = None,
    scale_xy: Optional[Tuple[float, float]] = None,
    show_gt_legend: bool = False,
) -> Image.Image:
    base = image.convert("RGB")
    overlay = base.copy()

    use_calib = calib is not None
    if use_calib:
        gt_smooth = _smooth_points(gt_points, 80)
        gt_uv = _project_points_with_calib(gt_smooth, calib) if len(gt_smooth) >= 2 else []
        model_uvs = []
        for _, pts, _ in model_points:
            smooth = _smooth_points(pts, 80)
            model_uvs.append(_project_points_with_calib(smooth, calib) if len(smooth) >= 2 else [])
        if len(gt_uv) < 2 and all(len(uv) < 2 for uv in model_uvs):
            use_calib = False

        if use_calib and scale_xy is not None:
            sx, sy = scale_xy

            def _scale_uv(uvs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
                return [(float(u) * sx, float(v) * sy) for u, v in uvs]

            gt_uv = _scale_uv(gt_uv)
            model_uvs = [_scale_uv(uvs) for uvs in model_uvs]

    if use_calib:
        if len(gt_uv) >= 2:
            _draw_polyline(overlay, gt_uv, color=(0, 200, 0), width=line_width)
        for (name, pts, color), uv in zip(model_points, model_uvs):
            if len(uv) < 2:
                continue
            _draw_polyline(overlay, uv, color=color, width=line_width)
    else:
        point_sets: List[Optional[Sequence[Sequence[float]]]] = [gt_points]
        point_sets.extend([pts for _, pts, _ in model_points])
        shared_scale = _shared_traj_scale(
            overlay.size,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
            point_sets=point_sets,
        )

        if len(gt_points) >= 2:
            smooth_gt = _smooth_points(gt_points, 80)
            uv_gt, _ = _traj_xyz_to_pixels(
                smooth_gt,
                overlay.size,
                forward_axis=forward_axis,
                flip_lateral=flip_lateral,
                scale_px_per_meter=shared_scale,
            )
            _draw_polyline(overlay, uv_gt, color=(0, 200, 0), width=line_width)

        for _, pts, color in model_points:
            if len(pts) < 2:
                continue
            smooth = _smooth_points(pts, 80)
            uv, _ = _traj_xyz_to_pixels(
                smooth,
                overlay.size,
                forward_axis=forward_axis,
                flip_lateral=flip_lateral,
                scale_px_per_meter=shared_scale,
            )
            _draw_polyline(overlay, uv, color=color, width=line_width)

    if show_gt_legend:
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        pad = 6
        box_w = 38
        box_h = 18
        draw.rectangle([(pad, pad), (pad + box_w, pad + box_h)], fill=(255, 255, 255))
        draw.rectangle([(pad, pad), (pad + box_w, pad + box_h)], outline=(200, 200, 200), width=1)
        draw.text((pad + 4, pad + 2), "GT", fill=(0, 200, 0), font=font)

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

        plot_pts = _align_pred_to_gt(pts, gt_points)
        mean_l2 = _mean_l2(gt_points, plot_pts)
        final_l2 = _final_l2(gt_points, plot_pts)
        metrics: List[str] = []
        if mean_l2 is not None:
            metrics.append(f"mean_L2={mean_l2:.3f}")
        if final_l2 is not None:
            metrics.append(f"final_L2={final_l2:.3f}")

        label = f"{name} ({', '.join(metrics)})" if metrics else name
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
            pred_points=plot_pts,
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
    line_width: int,
    show_gt_legend: bool = False,
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

    def _z_series(points: List[List[float]]) -> List[float]:
        if not points or len(points) < 2:
            return []
        smooth = _smooth_points(points, 80)
        z0 = float(smooth[0][2]) if len(smooth[0]) > 2 else 0.0
        out: List[float] = []
        for p in smooth:
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
            draw.line([a, b], fill=color, width=max(1, int(line_width)))

    if show_gt_legend:
        pad = 6
        box_w = 38
        box_h = 18
        draw.rectangle([(pad, pad), (pad + box_w, pad + box_h)], fill=(255, 255, 255))
        draw.rectangle([(pad, pad), (pad + box_w, pad + box_h)], outline=(200, 200, 200), width=1)
        draw.text((pad + 4, pad + 2), "GT", fill=(0, 200, 0), font=font)

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
    calib: Optional[Calib] = None,
) -> Image.Image:
    base = image.convert("RGB")
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    bar_h = 40
    draw.rectangle([(0, 0), (overlay.size[0], bar_h)], fill=(20, 20, 20))
    draw.text((10, 6), header, fill=(255, 255, 255), font=font)

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
    model_points = [
        (out.name, _align_pred_to_gt(out.trajectory_points, gt_points), color)
        for out, color in model_outputs
    ]
    z_series = [("GT", gt_points, (0, 200, 0))]
    z_series.extend([(name, pts, color) for name, pts, color in model_points])
    z_panel = _render_z_trend_panel(size=(top.size[0], z_panel_h), series=z_series, line_width=line_width)

    out = Image.new("RGB", (top.size[0], top.size[1] + z_panel.size[1]), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(z_panel, (0, top.size[1]))
    return out


def _smooth_points(points: List[List[float]], target_len: int) -> List[List[float]]:
    if not points or len(points) < 2:
        return points
    target_len = max(2, int(target_len))
    return _resample_points_by_index(points, target_len)


def _language_only_text(text: str) -> str:
    if not text:
        return ""
    m = _TRAJ_TOKEN_RE.search(text)
    if not m:
        return text.strip()
    return text[: m.start()].strip()


def _render_xy_plot_panel(
    *,
    size: Tuple[int, int],
    series: Sequence[Tuple[str, List[List[float]], Tuple[int, int, int]]],
    forward_axis: str,
    flip_lateral: bool,
    line_width: int,
    show_gt_legend: bool = False,
) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    point_sets: List[Optional[Sequence[Sequence[float]]]] = []
    for _, pts, _ in series:
        point_sets.append(pts)
    shared_scale = _shared_traj_scale(
        img.size,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        point_sets=point_sets,
    )

    for _, pts, color in series:
        if not pts or len(pts) < 2:
            continue
        smooth = _smooth_points(pts, 80)
        uv, _ = _traj_xyz_to_pixels(
            smooth,
            img.size,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
            scale_px_per_meter=shared_scale,
        )
        _draw_polyline(img, uv, color=color, width=max(1, int(line_width)))

    if show_gt_legend:
        pad = 6
        box_w = 38
        box_h = 18
        draw.rectangle([(pad, pad), (pad + box_w, pad + box_h)], fill=(255, 255, 255))
        draw.rectangle([(pad, pad), (pad + box_w, pad + box_h)], outline=(200, 200, 200), width=1)
        draw.text((pad + 4, pad + 2), "GT", fill=(0, 200, 0), font=font)

    return img


def _resize_for_display(image: Image.Image, max_size: int) -> Tuple[Image.Image, Tuple[float, float]]:
    if max_size <= 0:
        return image, (1.0, 1.0)
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= max_size:
        return image, (1.0, 1.0)
    scale = float(max_size) / float(max_dim)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = image.resize(new_size, Image.BICUBIC)
    return resized, (scale, scale)


def _safe_file_stem(value: str) -> str:
    return _safe_dir_name(value).replace(".", "_")


def _load_manifest_outputs(
    manifest_path: Path,
    *,
    adapter_name: str,
    adapter_path: str,
) -> Dict[str, ModelOutput]:
    outputs: Dict[str, ModelOutput] = {}
    if not manifest_path.is_file():
        return outputs
    for raw in manifest_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        key = obj.get("key")
        if not isinstance(key, str) or not key:
            continue
        outputs[key] = ModelOutput(
            name=adapter_name,
            adapter_path=adapter_path,
            output_text=str(obj.get("output_text") or ""),
            trajectory_points=list(obj.get("trajectory_points") or []),
            valid=bool(obj.get("valid")),
        )
    return outputs


def _mean_xy_error(pred_points: List[List[float]], gt_points: List[List[float]]) -> Optional[float]:
    if len(pred_points) < 2 or len(gt_points) < 2:
        return None
    pred = pred_points
    if len(pred_points) != len(gt_points):
        pred = _resample_points_by_index(pred_points, len(gt_points))
    total = 0.0
    n = 0
    for g, p in zip(gt_points, pred):
        total += _l2_point_xy(g, p)
        n += 1
    return (total / n) if n else None


def _mean_z_error(pred_points: List[List[float]], gt_points: List[List[float]]) -> Optional[float]:
    if len(pred_points) < 2 or len(gt_points) < 2:
        return None
    pred = pred_points
    if len(pred_points) != len(gt_points):
        pred = _resample_points_by_index(pred_points, len(gt_points))
    z0_gt = float(gt_points[0][2]) if len(gt_points[0]) > 2 else 0.0
    z0_pr = float(pred[0][2]) if len(pred[0]) > 2 else 0.0
    total = 0.0
    n = 0
    for g, p in zip(gt_points, pred):
        zg = (float(g[2]) if len(g) > 2 else 0.0) - z0_gt
        zp = (float(p[2]) if len(p) > 2 else 0.0) - z0_pr
        total += abs(zg - zp)
        n += 1
    return (total / n) if n else None


def _load_results_by_model(adapters: Sequence[AdapterSpec], out_dir: Path) -> Dict[str, Dict[str, ModelOutput]]:
    results: Dict[str, Dict[str, ModelOutput]] = {}
    for adapter in adapters:
        adapter_dir = out_dir / _safe_dir_name(adapter.name)
        manifest_path = adapter_dir / "manifest.jsonl"
        results[adapter.name] = _load_manifest_outputs(
            manifest_path,
            adapter_name=adapter.name,
            adapter_path=adapter.path,
        )
    return results


def _get_adapter(adapters: Sequence[AdapterSpec], name: str) -> AdapterSpec:
    for adapter in adapters:
        if adapter.name == name:
            return adapter
    raise SystemExit(f"Adapter named '{name}' not found. Available: {[a.name for a in adapters]}")


def _render_triplet_outputs(
    *,
    stage_dir: Path,
    item: SampleItem,
    base_image: Image.Image,
    scale_xy: Tuple[float, float],
    gt_points: List[List[float]],
    model_points: Sequence[Tuple[str, List[List[float]], Tuple[int, int, int]]],
    model_outputs: Sequence[Tuple[ModelOutput, Tuple[int, int, int]]],
    forward_axis: str,
    flip_lateral: bool,
    line_width: int,
    panel_width: int,
    calib: Optional[Calib],
) -> None:
    key_stem = _safe_file_stem(item.key)
    overlay = _render_overlay_multimodel(
        image=base_image,
        gt_points=gt_points,
        model_points=model_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        line_width=line_width,
        calib=calib,
        scale_xy=scale_xy,
        show_gt_legend=False,
    )
    overlay_path = stage_dir / f"{key_stem}_image.png"
    overlay.save(overlay_path)

    panel_width = max(260, int(panel_width))
    xy_panel = _render_xy_plot_panel(
        size=(panel_width, base_image.size[1]),
        series=[("GT", gt_points, (0, 200, 0))] + [
            (name, pts, color) for name, pts, color in model_points
        ],
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
        line_width=line_width,
        show_gt_legend=False,
    )
    panel_path = stage_dir / f"{key_stem}_xy.png"
    xy_panel.save(panel_path)

    z_panel_w = max(base_image.size[0], panel_width)
    z_panel_h = max(140, int(base_image.size[1] // 4))
    z_series = [("GT", gt_points, (0, 200, 0))]
    z_series.extend([(name, pts, color) for name, pts, color in model_points])
    z_panel = _render_z_trend_panel(
        size=(z_panel_w, z_panel_h),
        series=z_series,
        line_width=line_width,
        show_gt_legend=False,
    )
    z_path = stage_dir / f"{key_stem}_z.png"
    z_panel.save(z_path)

    for output, _ in model_outputs:
        text = _language_only_text(output.output_text)
        out_name = _safe_file_stem(output.name)
        txt_path = stage_dir / f"{key_stem}_{out_name}.txt"
        txt_path.write_text(text, encoding="utf-8")


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




def _parse_num_samples(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in ("all", "full"):
        return None
    try:
        parsed = int(raw)
    except Exception as exc:
        raise SystemExit(f"Invalid --num-samples value: {value}") from exc
    if parsed <= 0:
        return None
    return parsed

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

    num_samples = _parse_num_samples(args.num_samples)

    rng = random.Random(args.seed)
    if num_samples is not None:
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
        if num_samples is not None and len(items) >= num_samples:
            break

    return items


def _run_inference_for_adapter(
    *,
    adapter: AdapterSpec,
    items: Sequence[SampleItem],
    args: argparse.Namespace,
) -> Tuple[str, int, int, Path]:
    model, processor = _load_model_and_processor(
        base_model=args.base_model,
        adapter=adapter.path,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )

    adapter_dir = args.out_dir / _safe_dir_name(adapter.name)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = adapter_dir / "manifest.jsonl"
    done_keys = _load_manifest_keys(manifest_path)
    if done_keys:
        print(f"[RESUME {adapter.name}] {len(done_keys)} samples already in {manifest_path}")

    written = 0
    skipped = 0
    skipped_f = None
    if bool(args.debug_save_skipped):
        skipped_path = adapter_dir / "skipped_outputs.jsonl"
        skipped_f = skipped_path.open("w", encoding="utf-8")

    with manifest_path.open("a", encoding="utf-8") as f:
        for idx, item in enumerate(items, start=1):
            if item.key in done_keys:
                skipped += 1
                continue

            try:
                image = Image.open(item.image_path).convert("RGB")
            except Exception:
                result = {
                    "key": item.key,
                    "image_path": str(item.image_path),
                    "gt_trajectory_points": item.gt_points,
                    "output_text": "",
                    "trajectory_points": [],
                    "valid": False,
                    "adapter": adapter.name,
                    "adapter_path": adapter.path,
                    "meta": item.meta,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                done_keys.add(item.key)
                written += 1
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

            result = {
                "key": item.key,
                "image_path": str(item.image_path),
                "gt_trajectory_points": item.gt_points,
                "output_text": out_text,
                "trajectory_points": traj_points,
                "valid": valid,
                "adapter": adapter.name,
                "adapter_path": adapter.path,
                "meta": item.meta,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done_keys.add(item.key)
            written += 1

    if skipped_f is not None:
        skipped_f.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter.name, written, skipped, manifest_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export ORAD-3D LoRA adapter predictions with GT (no images/metrics).")
    ap.add_argument("--base-model", type=str, default=_DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="Adapter spec as name=path (repeatable).",
    )
    ap.add_argument("--sft-name", type=str, default="sft", help="Adapter name for SFT baseline.")
    ap.add_argument("--refine-name", type=str, default="sft_refine", help="Adapter name for SFT refine.")
    ap.add_argument("--orpo-name", type=str, default="orpo", help="Adapter name for ORPO.")
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

    ap.add_argument("--debug-metrics", action="store_true", help="Save per-sample ADE/FDE/HitRate/Coverage stats for debugging.")

    ap.add_argument("--image", action="append", default=None)
    ap.add_argument("--orad-root", type=Path, default=None)
    ap.add_argument("--gt-jsonl", type=Path, default=None)
    ap.add_argument("--gt-key", type=str, default="trajectory_ins")
    ap.add_argument("--split", type=str, default="training", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])
    ap.add_argument("--num-samples", type=str, default="10", help="Number of samples to run (use all for full split).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scan", type=int, default=None)

    ap.add_argument(
        "--traj-step-sec",
        type=float,
        default=0.1,
        help="Fallback trajectory timestep in seconds when poses.txt is missing or frames-per-point <= 0.",
    )
    ap.add_argument(
        "--metric-horizons",
        type=str,
        default="1,2,3",
        help="Comma-separated horizon seconds for ADE_{h}s segment metrics (e.g., 1,2,3).",
    )
    ap.add_argument(
        "--frames-per-point",
        type=int,
        default=7,
        help="Frames per trajectory point when mapping seconds from poses.txt (<= 0 disables).",
    )

    ap.add_argument(
        "--estimate-frames-per-point",
        action="store_true",
        help="Estimate frames-per-point from poses.txt and trajectory spacing for the sampled scenes.",
    )
    ap.add_argument(
        "--estimate-samples",
        type=int,
        default=20,
        help="Number of local_path JSONs per scene to sample when estimating frames-per-point.",
    )
    ap.add_argument(
        "--estimate-max-k",
        type=int,
        default=20,
        help="Max frames-per-point candidate to consider when estimating.",
    )
    ap.add_argument(
        "--auto-frames-per-point",
        action="store_true",
        help="Override --frames-per-point with the estimated median when available.",
    )

    ap.add_argument(
        "--hit-threshold",
        type=float,
        default=2.0,
        help="HitRate threshold in meters (max L2 < threshold).",
    )
    ap.add_argument(
        "--failure-threshold",
        type=float,
        default=10.0,
        help="Failure threshold in meters for L2 at the final point.",
    )

    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true")

    ap.add_argument(
        "--projection",
        choices=["simple", "calib"],
        default="calib",
        help="Overlay projection mode: 'calib' uses cam_K/cam_RT (and lidar_R if present), 'simple' uses ego-plane scaling.",
    )
    ap.add_argument("--panel-width", type=int, default=620)
    ap.add_argument("--line-width", type=int, default=4)
    ap.add_argument(
        "--max-image-size",
        type=int,
        default=960,
        help="Resize images so the longer side <= this value (0 disables).",
    )
    ap.add_argument(
        "--max-visualizations",
        type=int,
        default=0,
        help="Max visualizations per stage (0 = no limit).",
    )
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
    if len(adapters) < 1:
        raise SystemExit("Provide at least one adapter.")

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

    for adapter in adapters:
        print(f"[LOAD] {adapter.name} -> {adapter.path}")
        name, written, skipped, manifest_path = _run_inference_for_adapter(adapter=adapter, items=items, args=args)
        print(f"[DONE {name}] {written} new, {skipped} skipped -> {manifest_path}")

    results_by_model = _load_results_by_model(adapters, args.out_dir)
    sft_adapter = _get_adapter(adapters, args.sft_name)
    refine_adapter = _get_adapter(adapters, args.refine_name)
    orpo_adapter = _get_adapter(adapters, args.orpo_name)

    max_viz = int(args.max_visualizations) if int(args.max_visualizations) > 0 else None

    stage1_dir = args.out_dir / "viz_sft_refine"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage1_count = 0
    for item in items:
        sft_out = results_by_model.get(sft_adapter.name, {}).get(item.key)
        refine_out = results_by_model.get(refine_adapter.name, {}).get(item.key)
        if sft_out is None or refine_out is None:
            continue
        if not sft_out.valid or not refine_out.valid:
            continue
        sft_xy = _mean_xy_error(sft_out.trajectory_points, item.gt_points)
        refine_xy = _mean_xy_error(refine_out.trajectory_points, item.gt_points)
        if sft_xy is None or refine_xy is None:
            continue
        if refine_xy >= sft_xy:
            continue

        try:
            raw_image = Image.open(item.image_path).convert("RGB")
        except Exception:
            continue

        display_image, scale_xy = _resize_for_display(raw_image, int(args.max_image_size))

        calib = None
        if args.projection == "calib":
            seq_dir = _infer_seq_dir_for_item(item, orad_root=args.orad_root)
            ts = str(item.meta.get("timestamp") or item.image_path.stem)
            if seq_dir is not None and ts:
                calib = _load_calib(seq_dir, ts)

        model_points = [
            (sft_out.name, sft_out.trajectory_points, sft_adapter.color),
            (refine_out.name, refine_out.trajectory_points, refine_adapter.color),
        ]
        model_outputs = [
            (sft_out, sft_adapter.color),
            (refine_out, refine_adapter.color),
        ]
        _render_triplet_outputs(
            stage_dir=stage1_dir,
            item=item,
            base_image=display_image,
            scale_xy=scale_xy,
            gt_points=item.gt_points,
            model_points=model_points,
            model_outputs=model_outputs,
            forward_axis=args.forward_axis,
            flip_lateral=bool(args.flip_lateral),
            line_width=int(args.line_width),
            panel_width=int(args.panel_width),
            calib=calib,
        )
        stage1_count += 1
        if max_viz is not None and stage1_count >= max_viz:
            break

    stage2_dir = args.out_dir / "viz_refine_orpo"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage2_count = 0
    for item in items:
        refine_out = results_by_model.get(refine_adapter.name, {}).get(item.key)
        orpo_out = results_by_model.get(orpo_adapter.name, {}).get(item.key)
        if refine_out is None or orpo_out is None:
            continue
        if not refine_out.valid or not orpo_out.valid:
            continue
        refine_xy = _mean_xy_error(refine_out.trajectory_points, item.gt_points)
        orpo_xy = _mean_xy_error(orpo_out.trajectory_points, item.gt_points)
        refine_z = _mean_z_error(refine_out.trajectory_points, item.gt_points)
        orpo_z = _mean_z_error(orpo_out.trajectory_points, item.gt_points)
        if refine_xy is None or orpo_xy is None or refine_z is None or orpo_z is None:
            continue
        if not (orpo_xy < refine_xy and orpo_z < refine_z):
            continue

        try:
            raw_image = Image.open(item.image_path).convert("RGB")
        except Exception:
            continue

        display_image, scale_xy = _resize_for_display(raw_image, int(args.max_image_size))

        calib = None
        if args.projection == "calib":
            seq_dir = _infer_seq_dir_for_item(item, orad_root=args.orad_root)
            ts = str(item.meta.get("timestamp") or item.image_path.stem)
            if seq_dir is not None and ts:
                calib = _load_calib(seq_dir, ts)

        model_points = [
            (refine_out.name, refine_out.trajectory_points, refine_adapter.color),
            (orpo_out.name, orpo_out.trajectory_points, orpo_adapter.color),
        ]
        model_outputs = [
            (refine_out, refine_adapter.color),
            (orpo_out, orpo_adapter.color),
        ]
        _render_triplet_outputs(
            stage_dir=stage2_dir,
            item=item,
            base_image=display_image,
            scale_xy=scale_xy,
            gt_points=item.gt_points,
            model_points=model_points,
            model_outputs=model_outputs,
            forward_axis=args.forward_axis,
            flip_lateral=bool(args.flip_lateral),
            line_width=int(args.line_width),
            panel_width=int(args.panel_width),
            calib=calib,
        )
        stage2_count += 1
        if max_viz is not None and stage2_count >= max_viz:
            break

    print(
        f"[DONE] wrote per-adapter manifests -> {args.out_dir} | "
        f"viz_sft_refine={stage1_count}, viz_refine_orpo={stage2_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
