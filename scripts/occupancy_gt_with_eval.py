#!/usr/bin/env python3
"""
Visualize ORAD-3D occupancy GT with GT path + multiple model inference trajectories.

- Background: occupancy (3D grid or raw point grid)
- Paths: GT (local_path_* or poses) + N model predictions
- Mark "넘어간거": points on non-road / obstacle labels along the path
  (default highlight label ids: {2,3,4,5,7} -> car, people, water, snow, rock)

Example (3 adapters):
python scripts/occupancy_gt_with_eval.py \
  --dataset-root /home/work/datasets/bg/ORAD-3D --split testing --scene y0602_1309 \
  --mode topdown --save-dir occupancy_plots_multi \
  --path-source local_path_ins --path-frame local \
  --grid-axis auto --occ-all --require-local-path \
  --use-labels --label-proj bottom \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --adapter sft_refine=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_refine/checkpoint-1044 \
  --adapter sft=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_8/checkpoint-1044 \
  --adapter orpo=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/orpo2/checkpoint-1044 \
  --image-folder image_data \
  --temperature 1e-6

python scripts/occupancy_gt_with_eval.py \
  --dataset-root /home/work/datasets/bg/ORAD-3D --split testing --scene y2021_0223_1454 \
  --mode topdown --save-dir occupancy_plots_multi \
  --path-source local_path_ins --path-frame local \
  --grid-axis auto --require-local-path \
  --use-labels --label-proj bottom \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --adapter sft_refine=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_refine/checkpoint-1044 \
  --adapter sft=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_8/checkpoint-1044 \
  --adapter orpo=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/orpo2/checkpoint-1044 \
  --image-folder image_data \
  --temperature 1e-6

Notes
- Model inference uses camera image {scene_dir}/{image-folder}/{ts}.png
- Prediction points are assumed to be in the same ego/local coordinate convention as GT local_path.
  If your model output axis is different, use --pred-path-axis-order / --pred-path-rot-deg / --pred-path-scale.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_LABELS = [
    "road",
    "safe-road",
    "car",
    "people",
    "water",
    "snow",
    "grass-on-road",
    "rock",
]

DEFAULT_LABEL_COLORS = {
    0: (0.45, 0.45, 0.45),  # road - gray
    1: (0.60, 0.20, 0.80),  # safe-road - purple
    2: (0.85, 0.10, 0.10),  # car - red
    3: (1.00, 0.85, 0.10),  # people - yellow
    4: (0.10, 0.40, 0.90),  # water - blue
    5: (0.95, 0.95, 0.95),  # snow - white
    6: (0.10, 0.70, 0.20),  # grass-on-road - green
    7: (0.55, 0.35, 0.20),  # rock - brown
}

# mark "넘어감" 대상 라벨
DEFAULT_VIOLATION_LABEL_IDS = {2, 3, 4, 5, 7}

# default grid mapping (fallback when occupancy meta is missing)
DEFAULT_GRID_ORIGIN = (-25.0, 5.7, -3.0)
DEFAULT_GRID_RES = (0.5,)

# 모델 색 (N개 지원)
MODEL_COLORS: List[Tuple[float, float, float]] = [
    (0.86, 0.08, 0.24),  # crimson-ish
    (0.12, 0.56, 1.00),  # dodgerblue
    (1.00, 0.55, 0.00),  # orange
    (0.54, 0.17, 0.89),  # blueviolet
    (0.00, 0.81, 0.82),  # turquoise
    (0.63, 0.32, 0.18),  # sienna
]

# trajectory parsing (same as your inference script style)
_TRAJ_TOKEN_RE = re.compile(r"<\s*trajectory\s*>", re.IGNORECASE)
_POINT_RE = re.compile(
    r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]"
)

# -----------------------------
# Occupancy + pose helpers (your original)
# -----------------------------
def _split_floats(line: str) -> List[float]:
    line = line.replace(",", " ").strip()
    if not line:
        return []
    return [float(x) for x in line.split()]

def _iter_occ_files(occ_dir: str, pattern: Optional[str]) -> List[str]:
    if pattern:
        return sorted(glob.glob(os.path.join(occ_dir, pattern)))
    exts = (".npz", ".npy", ".bin", ".raw")
    out: List[str] = []
    for root, _, files in os.walk(occ_dir):
        for name in files:
            if name.lower().endswith(exts):
                out.append(os.path.join(root, name))
    return sorted(out)

def _pick_default_key(npz: np.lib.npyio.NpzFile, preferred: Optional[str]) -> str:
    if preferred:
        if preferred not in npz.files:
            raise KeyError(f"npz key not found: {preferred}. available: {npz.files}")
        return preferred
    if not npz.files:
        raise ValueError("npz file has no arrays")
    return npz.files[0]

def _extract_meta_from_npz(data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    meta: Dict[str, np.ndarray] = {}
    for key in data.files:
        lower = key.lower()
        if lower in {"origin", "grid_origin", "min_bound"}:
            meta["origin"] = np.array(data[key]).astype(np.float32).reshape(-1)
        if lower in {"voxel_size", "resolution", "voxel_size_xyz"}:
            meta["voxel_size"] = np.array(data[key]).astype(np.float32).reshape(-1)
    return meta

def load_occupancy(
    path: str,
    shape: Optional[Tuple[int, int, int]],
    dtype: str,
    npz_key: Optional[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    ext = os.path.splitext(path)[1].lower()
    meta: Dict[str, np.ndarray] = {}
    if ext == ".npz":
        with np.load(path) as data:
            key = _pick_default_key(data, npz_key)
            arr = data[key]
            meta = _extract_meta_from_npz(data)
    elif ext == ".npy":
        arr = np.load(path)
    elif ext in (".bin", ".raw"):
        if shape is None:
            raise ValueError("shape is required for .bin/.raw files")
        arr = np.fromfile(path, dtype=dtype).reshape(shape)
    else:
        raise ValueError(f"unsupported occupancy format: {ext}")

    if arr.ndim == 3:
        pass
    elif arr.ndim == 2 and arr.shape[1] in (3, 4):
        meta["raw_points"] = arr.astype(np.float32)
        arr = np.zeros((1, 1, 1), dtype=arr.dtype)
    else:
        raise ValueError(
            f"unsupported occupancy format: got shape {arr.shape}, expected 3D grid or (N, 3/4) points"
        )
    return arr, meta

def _coerce_vec3(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.array(values, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr[0], 3)
    if arr.size != 3:
        raise ValueError(f"{name} must have 1 or 3 values")
    return arr

def _maybe_vec3(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        arr = np.array(values, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 1:
        arr = np.repeat(arr[0], 3)
    if arr.size != 3:
        return None
    return arr

def _is_default_grid(values: Sequence[float], default: Sequence[float]) -> bool:
    try:
        return np.allclose(_coerce_vec3(values, "grid"), _coerce_vec3(default, "grid"))
    except Exception:
        return False

def _rotation_to_rpy(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)

def load_poses_with_meta(poses_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = []
    timestamps = []
    rpy_list = []
    with open(poses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = _split_floats(line)
            if not vals:
                continue
            ts = None
            rpy = None
            if len(vals) == 12:
                mat = np.array(vals, dtype=np.float32).reshape(3, 4)
                t = mat[:, 3]
                rpy = _rotation_to_rpy(mat[:, :3])
            elif len(vals) == 16:
                mat = np.array(vals, dtype=np.float32).reshape(4, 4)
                t = mat[:3, 3]
                rpy = _rotation_to_rpy(mat[:3, :3])
            elif len(vals) >= 7 and vals[0] > 1e9:
                ts = vals[0]
                t = np.array(vals[1:4], dtype=np.float32)
                if len(vals) == 7:
                    rpy = np.array(vals[4:7], dtype=np.float32)
            elif len(vals) >= 6:
                t = np.array(vals[0:3], dtype=np.float32)
                rpy = np.array(vals[3:6], dtype=np.float32)
            elif len(vals) >= 3:
                t = np.array(vals[:3], dtype=np.float32)
            else:
                continue
            poses.append(t)
            timestamps.append(np.nan if ts is None else ts)
            if rpy is None:
                rpy_list.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
            else:
                rpy_list.append(rpy)
    if not poses:
        raise ValueError(f"no poses parsed from {poses_path}")
    return (
        np.stack(poses, axis=0),
        np.array(timestamps, dtype=np.float32),
        np.stack(rpy_list, axis=0),
    )

def _load_local_path(local_path_dir: str, ts: str, key: str) -> Optional[np.ndarray]:
    path = os.path.join(local_path_dir, f"{ts}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        return None
    arr = np.array(data[key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
    return arr[:, :3]

def _find_nearest_pose_idx(timestamps: np.ndarray, ts: Optional[float]) -> int:
    if ts is None or not np.isfinite(ts) or timestamps.size == 0:
        return 0
    diffs = np.abs(timestamps - ts)
    return int(np.argmin(diffs))

def _to_local_frame(path: np.ndarray, pose_t: np.ndarray, yaw: Optional[float]) -> np.ndarray:
    local = path - pose_t
    if yaw is None or not np.isfinite(yaw):
        return local
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (R @ local.T).T

def _apply_pose_transform(points: np.ndarray, T: Optional[np.ndarray]) -> np.ndarray:
    if T is None:
        return points
    if T.shape != (4, 4):
        raise ValueError("pose transform must be 4x4")
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    out = (T @ homog.T).T[:, :3]
    return out

def _reorder_points(points: np.ndarray, axis_order: str) -> np.ndarray:
    if axis_order == "xyz":
        return points
    if sorted(axis_order) != ["x", "y", "z"] or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of xyz")
    order = [axis_order.index("x"), axis_order.index("y"), axis_order.index("z")]
    return points[:, order]

def _map_local_to_grid(
    path_local: np.ndarray,
    origin: np.ndarray,
    res: np.ndarray,
    axis_order: str,
) -> np.ndarray:
    if sorted(axis_order) != ["x", "y", "z"] or len(axis_order) != 3:
        raise ValueError("grid_axis must be a permutation of xyz")
    scaled = (path_local - origin) / res
    order = [axis_order.index("x"), axis_order.index("y"), axis_order.index("z")]
    return scaled[:, order]

def _best_grid_axis(
    path_local: np.ndarray,
    origin: np.ndarray,
    res: np.ndarray,
    occ_min: np.ndarray,
    occ_max: np.ndarray,
) -> Tuple[str, np.ndarray, float]:
    axes = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")
    best_axis = "xyz"
    best_ratio = -1.0
    best_path = _map_local_to_grid(path_local, origin, res, best_axis)
    for axis in axes:
        cand = _map_local_to_grid(path_local, origin, res, axis)
        mask = np.logical_and(cand >= occ_min, cand <= occ_max).all(axis=1)
        ratio = float(mask.mean()) if cand.size else 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_axis = axis
            best_path = cand
    return best_axis, best_path, best_ratio

def _grid_from_occ_points(
    points: np.ndarray, labels: Optional[np.ndarray], proj: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), None
    x = points[:, 0].astype(np.int32)
    y = points[:, 1].astype(np.int32)
    z = points[:, 2].astype(np.int32)
    w = int(y.max()) + 1
    h = int(x.max()) + 1
    key = x * w + y

    if labels is None:
        grid = np.zeros((h, w), dtype=np.uint8)
        grid[x, y] = 1
        return grid, None

    flat = np.zeros(h * w, dtype=np.int32)
    if proj == "max":
        np.maximum.at(flat, key, labels.astype(np.int32))
    else:
        order = np.lexsort((-z if proj == "top" else z, key))
        key_sorted = key[order]
        lab_sorted = labels[order].astype(np.int32)
        first = np.concatenate(([0], np.where(key_sorted[1:] != key_sorted[:-1])[0] + 1))
        flat[key_sorted[first]] = lab_sorted[first]
    return flat.reshape((h, w)), flat.reshape((h, w))

def _parse_label_map(path: Optional[str]) -> Dict[int, Tuple[float, float, float]]:
    if not path:
        return {}
    out: Dict[int, Tuple[float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            r, g, b = [float(x) for x in parts[1:4]]
            if r > 1.0 or g > 1.0 or b > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            out[idx] = (r, g, b)
    return out

def _load_label_map(
    path: Optional[str],
) -> Tuple[Dict[int, str], Dict[int, Tuple[float, float, float]]]:
    if not path:
        return {}, {}
    ext = os.path.splitext(path)[1].lower()
    label_names: Dict[int, str] = {}
    label_colors: Dict[int, Tuple[float, float, float]] = {}

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "labels" in data:
            labels_data = data["labels"]
        else:
            labels_data = data

        if isinstance(labels_data, list):
            for idx, name in enumerate(labels_data):
                label_names[int(idx)] = str(name)
        elif isinstance(labels_data, dict):
            for key, val in labels_data.items():
                label_names[int(key)] = str(val)

        if isinstance(data, dict) and "colors" in data and isinstance(data["colors"], dict):
            for key, val in data["colors"].items():
                rgb = [float(x) for x in val]
                if max(rgb) > 1.0:
                    rgb = [x / 255.0 for x in rgb]
                label_colors[int(key)] = (rgb[0], rgb[1], rgb[2])
        return label_names, label_colors

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            label_id = int(parts[0])
            name = parts[1] if len(parts) >= 2 else f"class_{label_id}"
            label_names[label_id] = name
            if len(parts) >= 5:
                rgb = [float(x) for x in parts[2:5]]
                if max(rgb) > 1.0:
                    rgb = [x / 255.0 for x in rgb]
                label_colors[label_id] = (rgb[0], rgb[1], rgb[2])
    return label_names, label_colors

def _find_poses(scene_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(scene_dir, "poses.txt"),
        os.path.join(scene_dir, "scene_data", "poses.txt"),
        os.path.join(scene_dir, "scene_data_refine", "poses.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

def _load_pose_transform(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.extend([float(x) for x in line.split()])
    if len(vals) == 16:
        return np.array(vals, dtype=np.float32).reshape(4, 4)
    if len(vals) == 12:
        mat = np.array(vals, dtype=np.float32).reshape(3, 4)
        T = np.eye(4, dtype=np.float32)
        T[:3, :] = mat
        return T
    raise ValueError("pose transform must have 12 or 16 floats")

# -----------------------------
# Inference helpers (minimal subset from your compare script)
# -----------------------------
@dataclass(frozen=True)
class AdapterSpec:
    name: str
    path: str
    color: Tuple[float, float, float]

def _extract_trajectory_section(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TRAJ_TOKEN_RE.search(text)
    if not m:
        return None
    return text[m.end():].strip() or ""

def _extract_trajectory_points(text: str) -> List[List[float]]:
    pts: List[List[float]] = []
    for m in _POINT_RE.finditer(text or ""):
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

def _parse_adapter_specs(values: Sequence[str]) -> List[AdapterSpec]:
    if not values:
        return []
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
        color = MODEL_COLORS[len(specs) % len(MODEL_COLORS)]
        specs.append(AdapterSpec(name=name, path=path, color=color))
    return specs

def _maybe_set_cache_env(cache_dir: Optional[str]) -> None:
    if not cache_dir:
        return
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))

def _load_model_and_processor(
    *,
    base_model: str,
    adapter: str,
    cache_dir: Optional[str],
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
):
    # lazy import (inference 없으면 torch/transformers 로딩 안 함)
    import torch
    from peft import PeftModel
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoProcessor,
        AutoTokenizer,
    )

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
            processor.tokenizer = tokenizer
        except Exception:
            pass

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)

    model = None
    torch_dtype: Any = "auto"
    v = (dtype or "").strip().lower()
    if v in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16
    elif v in ("fp16", "float16"):
        torch_dtype = torch.float16
    elif v in ("fp32", "float32"):
        torch_dtype = torch.float32

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
        messages.append({"role": "system", "content": [{"type": "text", "text": system_text.strip()}]})
    messages.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": (prompt_text or "").strip()}]}
    )
    return messages

def _maybe_prefix_image_token(prompt_text: str, *, use_sharegpt_format: bool) -> str:
    txt = (prompt_text or "").strip()
    if not use_sharegpt_format:
        return txt
    if txt.lower().startswith("<image>"):
        return txt
    return f"<image>\n{txt}".strip()

def _prepare_inputs(processor, *, image, system_text: str, prompt_text: str, use_sharegpt_format: bool):
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

def _run_inference_one_image(
    *,
    image_path: Path,
    adapters: Sequence[AdapterSpec],
    base_model: str,
    cache_dir: Optional[str],
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
    system_text: str,
    prompt_text: str,
    use_sharegpt_format: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    skip_special_tokens: bool,
    debug: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {adapter_name: {"points": np.ndarray (N,3) or None, "raw_text": str, "valid": bool}}
    """
    from PIL import Image
    import torch

    if not image_path.is_file():
        return {a.name: {"points": None, "raw_text": "", "valid": False} for a in adapters}

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return {a.name: {"points": None, "raw_text": "", "valid": False} for a in adapters}

    outputs: Dict[str, Dict[str, Any]] = {}

    # 모델은 어댑터별로 로드 (원하면 캐시로 최적화 가능)
    for adapter in adapters:
        try:
            model, processor = _load_model_and_processor(
                base_model=base_model,
                adapter=adapter.path,
                cache_dir=cache_dir,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            if debug:
                print(f"[INFER] load failed: {adapter.name}: {e}", file=sys.stderr)
            outputs[adapter.name] = {"points": None, "raw_text": "", "valid": False}
            continue

        try:
            inputs, input_len = _prepare_inputs(
                processor,
                image=image,
                system_text=system_text,
                prompt_text=prompt_text,
                use_sharegpt_format=use_sharegpt_format,
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

            # decode
            gen_ids = out_ids
            if (
                input_len > 0
                and isinstance(out_ids, torch.Tensor)
                and out_ids.ndim == 2
                and out_ids.shape[1] > input_len
            ):
                gen_ids = out_ids[:, input_len:]

            try:
                out_text = processor.batch_decode(gen_ids, skip_special_tokens=bool(skip_special_tokens))[0]
            except Exception:
                out_text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=bool(skip_special_tokens))[0]
            out_text = _clean_output_text(out_text)

            traj_section = _extract_trajectory_section(out_text)
            pts = _extract_trajectory_points(traj_section) if traj_section is not None else []
            valid = len(pts) >= 2
            outputs[adapter.name] = {
                "points": (np.array(pts, dtype=np.float32) if valid else None),
                "raw_text": out_text,
                "valid": valid,
            }

            if debug and not valid:
                head = (out_text[:180] + ("..." if len(out_text) > 180 else "")).replace("\n", "\\n")
                print(f"[INFER][SKIP] {adapter.name} {image_path.name}: {head}", file=sys.stderr)

        except Exception as e:
            if debug:
                print(f"[INFER] failed: {adapter.name}: {e}", file=sys.stderr)
            outputs[adapter.name] = {"points": None, "raw_text": "", "valid": False}

        # cleanup
        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outputs

# -----------------------------
# Plotting
# -----------------------------
def _labels_to_rgba(labels: np.ndarray, palette: Dict[int, Tuple[float, float, float]], alpha: float) -> np.ndarray:
    import matplotlib.pyplot as plt
    uniq = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(1, len(uniq)))
    colors = []
    for i, lab in enumerate(uniq):
        lab_int = int(lab)
        if lab_int in palette:
            r, g, b = palette[lab_int]
        else:
            r, g, b, _ = cmap(i)
        a = 0.0 if lab_int == 0 else alpha
        colors.append((r, g, b, a))
    colors = np.array(colors, dtype=np.float32)
    idx = np.searchsorted(uniq, labels)
    return colors[idx]

def _add_label_legend(ax, labels: np.ndarray, label_names: Dict[int, str],
                      label_colors: Dict[int, Tuple[float, float, float]]) -> None:
    try:
        from matplotlib.patches import Patch
    except Exception:
        return
    uniq = sorted({int(x) for x in np.unique(labels) if int(x) >= 0})
    handles = []
    for lab in uniq:
        name = label_names.get(lab, f"class_{lab}")
        color = label_colors.get(lab, (0.6, 0.6, 0.6))
        handles.append(Patch(facecolor=color, edgecolor="none", label=name))
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)

def _plot_path(ax, path_xy: np.ndarray, color: Optional[Tuple[float, float, float]], width: float, gradient: bool) -> None:
    if path_xy.shape[0] == 0:
        return
    if not gradient and color is not None:
        ax.plot(path_xy[:, 0], path_xy[:, 1], color=color, linewidth=width)
    else:
        from matplotlib.collections import LineCollection
        segs = np.stack([path_xy[:-1], path_xy[1:]], axis=1)
        values = np.linspace(0.0, 1.0, segs.shape[0])
        lc = LineCollection(segs, cmap="viridis", linewidths=width)
        lc.set_array(values)
        ax.add_collection(lc)

def _mark_violations(
    *,
    ax,
    path_xy: np.ndarray,
    label_grid: np.ndarray,
    axes: str,
    violation_ids: set[int],
    marker_edge: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    label: str = "violation",
) -> None:
    """
    label_grid: shape (H,W) where index is [gx,gy] in grid coordinates (same as earlier code)
    axes: "xy" or "yx" supported for label lookup
    """
    if label_grid is None or path_xy.size == 0 or axes not in ("xy", "yx"):
        return
    xi = np.rint(path_xy[:, 0]).astype(int)
    yi = np.rint(path_xy[:, 1]).astype(int)
    if axes == "xy":
        gx, gy = xi, yi
    else:
        gx, gy = yi, xi
    valid = (gx >= 0) & (gy >= 0) & (gx < label_grid.shape[0]) & (gy < label_grid.shape[1])
    if not np.any(valid):
        return
    labs = label_grid[gx[valid], gy[valid]].astype(int)
    hit = np.isin(labs, list(violation_ids))
    if not np.any(hit):
        return
    hit_xy = path_xy[valid][hit]
    ax.scatter(
        hit_xy[:, 0],
        hit_xy[:, 1],
        s=36,
        facecolors="none",
        edgecolors=marker_edge,
        linewidths=1.6,
        label=label,
    )

def _plot_topdown_multi(
    *,
    occ: np.ndarray,
    occ_points: Optional[np.ndarray],
    occ_labels: Optional[np.ndarray],
    points_are_grid: bool,
    use_labels: bool,
    label_map: Dict[int, Tuple[float, float, float]],
    label_proj: str,
    label_alpha: float,
    gt_path: np.ndarray,
    model_paths: Sequence[Tuple[str, Optional[np.ndarray], Tuple[float, float, float]]],
    title: str,
    save: Optional[str],
    axes: str,
    flip_x: bool,
    flip_y: bool,
    path_width: float,
    path_gradient: bool,
    label_names: Optional[Dict[int, str]] = None,
    label_legend: bool = False,
    label_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    violation_ids: Optional[set[int]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:
        raise RuntimeError("matplotlib is required for topdown view") from exc

    violation_ids = violation_ids or set(DEFAULT_VIOLATION_LABEL_IDS)

    axes = axes.lower()
    if len(axes) != 2 or any(a not in "xyz" for a in axes) or axes[0] == axes[1]:
        raise ValueError("topdown-axes must be one of: xy, yx, xz, zx, yz, zy")

    fig, ax = plt.subplots(figsize=(7, 7))
    label_grid_for_path = None

    # --------- background (occupancy) ----------
    if occ.shape != (1, 1, 1):
        axis_map = {"x": 0, "y": 1, "z": 2}
        proj_axis = next(a for a in "xyz" if a not in axes)
        proj_idx = axis_map[proj_axis]
        mask = occ > 0
        proj = mask.max(axis=proj_idx).astype(np.float32)
        proj_order = "".join([a for a in "xyz" if a != proj_axis])
        img = proj.T if axes == proj_order else proj
        cmap = ListedColormap(["white", "#7a7a7a"])
        ax.imshow(img, cmap=cmap, origin="lower")
    elif points_are_grid and occ_points is not None:
        grid, label_grid = _grid_from_occ_points(occ_points, occ_labels, label_proj)
        label_grid_for_path = label_grid
        if use_labels and label_grid is not None:
            palette = dict(label_colors) if label_colors else dict(label_map)
            # fill palette if missing ids
            uniq = [int(x) for x in np.unique(label_grid) if int(x) >= 0]
            for idx, lab in enumerate(uniq):
                if lab not in palette:
                    palette[lab] = MODEL_COLORS[idx % len(MODEL_COLORS)]
            rgba = _labels_to_rgba(label_grid.T, palette, label_alpha)
            ax.imshow(rgba, origin="lower")
            if label_legend and label_names:
                _add_label_legend(ax, label_grid, label_names, palette)
        else:
            cmap = ListedColormap(["white", "#7a7a7a"])
            ax.imshow(grid.T, cmap=cmap, origin="lower")
    else:
        ax.text(0.5, 0.5, "no occupied voxels", ha="center", va="center", transform=ax.transAxes)

    # --------- paths ----------
    idx = {"x": 0, "y": 1, "z": 2}

    def prep_path(p: np.ndarray) -> np.ndarray:
        xy = p[:, [idx[axes[0]], idx[axes[1]]]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if xy.size:
            if flip_x:
                xy[:, 0] = -xy[:, 0]
            if flip_y:
                xy[:, 1] = -xy[:, 1]
        return xy

    # GT
    gt_xy = prep_path(gt_path)
    if gt_xy.size:
        _plot_path(ax, gt_xy, (0.0, 0.75, 0.0), path_width, False)
        ax.scatter(gt_xy[0, 0], gt_xy[0, 1], color=(0.0, 0.75, 0.0), s=30, label="GT start")
        ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], color=(0.0, 0.55, 0.0), s=30, label="GT end")
        if use_labels and label_grid_for_path is not None and axes in ("xy", "yx"):
            _mark_violations(
                ax=ax,
                path_xy=gt_xy,
                label_grid=label_grid_for_path,
                axes=axes,
                violation_ids=violation_ids,
                marker_edge=(0.0, 0.5, 0.0),
                label="GT violation",
            )

    # Models
    for name, mp, color in model_paths:
        if mp is None or mp.size == 0:
            continue
        mxy = prep_path(mp)
        if mxy.size < 2:
            continue
        _plot_path(ax, mxy, None if path_gradient else color, path_width, path_gradient)
        ax.scatter(mxy[0, 0], mxy[0, 1], color=color, s=20, label=f"{name} start")
        ax.scatter(mxy[-1, 0], mxy[-1, 1], color=color, s=20, alpha=0.85, label=f"{name} end")
        if use_labels and label_grid_for_path is not None and axes in ("xy", "yx"):
            _mark_violations(
                ax=ax,
                path_xy=mxy,
                label_grid=label_grid_for_path,
                axes=axes,
                violation_ids=violation_ids,
                marker_edge=color,
                label=f"{name} violation",
            )

    ax.set_title(title)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
    else:
        plt.show()

# -----------------------------
# Args / Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ORAD-3D occupancy GT with GT path + model predictions")
    parser.add_argument("--dataset-root", required=True, help="root containing training/validation/testing")
    parser.add_argument("--split", choices=("training", "validation", "testing"), required=True)
    parser.add_argument("--scene", help="specific scene id (folder name)")
    parser.add_argument("--all", action="store_true", help="process all scenes with occupancy+poses")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-label-legend", dest="label_legend", action="store_false")
    parser.set_defaults(label_legend=True)

    parser.add_argument("--occ-pattern", help="glob pattern inside occupancy dir (e.g. *.npz)")
    parser.add_argument("--occ-index", type=int, default=0)
    parser.add_argument("--occ-all", action="store_true", help="process all occupancy files in scene")
    parser.add_argument("--npz-key", help="npz array key for occupancy")
    parser.add_argument("--shape", nargs=3, type=int, help="shape for raw/bin files, e.g. 256 256 32")
    parser.add_argument("--dtype", default="uint8", help="dtype for raw/bin files")
    parser.add_argument("--mode", choices=("topdown",), default="topdown")
    parser.add_argument("--save-dir", help="save images into directory (per scene)")
    parser.add_argument("--pose-to-occ", help="optional 4x4 transform file for pose to occupancy frame")

    # GT path options (as-is)
    parser.add_argument(
        "--path-source",
        default="local_path_ins",
        choices=("poses", "local_path_ins", "local_path_hmi", "local_path_ins_past", "local_path_hmi_past"),
    )
    parser.add_argument("--require-local-path", action="store_true", help="skip frames without local_path")
    parser.add_argument("--path-frame", default="local", choices=("local", "global"))
    parser.add_argument("--path-axis-order", default="xyz", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"))
    parser.add_argument("--path-scale", type=float, default=1.0)
    parser.add_argument("--path-rot-deg", type=float, default=0.0)
    parser.add_argument("--path-clip", dest="path_clip", action="store_true", default=True)
    parser.add_argument("--no-path-clip", dest="path_clip", action="store_false")

    # grid mapping
    parser.add_argument("--grid-origin", nargs=3, type=float, default=list(DEFAULT_GRID_ORIGIN))
    parser.add_argument("--grid-res", nargs="+", type=float, default=list(DEFAULT_GRID_RES))
    parser.add_argument("--grid-axis", default="xyz", help="map local xyz to grid axes or auto")
    parser.add_argument("--occ-axis-order", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"), default="xyz")

    # visualization
    parser.add_argument("--topdown-axes", default="xy", help="axes to show in topdown view")
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--path-width", type=float, default=4.0)
    parser.add_argument("--path-gradient", action="store_true")

    # labels
    parser.add_argument("--use-labels", action="store_true", help="colorize occupancy labels")
    parser.add_argument("--label-map", help="label map file")
    parser.add_argument("--label-proj", choices=("bottom", "top", "max"), default="bottom")
    parser.add_argument("--label-alpha", type=float, default=1.0)

    # violation ids override
    parser.add_argument("--violation-label-ids", type=str, default="", help="e.g. 2,3,4,5,7")

    # -------- model inference options --------
    parser.add_argument("--adapter", action="append", default=None, help="Adapter spec name=path (repeatable).")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--skip-special-tokens", action="store_true")
    parser.add_argument("--use-sharegpt-format", action="store_true")

    parser.add_argument(
        "--system",
        type=str,
        default=(
            "You are an off-road autonomous driving agent. "
            "Given an input camera image, describe the scene and provide a safe drivable trajectory. "
            "Output the trajectory after a <trajectory> token as a comma-separated list of [x,y,z] points."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="I am seeing an off-road driving image. Please generate a safe drivable trajectory for my vehicle to follow.",
    )

    parser.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])

    # pred path transform knobs (모델 출력 좌표 보정용)
    parser.add_argument("--pred-path-axis-order", default="xyz", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"))
    parser.add_argument("--pred-path-scale", type=float, default=1.0)
    parser.add_argument("--pred-path-rot-deg", type=float, default=0.0)

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def _parse_violation_ids(s: str) -> set[int]:
    s = (s or "").strip()
    if not s:
        return set(DEFAULT_VIOLATION_LABEL_IDS)
    out: set[int] = set()
    for p in re.split(r"[\s,]+", s):
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            pass
    return out if out else set(DEFAULT_VIOLATION_LABEL_IDS)

def main() -> int:
    args = parse_args()
    label_names, label_colors = _load_label_map(args.label_map)
    if not label_names:
        label_names = {i: n for i, n in enumerate(DEFAULT_LABELS)}
    if not label_colors:
        label_colors = DEFAULT_LABEL_COLORS

    split_dir = os.path.join(args.dataset_root, args.split)
    if not os.path.isdir(split_dir):
        print(f"error: split dir not found: {split_dir}", file=sys.stderr)
        return 2

    # scene list
    if args.scene and not args.all:
        scenes = [args.scene]
    else:
        scenes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    # valid scenes
    valid: List[str] = []
    for scene in scenes:
        scene_dir = os.path.join(split_dir, scene)
        occ_dir = os.path.join(scene_dir, "occupancy")
        poses_path = _find_poses(scene_dir)
        if os.path.isdir(occ_dir) and poses_path:
            valid.append(scene)
    if not valid:
        print("error: no scenes with occupancy + poses.txt found", file=sys.stderr)
        return 2

    if args.scene and not args.all:
        sample_scenes = valid
    elif args.all:
        sample_scenes = valid
    else:
        rnd = random.Random(args.seed)
        k = min(args.num_samples, len(valid))
        sample_scenes = rnd.sample(valid, k)

    T_pose_to_occ = _load_pose_transform(args.pose_to_occ)
    label_map = _parse_label_map(args.label_map)
    adapters = _parse_adapter_specs(args.adapter or [])
    violation_ids = _parse_violation_ids(args.violation_label_ids)

    for scene in sample_scenes:
        scene_dir = os.path.join(split_dir, scene)
        occ_dir = os.path.join(scene_dir, "occupancy")
        occ_files = _iter_occ_files(occ_dir, args.occ_pattern)
        if not occ_files:
            print(f"skip: no occupancy files in {occ_dir}", file=sys.stderr)
            continue

        occ_list = occ_files if args.occ_all else ([occ_files[args.occ_index]] if 0 <= args.occ_index < len(occ_files) else [])
        if not occ_list:
            print(f"skip: occ-index out of range for {scene}", file=sys.stderr)
            continue

        poses_path = _find_poses(scene_dir)
        if not poses_path:
            print(f"skip: no poses.txt for {scene}", file=sys.stderr)
            continue

        for occ_path in occ_list:
            occ_ts = os.path.splitext(os.path.basename(occ_path))[0]
            occ_ts_val = None
            try:
                occ_ts_val = float(occ_ts)
            except ValueError:
                pass

            occ, meta = load_occupancy(
                occ_path, tuple(args.shape) if args.shape else None, args.dtype, args.npz_key
            )

            occ_points = None
            occ_labels = None
            points_are_grid = False
            if "raw_points" in meta:
                raw = meta["raw_points"].astype(np.float32)
                if raw.shape[1] == 4:
                    occ_points = raw[:, :3]
                    occ_labels = raw[:, 3].astype(np.int32)
                else:
                    occ_points = raw[:, :3]
                occ_points = _reorder_points(occ_points, args.occ_axis_order)
                points_are_grid = True

            # prefer occupancy meta for grid mapping when args are left at defaults
            origin = _coerce_vec3(args.grid_origin, "grid-origin")
            res_vals = _coerce_vec3(args.grid_res, "grid-res")
            meta_origin = _maybe_vec3(meta.get("origin"))
            if meta_origin is not None and _is_default_grid(args.grid_origin, DEFAULT_GRID_ORIGIN):
                origin = meta_origin
            meta_res = _maybe_vec3(meta.get("voxel_size"))
            if meta_res is not None and _is_default_grid(args.grid_res, DEFAULT_GRID_RES):
                res_vals = meta_res

            # -------- GT path (원래 로직 유지) ----------
            gt_path = None
            if args.path_source.startswith("local_path"):
                local_dir = os.path.join(scene_dir, "local_path")
                key_map = {
                    "local_path_ins": "trajectory_ins",
                    "local_path_hmi": "trajectory_hmi",
                    "local_path_ins_past": "trajectory_ins_past",
                    "local_path_hmi_past": "trajectory_hmi_past",
                }
                key = key_map.get(args.path_source, "trajectory_ins")
                gt_path = _load_local_path(local_dir, occ_ts, key)

            if gt_path is None and args.require_local_path:
                if args.debug:
                    print(f"[{scene} {occ_ts}] skip: no local_path", file=sys.stderr)
                continue

            if gt_path is None:
                poses, timestamps, rpy = load_poses_with_meta(poses_path)
                gt_path = poses
                if args.path_frame == "local":
                    idx = _find_nearest_pose_idx(timestamps, occ_ts_val)
                    yaw = rpy[idx, 2] if rpy.size else None
                    gt_path = _to_local_frame(gt_path, poses[idx], yaw)

            gt_path = _reorder_points(gt_path, args.path_axis_order)
            gt_path = gt_path * float(args.path_scale)
            if abs(args.path_rot_deg) > 1e-6:
                rad = np.deg2rad(args.path_rot_deg)
                c, s = np.cos(rad), np.sin(rad)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                gt_path = (R @ gt_path.T).T

            if T_pose_to_occ is not None:
                gt_path = _apply_pose_transform(gt_path, T_pose_to_occ)

            # -------- Model inference paths ----------
            model_paths_local: List[Tuple[str, Optional[np.ndarray], Tuple[float, float, float]]] = []
            if adapters:
                img_path = Path(scene_dir) / args.image_folder / f"{occ_ts}.png"
                infer_out = _run_inference_one_image(
                    image_path=img_path,
                    adapters=adapters,
                    base_model=str(args.base_model),
                    cache_dir=args.cache_dir,
                    dtype=str(args.dtype),
                    device_map=str(args.device_map),
                    trust_remote_code=bool(args.trust_remote_code),
                    system_text=str(args.system),
                    prompt_text=str(args.prompt),
                    use_sharegpt_format=bool(args.use_sharegpt_format),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    skip_special_tokens=bool(args.skip_special_tokens),
                    debug=bool(args.debug),
                )

                for ad in adapters:
                    pts = infer_out.get(ad.name, {}).get("points")
                    if pts is None:
                        model_paths_local.append((ad.name, None, ad.color))
                        continue

                    # pred transform knobs
                    pts = _reorder_points(pts, args.pred_path_axis_order)
                    pts = pts * float(args.pred_path_scale)
                    if abs(args.pred_path_rot_deg) > 1e-6:
                        rad = np.deg2rad(args.pred_path_rot_deg)
                        c, s = np.cos(rad), np.sin(rad)
                        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                        pts = (R @ pts.T).T

                    if T_pose_to_occ is not None:
                        pts = _apply_pose_transform(pts, T_pose_to_occ)

                    model_paths_local.append((ad.name, pts, ad.color))

            # -------- grid mapping (GT + models should be mapped same way) ----------
            has_grid = points_are_grid or occ.shape != (1, 1, 1)
            if has_grid:
                if points_are_grid and occ_points is not None:
                    occ_min = occ_points.min(axis=0)
                    occ_max = occ_points.max(axis=0)
                else:
                    occ_shape = np.array(occ.shape, dtype=np.float32)
                    if occ_shape.size != 3:
                        raise ValueError("occupancy grid must be 3D for grid mapping")
                    occ_min = np.zeros(3, dtype=np.float32)
                    occ_max = occ_shape - 1.0

                if args.grid_axis == "auto":
                    axis, gt_path_grid, ratio = _best_grid_axis(gt_path, origin, res_vals, occ_min, occ_max)
                    if args.debug:
                        print(f"[{scene} {occ_ts}] auto grid-axis={axis} in_ratio={ratio:.3f}", file=sys.stderr)
                    gt_path = gt_path_grid
                    mapped_models: List[Tuple[str, Optional[np.ndarray], Tuple[float, float, float]]] = []
                    for name, mp, col in model_paths_local:
                        if mp is None:
                            mapped_models.append((name, None, col))
                        else:
                            mapped_models.append((name, _map_local_to_grid(mp, origin, res_vals, axis), col))
                    model_paths_local = mapped_models
                else:
                    gt_path = _map_local_to_grid(gt_path, origin, res_vals, args.grid_axis)
                    mapped_models = []
                    for name, mp, col in model_paths_local:
                        if mp is None:
                            mapped_models.append((name, None, col))
                        else:
                            mapped_models.append((name, _map_local_to_grid(mp, origin, res_vals, args.grid_axis), col))
                    model_paths_local = mapped_models

                # clip
                if args.path_clip:
                    mask = np.logical_and(gt_path >= occ_min, gt_path <= occ_max).all(axis=1)
                    gt_path = gt_path[mask]
                    clipped_models = []
                    for name, mp, col in model_paths_local:
                        if mp is None:
                            clipped_models.append((name, None, col))
                            continue
                        m = np.logical_and(mp >= occ_min, mp <= occ_max).all(axis=1)
                        clipped_models.append((name, mp[m], col))
                    model_paths_local = clipped_models

            title = f"{scene} | {os.path.basename(occ_path)}"
            save = None
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                save = os.path.join(args.save_dir, f"{scene}_{occ_ts}.png")

            use_labels = args.use_labels or (occ_labels is not None and np.max(occ_labels) > 1)

            _plot_topdown_multi(
                occ=occ,
                occ_points=occ_points,
                occ_labels=occ_labels,
                points_are_grid=points_are_grid,
                use_labels=use_labels,
                label_map=label_map,
                label_proj=args.label_proj,
                label_alpha=args.label_alpha,
                gt_path=gt_path,
                model_paths=model_paths_local,
                title=title,
                save=save,
                axes=args.topdown_axes,
                flip_x=args.flip_x,
                flip_y=args.flip_y,
                path_width=args.path_width,
                path_gradient=args.path_gradient,
                label_names=label_names,
                label_colors=label_colors,
                label_legend=args.label_legend,
                violation_ids=violation_ids,
            )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
