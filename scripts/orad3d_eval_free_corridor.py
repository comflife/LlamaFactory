#!/usr/bin/env python3
"""
Evaluate ORAD-3D VLM trajectory outputs with:
1) Free-space corridor HitRate (occupancy-based, no GT path).
2) Z-only similarity vs GT (arc-length aligned).

This script consumes saved manifest.jsonl files (from orad3d_vlm_trajectory_fewshot_v3.py)
and computes:
  - HitFree@d: path fully free AND min clearance >= d
  - Reachable distance@d: distance until first non-free / clearance < d
  - Z metrics: z-MSE / z-RMSE and dz/ds trend MSE (+ optional corr)

Default model specs match the 6 requested models under:
/home/work/byounggun/LlamaFactory/orad3d_all_models
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required for distance_transform_edt") from exc

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

# mark "넘어감" 대상 라벨
DEFAULT_VIOLATION_LABEL_IDS = {2, 3, 4, 5, 7}

# default grid mapping (fallback when occupancy meta is missing)
DEFAULT_GRID_ORIGIN = (-25.0, 5.7, -3.0)
DEFAULT_GRID_RES = (0.5,)

DEFAULT_MODEL_SPECS = [
    ("qwen3-vl-8b-instruct", "qwen3-vl-8b-instruct"),
    ("qwen3-vl-32b-instruct", "qwen3-vl-32b-instruct"),
    ("gpt-4o", "gpt-4o"),
    ("gemini-3-flash-preview", "gemini-3-flash-preview"),
    ("llama-4-scout", "llama-4-scout"),
    ("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct"),
]


# -----------------------------
# Occupancy helpers (adapted)
# -----------------------------
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
            key = npz_key or (data.files[0] if data.files else None)
            if not key:
                raise ValueError("npz file has no arrays")
            if key not in data.files:
                raise KeyError(f"npz key not found: {key}. available: {data.files}")
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


def _path_to_axes(
    path_xyz: np.ndarray,
    axes: str,
    *,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    if path_xyz is None or path_xyz.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    idx = {"x": 0, "y": 1, "z": 2}
    xy = path_xyz[:, [idx[axes[0]], idx[axes[1]]]]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.size:
        if flip_x:
            xy[:, 0] = -xy[:, 0]
        if flip_y:
            xy[:, 1] = -xy[:, 1]
    return xy


# -----------------------------
# Manifest + metrics
# -----------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


def _parse_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    if args.model_spec:
        for raw in args.model_spec:
            if "=" in raw:
                name, rel = raw.split("=", 1)
            else:
                name, rel = raw, raw
            specs.append(ModelSpec(name=name.strip(), path=(Path(args.manifest_root) / rel.strip())))
    else:
        for name, rel in DEFAULT_MODEL_SPECS:
            specs.append(ModelSpec(name=name, path=(Path(args.manifest_root) / rel)))
    return specs


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        path = path / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _find_occ_file(scene_dir: Path, ts: str) -> Optional[Path]:
    occ_dir = scene_dir / "occupancy"
    if not occ_dir.is_dir():
        return None
    for ext in (".npy", ".npz", ".bin", ".raw"):
        cand = occ_dir / f"{ts}{ext}"
        if cand.exists():
            return cand
    return None


def _rotate_xy(points: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-6:
        return points
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (R @ points.T).T


def _arc_length_xy(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.array([0.0], dtype=np.float32)
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    ds = np.sqrt(dx * dx + dy * dy)
    return np.concatenate([[0.0], np.cumsum(ds)], axis=0)


def _compute_z_metrics(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    max_dist: float,
    step: float,
    *,
    center: bool,
    score_beta1: float,
    score_beta2: float,
    score_gamma: float,
) -> Optional[Dict[str, Optional[float]]]:
    if gt_points.shape[0] < 2 or pred_points.shape[0] < 2:
        return None
    s_gt = _arc_length_xy(gt_points)
    s_pred = _arc_length_xy(pred_points)
    max_s = min(float(s_gt[-1]), float(s_pred[-1]), float(max_dist))
    if max_s < step or max_s <= 0.0:
        return None
    s_samples = np.arange(0.0, max_s + 1e-6, step, dtype=np.float32)
    z_gt = np.interp(s_samples, s_gt, gt_points[:, 2])
    z_pred = np.interp(s_samples, s_pred, pred_points[:, 2])
    if center:
        z_gt = z_gt - float(np.mean(z_gt))
        z_pred = z_pred - float(np.mean(z_pred))
    diff = z_pred - z_gt
    z_mse = float(np.mean(diff * diff))
    z_rmse = math.sqrt(z_mse)
    # dz/ds trend
    dz_gt = np.gradient(z_gt, s_samples)
    dz_pred = np.gradient(z_pred, s_samples)
    dz_diff = dz_pred - dz_gt
    dz_mse = float(np.mean(dz_diff * dz_diff))
    dz_corr = None
    if np.std(dz_gt) > 1e-6 and np.std(dz_pred) > 1e-6:
        corr = np.corrcoef(dz_gt, dz_pred)[0, 1]
        dz_corr = float(corr)
    # trend score (focus on dz/ds)
    dz_corr_safe = dz_corr if dz_corr is not None else 0.0
    z_score = (score_beta1 * math.exp(-score_gamma * dz_mse)) + (
        score_beta2 * max(0.0, dz_corr_safe)
    )
    return {
        "z_mse": z_mse,
        "z_rmse": z_rmse,
        "dz_mse": dz_mse,
        "dz_corr": dz_corr,
        "z_score": z_score,
    }


def _compute_occ_metrics(
    *,
    path_grid: np.ndarray,
    gt_grid: np.ndarray,
    obstacle_mask: np.ndarray,
    dist_map: np.ndarray,
    axes: str,
    flip_x: bool,
    flip_y: bool,
    axis_res: Dict[str, float],
    clearance_thresholds: Sequence[float],
    soft_mode: str,
    soft_tau: float,
    soft_alpha: float,
    score_weights: Tuple[float, float, float],
    eps: float,
) -> Optional[Dict[str, Any]]:
    if path_grid is None or path_grid.size == 0:
        return None
    if axes not in ("xy", "yx"):
        raise ValueError("topdown-axes must be xy or yx for hitfree eval")
    path_xy = _path_to_axes(path_grid, axes, flip_x=flip_x, flip_y=flip_y)
    if path_xy.size == 0:
        return None
    xi = np.rint(path_xy[:, 0]).astype(int)
    yi = np.rint(path_xy[:, 1]).astype(int)
    if axes == "xy":
        gx, gy = xi, yi
    else:
        gx, gy = yi, xi
    h, w = obstacle_mask.shape
    valid = (gx >= 0) & (gy >= 0) & (gx < h) & (gy < w)
    if not np.any(valid):
        return {
            "all_free": False,
            "min_clearance": float("-inf"),
            "hitfree": {d: 0 for d in clearance_thresholds},
            "reachable": {d: 0.0 for d in clearance_thresholds},
            "safe_len": {d: 0.0 for d in clearance_thresholds},
            "bad_ratio_len": {d: 1.0 for d in clearance_thresholds},
            "softfree": {d: 0.0 for d in clearance_thresholds},
            "extra_bonus": {d: 0.0 for d in clearance_thresholds},
            "score": {d: 0.0 for d in clearance_thresholds},
            "gt_len": 0.0,
        }
    obstacle_here = np.zeros_like(valid, dtype=bool)
    obstacle_here[valid] = obstacle_mask[gx[valid], gy[valid]]
    all_free = bool(valid.all() and (~obstacle_here).all())

    # distance map is in meters already (sampling applied)
    clearance = np.full_like(path_xy[:, 0], fill_value=float("-inf"), dtype=np.float32)
    clearance[valid] = dist_map[gx[valid], gy[valid]]
    min_clearance = float(np.min(clearance)) if clearance.size else float("-inf")

    # path length in meters (2D on axes)
    res_a = float(axis_res[axes[0]])
    res_b = float(axis_res[axes[1]])
    if path_xy.shape[0] >= 2:
        dxy = np.diff(path_xy, axis=0)
        seg = np.sqrt((dxy[:, 0] * res_a) ** 2 + (dxy[:, 1] * res_b) ** 2)
        cum = np.concatenate([[0.0], np.cumsum(seg)], axis=0)
    else:
        seg = np.array([], dtype=np.float32)
        cum = np.array([0.0], dtype=np.float32)

    # GT length in meters (same axis/res)
    gt_xy = _path_to_axes(gt_grid, axes, flip_x=flip_x, flip_y=flip_y)
    if gt_xy.shape[0] >= 2:
        dgt = np.diff(gt_xy, axis=0)
        gt_seg = np.sqrt((dgt[:, 0] * res_a) ** 2 + (dgt[:, 1] * res_b) ** 2)
        gt_len = float(gt_seg.sum())
    else:
        gt_len = 0.0

    hitfree: Dict[float, int] = {}
    reachable: Dict[float, float] = {}
    safe_len: Dict[float, float] = {}
    bad_ratio_len: Dict[float, float] = {}
    softfree: Dict[float, float] = {}
    extra_bonus: Dict[float, float] = {}
    score: Dict[float, float] = {}
    for d in clearance_thresholds:
        ok = all_free and (min_clearance >= d)
        hitfree[d] = 1 if ok else 0
        # reachable distance until first violation
        bad = (~valid) | obstacle_here | (clearance < d)
        if np.any(bad):
            first = int(np.argmax(bad))
            if not bad[0]:
                reachable[d] = float(cum[first - 1]) if first > 0 else 0.0
            else:
                reachable[d] = 0.0
        else:
            reachable[d] = float(cum[-1]) if cum.size else 0.0

        # length-weighted bad ratio + total safe length
        if seg.size:
            seg_bad = bad[:-1] | bad[1:]
            bad_len = float(seg[seg_bad].sum())
            total_len = float(seg.sum())
            safe_len[d] = total_len - bad_len
            bad_ratio_len[d] = (bad_len / total_len) if total_len > 1e-9 else 0.0
        else:
            safe_len[d] = 0.0
            bad_ratio_len[d] = 0.0

        # softfree score
        if soft_mode == "linear":
            softfree[d] = max(0.0, 1.0 - (bad_ratio_len[d] / max(soft_tau, 1e-9)))
        else:
            softfree[d] = math.exp(-soft_alpha * bad_ratio_len[d])

        # bonus for exceeding GT length safely
        if gt_len > eps:
            extra = max(0.0, safe_len[d] - gt_len)
            extra_bonus[d] = min(1.0, extra / (gt_len + eps))
            ratio = safe_len[d] / (gt_len + eps)
        else:
            extra_bonus[d] = 0.0
            ratio = 0.0

        # combined occupancy score
        w1, w2, w3 = score_weights
        score[d] = (w1 * ratio) + (w2 * softfree[d]) + (w3 * extra_bonus[d])

    return {
        "all_free": all_free,
        "min_clearance": min_clearance,
        "hitfree": hitfree,
        "reachable": reachable,
        "safe_len": safe_len,
        "bad_ratio_len": bad_ratio_len,
        "softfree": softfree,
        "extra_bonus": extra_bonus,
        "score": score,
        "gt_len": gt_len,
    }


def _parse_violation_ids(text: str) -> set[int]:
    if not text:
        return set(DEFAULT_VIOLATION_LABEL_IDS)
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ORAD-3D occupancy + Z-only eval for saved manifests.")
    ap.add_argument("--dataset-root", required=True, help="root containing training/validation/testing")
    ap.add_argument("--split", choices=("training", "validation", "testing"), required=True)
    ap.add_argument("--manifest-root", default="/home/work/byounggun/LlamaFactory/orad3d_all_models")
    ap.add_argument(
        "--model-spec",
        action="append",
        default=None,
        help="Model spec name=dir (repeatable). If omitted, uses default 6 models.",
    )
    ap.add_argument("--out-json", help="optional JSON output path")
    ap.add_argument("--clearance-thresholds", default="0,0.5,1.0", help="comma-separated meters")
    ap.add_argument("--z-max-dist", type=float, default=20.0, help="max arc-length for Z metrics (meters)")
    ap.add_argument("--z-sample-step", type=float, default=0.5, help="arc-length sampling step (meters)")
    ap.add_argument("--z-center", action="store_true", help="center z before computing z_mse/rmse")
    ap.add_argument("--z-score-beta1", type=float, default=0.7)
    ap.add_argument("--z-score-beta2", type=float, default=0.3)
    ap.add_argument("--z-score-gamma", type=float, default=1.0)

    # grid mapping
    ap.add_argument("--grid-origin", nargs=3, type=float, default=list(DEFAULT_GRID_ORIGIN))
    ap.add_argument("--grid-res", nargs="+", type=float, default=list(DEFAULT_GRID_RES))
    ap.add_argument("--grid-axis", default="auto", help="map local xyz to grid axes or auto")
    ap.add_argument("--occ-axis-order", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"), default="xyz")

    # occupancy options
    ap.add_argument("--npz-key", help="npz array key for occupancy")
    ap.add_argument("--shape", nargs=3, type=int, help="shape for raw/bin files, e.g. 256 256 32")
    ap.add_argument("--dtype", default="uint8", help="dtype for raw/bin files")
    ap.add_argument("--label-proj", choices=("bottom", "top", "max"), default="bottom")
    ap.add_argument("--violation-label-ids", type=str, default="", help="e.g. 2,3,4,5,7")

    # pred transform knobs
    ap.add_argument("--pred-path-axis-order", default="xyz", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"))
    ap.add_argument("--pred-path-scale", type=float, default=1.0)
    ap.add_argument("--pred-path-rot-deg", type=float, default=0.0)

    # topdown axes
    ap.add_argument("--topdown-axes", default="xy", help="axes for occupancy eval (xy or yx)")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")

    # soft occupancy scoring
    ap.add_argument("--soft-mode", choices=("exp", "linear"), default="exp")
    ap.add_argument("--soft-tau", type=float, default=0.05, help="linear tolerance for bad ratio")
    ap.add_argument("--soft-alpha", type=float, default=6.0, help="exp penalty weight for bad ratio")
    ap.add_argument("--score-weights", nargs=3, type=float, default=[0.6, 0.3, 0.1], help="w1 w2 w3")
    ap.add_argument("--score-eps", type=float, default=1e-6)

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    split_dir = Path(args.dataset_root) / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split not found: {split_dir}")

    clearance_thresholds = [float(x) for x in args.clearance_thresholds.split(",") if x.strip()]
    if not clearance_thresholds:
        clearance_thresholds = [0.0]

    model_specs = _parse_model_specs(args)
    if not model_specs:
        raise ValueError("no model specs provided")

    # load manifests
    manifests: Dict[str, List[Dict[str, Any]]] = {}
    for spec in model_specs:
        manifests[spec.name] = _load_manifest(spec.path)

    # build sample registry from any manifest entries
    samples: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entries in manifests.values():
        for ent in entries:
            seq = str(ent.get("sequence") or "")
            ts = str(ent.get("timestamp") or "")
            if not seq or not ts:
                continue
            key = (seq, ts)
            if key not in samples and ent.get("gt_points"):
                samples[key] = {
                    "sequence": seq,
                    "timestamp": ts,
                    "gt_points": ent.get("gt_points") or [],
                    "key": ent.get("key") or "",
                }
    if not samples:
        raise ValueError("no samples with gt_points found in manifests")

    # index predictions by model
    pred_index: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
    for name, entries in manifests.items():
        idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for ent in entries:
            seq = str(ent.get("sequence") or "")
            ts = str(ent.get("timestamp") or "")
            if not seq or not ts:
                continue
            key = (seq, ts)
            idx[key] = ent
        pred_index[name] = idx

    # accumulators
    model_stats: Dict[str, Dict[str, Any]] = {}
    for spec in model_specs:
        model_stats[spec.name] = {
            "total_samples": 0,
            "occ_samples": 0,
            "occ_missing": 0,
            "invalid_path": 0,
            "hit_counts": {d: 0 for d in clearance_thresholds},
            "reach_sum": {d: 0.0 for d in clearance_thresholds},
            "reach_count": {d: 0 for d in clearance_thresholds},
            "safe_len_sum": {d: 0.0 for d in clearance_thresholds},
            "safe_len_count": {d: 0 for d in clearance_thresholds},
            "bad_ratio_sum": {d: 0.0 for d in clearance_thresholds},
            "bad_ratio_count": {d: 0 for d in clearance_thresholds},
            "softfree_sum": {d: 0.0 for d in clearance_thresholds},
            "softfree_count": {d: 0 for d in clearance_thresholds},
            "bonus_sum": {d: 0.0 for d in clearance_thresholds},
            "bonus_count": {d: 0 for d in clearance_thresholds},
            "score_sum": {d: 0.0 for d in clearance_thresholds},
            "score_count": {d: 0 for d in clearance_thresholds},
            "gt_len_sum": 0.0,
            "gt_len_count": 0,
            "min_clear_sum": 0.0,
            "min_clear_count": 0,
            "z_count": 0,
            "z_mse_sum": 0.0,
            "z_rmse_sum": 0.0,
            "dz_mse_sum": 0.0,
            "dz_corr_sum": 0.0,
            "dz_corr_count": 0,
            "z_score_sum": 0.0,
            "z_score_count": 0,
        }

    violation_ids = _parse_violation_ids(args.violation_label_ids)

    # cache occupancy contexts per sample
    occ_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for key, sample in samples.items():
        seq, ts = sample["sequence"], sample["timestamp"]
        gt_points_raw = np.array(sample["gt_points"], dtype=np.float32)
        if gt_points_raw.size == 0:
            continue

        # occupancy context
        occ_key = (seq, ts)
        if occ_key not in occ_cache:
            scene_dir = split_dir / seq
            occ_path = _find_occ_file(scene_dir, ts)
            if occ_path is None:
                occ_cache[occ_key] = {"ok": False}
            else:
                occ, meta = load_occupancy(
                    str(occ_path),
                    tuple(args.shape) if args.shape else None,
                    args.dtype,
                    args.npz_key,
                )
                occ_points = None
                occ_labels = None
                points_are_grid = False
                raw = meta.get("raw_points")
                if raw is not None:
                    raw = np.array(raw, dtype=np.float32)
                    if raw.shape[1] >= 4:
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
                meta_res = _maybe_vec3(meta.get("voxel_size"))
                if meta_origin is not None and _is_default_grid(args.grid_origin, DEFAULT_GRID_ORIGIN):
                    origin = meta_origin
                if meta_res is not None and _is_default_grid(args.grid_res, DEFAULT_GRID_RES):
                    res_vals = meta_res

                if points_are_grid and occ_points is not None:
                    occ_min = occ_points.min(axis=0)
                    occ_max = occ_points.max(axis=0)
                else:
                    occ_shape = np.array(occ.shape, dtype=np.float32)
                    if occ_shape.size != 3:
                        occ_cache[occ_key] = {"ok": False}
                        continue
                    occ_min = np.zeros(3, dtype=np.float32)
                    occ_max = occ_shape - 1.0

                if args.grid_axis == "auto":
                    axis, _, _ = _best_grid_axis(gt_points_raw, origin, res_vals, occ_min, occ_max)
                else:
                    axis = args.grid_axis

                # build obstacle grid
                if points_are_grid and occ_points is not None:
                    grid, label_grid = _grid_from_occ_points(occ_points, occ_labels, args.label_proj)
                    if label_grid is None:
                        # fallback: treat occupancy points as obstacles
                        obstacle_mask = grid.astype(bool)
                    else:
                        obstacle_mask = np.isin(label_grid, list(violation_ids))
                else:
                    if occ.shape == (1, 1, 1):
                        occ_cache[occ_key] = {"ok": False}
                        continue
                    # project occupied voxels to 2D (x,y)
                    proj = (occ > 0).max(axis=2)
                    obstacle_mask = proj.astype(bool)

                order = [axis.index("x"), axis.index("y"), axis.index("z")]
                res_grid = res_vals[order]
                axis_res = {"x": float(res_grid[0]), "y": float(res_grid[1]), "z": float(res_grid[2])}
                dist_map = distance_transform_edt(~obstacle_mask, sampling=(axis_res["x"], axis_res["y"]))

                occ_cache[occ_key] = {
                    "ok": True,
                    "origin": origin,
                    "res": res_vals,
                    "axis": axis,
                    "obstacle_mask": obstacle_mask,
                    "dist_map": dist_map,
                    "axis_res": axis_res,
                }

        occ_ctx = occ_cache.get(occ_key, {"ok": False})

        # per-model evaluation
        for spec in model_specs:
            stats = model_stats[spec.name]
            stats["total_samples"] += 1
            entry = pred_index.get(spec.name, {}).get((seq, ts))
            pred_pts_raw: Optional[np.ndarray] = None
            if entry and entry.get("trajectory_points"):
                pred_pts_raw = np.array(entry.get("trajectory_points"), dtype=np.float32)

            # z metrics (GT vs pred)
            if pred_pts_raw is not None and pred_pts_raw.size > 0:
                pred_pts_z = _reorder_points(pred_pts_raw, args.pred_path_axis_order)
                pred_pts_z = pred_pts_z * float(args.pred_path_scale)
                pred_pts_z = _rotate_xy(pred_pts_z, args.pred_path_rot_deg)
                z_metrics = _compute_z_metrics(
                    gt_points_raw,
                    pred_pts_z,
                    max_dist=args.z_max_dist,
                    step=args.z_sample_step,
                    center=args.z_center,
                    score_beta1=args.z_score_beta1,
                    score_beta2=args.z_score_beta2,
                    score_gamma=args.z_score_gamma,
                )
                if z_metrics:
                    stats["z_count"] += 1
                    stats["z_mse_sum"] += z_metrics["z_mse"] or 0.0
                    stats["z_rmse_sum"] += z_metrics["z_rmse"] or 0.0
                    stats["dz_mse_sum"] += z_metrics["dz_mse"] or 0.0
                    if z_metrics.get("dz_corr") is not None:
                        stats["dz_corr_sum"] += z_metrics["dz_corr"]
                        stats["dz_corr_count"] += 1
                    stats["z_score_sum"] += z_metrics["z_score"] or 0.0
                    stats["z_score_count"] += 1

            # occupancy metrics
            if not occ_ctx.get("ok"):
                stats["occ_missing"] += 1
                continue
            stats["occ_samples"] += 1

            if pred_pts_raw is None or pred_pts_raw.size == 0:
                stats["invalid_path"] += 1
                # count as fail for hitfree + reachable
                for d in clearance_thresholds:
                    stats["hit_counts"][d] += 0
                    stats["reach_sum"][d] += 0.0
                    stats["reach_count"][d] += 1
                    stats["safe_len_sum"][d] += 0.0
                    stats["safe_len_count"][d] += 1
                    stats["bad_ratio_sum"][d] += 1.0
                    stats["bad_ratio_count"][d] += 1
                    stats["softfree_sum"][d] += 0.0
                    stats["softfree_count"][d] += 1
                    stats["bonus_sum"][d] += 0.0
                    stats["bonus_count"][d] += 1
                    stats["score_sum"][d] += 0.0
                    stats["score_count"][d] += 1
                continue

            pred_pts = _reorder_points(pred_pts_raw, args.pred_path_axis_order)
            pred_pts = pred_pts * float(args.pred_path_scale)
            pred_pts = _rotate_xy(pred_pts, args.pred_path_rot_deg)

            origin = occ_ctx["origin"]
            res_vals = occ_ctx["res"]
            axis = occ_ctx["axis"]
            gt_grid = _map_local_to_grid(gt_points_raw, origin, res_vals, axis)
            pred_grid = _map_local_to_grid(pred_pts, origin, res_vals, axis)

            hit_metrics = _compute_occ_metrics(
                path_grid=pred_grid,
                gt_grid=gt_grid,
                obstacle_mask=occ_ctx["obstacle_mask"],
                dist_map=occ_ctx["dist_map"],
                axes=args.topdown_axes,
                flip_x=args.flip_x,
                flip_y=args.flip_y,
                axis_res=occ_ctx["axis_res"],
                clearance_thresholds=clearance_thresholds,
                soft_mode=args.soft_mode,
                soft_tau=args.soft_tau,
                soft_alpha=args.soft_alpha,
                score_weights=tuple(float(x) for x in args.score_weights),
                eps=args.score_eps,
            )

            if hit_metrics is None:
                stats["invalid_path"] += 1
                for d in clearance_thresholds:
                    stats["hit_counts"][d] += 0
                    stats["reach_sum"][d] += 0.0
                    stats["reach_count"][d] += 1
                    stats["safe_len_sum"][d] += 0.0
                    stats["safe_len_count"][d] += 1
                    stats["bad_ratio_sum"][d] += 1.0
                    stats["bad_ratio_count"][d] += 1
                    stats["softfree_sum"][d] += 0.0
                    stats["softfree_count"][d] += 1
                    stats["bonus_sum"][d] += 0.0
                    stats["bonus_count"][d] += 1
                    stats["score_sum"][d] += 0.0
                    stats["score_count"][d] += 1
                continue

            if math.isfinite(hit_metrics.get("min_clearance", float("nan"))):
                stats["min_clear_sum"] += float(hit_metrics["min_clearance"])
                stats["min_clear_count"] += 1
            if hit_metrics.get("gt_len") is not None:
                stats["gt_len_sum"] += float(hit_metrics["gt_len"])
                stats["gt_len_count"] += 1

            for d in clearance_thresholds:
                stats["hit_counts"][d] += int(hit_metrics["hitfree"][d])
                stats["reach_sum"][d] += float(hit_metrics["reachable"][d])
                stats["reach_count"][d] += 1
                stats["safe_len_sum"][d] += float(hit_metrics["safe_len"][d])
                stats["safe_len_count"][d] += 1
                stats["bad_ratio_sum"][d] += float(hit_metrics["bad_ratio_len"][d])
                stats["bad_ratio_count"][d] += 1
                stats["softfree_sum"][d] += float(hit_metrics["softfree"][d])
                stats["softfree_count"][d] += 1
                stats["bonus_sum"][d] += float(hit_metrics["extra_bonus"][d])
                stats["bonus_count"][d] += 1
                stats["score_sum"][d] += float(hit_metrics["score"][d])
                stats["score_count"][d] += 1

    # summarize
    out: Dict[str, Any] = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "split": args.split,
            "manifest_root": str(args.manifest_root),
            "clearance_thresholds": clearance_thresholds,
            "z_max_dist": args.z_max_dist,
            "z_sample_step": args.z_sample_step,
            "z_center": args.z_center,
            "z_score_beta1": args.z_score_beta1,
            "z_score_beta2": args.z_score_beta2,
            "z_score_gamma": args.z_score_gamma,
            "grid_origin": list(map(float, args.grid_origin)),
            "grid_res": list(map(float, args.grid_res)),
            "grid_axis": args.grid_axis,
            "topdown_axes": args.topdown_axes,
            "violation_label_ids": sorted(list(violation_ids)),
            "soft_mode": args.soft_mode,
            "soft_tau": args.soft_tau,
            "soft_alpha": args.soft_alpha,
            "score_weights": [float(x) for x in args.score_weights],
            "score_eps": args.score_eps,
        },
        "models": {},
    }

    for spec in model_specs:
        stats = model_stats[spec.name]
        model_out: Dict[str, Any] = {
            "total_samples": stats["total_samples"],
            "occ_samples": stats["occ_samples"],
            "occ_missing": stats["occ_missing"],
            "invalid_path": stats["invalid_path"],
            "hitfree": {},
            "reachable": {},
            "safe_len": {},
            "bad_ratio_len": {},
            "softfree": {},
            "extra_bonus": {},
            "occ_score": {},
            "gt_len_mean": None,
            "min_clearance_mean": None,
            "z_metrics": {
                "count": stats["z_count"],
                "z_mse_mean": None,
                "z_rmse_mean": None,
                "dz_mse_mean": None,
                "dz_corr_mean": None,
                "z_score_mean": None,
            },
        }

        for d in clearance_thresholds:
            total = stats["occ_samples"]
            hits = stats["hit_counts"][d]
            model_out["hitfree"][str(d)] = {
                "hits": hits,
                "total": total,
                "rate": (hits / total) if total else None,
            }
            cnt = stats["reach_count"][d]
            model_out["reachable"][str(d)] = {
                "mean": (stats["reach_sum"][d] / cnt) if cnt else None,
                "count": cnt,
            }
            safe_cnt = stats["safe_len_count"][d]
            model_out["safe_len"][str(d)] = {
                "mean": (stats["safe_len_sum"][d] / safe_cnt) if safe_cnt else None,
                "count": safe_cnt,
            }
            bad_cnt = stats["bad_ratio_count"][d]
            model_out["bad_ratio_len"][str(d)] = {
                "mean": (stats["bad_ratio_sum"][d] / bad_cnt) if bad_cnt else None,
                "count": bad_cnt,
            }
            soft_cnt = stats["softfree_count"][d]
            model_out["softfree"][str(d)] = {
                "mean": (stats["softfree_sum"][d] / soft_cnt) if soft_cnt else None,
                "count": soft_cnt,
            }
            bonus_cnt = stats["bonus_count"][d]
            model_out["extra_bonus"][str(d)] = {
                "mean": (stats["bonus_sum"][d] / bonus_cnt) if bonus_cnt else None,
                "count": bonus_cnt,
            }
            score_cnt = stats["score_count"][d]
            model_out["occ_score"][str(d)] = {
                "mean": (stats["score_sum"][d] / score_cnt) if score_cnt else None,
                "count": score_cnt,
            }

        if stats["gt_len_count"]:
            model_out["gt_len_mean"] = stats["gt_len_sum"] / stats["gt_len_count"]

        if stats["min_clear_count"]:
            model_out["min_clearance_mean"] = stats["min_clear_sum"] / stats["min_clear_count"]

        if stats["z_count"]:
            model_out["z_metrics"]["z_mse_mean"] = stats["z_mse_sum"] / stats["z_count"]
            model_out["z_metrics"]["z_rmse_mean"] = stats["z_rmse_sum"] / stats["z_count"]
            model_out["z_metrics"]["dz_mse_mean"] = stats["dz_mse_sum"] / stats["z_count"]
            model_out["z_metrics"]["z_score_mean"] = stats["z_score_sum"] / stats["z_score_count"] if stats["z_score_count"] else None
        if stats["dz_corr_count"]:
            model_out["z_metrics"]["dz_corr_mean"] = stats["dz_corr_sum"] / stats["dz_corr_count"]

        out["models"][spec.name] = model_out

    # pretty print
    for spec in model_specs:
        m = out["models"][spec.name]
        print(f"[MODEL] {spec.name}")
        print(f"  occ_samples={m['occ_samples']}  invalid_path={m['invalid_path']}  occ_missing={m['occ_missing']}")
        for d in clearance_thresholds:
            h = m["hitfree"][str(d)]
            r = m["reachable"][str(d)]
            sl = m["safe_len"][str(d)]
            sf = m["softfree"][str(d)]
            sc = m["occ_score"][str(d)]
            bn = m["extra_bonus"][str(d)]
            rate = h["rate"]
            rate_str = f"{rate:.3f}" if rate is not None else "n/a"
            mean_reach = r["mean"]
            reach_str = f"{mean_reach:.2f}m" if mean_reach is not None else "n/a"
            safe_str = f"{sl['mean']:.2f}m" if sl["mean"] is not None else "n/a"
            soft_str = f"{sf['mean']:.3f}" if sf["mean"] is not None else "n/a"
            score_str = f"{sc['mean']:.3f}" if sc["mean"] is not None else "n/a"
            bonus_str = f"{bn['mean']:.3f}" if bn["mean"] is not None else "n/a"
            print(
                f"  d={d:.2f}m | HitFree {h['hits']}/{h['total']} ({rate_str}) | "
                f"Reachable {reach_str} | SafeLen {safe_str} | SoftFree {soft_str} | "
                f"Bonus {bonus_str} | OccScore {score_str}"
            )
        z = m["z_metrics"]
        if z["count"]:
            print(
                f"  Z: dz_mse={z['dz_mse_mean']:.4f} dz_corr={z['dz_corr_mean']} "
                f"z_score={z['z_score_mean']}"
            )
        else:
            print("  Z: n/a")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
