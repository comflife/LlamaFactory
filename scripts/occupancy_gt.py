#!/usr/bin/env python3
"""
Visualize ORAD-3D occupancy GT with GT path.

Examples
  python scripts/occupancy_gt.py \
    --dataset-root /home/work/datasets/bg/ORAD-3D --split testing --scene y0602_1309 \
    --mode topdown --save-dir occupancy_plots --mode topdown --grid-axis xyz --path-source local_path_ins --path-frame local --grid-origin -50 0 -3 --grid-res 1.0

  python scripts/occupancy_gt.py \
  --dataset-root /home/work/datasets/bg/ORAD-3D --split testing --scene y0602_1309 \
  --mode topdown --save-dir occupancy_plots \
  --path-source local_path_ins --path-frame local \
  --grid-axis auto --debug

"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


def _split_floats(line: str) -> List[float]:
    line = line.replace(",", " ").strip()
    if not line:
        return []
    return [float(x) for x in line.split()]


def _iter_occ_files(occ_dir: str, pattern: Optional[str]) -> List[str]:
    if pattern:
        return sorted(glob.glob(os.path.join(occ_dir, pattern)))
    exts = (".npz", ".npy", ".bin", ".raw")
    out = []
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
    meta = {}
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

def _find_nearest_local_path(local_path_dir: str, ts: str, key: str) -> Optional[np.ndarray]:
    try:
        target = float(ts)
    except ValueError:
        return None
    files = glob.glob(os.path.join(local_path_dir, "*.json"))
    if not files:
        return None
    best = None
    best_diff = None
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        try:
            val = float(stem)
        except ValueError:
            continue
        diff = abs(val - target)
        if best_diff is None or diff < best_diff:
            best = stem
            best_diff = diff
    if best is None:
        return None
    return _load_local_path(local_path_dir, best, key)



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


def _labels_to_rgba(
    labels: np.ndarray, label_map: Dict[int, Tuple[float, float, float]], alpha: float
) -> np.ndarray:
    import matplotlib.pyplot as plt

    uniq = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(1, len(uniq)))
    colors = []
    for i, lab in enumerate(uniq):
        if lab in label_map:
            r, g, b = label_map[lab]
        else:
            r, g, b, _ = cmap(i)
        a = 0.0 if lab == 0 else alpha
        colors.append((r, g, b, a))
    colors = np.array(colors, dtype=np.float32)
    idx = np.searchsorted(uniq, labels)
    return colors[idx]


def _plot_path(ax, path_xy: np.ndarray, color: Optional[str], width: float, gradient: bool) -> None:
    if path_xy.shape[0] == 0:
        return
    if not gradient and color:
        ax.plot(path_xy[:, 0], path_xy[:, 1], color=color, linewidth=width)
    else:
        from matplotlib.collections import LineCollection

        segs = np.stack([path_xy[:-1], path_xy[1:]], axis=1)
        values = np.linspace(0.0, 1.0, segs.shape[0])
        lc = LineCollection(segs, cmap="viridis", linewidths=width)
        lc.set_array(values)
        ax.add_collection(lc)


def _plot_topdown(
    occ: np.ndarray,
    occ_points: Optional[np.ndarray],
    occ_labels: Optional[np.ndarray],
    points_are_grid: bool,
    use_labels: bool,
    label_map: Dict[int, Tuple[float, float, float]],
    label_proj: str,
    label_alpha: float,
    path: np.ndarray,
    title: str,
    save: Optional[str],
    axes: str,
    flip_x: bool,
    flip_y: bool,
    path_width: float,
    path_color: Optional[str],
    path_gradient: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:
        raise RuntimeError("matplotlib is required for topdown view") from exc

    axes = axes.lower()
    if len(axes) != 2 or any(a not in "xyz" for a in axes) or axes[0] == axes[1]:
        raise ValueError("topdown-axes must be one of: xy, yx, xz, zx, yz, zy")

    fig, ax = plt.subplots(figsize=(7, 7))

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
        if use_labels and label_grid is not None:
            rgba = _labels_to_rgba(label_grid.T, label_map, label_alpha)
            ax.imshow(rgba, origin="lower")
        else:
            cmap = ListedColormap(["white", "#7a7a7a"])
            ax.imshow(grid.T, cmap=cmap, origin="lower")
    else:
        ax.text(0.5, 0.5, "no occupied voxels", ha="center", va="center", transform=ax.transAxes)

    idx = {"x": 0, "y": 1, "z": 2}
    path_xy = path[:, [idx[axes[0]], idx[axes[1]]]]
    path_xy = path_xy[np.isfinite(path_xy).all(axis=1)]
    if path_xy.size:
        if flip_x:
            path_xy[:, 0] = -path_xy[:, 0]
        if flip_y:
            path_xy[:, 1] = -path_xy[:, 1]
        _plot_path(ax, path_xy, path_color, path_width, path_gradient)
        ax.scatter(path_xy[0, 0], path_xy[0, 1], color="green", s=25, label="start")
        ax.scatter(path_xy[-1, 0], path_xy[-1, 1], color="orange", s=25, label="end")

    ax.set_title(title)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
    else:
        plt.show()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ORAD-3D occupancy GT with path")
    parser.add_argument("--dataset-root", required=True, help="root containing training/validation/testing")
    parser.add_argument("--split", choices=("training", "validation", "testing"), required=True)
    parser.add_argument("--scene", help="specific scene id (folder name)")
    parser.add_argument("--all", action="store_true", help="process all scenes with occupancy+poses")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--occ-pattern", help="glob pattern inside occupancy dir (e.g. *.npz)")
    parser.add_argument("--occ-index", type=int, default=0)
    parser.add_argument("--occ-all", action="store_true", help="process all occupancy files in scene")
    parser.add_argument("--npz-key", help="npz array key for occupancy")
    parser.add_argument("--shape", nargs=3, type=int, help="shape for raw/bin files, e.g. 256 256 32")
    parser.add_argument("--dtype", default="uint8", help="dtype for raw/bin files")
    parser.add_argument("--mode", choices=("topdown",), default="topdown")
    parser.add_argument("--save-dir", help="save images into directory (per scene)")
    parser.add_argument("--pose-to-occ", help="optional 4x4 transform file for pose to occupancy frame")

    parser.add_argument(
        "--path-source",
        default="local_path_ins",
        choices=(
            "poses",
            "local_path_ins",
            "local_path_hmi",
            "local_path_ins_past",
            "local_path_hmi_past",
        ),
    )
    parser.add_argument("--local-path-nearest", dest="local_path_nearest", action="store_true", default=True)
    parser.add_argument("--no-local-path-nearest", dest="local_path_nearest", action="store_false")
    parser.add_argument("--path-frame", default="local", choices=("local", "global"))
    parser.add_argument("--path-axis-order", default="xyz", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"))
    parser.add_argument("--path-scale", type=float, default=1.0, help="scale path units")
    parser.add_argument("--path-rot-deg", type=float, default=0.0, help="extra yaw rotation in degrees")
    parser.add_argument("--path-clip", dest="path_clip", action="store_true", default=True)
    parser.add_argument("--no-path-clip", dest="path_clip", action="store_false")

    parser.add_argument("--grid-origin", nargs=3, type=float, default=[-25.0, 5.7, -3.0])
    parser.add_argument("--grid-res", nargs="+", type=float, default=[0.5])
    parser.add_argument("--grid-axis", default="xyz", help="map local xyz to grid axes or auto")
    parser.add_argument(
        "--occ-axis-order",
        choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"),
        default="xyz",
        help="axis order of occupancy point columns",
    )

    parser.add_argument("--topdown-axes", default="xy", help="axes to show in topdown view")
    parser.add_argument("--flip-x", action="store_true", help="flip topdown x axis")
    parser.add_argument("--flip-y", action="store_true", help="flip topdown y axis")
    parser.add_argument("--path-width", type=float, default=4.0)
    parser.add_argument("--path-color", default="#7e3f98")
    parser.add_argument("--path-gradient", action="store_true")

    parser.add_argument("--use-labels", action="store_true", help="colorize occupancy labels")
    parser.add_argument("--label-map", help="label color map file: id r g b [name]")
    parser.add_argument("--label-proj", choices=("bottom", "top", "max"), default="bottom")
    parser.add_argument("--label-alpha", type=float, default=1.0)

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    split_dir = os.path.join(args.dataset_root, args.split)
    if not os.path.isdir(split_dir):
        print(f"error: split dir not found: {split_dir}", file=sys.stderr)
        return 2

    if args.scene and not args.all:
        scenes = [args.scene]
    else:
        scenes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    valid = []
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

    for scene in sample_scenes:
        scene_dir = os.path.join(split_dir, scene)
        occ_dir = os.path.join(scene_dir, "occupancy")
        occ_files = _iter_occ_files(occ_dir, args.occ_pattern)
        if not occ_files:
            print(f"skip: no occupancy files in {occ_dir}", file=sys.stderr)
            continue

        if args.occ_all:
            occ_list = occ_files
        else:
            if args.occ_index < 0 or args.occ_index >= len(occ_files):
                print(f"skip: occ-index out of range for {scene}", file=sys.stderr)
                continue
            occ_list = [occ_files[args.occ_index]]

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

            path = None
            if args.path_source.startswith("local_path"):
                local_dir = os.path.join(scene_dir, "local_path")
                key_map = {
                    "local_path_ins": "trajectory_ins",
                    "local_path_hmi": "trajectory_hmi",
                    "local_path_ins_past": "trajectory_ins_past",
                    "local_path_hmi_past": "trajectory_hmi_past",
                }
                key = key_map.get(args.path_source, "trajectory_ins")
                path = _load_local_path(local_dir, occ_ts, key)
                if path is None and args.local_path_nearest:
                    path = _find_nearest_local_path(local_dir, occ_ts, key)

            if path is None:
                poses, timestamps, rpy = load_poses_with_meta(poses_path)
                path = poses
                if args.path_frame == "local":
                    idx = _find_nearest_pose_idx(timestamps, occ_ts_val)
                    yaw = rpy[idx, 2] if rpy.size else None
                    path = _to_local_frame(path, poses[idx], yaw)

            path = _reorder_points(path, args.path_axis_order)
            path = path * float(args.path_scale)
            if abs(args.path_rot_deg) > 1e-6:
                rad = np.deg2rad(args.path_rot_deg)
                c, s = np.cos(rad), np.sin(rad)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                path = (R @ path.T).T

            if T_pose_to_occ is not None:
                path = _apply_pose_transform(path, T_pose_to_occ)

            if points_are_grid and occ_points is not None:
                origin = np.array(args.grid_origin, dtype=np.float32)
                res_vals = np.array(args.grid_res, dtype=np.float32)
                if res_vals.size == 1:
                    res_vals = np.array([res_vals[0]] * 3, dtype=np.float32)

                occ_min = occ_points.min(axis=0)
                occ_max = occ_points.max(axis=0)

                if args.grid_axis == "auto":
                    axis, path, ratio = _best_grid_axis(path, origin, res_vals, occ_min, occ_max)
                    if args.debug:
                        print(
                            f"[{scene} {occ_ts}] auto grid-axis={axis} in_ratio={ratio:.3f}",
                            file=sys.stderr,
                        )
                else:
                    path = _map_local_to_grid(path, origin, res_vals, args.grid_axis)

                if args.path_clip:
                    mask = np.logical_and(path >= occ_min, path <= occ_max).all(axis=1)
                    path = path[mask]

                if args.debug:
                    in_ratio = np.mean(
                        np.logical_and(path >= occ_min, path <= occ_max).all(axis=1)
                    )
                    print(
                        f"[{scene} {occ_ts}] path in occ bounds ratio: {in_ratio:.3f}",
                        file=sys.stderr,
                    )

            title = f"{scene} | {os.path.basename(occ_path)}"
            save = None
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                save = os.path.join(args.save_dir, f"{scene}_{occ_ts}.png")

            use_labels = args.use_labels or (occ_labels is not None and np.max(occ_labels) > 1)

            _plot_topdown(
                occ=occ,
                occ_points=occ_points,
                occ_labels=occ_labels,
                points_are_grid=points_are_grid,
                use_labels=use_labels,
                label_map=label_map,
                label_proj=args.label_proj,
                label_alpha=args.label_alpha,
                path=path,
                title=title,
                save=save,
                axes=args.topdown_axes,
                flip_x=args.flip_x,
                flip_y=args.flip_y,
                path_width=args.path_width,
                path_color=None if args.path_gradient else args.path_color,
                path_gradient=args.path_gradient,
            )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
