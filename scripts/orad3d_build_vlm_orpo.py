#!/usr/bin/env python3
"""Build multimodal preference datasets (ORPO/DPO-style or KTO-style) for LLaMAFactory VLM fine-tuning.

This script scans ORAD-3D extracted folders and emits JSONL examples compatible with LLaMAFactory's
"sharegpt" converter.

Supported preference formats:

- ORPO/DPO-style ranking (--orpo): one example contains:

    - messages: prompt turns (system + user)
    - chosen: a better assistant message
    - rejected: a worse assistant message
    - images: image paths aligned with <image> tokens

    This matches LLaMAFactory "preference dataset" (ranking=true) expected schema.

- KTO-style boolean feedback (--kto): emits one example per completion with boolean `kto_tag`.

Hard negatives are sampled from other frames.

Note: for rejected/false examples we keep text and trajectory coherent by using the scene_text that
belongs to the sampled wrong frame (same-scene text+trajectory).

You can enforce *both*:
- language similarity: the wrong frame scene_text must be similar to the positive frame scene_text,
  to avoid pulling an unrelated scene.
- trajectory dissimilarity: the wrong frame XY trajectory must be sufficiently different.

We compute language similarity with a lightweight bag-of-words cosine (no external model) and compute
trajectory similarity in XY only (ignore z) using delta-cosine + heading similarity.

Example (ORPO pairwise dataset for training split; SBERT text similarity + XY trajectory dissimilarity):

    # If you use SBERT backend:
    #   pip install sentence-transformers
    #
    # This writes:
    #   - /data3/orad3d_orpo/orad3d_train_orpo.jsonl
    #   - /data3/orad3d_orpo/dataset_info.json (required by LLaMAFactory)
    #   - /data3/orad3d_orpo/dataset_info_orpo.json (alias for convenience)
    python3 scripts/orad3d_build_vlm_orpo.py \
        --orad-root /data3/ORAD-3D \
        --splits training \
        --image-folder image_data \
        --trajectory-key trajectory_ins \
        --num-points 0 \
        --relative-media \
        --media-root /data3/ORAD-3D \
        --orpo \
        --out /data3/orad3d_orpo/orad3d_train_orpo.jsonl \
        --write-dataset-info \
        --dataset-name orad3d_orpo \
        --text-sim-backend sbert \
        --sbert-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --sbert-device cpu \
        --min-text-similarity 0.35 \
        --max-traj-similarity 0.45 \
        --train-exclude-bottom-quantile 0.25 \
        --wrong-pool-size 256 \
        --seed 0

Notes:
- <trajectory> is treated as a special token in LLaMAFactory via training config (e.g., add_special_tokens: "<trajectory>").
- If negative sampling becomes too strict (many frames skipped), relax thresholds or increase --wrong-pool-size.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import random
import re
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Tuple


_TEXT_SIM = None


_TS_RE = re.compile(r"^(?P<ts>\d+)")


def _tqdm(iterable, *, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


@dataclass(frozen=True)
class BuildOptions:
    orad_root: Path
    splits: tuple[str, ...]
    out_path: Path
    image_folder: str
    prompt_text: str
    system_text: str
    trajectory_key: str
    num_points: int
    relative_media: bool
    media_root: Path
    max_samples: int | None
    max_per_sequence: int | None
    write_dataset_info: bool
    dataset_name: str

    preference_format: Literal["none", "kto", "orpo"]

    # Trajectory hard-negative constraints (XY only).
    min_heading_diff_deg: float | None
    max_delta_cosine: float | None
    max_negative_tries: int

    # Negative sampling diversification.
    wrong_pool_size: int
    # Training split only: exclude the most-different bottom quantile by XY similarity.
    train_exclude_bottom_quantile: float
    seed: int

    # Thresholded negative selection.
    min_text_similarity: float | None
    max_traj_similarity: float | None

    # Text similarity backend.
    text_sim_backend: Literal["bow", "sbert"]
    sbert_model: str
    sbert_device: str
    sbert_cache_size: int
    hf_cache_dir: str | None

    # Progress
    use_tqdm: bool

    # Multi-output mode
    out_dir: Path | None
    prefix: str


def _configure_hf_cache_dir(hf_cache_dir: str | None) -> str | None:
    """Configure a writable Hugging Face cache dir.

    This avoids common permission issues when ~/.cache/huggingface is owned by root.
    Returns the directory used (or None if no change applied).
    """

    def _is_writable_dir(p: Path) -> bool:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_path = p / ".__hf_write_test__"
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    if hf_cache_dir:
        target = Path(hf_cache_dir).expanduser()
    else:
        default = Path.home() / ".cache" / "huggingface"
        if _is_writable_dir(default):
            return None
        target = Path.home() / ".hf_cache"

    if not _is_writable_dir(target):
        return None

    # Set before importing transformers/sentence-transformers.
    os.environ["HF_HOME"] = str(target)
    os.environ["HF_HUB_CACHE"] = str(target / "hub")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(target)
    return str(target)


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


def _choose_points(points: list[list[float]], num_points: int) -> list[list[float]]:
    if not points:
        return []
    # num_points <= 0 means: keep the original full trajectory.
    if num_points <= 0:
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


def _format_trajectory(points: list[list[float]]) -> str:
    formatted = []
    for p in points:
        if not isinstance(p, list) or len(p) < 3:
            continue
        x, y, z = p[0], p[1], p[2]
        formatted.append(f"[{x:.3f},{y:.3f},{z:.3f}]")
    return ",".join(formatted)


def _to_xy(points: list[list[float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points:
        if not isinstance(p, list) or len(p) < 2:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _normalize_xy(xy: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not xy:
        return []
    x0, y0 = xy[0]
    return [(x - x0, y - y0) for (x, y) in xy]


def _heading_angle_rad(xy_norm: list[tuple[float, float]]) -> float | None:
    if len(xy_norm) < 2:
        return None
    dx = xy_norm[-1][0]
    dy = xy_norm[-1][1]
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None
    return math.atan2(dy, dx)


def _angle_diff_rad(a: float, b: float) -> float:
    diff = abs(a - b) % (2.0 * math.pi)
    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return diff


def _delta_cosine(xy_norm_a: list[tuple[float, float]], xy_norm_b: list[tuple[float, float]]) -> float | None:
    if len(xy_norm_a) < 2 or len(xy_norm_b) < 2:
        return None

    def to_deltas(xy_norm: list[tuple[float, float]]) -> list[float]:
        flat: list[float] = []
        for (x0, y0), (x1, y1) in zip(xy_norm[:-1], xy_norm[1:]):
            flat.append(float(x1 - x0))
            flat.append(float(y1 - y0))
        return flat

    a = to_deltas(xy_norm_a)
    b = to_deltas(xy_norm_b)
    if len(a) != len(b) or len(a) == 0:
        return None

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return dot / (na * nb)


def _trajectory_similarity_xy(xy_norm_a: list[tuple[float, float]], xy_norm_b: list[tuple[float, float]]) -> float:
    """Return a heuristic similarity in [0, 1] using XY only.

    Higher means more similar. This combines:
    - Per-step delta cosine similarity
    - Overall heading similarity (end-to-end direction)
    """
    delta_cos = _delta_cosine(xy_norm_a, xy_norm_b)
    if delta_cos is None:
        delta_sim = 0.0
    else:
        delta_sim = 0.5 * (max(-1.0, min(1.0, float(delta_cos))) + 1.0)

    head_a = _heading_angle_rad(xy_norm_a)
    head_b = _heading_angle_rad(xy_norm_b)
    if head_a is None or head_b is None:
        heading_sim = 0.5
    else:
        heading_sim = 0.5 * (math.cos(_angle_diff_rad(head_a, head_b)) + 1.0)

    return max(0.0, min(1.0, 0.8 * delta_sim + 0.2 * heading_sim))


def _quantile(values: list[float], q: float) -> float:
    """Return q-quantile (0..1) using nearest-rank on sorted values."""
    if not values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    sorted_vals = sorted(values)
    idx = int(round(q * (len(sorted_vals) - 1)))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return float(sorted_vals[idx])


def _text_tokens(text: str) -> list[str]:
    # Keep it lightweight and unicode-friendly (\w includes many unicode word chars).
    return [t for t in re.findall(r"\w+", text.lower()) if t]


def _text_cosine_similarity(a: str, b: str) -> float:
    """Cosine similarity over bag-of-words counts in [0, 1]."""
    a_tokens = _text_tokens(a)
    b_tokens = _text_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0

    # Count frequencies
    counts_a: dict[str, int] = {}
    for t in a_tokens:
        counts_a[t] = counts_a.get(t, 0) + 1
    counts_b: dict[str, int] = {}
    for t in b_tokens:
        counts_b[t] = counts_b.get(t, 0) + 1

    # Dot product over intersection
    dot = 0.0
    for t, va in counts_a.items():
        vb = counts_b.get(t)
        if vb:
            dot += float(va * vb)

    na = math.sqrt(sum(float(v * v) for v in counts_a.values()))
    nb = math.sqrt(sum(float(v * v) for v in counts_b.values()))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos = dot / (na * nb)
    # Clamp to [0,1] for convenience.
    return max(0.0, min(1.0, float(cos)))


class _TextSimilarity:
    def __init__(self, opt: BuildOptions):
        self.backend = opt.text_sim_backend
        self.sbert_model_name = opt.sbert_model
        self.sbert_device = opt.sbert_device
        self.cache_size = int(opt.sbert_cache_size)
        self.hf_cache_dir = opt.hf_cache_dir
        if self.cache_size <= 0:
            self.cache_size = 0

        self._model = None
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    def _get_sbert(self):
        if self._model is not None:
            return self._model

        cache_dir = _configure_hf_cache_dir(self.hf_cache_dir)

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as err:
            raise RuntimeError(
                "SBERT backend requested but `sentence-transformers` is not installed. "
                "Install with: pip install sentence-transformers"
            ) from err

        kwargs = {"device": self.sbert_device}
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        self._model = SentenceTransformer(self.sbert_model_name, **kwargs)
        return self._model

    def _embed_sbert(self, text: str):
        # Cache by exact text; normalize embeddings so cosine is dot.
        if self.cache_size and text in self._cache:
            vec = self._cache.pop(text)
            self._cache[text] = vec
            return vec

        model = self._get_sbert()
        # `normalize_embeddings=True` returns unit vectors.
        vec = model.encode([text], normalize_embeddings=True)[0]

        if self.cache_size:
            self._cache[text] = vec
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return vec

    def similarity01(self, a: str, b: str) -> float:
        if self.backend == "bow":
            return _text_cosine_similarity(a, b)
        elif self.backend == "sbert":
            va = self._embed_sbert(a)
            vb = self._embed_sbert(b)
            # Cosine in [-1,1], but typically >=0 for related texts.
            # Clamp to [0,1] for thresholding parity with BoW.
            try:
                import numpy as np

                cos = float(np.dot(va, vb))
            except Exception:
                cos = float(sum(float(x) * float(y) for x, y in zip(va, vb)))
            return max(0.0, min(1.0, cos))
        else:
            raise ValueError(f"Unknown text_sim_backend: {self.backend}")


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


# Collect all valid trajectories (with their key + scene_text) for wrong sampling in preference modes.
def _collect_trajectory_pool(opt: BuildOptions) -> List[Tuple[Tuple[str, str, str], list[list[float]], str]]:
    pool = []
    for split in opt.splits:
        split_dir = opt.orad_root / split
        seq_dirs = list(_iter_sequences(split_dir))
        for seq_dir in _tqdm(
            seq_dirs,
            enabled=bool(opt.use_tqdm),
            desc=f"Collect pool ({split})",
            unit="seq",
            leave=False,
        ):
            image_dir = seq_dir / opt.image_folder
            scene_dir = seq_dir / "scene_data"
            local_dir = seq_dir / "local_path"

            if not (image_dir.exists() and scene_dir.exists() and local_dir.exists()):
                continue

            for img_path in image_dir.iterdir():
                if not img_path.is_file():
                    continue
                ts = _extract_timestamp(img_path.name)
                if ts is None:
                    continue

                scene_path = scene_dir / f"{ts}.txt"
                local_path = local_dir / f"{ts}.json"
                if not (scene_path.exists() and local_path.exists()):
                    continue

                try:
                    scene_text = _read_text(scene_path)
                    local_obj = _read_json(local_path)
                except Exception:
                    continue

                raw_points = local_obj.get(opt.trajectory_key)
                if not isinstance(raw_points, list) or len(raw_points) == 0:
                    continue

                try:
                    points = [list(map(float, p[:3])) for p in raw_points if isinstance(p, list) and len(p) >= 3]
                except Exception:
                    continue

                if len(points) == 0:
                    continue

                if not scene_text:
                    continue

                key = (split, seq_dir.name, ts)
                pool.append((key, points, scene_text))

    return pool


def _pick_wrong_trajectory(
    *,
    current_key: Tuple[str, str, str],
    correct_points: list[list[float]],
    correct_scene_text: str,
    traj_pool: List[Tuple[Tuple[str, str, str], list[list[float]], str]],
    opt: BuildOptions,
) -> Tuple[Tuple[str, str, str] | None, list[list[float]] | None, str | None]:
    if len(traj_pool) <= 1:
        return None, None, None

    # Compare on the same resolution as what we will emit.
    correct_xy = _normalize_xy(_to_xy(_choose_points(correct_points, opt.num_points)))
    correct_heading = _heading_angle_rad(correct_xy)

    # Build a candidate subset to avoid repeatedly choosing the same global outlier.
    pool_wo_self = [(k, pts, txt) for (k, pts, txt) in traj_pool if k != current_key]
    if not pool_wo_self:
        return None, None, None

    k = int(opt.wrong_pool_size)
    if k <= 0:
        k = 256
    k = min(k, len(pool_wo_self))
    candidates = random.sample(pool_wo_self, k=k)

    # Apply language similarity threshold first to prevent unrelated scenes.
    if opt.min_text_similarity is not None:
        min_sim = float(opt.min_text_similarity)
        candidates_lang = []
        for wrong_key, wrong_raw_points, wrong_scene_text in candidates:
            if not wrong_scene_text:
                continue
            sim = _TEXT_SIM.similarity01(correct_scene_text, wrong_scene_text)  # type: ignore[union-attr]
            if sim >= min_sim:
                candidates_lang.append((wrong_key, wrong_raw_points, wrong_scene_text))
        if candidates_lang:
            candidates = candidates_lang

    # Training split only: exclude the most-different bottom quantile.
    if current_key[0] == "training" and float(opt.train_exclude_bottom_quantile) > 0.0:
        sims = []
        for _, wrong_raw_points, _ in candidates:
            wrong_xy = _normalize_xy(_to_xy(_choose_points(wrong_raw_points, opt.num_points)))
            sims.append(_trajectory_similarity_xy(correct_xy, wrong_xy))

        threshold = _quantile(sims, float(opt.train_exclude_bottom_quantile))
        # Exclude the most different: keep strictly above threshold.
        filtered = [cand for cand, sim in zip(candidates, sims) if sim > threshold]
        if filtered:
            candidates = filtered

    # We want: language similarity high + trajectory similarity low.
    # Language similarity is maximized by construction (we reuse the same scene_text);
    # here we explicitly minimize XY trajectory similarity.
    scored: list[tuple[float, float, Tuple[str, str, str], list[list[float]], str]] = []
    for wrong_key, wrong_raw_points, wrong_scene_text in candidates:
        wrong_xy = _normalize_xy(_to_xy(_choose_points(wrong_raw_points, opt.num_points)))
        traj_sim = _trajectory_similarity_xy(correct_xy, wrong_xy)
        text_sim = _TEXT_SIM.similarity01(correct_scene_text, wrong_scene_text) if wrong_scene_text else 0.0  # type: ignore[union-attr]
        scored.append((traj_sim, text_sim, wrong_key, wrong_raw_points, wrong_scene_text))

    scored.sort(key=lambda t: t[0])  # ascending similarity => more different

    # Apply trajectory dissimilarity threshold (keep sufficiently different).
    if opt.max_traj_similarity is not None:
        max_sim = float(opt.max_traj_similarity)
        filtered_scored = [row for row in scored if row[0] <= max_sim]
        if filtered_scored:
            scored = filtered_scored

    tries = max(1, int(opt.max_negative_tries))
    for traj_sim, _, wrong_key, wrong_raw_points, wrong_scene_text in scored[: min(tries, len(scored))]:
        wrong_xy = _normalize_xy(_to_xy(_choose_points(wrong_raw_points, opt.num_points)))

        if opt.min_heading_diff_deg is not None and correct_heading is not None:
            wrong_heading = _heading_angle_rad(wrong_xy)
            if wrong_heading is None:
                continue
            heading_diff_deg = math.degrees(_angle_diff_rad(correct_heading, wrong_heading))
            if heading_diff_deg < float(opt.min_heading_diff_deg):
                continue

        if opt.max_delta_cosine is not None:
            cos = _delta_cosine(correct_xy, wrong_xy)
            if cos is not None and cos > float(opt.max_delta_cosine):
                continue

        if not wrong_scene_text:
            continue

        return wrong_key, wrong_raw_points, wrong_scene_text

    # Fallback: pick the least similar candidate even if constraints could not be satisfied.
    if scored:
        _, _, wrong_key, wrong_raw_points, wrong_scene_text = scored[0]
        return wrong_key, wrong_raw_points, wrong_scene_text

    return None, None, None


def _iter_frame_samples(
    seq_dir: Path,
    split: str,
    opt: BuildOptions,
    traj_pool: List[Tuple[Tuple[str, str, str], list[list[float]], str]],
) -> Iterator[dict]:
    image_dir = seq_dir / opt.image_folder
    scene_dir = seq_dir / "scene_data"
    local_dir = seq_dir / "local_path"

    if not (image_dir.exists() and scene_dir.exists() and local_dir.exists()):
        return

    image_files = [p for p in sorted(image_dir.iterdir(), key=lambda p: p.name) if p.is_file()]
    produced = 0

    current_key_base = (split, seq_dir.name)

    for img_path in image_files:
        ts = _extract_timestamp(img_path.name)
        if ts is None:
            continue

        scene_path = scene_dir / f"{ts}.txt"
        local_path = local_dir / f"{ts}.json"

        if not (scene_path.exists() and local_path.exists()):
            continue

        try:
            scene_text = _read_text(scene_path)
            local_obj = _read_json(local_path)
        except Exception:
            continue

        raw_points = local_obj.get(opt.trajectory_key)
        if not isinstance(raw_points, list) or len(raw_points) == 0:
            continue

        try:
            correct_points = [list(map(float, p[:3])) for p in raw_points if isinstance(p, list) and len(p) >= 3]
        except Exception:
            continue

        if len(correct_points) == 0:
            continue

        current_key = (*current_key_base, ts)

        chosen_correct = _choose_points(correct_points, opt.num_points)
        traj_correct = _format_trajectory(chosen_correct)
        if not traj_correct:
            continue

        assistant_correct = f"{scene_text}\n<trajectory>\n{traj_correct}"
        user_text = f"<image>\n{opt.prompt_text}"

        base_messages = [
            {"role": "system", "content": opt.system_text},
            {"role": "user", "content": user_text},
        ]
        images = [_to_media_path(img_path, opt)]

        if opt.preference_format == "none":
            # Standard supervised (ShareGPT/OpenAI style) example.
            sample = {
                "messages": base_messages + [{"role": "assistant", "content": assistant_correct}],
                "images": images,
                "meta": {"split": split, "sequence": seq_dir.name, "timestamp": ts},
            }
            yield sample
        elif opt.preference_format == "kto":
            # KTO: boolean feedback on a single completion.
            sample = {
                "messages": base_messages + [{"role": "assistant", "content": assistant_correct}],
                "images": images,
                "kto_tag": True,
                "meta": {"split": split, "sequence": seq_dir.name, "timestamp": ts},
            }
            yield sample
        elif opt.preference_format == "orpo":
            wrong_key, wrong_raw_points, wrong_scene_text = _pick_wrong_trajectory(
                current_key=current_key,
                correct_points=correct_points,
                correct_scene_text=scene_text,
                traj_pool=traj_pool,
                opt=opt,
            )
            if wrong_raw_points is None:
                continue

            wrong_chosen = _choose_points(wrong_raw_points, opt.num_points)
            traj_wrong = _format_trajectory(wrong_chosen)
            if not traj_wrong:
                continue

            # Rejected should be coherent: use the scene_text belonging to the wrong trajectory frame.
            if not wrong_scene_text:
                continue
            assistant_wrong = f"{wrong_scene_text}\n<trajectory>\n{traj_wrong}"

            # Add debug similarities (helps verify thresholds).
            wrong_xy_norm = _normalize_xy(_to_xy(_choose_points(wrong_raw_points, opt.num_points)))
            traj_sim = _trajectory_similarity_xy(_normalize_xy(_to_xy(_choose_points(correct_points, opt.num_points))), wrong_xy_norm)
            text_sim = _TEXT_SIM.similarity01(scene_text, wrong_scene_text)  # type: ignore[union-attr]

            # ORPO/DPO-style preference example: prompt messages end with user (odd turns),
            # chosen/rejected are assistant messages.
            sample = {
                "messages": base_messages,
                "images": images,
                "chosen": {"role": "assistant", "content": assistant_correct},
                "rejected": {"role": "assistant", "content": assistant_wrong},
                "meta": {
                    "split": split,
                    "sequence": seq_dir.name,
                    "timestamp": ts,
                    "negative_from": {"split": wrong_key[0], "sequence": wrong_key[1], "timestamp": wrong_key[2]},
                    "neg_text_sim": round(float(text_sim), 6),
                    "neg_traj_sim": round(float(traj_sim), 6),
                },
            }
            yield sample
        else:
            raise ValueError(f"Unknown preference_format: {opt.preference_format}")

        produced += 1
        if opt.max_per_sequence is not None and produced >= opt.max_per_sequence:
            return


def build_samples(opt: BuildOptions) -> Iterable[dict]:
    traj_pool = _collect_trajectory_pool(opt) if opt.preference_format in {"kto", "orpo"} else []

    emitted = 0
    for split in opt.splits:
        split_dir = opt.orad_root / split
        seq_dirs = list(_iter_sequences(split_dir))
        for seq_dir in _tqdm(
            seq_dirs,
            enabled=bool(opt.use_tqdm),
            desc=f"Build samples ({split})",
            unit="seq",
        ):
            for sample in _iter_frame_samples(seq_dir, split, opt, traj_pool):
                yield sample
                emitted += 1
                if opt.max_samples is not None and emitted >= opt.max_samples:
                    return


def build_split_samples(opt: BuildOptions, split: str) -> Iterable[dict]:
    """Build samples for a single split.

    - training: ORPO pairwise preference (chosen/rejected)
    - validation/testing: ShareGPT supervised samples
    """
    pref: Literal["none", "kto", "orpo"] = "orpo" if split == "training" else "none"
    opt_split = replace(opt, splits=(split,), preference_format=pref)

    traj_pool = _collect_trajectory_pool(opt_split) if pref in {"kto", "orpo"} else []

    split_dir = opt_split.orad_root / split
    seq_dirs = list(_iter_sequences(split_dir))
    for seq_dir in _tqdm(
        seq_dirs,
        enabled=bool(opt_split.use_tqdm),
        desc=f"Build samples ({split})",
        unit="seq",
    ):
        for sample in _iter_frame_samples(seq_dir, split, opt_split, traj_pool):
            yield sample


def _write_dataset_info(out_dir: Path, opt: BuildOptions) -> None:
    info: dict = {
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
            },
        }
    }

    if opt.preference_format == "kto":
        info[opt.dataset_name]["columns"]["kto_tag"] = "kto_tag"
    elif opt.preference_format == "orpo":
        info[opt.dataset_name]["ranking"] = True
        info[opt.dataset_name]["columns"]["chosen"] = "chosen"
        info[opt.dataset_name]["columns"]["rejected"] = "rejected"

    # LLaMAFactory loads dataset metadata from `dataset_info.json`.
    payload = json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "dataset_info.json").write_text(payload, encoding="utf-8")
    # Keep a secondary alias for convenience when inspecting generated artifacts.
    if opt.preference_format == "orpo":
        (out_dir / "dataset_info_orpo.json").write_text(payload, encoding="utf-8")


def _write_dataset_info_sharegpt_multi(out_dir: Path, datasets: dict[str, str]) -> None:
    """Write dataset_info.json with multiple ShareGPT datasets.

    Args:
        datasets: mapping dataset_name -> file_name
    """
    info: dict = {}
    for name, file_name in datasets.items():
        info[name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }

    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ORAD-3D VLM dataset (ShareGPT/KTO/ORPO) for LLaMAFactory.")
    ap.add_argument("--orad-root", default="/data3/ORAD-3D", help="Root folder containing training/validation/testing")
    ap.add_argument("--splits", nargs="+", default=["training"], choices=["training", "validation", "testing"])
    ap.add_argument("--out", default=None, help="Output JSONL path (single-file mode)")
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for multi-split mode. If set, training is written as ORPO JSONL and "
            "validation/testing are written as ShareGPT JSONL."
        ),
    )
    ap.add_argument(
        "--prefix",
        default="orad3d",
        help="Filename prefix used in --out-dir mode.",
    )
    ap.add_argument("--image-folder", default="image_data", choices=["gt_image", "image_data"])
    ap.add_argument("--prompt", default="I am seeing an off-road driving image. Please generate a safe drivable trajectory for my vehicle to follow.")
    ap.add_argument("--system", default="You are an off-road autonomous driving agent. Given an input camera image, describe the scene and provide a safe drivable trajectory. Output the trajectory after a <trajectory> token as a comma-separated list of [x,y,z] points.")
    ap.add_argument("--trajectory-key", default="trajectory_ins", choices=["trajectory_ins", "trajectory_hmi", "trajectory_ins_past", "trajectory_hmi_past"])
    ap.add_argument(
        "--num-points",
        type=int,
        default=0,
        help="How many trajectory points to output. 0 means use the full original trajectory.",
    )
    ap.add_argument("--relative-media", action="store_true")
    ap.add_argument("--media-root", default="/data3/ORAD-3D")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-per-seq", type=int, default=None)
    ap.add_argument("--write-dataset-info", action="store_true")
    ap.add_argument("--dataset-name", default="orad3d_vlm")

    pref = ap.add_mutually_exclusive_group()
    pref.add_argument("--kto", action="store_true", help="Emit KTO format (boolean feedback via kto_tag=true)")
    pref.add_argument("--orpo", action="store_true", help="Emit ORPO/DPO ranking format (chosen/rejected)")

    ap.add_argument(
        "--min-heading-diff-deg",
        type=float,
        default=None,
        help="Optional hard-negative constraint: require heading angle difference >= this (degrees), XY only.",
    )
    ap.add_argument(
        "--max-delta-cosine",
        type=float,
        default=None,
        help=(
            "Optional hard-negative constraint: require cosine(similarity) between per-step XY deltas <= this. "
            "Lower => more different turning/shape."
        ),
    )
    ap.add_argument(
        "--max-negative-tries",
        type=int,
        default=50,
        help="How many attempts to find a wrong trajectory that satisfies constraints.",
    )

    ap.add_argument(
        "--wrong-pool-size",
        type=int,
        default=256,
        help="How many candidate wrong trajectories to sample per example (prevents global outlier dominating).",
    )
    ap.add_argument(
        "--train-exclude-bottom-quantile",
        type=float,
        default=0.25,
        help="Training split only: exclude the most-different bottom quantile by XY similarity when sampling negatives.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")

    ap.add_argument(
        "--min-text-similarity",
        type=float,
        default=0.2,
        help=(
            "Minimum bag-of-words cosine similarity between positive and negative scene_text. "
            "Higher => more semantically related scenes (lightweight heuristic)."
        ),
    )
    ap.add_argument(
        "--max-traj-similarity",
        type=float,
        default=0.5,
        help="Maximum allowed XY trajectory similarity for negatives (lower => more different).",
    )

    ap.add_argument(
        "--text-sim-backend",
        choices=["bow", "sbert"],
        default="sbert",
        help="Text similarity backend used for --min-text-similarity. Default: sbert.",
    )
    ap.add_argument(
        "--sbert-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-Transformers model name/path used when --text-sim-backend sbert.",
    )
    ap.add_argument(
        "--sbert-device",
        default="cpu",
        help="Device for SBERT encoding (e.g., cpu, cuda).",
    )
    ap.add_argument(
        "--sbert-cache-size",
        type=int,
        default=50000,
        help="LRU cache size for SBERT embeddings keyed by scene_text. 0 disables caching.",
    )
    ap.add_argument(
        "--hf-cache-dir",
        default=None,
        help=(
            "Writable Hugging Face cache directory for downloading SBERT/transformers models. "
            "If not set, uses ~/.cache/huggingface when writable, otherwise falls back to ~/.hf_cache."
        ),
    )
    ap.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    args = ap.parse_args()

    random.seed(int(args.seed))

    preference_format: Literal["none", "kto", "orpo"] = "none"
    if args.kto:
        preference_format = "kto"
    elif args.orpo:
        preference_format = "orpo"

    out_dir = Path(args.out_dir) if args.out_dir else None
    out_path = Path(args.out) if args.out else Path(".")
    if out_dir is None and args.out is None:
        raise SystemExit("[ERR] Provide either --out (single-file) or --out-dir (multi-split).")

    opt = BuildOptions(
        orad_root=Path(args.orad_root),
        splits=tuple(args.splits),
        out_path=out_path,
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
        preference_format=preference_format,
        min_heading_diff_deg=args.min_heading_diff_deg,
        max_delta_cosine=args.max_delta_cosine,
        max_negative_tries=int(args.max_negative_tries),
        wrong_pool_size=int(args.wrong_pool_size),
        train_exclude_bottom_quantile=float(args.train_exclude_bottom_quantile),
        seed=int(args.seed),
        min_text_similarity=float(args.min_text_similarity) if args.min_text_similarity is not None else None,
        max_traj_similarity=float(args.max_traj_similarity) if args.max_traj_similarity is not None else None,
        text_sim_backend=args.text_sim_backend,
        sbert_model=str(args.sbert_model),
        sbert_device=str(args.sbert_device),
        sbert_cache_size=int(args.sbert_cache_size),
        hf_cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
        use_tqdm=(not bool(args.no_tqdm)) and bool(getattr(sys.stderr, "isatty", lambda: False)()),
        out_dir=out_dir,
        prefix=str(args.prefix),
    )

    global _TEXT_SIM
    _TEXT_SIM = _TextSimilarity(opt)

    # Multi-split mode: training -> ORPO, validation/testing -> ShareGPT.
    if opt.out_dir is not None:
        opt.out_dir.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        sharegpt_datasets: dict[str, str] = {}

        for split in opt.splits:
            if split == "training":
                out_file = opt.out_dir / f"{opt.prefix}_training_orpo.jsonl"
                opt_train = replace(opt, out_path=out_file, preference_format="orpo")
                count = 0
                with out_file.open("w", encoding="utf-8") as f:
                    for sample in build_split_samples(opt_train, "training"):
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1

                written.append(f"{out_file} ({count} samples)")
                if opt.write_dataset_info:
                    _write_dataset_info(opt.out_dir, opt_train)

            elif split in {"validation", "testing"}:
                out_file = opt.out_dir / f"{opt.prefix}_{split}_sharegpt.jsonl"
                opt_eval = replace(opt, out_path=out_file, preference_format="none")
                count = 0
                with out_file.open("w", encoding="utf-8") as f:
                    for sample in build_split_samples(opt_eval, split):
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1

                written.append(f"{out_file} ({count} samples)")
                sharegpt_datasets[f"{opt.dataset_name}_{split}"] = out_file.name

        if opt.write_dataset_info and sharegpt_datasets:
            _write_dataset_info_sharegpt_multi(opt.out_dir, sharegpt_datasets)

        for line in written:
            print(f"[OK] wrote {line}")
        if opt.write_dataset_info:
            if "training" in opt.splits:
                print(f"[OK] wrote dataset_info_orpo.json -> {opt.out_dir / 'dataset_info_orpo.json'}")
            if sharegpt_datasets:
                print(f"[OK] wrote dataset_info.json -> {opt.out_dir / 'dataset_info.json'}")

    else:
        # Single-file mode (backward compatible)
        opt.out_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with opt.out_path.open("w", encoding="utf-8") as f:
            for sample in build_samples(opt):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1

        if opt.write_dataset_info:
            _write_dataset_info(opt.out_path.parent, opt)

        mode = (
            "ORPO ranking (chosen/rejected)"
            if opt.preference_format == "orpo"
            else "KTO (kto_tag)"
            if opt.preference_format == "kto"
            else "ShareGPT"
        )
        print(f"[OK] wrote {count} samples ({mode}) -> {opt.out_path}")
        if opt.write_dataset_info:
            print(f"[OK] wrote dataset_info.json -> {opt.out_path.parent / 'dataset_info.json'}")
            if opt.preference_format == "orpo":
                print(f"[OK] wrote dataset_info_orpo.json -> {opt.out_path.parent / 'dataset_info_orpo.json'}")

        if count == 0:
            print("[WARN] no samples were emitted. Check paths and extracted files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())