#!/usr/bin/env python3
"""Visualize ORPO preference pairs (chosen vs rejected) for ORAD-3D VLM datasets.

Two modes:

1) JSONL mode (offline):
    Reads an ORPO JSONL produced by scripts/orad3d_build_vlm_orpo.py (training split output)
    where each record contains:
      - images: ["relative/or/absolute/path.png"]
      - chosen:  "<scene text>\n<trajectory>\n[x,y,z],[x,y,z],..."
      - rejected:"<scene text>\n<trajectory>\n[x,y,z],[x,y,z],..."

2) Live mode (online):
    Scans ORAD-3D folders and samples (anchor, negative) pairs on-the-fly, then renders
    chosen/rejected overlays similarly. This is useful for quickly testing sampling settings
    before generating full JSONL.

For each sampled record, this script:
  - loads the image
  - parses chosen/rejected texts + trajectories
  - overlays both trajectories in ego-centric XY on the same image
      - chosen: green
      - rejected: red
    (trajectory comparison is XY-only; z is ignored for drawing)
  - renders chosen/rejected scene text in a right-side panel
  - saves a single composite PNG per sample

Example:
python3 scripts/orad3d_visualize_orpo_pairs.py   --orad-root /data3/ORAD-3D   --split training   --image-folder image_data   --trajectory-key trajectory_ins   --text-sim-backend sbert   --require-sbert   --hf-cache-dir /home/byounggun/.hf_cache   --wrong-pool-size 1024   --min-text-similarity 0.3   --max-traj-similarity 0.9   --train-exclude-bottom-quantile 0.0   --media-root /data3/ORAD-3D   --out-dir /home/byounggun/LlamaFactory/orad3d_orpo_pair_viz   --num-samples 5   --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_TRAJ_TOKEN = "<trajectory>"
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_TS_RE = re.compile(r"^(?P<ts>\d+)")


def _configure_hf_cache_dir(hf_cache_dir: Optional[str]) -> Optional[str]:
    """Configure a writable Hugging Face cache dir.

    Avoids permission issues when ~/.cache/huggingface is owned by root.
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

    os.environ["HF_HOME"] = str(target)
    os.environ["HF_HUB_CACHE"] = str(target / "hub")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(target)
    return str(target)


class _TextSimilarity:
    def __init__(
        self,
        backend: str,
        sbert_model: str,
        sbert_device: str,
        cache_size: int,
        require_sbert: bool,
        hf_cache_dir: Optional[str],
    ) -> None:
        self._backend = backend
        self._cache_size = int(cache_size)
        self._cache: dict[str, np.ndarray] = {}
        self._order: List[str] = []

        self._model = None
        if backend == "sbert":
            cache_dir = _configure_hf_cache_dir(hf_cache_dir)
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as e:
                if require_sbert:
                    raise SystemExit(
                        "[ERR] text-sim backend 'sbert' requires sentence-transformers. "
                        "Install it or use --text-sim-backend bow"
                    ) from e

                # Graceful fallback for quick visualization.
                print(
                    "[WARN] sentence-transformers not available; falling back to BoW text similarity. "
                    "(Tip: activate the right env then `pip install sentence-transformers`)",
                    file=sys.stderr,
                )
                self._backend = "bow"
                return

            kwargs = {"device": sbert_device}
            if cache_dir:
                kwargs["cache_folder"] = cache_dir
            self._model = SentenceTransformer(sbert_model, **kwargs)

    def similarity(self, a: str, b: str) -> float:
        if self._backend == "sbert":
            va = self._embed(a)
            vb = self._embed(b)
            denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
            if denom <= 1e-12:
                return 0.0
            return float(np.dot(va, vb) / denom)
        return _bow_cosine(a, b)

    def _embed(self, text: str) -> np.ndarray:
        key = text.strip()
        if key in self._cache:
            return self._cache[key]

        assert self._model is not None
        vec = self._model.encode([key], normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(vec[0], dtype=np.float32)

        self._cache[key] = emb
        self._order.append(key)
        if self._cache_size > 0 and len(self._order) > self._cache_size:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return emb


@dataclass(frozen=True)
class PairSample:
    index: int
    image_path: Path
    chosen_text: str
    rejected_text: str
    chosen_xyz: List[Tuple[float, float, float]]
    rejected_xyz: List[Tuple[float, float, float]]
    meta: dict


@dataclass(frozen=True)
class _Frame:
    split: str
    sequence: str
    timestamp: str
    image_path: Path
    scene_text: str
    xyz: List[Tuple[float, float, float]]


def _resolve_image_path(image_value: str, media_root: Optional[Path]) -> Path:
    p = Path(image_value)
    if p.is_absolute() or media_root is None:
        return p
    return media_root / p


def _split_text_and_traj(assistant_text: str) -> tuple[str, List[Tuple[float, float, float]]]:
    """Split assistant_text into (scene_text, xyz_points).

    Accepts the format generated by the ORPO builder.
    """

    if not isinstance(assistant_text, str):
        return "", []

    if _TRAJ_TOKEN not in assistant_text:
        return assistant_text.strip(), []

    before, after = assistant_text.split(_TRAJ_TOKEN, 1)
    scene_text = before.strip()

    # after typically starts with newlines; parse all [x,y,z] groups.
    traj_text = after.strip()
    points: List[Tuple[float, float, float]] = []
    for match in _BRACKET_RE.finditer(traj_text):
        raw = match.group(1)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2]) if len(parts) >= 3 else 0.0
        except Exception:
            continue
        points.append((x, y, z))

    return scene_text, points


def _extract_timestamp(name: str) -> str | None:
    match = _TS_RE.match(name)
    if not match:
        return None
    return match.group("ts")


def _bow_cosine(a: str, b: str) -> float:
    def to_counts(s: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tok in re.findall(r"[\w]+", s.lower()):
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    ca = to_counts(a)
    cb = to_counts(b)
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    dot = float(sum(ca[t] * cb[t] for t in common))
    na = float(sum(v * v for v in ca.values()))
    nb = float(sum(v * v for v in cb.values()))
    denom = (na * nb) ** 0.5
    if denom <= 1e-12:
        return 0.0
    return dot / denom


def _trajectory_xy_normalized(xyz: Sequence[Tuple[float, float, float]]) -> List[Tuple[float, float]]:
    if not xyz:
        return []
    x0, y0, _ = xyz[0]
    out: List[Tuple[float, float]] = []
    for x, y, _z in xyz:
        out.append((float(x - x0), float(y - y0)))
    return out


def _trajectory_similarity_xy(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    """Cheap similarity in [0,1]: 1 means very similar."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n < 2:
        return 0.0

    aa = np.asarray(a[:n], dtype=np.float32)
    bb = np.asarray(b[:n], dtype=np.float32)

    da = aa[1:] - aa[:-1]
    db = bb[1:] - bb[:-1]
    na = np.linalg.norm(da, axis=1) + 1e-8
    nb = np.linalg.norm(db, axis=1) + 1e-8
    da = da / na[:, None]
    db = db / nb[:, None]
    cos = np.sum(da * db, axis=1)
    cos = np.clip(cos, -1.0, 1.0)

    # Map average cosine from [-1,1] to [0,1]
    return float((np.mean(cos) + 1.0) / 2.0)


def _ego_xy_to_pixels(
    xyz: Sequence[Tuple[float, float, float]],
    image_size: Tuple[int, int],
    scale_px_per_meter: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Map ego-centric XY (x=lateral, y=forward) to image pixels.

    - Origin is bottom-center of the image.
    - The first point is treated as the origin (0,0).
    - Z is ignored.

    This follows the same convention as scripts/orad3d_make_samples.py.
    """

    if not xyz:
        return []

    width, height = image_size

    x0, y0, _ = xyz[0]
    lateral = np.asarray([x - x0 for x, _, _ in xyz], dtype=np.float64)
    forward = np.asarray([y - y0 for _, y, _ in xyz], dtype=np.float64)

    if scale_px_per_meter is None:
        max_forward = float(np.max(forward)) if forward.size else 0.0
        max_lateral = float(np.max(np.abs(lateral))) if lateral.size else 0.0

        scale_h = (0.80 * height / max_forward) if max_forward > 1e-6 else 10.0
        scale_w = (0.45 * width / max_lateral) if max_lateral > 1e-6 else 10.0
        scale_px_per_meter = max(1.0, min(scale_h, scale_w))

    u = (width / 2.0) + (lateral * scale_px_per_meter)
    v = (height - 1.0) - (forward * scale_px_per_meter)
    return list(zip(u.tolist(), v.tolist()))


def _auto_scale_for_both(
    chosen_xyz: Sequence[Tuple[float, float, float]],
    rejected_xyz: Sequence[Tuple[float, float, float]],
    image_size: Tuple[int, int],
) -> float:
    """Compute a single scale so both trajectories fit reasonably."""

    width, height = image_size

    def stats(xyz: Sequence[Tuple[float, float, float]]) -> tuple[float, float]:
        if not xyz:
            return 0.0, 0.0
        x0, y0, _ = xyz[0]
        lateral = np.asarray([x - x0 for x, _, _ in xyz], dtype=np.float64)
        forward = np.asarray([y - y0 for _, y, _ in xyz], dtype=np.float64)
        max_forward = float(np.max(forward)) if forward.size else 0.0
        max_lateral = float(np.max(np.abs(lateral))) if lateral.size else 0.0
        return max_forward, max_lateral

    mf1, ml1 = stats(chosen_xyz)
    mf2, ml2 = stats(rejected_xyz)
    max_forward = max(mf1, mf2)
    max_lateral = max(ml1, ml2)

    scale_h = (0.80 * height / max_forward) if max_forward > 1e-6 else 10.0
    scale_w = (0.45 * width / max_lateral) if max_lateral > 1e-6 else 10.0
    return max(1.0, min(scale_h, scale_w))


def _draw_polyline(
    image: Image.Image,
    uv: List[Tuple[float, float]],
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


def _render_pair_panel(
    title: str,
    chosen_text: str,
    rejected_text: str,
    panel_size: Tuple[int, int],
    extra_lines: Sequence[str] = (),
) -> Image.Image:
    panel_w, panel_h = panel_size
    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    x0, y = 14, 12
    draw.text((x0, y), title, fill=(0, 0, 0), font=font)
    y += 22

    for line in extra_lines:
        draw.text((x0, y), line, fill=(70, 70, 70), font=font)
        y += 14
    if extra_lines:
        y += 6

    # Chosen block
    draw.text((x0, y), "CHOSEN (positive)", fill=(0, 128, 0), font=font)
    y += 16
    wrapped_pos = textwrap.fill(chosen_text.strip().replace("\n", " "), width=80)
    draw.text((x0, y), wrapped_pos, fill=(0, 0, 0), font=font)
    y += 12 * (wrapped_pos.count("\n") + 1) + 18

    # Rejected block
    draw.text((x0, y), "REJECTED (negative)", fill=(200, 0, 0), font=font)
    y += 16
    wrapped_neg = textwrap.fill(rejected_text.strip().replace("\n", " "), width=80)
    draw.text((x0, y), wrapped_neg, fill=(0, 0, 0), font=font)

    return panel


def _draw_legend(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0 = 10, 10
    pad = 6

    # Semi-opaque background (approx using solid rectangle)
    lines = ["CHOSEN: green", "REJECTED: red"]
    max_w = max(draw.textlength(line, font=font) for line in lines)
    box_w = int(max_w) + pad * 2
    box_h = (len(lines) * 14) + pad * 2

    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(255, 255, 255))
    y = y0 + pad
    draw.text((x0 + pad, y), lines[0], fill=(0, 128, 0), font=font)
    y += 14
    draw.text((x0 + pad, y), lines[1], fill=(200, 0, 0), font=font)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _load_pairs(
    jsonl_path: Path,
    media_root: Optional[Path],
) -> List[PairSample]:
    pairs: List[PairSample] = []

    for idx, obj in enumerate(_iter_jsonl(jsonl_path)):
        images = obj.get("images")
        if not isinstance(images, list) or not images:
            continue

        chosen = obj.get("chosen")
        rejected = obj.get("rejected")
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            continue

        chosen_text, chosen_xyz = _split_text_and_traj(chosen)
        rejected_text, rejected_xyz = _split_text_and_traj(rejected)

        image_path = _resolve_image_path(str(images[0]), media_root)
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}

        pairs.append(
            PairSample(
                index=idx,
                image_path=image_path,
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                chosen_xyz=chosen_xyz,
                rejected_xyz=rejected_xyz,
                meta=meta,
            )
        )

    return pairs


def _iter_sequences(split_dir: Path) -> Iterable[Path]:
    if not split_dir.exists():
        return
    for child in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            yield child


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _choose_points(points: List[Tuple[float, float, float]], num_points: int) -> List[Tuple[float, float, float]]:
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
    selected: List[Tuple[float, float, float]] = []
    for idx in indices:
        idx = max(0, min(last_idx, int(idx)))
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(points[idx])

    while len(selected) < num_points:
        selected.append(points[last_idx])
    return selected


def _load_frames(
    orad_root: Path,
    split: str,
    image_folder: str,
    trajectory_key: str,
    num_points: int,
) -> List[_Frame]:
    split_dir = orad_root / split
    frames: List[_Frame] = []

    for seq_dir in _iter_sequences(split_dir):
        image_dir = seq_dir / image_folder
        scene_dir = seq_dir / "scene_data"
        local_dir = seq_dir / "local_path"
        if not image_dir.exists() or not scene_dir.exists() or not local_dir.exists():
            continue

        for img_path in sorted(image_dir.iterdir(), key=lambda p: p.name):
            if not img_path.is_file():
                continue
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

            raw_points = local_obj.get(trajectory_key)
            if not isinstance(raw_points, list) or not raw_points:
                continue

            xyz: List[Tuple[float, float, float]] = []
            try:
                for p in raw_points:
                    if not isinstance(p, list) or len(p) < 3:
                        continue
                    xyz.append((float(p[0]), float(p[1]), float(p[2])))
            except Exception:
                continue

            xyz = _choose_points(xyz, int(num_points))
            if not xyz:
                continue

            frames.append(
                _Frame(
                    split=split,
                    sequence=seq_dir.name,
                    timestamp=ts,
                    image_path=img_path,
                    scene_text=scene_text,
                    xyz=xyz,
                )
            )

    return frames


def _pick_negative(
    anchor: _Frame,
    frames: List[_Frame],
    text_sim: _TextSimilarity,
    wrong_pool_size: int,
    min_text_similarity: Optional[float],
    max_traj_similarity: Optional[float],
    train_exclude_bottom_quantile: float,
    allow_relax: bool,
) -> tuple[_Frame, float, float] | None:
    if len(frames) < 2:
        return None

    # Candidate pool from other frames.
    pool = [f for f in frames if not (f.sequence == anchor.sequence and f.timestamp == anchor.timestamp)]
    if not pool:
        return None
    if wrong_pool_size > 0 and len(pool) > wrong_pool_size:
        pool = random.sample(pool, k=int(wrong_pool_size))

    anchor_xy = _trajectory_xy_normalized(anchor.xyz)
    sims_xy: List[float] = []
    for cand in pool:
        sims_xy.append(_trajectory_similarity_xy(anchor_xy, _trajectory_xy_normalized(cand.xyz)))

    # Training-only: exclude most-different bottom quantile by XY similarity.
    if anchor.split == "training" and train_exclude_bottom_quantile > 1e-9 and len(pool) >= 8:
        q = float(train_exclude_bottom_quantile)
        q = max(0.0, min(0.95, q))
        k = int(len(pool) * q)
        if k >= 1:
            order = np.argsort(np.asarray(sims_xy, dtype=np.float32))  # low sim first (most different)
            keep_idx = set(order[k:].tolist())
            pool = [p for i, p in enumerate(pool) if i in keep_idx]
            sims_xy = [s for i, s in enumerate(sims_xy) if i in keep_idx]

    best: tuple[float, float, _Frame] | None = None  # (text_sim, traj_sim, frame)
    best_relaxed: tuple[float, float, _Frame] | None = None
    for cand, traj_sim in zip(pool, sims_xy):
        ts = float(text_sim.similarity(anchor.scene_text, cand.scene_text))

        # Track a relaxed best candidate regardless of thresholds.
        if best_relaxed is None:
            best_relaxed = (ts, traj_sim, cand)
        else:
            if ts > best_relaxed[0] + 1e-9 or (abs(ts - best_relaxed[0]) <= 1e-9 and traj_sim < best_relaxed[1]):
                best_relaxed = (ts, traj_sim, cand)

        if min_text_similarity is not None and ts < float(min_text_similarity):
            continue
        if max_traj_similarity is not None and traj_sim > float(max_traj_similarity):
            continue

        if best is None:
            best = (ts, traj_sim, cand)
            continue
        if ts > best[0] + 1e-9 or (abs(ts - best[0]) <= 1e-9 and traj_sim < best[1]):
            best = (ts, traj_sim, cand)

    if best is None:
        if allow_relax and best_relaxed is not None:
            return best_relaxed[2], float(best_relaxed[0]), float(best_relaxed[1])
        return None
    return best[2], float(best[0]), float(best[1])


def _build_live_pair_samples(
    frames: List[_Frame],
    text_sim: _TextSimilarity,
    num_samples: int,
    wrong_pool_size: int,
    min_text_similarity: Optional[float],
    max_traj_similarity: Optional[float],
    train_exclude_bottom_quantile: float,
    allow_relax: bool,
) -> List[PairSample]:
    if not frames:
        return []

    anchors = random.sample(frames, k=min(int(num_samples), len(frames)))
    out: List[PairSample] = []

    for i, anchor in enumerate(anchors):
        picked = _pick_negative(
            anchor=anchor,
            frames=frames,
            text_sim=text_sim,
            wrong_pool_size=int(wrong_pool_size),
            min_text_similarity=min_text_similarity,
            max_traj_similarity=max_traj_similarity,
            train_exclude_bottom_quantile=float(train_exclude_bottom_quantile),
            allow_relax=bool(allow_relax),
        )
        if picked is None:
            continue
        neg, text_s, traj_s = picked

        meta = {
            "split": anchor.split,
            "sequence": anchor.sequence,
            "timestamp": anchor.timestamp,
            "negative_from": {"split": neg.split, "sequence": neg.sequence, "timestamp": neg.timestamp},
            "neg_text_sim": round(float(text_s), 6),
            "neg_traj_sim": round(float(traj_s), 6),
            "relaxed": bool(allow_relax)
            and (
                (min_text_similarity is not None and float(text_s) < float(min_text_similarity))
                or (max_traj_similarity is not None and float(traj_s) > float(max_traj_similarity))
            ),
        }
        out.append(
            PairSample(
                index=i,
                image_path=anchor.image_path,
                chosen_text=anchor.scene_text,
                rejected_text=neg.scene_text,
                chosen_xyz=anchor.xyz,
                rejected_xyz=neg.xyz,
                meta=meta,
            )
        )

    return out


def build_composite(
    sample: PairSample,
    panel_width: int,
    scale_px_per_meter: Optional[float],
    line_width: int,
) -> Image.Image:
    base = Image.open(sample.image_path).convert("RGB")
    w, h = base.size

    if scale_px_per_meter is None:
        scale = _auto_scale_for_both(sample.chosen_xyz, sample.rejected_xyz, (w, h))
    else:
        scale = float(scale_px_per_meter)

    uv_pos = _ego_xy_to_pixels(sample.chosen_xyz, (w, h), scale_px_per_meter=scale)
    uv_neg = _ego_xy_to_pixels(sample.rejected_xyz, (w, h), scale_px_per_meter=scale)

    overlay = base.copy()
    # Draw negative first, then positive on top for visibility.
    _draw_polyline(overlay, uv_neg, color=(200, 0, 0), width=line_width)
    _draw_polyline(overlay, uv_pos, color=(0, 180, 0), width=line_width)
    _draw_legend(overlay)

    title = sample.meta.get("sequence")
    if title:
        title = f"{title} / {sample.meta.get('timestamp', '')}".strip(" /")
    else:
        title = f"index={sample.index}"

    extra_lines: List[str] = []
    if isinstance(sample.meta, dict):
        if "neg_text_sim" in sample.meta:
            extra_lines.append(f"neg_text_sim={sample.meta.get('neg_text_sim')}")
        if "neg_traj_sim" in sample.meta:
            extra_lines.append(f"neg_traj_sim={sample.meta.get('neg_traj_sim')}")

    panel = _render_pair_panel(
        title=title,
        chosen_text=sample.chosen_text,
        rejected_text=sample.rejected_text,
        panel_size=(panel_width, h),
        extra_lines=extra_lines,
    )

    canvas = Image.new("RGB", (w + panel_width, h), (255, 255, 255))
    canvas.paste(overlay, (0, 0))
    canvas.paste(panel, (w, 0))
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ORPO chosen/rejected pairs on one image.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path, default=None, help="ORPO training JSONL with chosen/rejected")
    src.add_argument("--orad-root", type=Path, default=None, help="ORAD-3D root to sample pairs live")

    # Live sampling options
    p.add_argument("--split", default="training", choices=["training", "validation", "testing"], help="Split to scan")
    p.add_argument(
        "--image-folder",
        default="image_data",
        choices=["gt_image", "image_data"],
        help="Which image folder to use",
    )
    p.add_argument(
        "--trajectory-key",
        default="trajectory_ins",
        choices=["trajectory_ins", "trajectory_hmi", "trajectory_ins_past", "trajectory_hmi_past"],
    )
    p.add_argument(
        "--num-points",
        type=int,
        default=0,
        help="How many trajectory points to draw. 0 means use the full original trajectory.",
    )

    p.add_argument("--wrong-pool-size", type=int, default=256, help="Candidate pool size for negatives")
    p.add_argument("--min-text-similarity", type=float, default=None)
    p.add_argument("--max-traj-similarity", type=float, default=None)
    p.add_argument(
        "--train-exclude-bottom-quantile",
        type=float,
        default=0.25,
        help="Training split only: exclude most-different bottom quantile by XY similarity.",
    )

    p.add_argument("--text-sim-backend", choices=["bow", "sbert"], default="sbert")
    p.add_argument("--sbert-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--sbert-device", default="cpu")
    p.add_argument("--sbert-cache-size", type=int, default=4096)
    p.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help=(
            "Writable Hugging Face cache directory for downloading SBERT/transformers models. "
            "If not set, uses ~/.cache/huggingface when writable, otherwise falls back to ~/.hf_cache."
        ),
    )
    p.add_argument(
        "--require-sbert",
        action="store_true",
        help="Fail if sentence-transformers is not installed when using --text-sim-backend sbert.",
    )

    p.add_argument(
        "--allow-relax",
        action="store_true",
        help=(
            "If strict thresholds yield no negative, fall back to the best-scoring candidate "
            "(max text similarity, then min trajectory similarity)."
        ),
    )

    p.add_argument(
        "--media-root",
        type=Path,
        default=None,
        help="Base directory to resolve relative image paths (e.g., /data3/ORAD-3D).",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for PNGs")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--panel-width",
        type=int,
        default=820,
        help="Right-side panel width (pixels) for chosen/rejected text",
    )
    p.add_argument(
        "--scale-px-per-meter",
        type=float,
        default=None,
        help="Fixed scale for XY overlay. If omitted, auto-scales per sample.",
    )
    p.add_argument("--line-width", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    if args.jsonl is not None:
        pairs = _load_pairs(args.jsonl, args.media_root)
    else:
        assert args.orad_root is not None
        frames = _load_frames(
            orad_root=args.orad_root,
            split=str(args.split),
            image_folder=str(args.image_folder),
            trajectory_key=str(args.trajectory_key),
            num_points=int(args.num_points),
        )
        if not frames:
            raise SystemExit(
                "No frames found. Check split structure and folders: "
                f"{args.orad_root}/{args.split}/<seq>/(image_data|gt_image, scene_data, local_path)"
            )

        text_sim = _TextSimilarity(
            backend=str(args.text_sim_backend),
            sbert_model=str(args.sbert_model),
            sbert_device=str(args.sbert_device),
            cache_size=int(args.sbert_cache_size),
            require_sbert=bool(args.require_sbert),
            hf_cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
        )
        pairs = _build_live_pair_samples(
            frames=frames,
            text_sim=text_sim,
            num_samples=int(args.num_samples),
            wrong_pool_size=int(args.wrong_pool_size),
            min_text_similarity=args.min_text_similarity,
            max_traj_similarity=args.max_traj_similarity,
            train_exclude_bottom_quantile=float(args.train_exclude_bottom_quantile),
            allow_relax=bool(args.allow_relax),
        )

    if not pairs:
        if args.jsonl is not None:
            raise SystemExit(f"No valid ORPO pairs found in: {args.jsonl}")
        msg = (
            f"No pairs could be sampled from: {args.orad_root} (split={args.split}). "
            f"Loaded frames: {len(frames)}. "
        )
        if args.text_sim_backend == "bow" and args.min_text_similarity is not None:
            msg += "BoW similarity is usually much lower; try --min-text-similarity 0.05~0.15 or use --text-sim-backend sbert. "
        if args.allow_relax is False:
            msg += "You can also add --allow-relax to force a fallback negative for visualization."
        raise SystemExit(msg)

    k = min(int(args.num_samples), len(pairs))
    picked = random.sample(pairs, k=k)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, s in enumerate(picked, start=1):
        try:
            comp = build_composite(
                s,
                panel_width=int(args.panel_width),
                scale_px_per_meter=args.scale_px_per_meter,
                line_width=int(args.line_width),
            )
        except Exception:
            continue

        seq = s.meta.get("sequence", "")
        ts = s.meta.get("timestamp", "")
        stem = f"{i:04d}"
        if seq or ts:
            stem = f"{stem}_{seq}_{ts}".strip("_")
        out = args.out_dir / f"{stem}.png"
        comp.save(out)
        saved += 1

    print(f"[OK] wrote {saved} visualizations -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
