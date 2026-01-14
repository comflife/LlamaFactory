#!/usr/bin/env python3
"""Compare raw vs refined ORAD-3D scene text using xAI chat API (no trajectory tokens, XY only).

Per frame:
  - read image, scene_data/<ts>.txt, scene_data_refine/<ts>.txt (optional), local_path/<ts>.json
  - build prompt with scene text + past ego XY path (z ignored)
  - call xAI chat twice (raw vs refined)
  - parse `[x,y]` points from output (first two numbers only)
  - overlay GT (optional), raw, refine on one image; save manifest JSONL
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import APIConnectionError, APIError, OpenAI, RateLimitError, Timeout
from PIL import Image, ImageDraw, ImageFont

_POINT2D_RE = re.compile(
    r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?\s*\]"
)


# ---------- data helpers ----------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _iter_sequence_dirs(split_dir: Path) -> Iterable[Path]:
    for child in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and not child.name.endswith(".zip"):
            yield child


def _downsample_points(points: Sequence[Sequence[float]], max_points: Optional[int]) -> List[List[float]]:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return [list(map(float, p[:2])) for p in points if len(p) >= 2]
    last_idx = len(points) - 1
    if max_points == 1:
        return [list(map(float, points[0][:2]))]
    indices = [round(i * last_idx / (max_points - 1)) for i in range(max_points)]
    out: List[List[float]] = []
    seen = set()
    for idx in indices:
        idx = max(0, min(last_idx, int(idx)))
        if idx in seen:
            continue
        seen.add(idx)
        out.append(list(map(float, points[idx][:2])))
    while len(out) < max_points:
        out.append(list(map(float, points[last_idx][:2])))
    return out


def _load_paths_from_local(path: Path, past_key: str, future_key: str, max_past_points: Optional[int]) -> Tuple[List[List[float]], Optional[List[List[float]]]]:
    obj = _read_json(path)
    def _norm(seq: Any) -> List[List[float]]:
        pts: List[List[float]] = []
        if isinstance(seq, list):
            for p in seq:
                if isinstance(p, list) and len(p) >= 2:
                    try:
                        pts.append([float(p[0]), float(p[1])])
                    except Exception:
                        continue
        return pts
    past = _downsample_points(_norm(obj.get(past_key, [])), max_past_points)
    future = _norm(obj.get(future_key, []))
    return past, (future if future else None)


def _iter_samples(orad_root: Path, split: str, image_folder: str, scene_folder: str, refine_folder: str, require_refine: bool) -> Iterable[Tuple[str, str, Path, Path, Optional[Path], Path]]:
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
            ref_path = refine_dir / f"{ts}.txt"
            local_path = local_dir / f"{ts}.json"
            if not scene_path.exists() or not local_path.exists():
                continue
            if require_refine and not ref_path.exists():
                continue
            yield (seq_dir.name, ts, img_path, scene_path, ref_path if ref_path.exists() else None, local_path)


# ---------- prompt / API ----------

def _encode_image_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _format_points_for_prompt(points: Sequence[Sequence[float]], max_points: Optional[int], decimals: int = 2) -> str:
    pts = _downsample_points(points, max_points=max_points)
    return ",".join(f"[{p[0]:.{decimals}f},{p[1]:.{decimals}f}]" for p in pts if len(p) >= 2)


def _build_prompt(scene_text: str, past_points: List[List[float]], target_points: int, min_points: int, forward_axis: str, flip_lateral: bool) -> str:
    past_str = _format_points_for_prompt(past_points, max_points=None)
    lat_axis = "y" if forward_axis == "x" else "x"
    lat_dir = "left" if flip_lateral else "right"
    coord_hint = (
        f"Coordinate convention: forward=+{forward_axis}; lateral={lat_axis} (positive {lat_axis} is {lat_dir}); z is ignored."
    )
    extra = (
        f"Predict at least {min_points} future points (no single-point outputs). "
        f"Target around {target_points} points. "
        "Output format: [x0,y0],[x1,y1],...[xN,yN] in meters, comma-separated, no extra text. "
        "Example (do not copy numbers): [0.0,0.0],[1.5,0.2],[3.0,0.6],[4.5,0.9]. "
        "Follow the past heading smoothly; if past curves left, keep a gentle left arc; if straight, extend forward with small lateral drift."
    )
    return (
        "You are an off-road driving annotator. "
        f"{coord_hint} "
        f"Scene description: {scene_text} "
        f"Past ego path (oldest->latest, meters, [x,y]): {past_str} "
        f"{extra} "
        "Return ONLY the coordinates list."
    )


def _call_xai_chat(client: OpenAI, model: str, system_text: str, user_text: str, image_path: Path, max_tokens: int, temperature: float, retries: int, retry_sleep: float) -> str:
    data_url = _encode_image_data_url(image_path)
    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = resp.choices[0].message.content
            return choice or ""
        except (RateLimitError, APIError, Timeout, APIConnectionError) as exc:
            last_err = exc
            if attempt >= retries:
                break
            import time
            time.sleep(retry_sleep * (2**attempt))
    raise RuntimeError(f"xAI API failed: {last_err}")


def _extract_points_xy(text: str) -> List[List[float]]:
    pts: List[List[float]] = []
    for m in _POINT2D_RE.finditer(text or ""):
        try:
            pts.append([float(m.group(1)), float(m.group(2))])
        except Exception:
            continue
    return pts


# ---------- visualization ----------

def _traj_xy_to_pixels(points_xy: List[List[float]], image_size: Tuple[int, int], scale_px_per_m: Optional[float] = None) -> Tuple[List[Tuple[float, float]], float]:
    if not points_xy or len(points_xy) < 2:
        return [], 0.0
    width, height = image_size
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    x0, y0 = xs[0], ys[0]
    dx = [x - x0 for x in xs]
    dy = [y - y0 for y in ys]
    if scale_px_per_m is None:
        max_fwd = max(dy) if dy else 0.0
        max_lat = max(abs(v) for v in dx) if dx else 0.0
        scale_h = (0.80 * height / max_fwd) if max_fwd > 1e-6 else 10.0
        scale_w = (0.45 * width / max_lat) if max_lat > 1e-6 else scale_h
        scale_px_per_m = max(1.0, min(scale_h, scale_w))
    uvs: List[Tuple[float, float]] = []
    for lat, fwd in zip(dx, dy):
        u = (width / 2.0) + (lat * scale_px_per_m)
        v = (height - 1.0) - (fwd * scale_px_per_m)
        uvs.append((float(u), float(v)))
    return uvs, float(scale_px_per_m)


def _draw_polyline(image: Image.Image, uv: List[Tuple[float, float]], color: Tuple[int, int, int], width: int = 3) -> None:
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


def _save_overlay(image: Image.Image, header: str, gt_xy: Optional[List[List[float]]], raw_xy: List[List[float]], ref_xy: List[List[float]], out_path: Path) -> None:
    base = image.convert("RGB")
    overlay = base.copy()

    scale_ref = gt_xy if gt_xy else (raw_xy if raw_xy else ref_xy)
    scale = None
    if scale_ref:
        _, scale = _traj_xy_to_pixels(scale_ref, base.size, scale_px_per_m=None)

    def uv(points: List[List[float]]) -> List[Tuple[float, float]]:
        return _traj_xy_to_pixels(points, base.size, scale_px_per_m=scale)[0] if points else []

    _draw_polyline(overlay, uv(gt_xy or []), color=(0, 200, 0))
    _draw_polyline(overlay, uv(raw_xy), color=(220, 20, 60))
    _draw_polyline(overlay, uv(ref_xy), color=(30, 144, 255))

    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    legend = [
        f"{header}",
        f"GT   (green): {len(gt_xy or [])} pts",
        f"Raw  (red):   {len(raw_xy)} pts",
        f"Ref  (blue):  {len(ref_xy)} pts",
    ]
    y = 10
    for line in legend:
        draw.text((12, y), line, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        y += 14

    overlay.save(out_path)


# ---------- main ----------


@dataclass
class RunOutcome:
    prompt: str
    output_text: str
    points_xy: List[List[float]]
    overlay_path: Optional[str]
    mean_l2_gt: Optional[float]
    final_l2_gt: Optional[float]


@dataclass
class PairResult:
    key: str
    split: str
    sequence: str
    timestamp: str
    image_path: str
    scene_text: str
    scene_text_refine: Optional[str]
    past_xy: List[List[float]]
    gt_xy: Optional[List[List[float]]]
    raw: RunOutcome
    refine: RunOutcome


def _mean_l2_xy(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Optional[float]:
    n = min(len(a), len(b))
    if n == 0:
        return None
    total = 0.0
    for i in range(n):
        dx = float(a[i][0]) - float(b[i][0])
        dy = float(a[i][1]) - float(b[i][1])
        total += math.sqrt(dx * dx + dy * dy)
    return total / n


def _final_l2_xy(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Optional[float]:
    if not a or not b:
        return None
    pa = a[-1]
    pb = b[-1]
    if len(pa) < 2 or len(pb) < 2:
        return None
    dx = float(pa[0]) - float(pb[0])
    dy = float(pa[1]) - float(pb[1])
    return math.sqrt(dx * dx + dy * dy)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare raw vs refined ORAD-3D scene text using xAI API (XY only).")
    ap.add_argument("--orad-root", type=Path, default=Path("/home/work/datasets/bg/ORAD-3D"))
    ap.add_argument("--split", type=str, default="validation", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])
    ap.add_argument("--scene-folder", type=str, default="scene_data")
    ap.add_argument("--refine-folder", type=str, default="scene_data_refine")
    ap.add_argument("--past-key", type=str, default="trajectory_ins_past")
    ap.add_argument("--future-key", type=str, default="trajectory_ins")
    ap.add_argument("--max-past-points", type=int, default=12)
    ap.add_argument("--target-points", type=int, default=12)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--require-refine", action="store_true")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--max-scan", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true")
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--base-url", type=str, default="https://api.x.ai/v1")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--api-key-env", type=str, default="XAI_API_KEY")
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--env-override", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-sleep", type=float, default=1.0)
    return ap.parse_args()


def _load_env_file(env_path: Path, override: bool = False) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("\"'")
        loaded[k] = v
        if override or k not in os.environ:
            os.environ[k] = v
    return loaded


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    env_path = args.env_file or (Path.cwd() / ".env")
    _load_env_file(env_path, override=args.env_override)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing {args.api_key_env} env var")
    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=120)

    samples = list(
        _iter_samples(
            orad_root=args.orad_root,
            split=args.split,
            image_folder=args.image_folder,
            scene_folder=args.scene_folder,
            refine_folder=args.refine_folder,
            require_refine=bool(args.require_refine),
        )
    )
    if args.max_scan is not None:
        samples = samples[: args.max_scan]
    if len(samples) > args.num_samples:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, k=int(args.num_samples))

    results: List[PairResult] = []
    saved = 0
    for seq, ts, img_path, scene_path, ref_path, local_path in samples:
        image = Image.open(img_path).convert("RGB")
        scene_text = _read_text(scene_path)
        scene_text_refine = _read_text(ref_path) if ref_path else None
        past_xy, gt_xy = _load_paths_from_local(
            local_path,
            past_key=args.past_key,
            future_key=args.future_key,
            max_past_points=args.max_past_points,
        )
        if not past_xy:
            print(f"[SKIP] {seq}/{ts}: missing past path")
            continue

        prompt_raw = _build_prompt(
            scene_text=scene_text,
            past_points=past_xy,
            target_points=int(args.target_points),
            min_points=int(args.min_points),
            forward_axis=args.forward_axis,
            flip_lateral=bool(args.flip_lateral),
        )
        prompt_ref = _build_prompt(
            scene_text=scene_text_refine or scene_text,
            past_points=past_xy,
            target_points=int(args.target_points),
            min_points=int(args.min_points),
            forward_axis=args.forward_axis,
            flip_lateral=bool(args.flip_lateral),
        )

        system = "You are an off-road autonomous driving assistant. Be concise and output only XY points."

        out_raw_text = _call_xai_chat(
            client=client,
            model=args.model,
            system_text=system,
            user_text=prompt_raw,
            image_path=img_path,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            retries=int(args.retries),
            retry_sleep=float(args.retry_sleep),
        )
        out_ref_text = _call_xai_chat(
            client=client,
            model=args.model,
            system_text=system,
            user_text=prompt_ref,
            image_path=img_path,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            retries=int(args.retries),
            retry_sleep=float(args.retry_sleep),
        )

        raw_xy = _extract_points_xy(out_raw_text)
        ref_xy = _extract_points_xy(out_ref_text)
        if len(raw_xy) < args.min_points or len(ref_xy) < args.min_points:
            print(f"[SKIP] {seq}/{ts}: too few points raw={len(raw_xy)} refine={len(ref_xy)} (min={args.min_points})")
            continue

        header = f"{args.split}_{seq}_{ts}"
        overlay_path = None
        if bool(args.save_overlays):
            overlay_path = args.out_dir / f"{saved:03d}_{header}.png"
            _save_overlay(
                image=image,
                header=header,
                gt_xy=gt_xy,
                raw_xy=raw_xy,
                ref_xy=ref_xy,
                out_path=overlay_path,
            )

        raw_mean = _mean_l2_xy(raw_xy, gt_xy) if gt_xy else None
        ref_mean = _mean_l2_xy(ref_xy, gt_xy) if gt_xy else None
        raw_final = _final_l2_xy(raw_xy, gt_xy) if gt_xy else None
        ref_final = _final_l2_xy(ref_xy, gt_xy) if gt_xy else None

        pair = PairResult(
            key=header,
            split=args.split,
            sequence=seq,
            timestamp=ts,
            image_path=str(img_path),
            scene_text=scene_text,
            scene_text_refine=scene_text_refine,
            past_xy=past_xy,
            gt_xy=gt_xy,
            raw=RunOutcome(
                prompt=prompt_raw,
                output_text=out_raw_text,
                points_xy=raw_xy,
                overlay_path=str(overlay_path) if overlay_path else None,
                mean_l2_gt=raw_mean,
                final_l2_gt=raw_final,
            ),
            refine=RunOutcome(
                prompt=prompt_ref,
                output_text=out_ref_text,
                points_xy=ref_xy,
                overlay_path=str(overlay_path) if overlay_path else None,
                mean_l2_gt=ref_mean,
                final_l2_gt=ref_final,
            ),
        )
        results.append(pair)
        saved += 1
        print(f"[OK] {header}: raw={len(raw_xy)} ref={len(ref_xy)} gt={len(gt_xy) if gt_xy else 0}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {len(results)} pairs -> {manifest_path}")
    if args.save_overlays:
        print(f"Overlays in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
