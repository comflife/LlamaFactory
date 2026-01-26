#!/usr/bin/env python3
"""
Compare multiple VLM models (local HF + OpenRouter) on ORAD-3D ego trajectory prediction with few-shot examples.

✅ 현재 버전 (요청 반영):
- 오버레이(이미지 위 projection) 완전 제거
- PNG composite에는 "카메라 이미지 + 그래프"만 포함:
    - 상단 좌측: 원본 카메라 이미지
    - 상단 우측: 모델별 XY plot 패널 (GT vs pred)
    - 하단 전체: Z trend 패널 (GT + all models)

Output:
- manifest.jsonl
- (옵션) PNG composites (image + plots) when --save-overlays

python scripts/orad3d_vlm_trajectory_fewshot.py   --model-spec openrouter:google/gemini-3-flash-preview   --model-spec openrouter:qwen/qwen3-vl-8b-instruct   --orad-root /home/work/datasets/bg/ORAD-3D   --split testing --num-samples 1 --num-examples 3   --out-dir /home/work/byounggun/LlamaFactory/orad3d_vlm_trajectory_compare   --use-sharegpt-format --save-overlays
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
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

try:
    import openai
except ImportError:
    openai = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class TrainExample:
    sequence: str
    timestamp: str
    image_path: Path
    past_points: List[List[float]]
    future_points: List[List[float]]


@dataclass(frozen=True)
class TestSample:
    key: str
    sequence: str
    timestamp: str
    image_path: Path
    past_points: List[List[float]]
    gt_points: List[List[float]]
    target_dist: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    model_type: str  # 'local' or 'openrouter'
    color: Tuple[int, int, int]


@dataclass
class ModelOutput:
    name: str
    output_text: str
    trajectory_points: List[List[float]]
    valid: bool


@dataclass
class SampleResult:
    key: str
    image_path: str
    past_points: List[List[float]]
    gt_points: List[List[float]]
    target_dist: float
    outputs: List[ModelOutput]
    composite_path: str


# -----------------------------
# Regex + colors
# -----------------------------
_POINT_RE = re.compile(
    r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]"
)

_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (220, 20, 60),    # crimson
    (30, 144, 255),   # dodger blue
    (255, 140, 0),    # dark orange
    (138, 43, 226),   # blue violet
    (0, 206, 209),    # dark turquoise
    (160, 82, 45),    # sienna
    (128, 0, 0),      # maroon
    (255, 20, 147),   # deep pink
]

# XY plot assumes:
# forward +y, lateral +x right (coord_hint default)
_FORWARD_AXIS = "y"
_FLIP_LATERAL = False


# -----------------------------
# Utilities
# -----------------------------
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ and value:
            os.environ[key] = value


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _load_paths(path: Path, past_key: str, future_key: str) -> Tuple[List[List[float]], List[List[float]]]:
    obj = _read_json(path)
    past_raw = obj.get(past_key, [])
    future_raw = obj.get(future_key, [])

    def normalize(seq: Any) -> List[List[float]]:
        pts: List[List[float]] = []
        if isinstance(seq, list):
            for p in seq:
                if isinstance(p, list) and len(p) >= 3:
                    try:
                        pts.append([float(p[0]), float(p[1]), float(p[2])])
                    except Exception:
                        pass
                elif isinstance(p, list) and len(p) == 2:
                    try:
                        pts.append([float(p[0]), float(p[1]), 0.0])
                    except Exception:
                        pass
        return pts

    past = normalize(past_raw)[-12:]
    future = normalize(future_raw)
    return past, future


def _compute_target_dist(gt_points: List[List[float]]) -> float:
    if not gt_points:
        return 10.0
    last = gt_points[-1]
    return math.sqrt(sum(float(x) ** 2 for x in last[:3]))


def _downsample_points(points: List[List[float]], max_n: int) -> List[List[float]]:
    if len(points) <= max_n:
        return points
    if max_n <= 1:
        return [points[-1]]
    indices = [round(i * (len(points) - 1) / (max_n - 1)) for i in range(max_n)]
    return [points[idx] for idx in indices]


def _format_points_str(points: List[List[float]], max_n: int = 8, decimals: int = 2) -> str:
    pts = _downsample_points(points, max_n)
    return " ".join(f"[{p[0]:.{decimals}f},{p[1]:.{decimals}f},{p[2]:.{decimals}f}]" for p in pts)


def _extract_traj_xyz(text: str) -> List[List[float]]:
    pts: List[List[float]] = []
    for m in _POINT_RE.finditer(text or ""):
        try:
            pts.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        except Exception:
            pass
    return pts


# -----------------------------
# ORAD sampling
# -----------------------------
def _iter_orad_samples(
    orad_root: Path,
    split: str,
    image_folder: str,
    past_key: str,
    future_key: str,
    max_scan: Optional[int] = None,
) -> Iterable[TestSample]:
    split_dir = orad_root / split
    count = 0
    for seq_dir in sorted(split_dir.glob("*/")):
        if not seq_dir.is_dir():
            continue
        img_dir = seq_dir / image_folder
        local_dir = seq_dir / "local_path"
        if not img_dir.is_dir() or not local_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            ts = img_path.stem
            local_json = local_dir / f"{ts}.json"
            if not local_json.exists():
                continue
            past_pts, gt_pts = _load_paths(local_json, past_key, future_key)
            if len(past_pts) < 2 or len(gt_pts) < 2:
                continue
            target_dist = _compute_target_dist(gt_pts)
            key = f"{split}_{seq_dir.name}_{ts}"
            yield TestSample(
                key=key,
                sequence=seq_dir.name,
                timestamp=ts,
                image_path=img_path,
                past_points=past_pts,
                gt_points=gt_pts,
                target_dist=target_dist,
            )
            count += 1
            if max_scan and count >= max_scan:
                return


def _select_train_examples(
    orad_root: Path,
    num: int,
    seed: int,
    image_folder: str,
    past_key: str,
    future_key: str,
    max_scan: int,
) -> List[TrainExample]:
    train_samples = list(_iter_orad_samples(orad_root, "training", image_folder, past_key, future_key, max_scan=max_scan))
    if not train_samples:
        return []
    rng = random.Random(seed)
    selected = rng.sample(train_samples, min(num, len(train_samples)))
    examples: List[TrainExample] = []
    for s in selected:
        past = _downsample_points(s.past_points, 12)
        examples.append(TrainExample(s.sequence, s.timestamp, s.image_path, past, s.gt_points))
    return examples


# -----------------------------
# Prompt building
# -----------------------------
def _build_fewshot_prompt(
    examples: List[TrainExample],
    test_sample: TestSample,
    system: str,
    coord_hint: str,
    min_points: int,
    use_sharegpt: bool,
) -> Tuple[str, List[Image.Image]]:
    lines: List[str] = []
    images: List[Image.Image] = []
    if system:
        lines.append(system.strip())
    lines.append("You will see examples of past ego trajectory and the correct future trajectory.")
    lines.append("Use the examples to predict the future trajectory for the final image.")
    lines.append("")
    for i, ex in enumerate(examples, 1):
        if use_sharegpt:
            lines.append("<image>")
        lines.append(f"Example {i}")
        lines.append("Past ego trajectory (meters, [x,y,z], oldest to newest):")
        lines.append(_format_points_str(ex.past_points))
        lines.append("Future ego trajectory (meters, [x,y,z], oldest to newest):")
        lines.append(_format_points_str(ex.future_points))
        lines.append("")
        images.append(Image.open(ex.image_path).convert("RGB"))

    if use_sharegpt:
        lines.append("<image>")
    lines.append("Now predict the future trajectory for the next image.")
    lines.append("Past ego trajectory (meters, [x,y,z], oldest to newest):")
    lines.append(_format_points_str(test_sample.past_points))
    lines.append(f"Target future length approx {test_sample.target_dist:.1f} meters (distance to last point).")
    lines.append(f"Predict at least {min_points} smooth drivable points ahead.")
    lines.append(coord_hint)
    lines.append("Output ONLY [x,y,z] points separated by spaces. No other text.")
    images.append(Image.open(test_sample.image_path).convert("RGB"))

    return "\n".join(lines), images


def _encode_image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _build_openrouter_messages(
    examples: List[TrainExample],
    test_sample: TestSample,
    system: str,
    coord_hint: str,
    min_points: int,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples, 1):
        content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(ex.image_path)}})
        content.append(
            {
                "type": "text",
                "text": (
                    f"Example {i}\n"
                    "Past ego trajectory (meters, [x,y,z], oldest to newest):\n"
                    f"{_format_points_str(ex.past_points)}\n"
                    "Future ego trajectory (meters, [x,y,z], oldest to newest):\n"
                    f"{_format_points_str(ex.future_points)}"
                ),
            }
        )
    content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(test_sample.image_path)}})
    content.append(
        {
            "type": "text",
            "text": (
                "Now predict the future trajectory for this image.\n"
                "Past ego trajectory (meters, [x,y,z], oldest to newest):\n"
                f"{_format_points_str(test_sample.past_points)}\n"
                f"Target future length approx {test_sample.target_dist:.1f} meters (distance to last point).\n"
                f"Predict at least {min_points} smooth drivable points ahead.\n"
                f"{coord_hint}\n"
                "Output ONLY [x,y,z] points separated by spaces. No other text."
            ),
        }
    )
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def _build_local_messages(
    examples: List[TrainExample],
    test_sample: TestSample,
    system: str,
    coord_hint: str,
    min_points: int,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples, 1):
        content.append({"type": "image"})
        content.append(
            {
                "type": "text",
                "text": (
                    f"Example {i}\n"
                    "Past ego trajectory (meters, [x,y,z], oldest to newest):\n"
                    f"{_format_points_str(ex.past_points)}\n"
                    "Future ego trajectory (meters, [x,y,z], oldest to newest):\n"
                    f"{_format_points_str(ex.future_points)}"
                ),
            }
        )
    content.append({"type": "image"})
    content.append(
        {
            "type": "text",
            "text": (
                "Now predict the future trajectory for this image.\n"
                "Past ego trajectory (meters, [x,y,z], oldest to newest):\n"
                f"{_format_points_str(test_sample.past_points)}\n"
                f"Target future length approx {test_sample.target_dist:.1f} meters (distance to last point).\n"
                f"Predict at least {min_points} smooth drivable points ahead.\n"
                f"{coord_hint}\n"
                "Output ONLY [x,y,z] points separated by spaces. No other text."
            ),
        }
    )
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


# -----------------------------
# Inference
# -----------------------------
def _load_local_model(
    model_id: str,
    cache_dir: str,
    dtype: str,
    device_map: str,
    trust_remote: bool,
) -> Tuple[torch.nn.Module, Any]:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote, cache_dir=cache_dir)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer
    _ = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote, cache_dir=cache_dir)

    torch_dtype = {
        "auto": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype.lower()]

    for cls in (AutoModelForCausalLM, AutoModelForVision2Seq, AutoModelForImageTextToText):
        try:
            model = cls.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote,
                cache_dir=cache_dir,
            )
            model.eval()
            return model, processor
        except Exception:
            continue
    raise RuntimeError(f"Failed to load {model_id}")


def _infer_local(
    model: torch.nn.Module,
    processor: Any,
    images: List[Image.Image],
    prompt_text: str,
    max_new: int,
    temp: float,
    top_p: float,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    prompt = prompt_text
    if messages:
        if hasattr(processor, "apply_chat_template"):
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "apply_chat_template"):
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new}
    if temp > 0:
        gen_kwargs.update(do_sample=True, temperature=temp, top_p=top_p)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    if prompt and prompt in decoded:
        return decoded.split(prompt, 1)[-1].strip()
    return decoded.strip()


def _infer_openrouter(model_id: str, api_key: str, messages: List[Dict[str, Any]], max_tokens: int, temp: float) -> str:
    if openai is None:
        raise ImportError("pip install openai")
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# Plot-only visualization (with camera image)
# -----------------------------
def _points_forward_lateral(
    points_xyz: List[List[float]],
    *,
    forward_axis: str,
    flip_lateral: bool,
) -> Tuple[List[float], List[float]]:
    if len(points_xyz) < 2:
        return [], []
    xs = [float(p[0]) for p in points_xyz]
    ys = [float(p[1]) for p in points_xyz]
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
    max_lat = max(
        max((abs(v) for v in lat_gt), default=0.0),
        max((abs(v) for v in lat_pr), default=0.0),
        1e-3,
    )

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
) -> None:
    x0, y0, x1, y1 = box
    font = ImageFont.load_default()
    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)

    if len(gt_points) < 2 and len(pred_points) < 2:
        draw.text((x0 + 8, y0 + 8), "(no trajectory)", fill=(0, 0, 0), font=font)
        return

    scale = _shared_plot_scale(
        box,
        gt_points,
        pred_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )
    if scale is None:
        draw.text((x0 + 8, y0 + 8), "(no trajectory)", fill=(0, 0, 0), font=font)
        return

    margin = 10
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    cx = x0 + w / 2.0
    bottom = y0 + h - margin
    draw.line([(cx, y0 + margin), (cx, y0 + h - margin)], fill=(220, 220, 220), width=1)
    draw.line([(x0 + margin, bottom), (x0 + w - margin, bottom)], fill=(220, 220, 220), width=1)
    draw.text((x1 - 18, y0 + 4), "m", fill=(110, 110, 110), font=font)

    # GT (green)
    if len(gt_points) >= 2:
        fwd_gt, lat_gt = _points_forward_lateral(gt_points, forward_axis=forward_axis, flip_lateral=flip_lateral)
        pts_gt = _plot_points_in_box(fwd_gt, lat_gt, box=box, scale=scale)
        for a, b in zip(pts_gt[:-1], pts_gt[1:]):
            draw.line([a, b], fill=(0, 160, 0), width=3)
        for p in pts_gt:
            r = 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(0, 160, 0))

    # Pred
    if len(pred_points) >= 2:
        fwd_pr, lat_pr = _points_forward_lateral(pred_points, forward_axis=forward_axis, flip_lateral=flip_lateral)
        pts_pr = _plot_points_in_box(fwd_pr, lat_pr, box=box, scale=scale)
        for a, b in zip(pts_pr[:-1], pts_pr[1:]):
            draw.line([a, b], fill=pred_color, width=3)
        for p in pts_pr:
            r = 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] - r], fill=pred_color)  # tiny stylized dot


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
    g = gt_points[n - 1]
    p = pred_points[n - 1]
    dx = float(g[0]) - float(p[0])
    dy = float(g[1]) - float(p[1])
    dz = float(g[2]) - float(p[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _render_model_panel(
    *,
    width: int,
    header: str,
    gt_points: List[List[float]],
    model_outputs: Sequence[Tuple[ModelOutput, Tuple[int, int, int]]],
    forward_axis: str,
    flip_lateral: bool,
) -> Image.Image:
    font = ImageFont.load_default()
    pad = 12
    header_h = 30
    gap = 12

    rows = max(1, len(model_outputs))
    row_h = 220 if rows <= 4 else 180
    height = header_h + pad + rows * row_h + (rows - 1) * gap + pad

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, 0), (width, header_h)], fill=(245, 245, 245))
    draw.text((pad, 8), header, fill=(0, 0, 0), font=font)

    if not model_outputs:
        draw.text((pad, header_h + 10), "(no models)", fill=(0, 0, 0), font=font)
        return img

    y = header_h + pad
    for out, color in model_outputs:
        label_h = 16

        metrics: List[str] = []
        if out.valid and len(out.trajectory_points) >= 2:
            m = _mean_l2(gt_points, out.trajectory_points)
            f = _final_l2(gt_points, out.trajectory_points)
            if m is not None:
                metrics.append(f"mean_L2={m:.3f}")
            if f is not None:
                metrics.append(f"final_L2={f:.3f}")
        label = f"{out.name} ({', '.join(metrics)})" if metrics else out.name
        draw.text((pad, y), label, fill=color, font=font)

        snippet_y = y + label_h
        if out.output_text:
            max_chars = max(40, (width - 2 * pad) // 6)
            snippet = " ".join((out.output_text or "").strip().split())
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3] + "..."
            draw.text((pad, snippet_y), snippet, fill=(80, 80, 80), font=font)
            plot_top = snippet_y + 14
        else:
            plot_top = snippet_y + 2

        plot_box = (pad, plot_top, width - pad, y + row_h)
        pred_pts = out.trajectory_points if (out.valid and len(out.trajectory_points) >= 2) else []
        _draw_traj_compare_plot(
            draw=draw,
            box=plot_box,
            gt_points=gt_points,
            pred_points=pred_pts,
            pred_color=color,
            forward_axis=forward_axis,
            flip_lateral=flip_lateral,
        )

        y += row_h + gap

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

    x0, y0, x1, y1 = (pad_left, pad_top, w - pad_right, h - pad_bottom)
    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)
    draw.text((12, 4), "Z trend (relative)", fill=(0, 0, 0), font=font)
    draw.text((12, 16), "z - z0", fill=(110, 110, 110), font=font)

    def z_series(points: List[List[float]]) -> List[float]:
        if not points or len(points) < 2:
            return []
        z0 = float(points[0][2]) if len(points[0]) > 2 else 0.0
        return [(float(p[2]) if len(p) > 2 else 0.0) - z0 for p in points]

    sdata: List[Tuple[str, List[float], Tuple[int, int, int]]] = [(n, z_series(pts), c) for n, pts, c in series]
    allz: List[float] = []
    for _, zs, _ in sdata:
        allz.extend(zs)

    zmin, zmax = (min(allz), max(allz)) if allz else (-1.0, 1.0)
    if abs(zmax - zmin) < 1e-6:
        zmin -= 1.0
        zmax += 1.0

    usable_w = max(1, x1 - x0)
    usable_h = max(1, y1 - y0)

    def to_xy(zs: List[float]) -> List[Tuple[float, float]]:
        if not zs:
            return []
        n = len(zs)
        xs = [x0 + (i * usable_w / max(1, n - 1)) for i in range(n)]
        ys = [y1 - ((z - zmin) / (zmax - zmin)) * usable_h for z in zs]
        return list(zip(xs, ys))

    if zmin <= 0.0 <= zmax:
        y_zero = y1 - ((0.0 - zmin) / (zmax - zmin)) * usable_h
        draw.line([(x0, y_zero), (x1, y_zero)], fill=(220, 220, 220), width=1)

    for name, zs, color in sdata:
        pts = to_xy(zs)
        if len(pts) < 2:
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            draw.line([a, b], fill=color, width=2)
        for p in pts:
            r = 2
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)

    lx, ly = x0 + 6, y0 + 6
    for name, _, color in sdata:
        draw.text((lx, ly), name, fill=color, font=font)
        ly += 12

    return img


def _make_image_plus_plots_composite(
    *,
    camera_image: Image.Image,
    sample: TestSample,
    outputs: List[ModelOutput],
    specs: List[ModelSpec],
    panel_width: int,
) -> Image.Image:
    # right panel (XY per model)
    model_outputs_for_panel: List[Tuple[ModelOutput, Tuple[int, int, int]]] = []
    for out, spec in zip(outputs, specs):
        model_outputs_for_panel.append((out, spec.color))

    panel = _render_model_panel(
        width=panel_width,
        header=f"{sample.key} | XY plots (GT green)",
        gt_points=sample.gt_points,
        model_outputs=model_outputs_for_panel,
        forward_axis=_FORWARD_AXIS,
        flip_lateral=_FLIP_LATERAL,
    )

    cam = camera_image.convert("RGB")
    top_w = cam.width + panel.width
    top_h = max(cam.height, panel.height)

    top = Image.new("RGB", (top_w, top_h), (255, 255, 255))
    top.paste(cam, (0, 0))
    top.paste(panel, (cam.width, 0))

    # bottom Z trend spans full width
    z_panel_h = max(220, top_h // 4)
    z_series: List[Tuple[str, List[List[float]], Tuple[int, int, int]]] = [("GT", sample.gt_points, (0, 160, 0))]
    for out, spec in zip(outputs, specs):
        if out.valid and len(out.trajectory_points) >= 2:
            z_series.append((spec.name, out.trajectory_points, spec.color))
    z_panel = _render_z_trend_panel(size=(top.width, z_panel_h), series=z_series)

    canvas = Image.new("RGB", (top.width, top.height + z_panel.height), (255, 255, 255))
    canvas.paste(top, (0, 0))
    canvas.paste(z_panel, (0, top.height))
    return canvas


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Compare VLM models on ORAD-3D trajectory few-shot (IMAGE + PLOTS).")
    ap.add_argument("--model-spec", action="append", required=True, help="local:ID or openrouter:ID")
    ap.add_argument("--openrouter-api-key", type=str, default=None)
    ap.add_argument("--cache-dir", type=str, default="/home/work/byounggun/.cache/hf")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--system", type=str, default="You are an off-road driving agent. Predict safe future ego trajectory [x,y,z] from image and past path.")
    ap.add_argument("--coord-hint", type=str, default="Coordinate system: forward +y, lateral +x right, up +z.")
    ap.add_argument("--orad-root", type=Path, default=Path("/home/work/datasets/bg/ORAD-3D"))
    ap.add_argument("--split", type=str, default="testing", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data")
    ap.add_argument("--past-key", type=str, default="trajectory_ins_past")
    ap.add_argument("--future-key", type=str, default="trajectory_ins")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--num-examples", type=int, default=3)
    ap.add_argument("--max-train-scan", type=int, default=200)
    ap.add_argument("--max-scan", type=int, default=500)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--use-sharegpt-format", action="store_true")
    ap.add_argument("--save-overlays", action="store_true", help="Save PNG composites (camera image + plots).")
    ap.add_argument("--panel-width", type=int, default=980)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    _load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    if not args.openrouter_api_key:
        args.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    args.out_dir.mkdir(exist_ok=True, parents=True)

    specs: List[ModelSpec] = []
    for i, spec_str in enumerate(args.model_spec):
        if ":" not in spec_str:
            raise ValueError(f"Invalid --model-spec: {spec_str}, use local:ID or openrouter:ID")
        typ, mid = spec_str.split(":", 1)
        typ = typ.strip().lower()
        if typ not in ("local", "openrouter"):
            raise ValueError(f"Invalid model type in --model-spec: {typ} (use local/openrouter)")
        name = Path(mid).name or mid.split("/")[-1]
        color = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
        specs.append(ModelSpec(name=name, model_id=mid, model_type=typ, color=color))

    if not args.openrouter_api_key and any(s.model_type == "openrouter" for s in specs):
        raise ValueError("Set --openrouter-api-key or OPENROUTER_API_KEY env")

    train_examples = _select_train_examples(
        args.orad_root,
        args.num_examples,
        args.seed,
        args.image_folder,
        args.past_key,
        args.future_key,
        args.max_train_scan,
    )
    print(f"[EXAMPLES] Selected {len(train_examples)} train examples")

    test_samples = list(
        _iter_orad_samples(
            args.orad_root,
            args.split,
            args.image_folder,
            args.past_key,
            args.future_key,
            max_scan=args.max_scan,
        )
    )
    rng = random.Random(args.seed)
    test_samples = rng.sample(test_samples, min(args.num_samples, len(test_samples)))
    print(f"[SAMPLES] {len(test_samples)} test samples")

    results_by_model: Dict[str, List[ModelOutput]] = {s.name: [] for s in specs}

    for spec in specs:
        print(f"[MODEL] Loading {spec.name} ({spec.model_type}:{spec.model_id})")
        model = None
        processor = None
        if spec.model_type == "local":
            model, processor = _load_local_model(
                spec.model_id, args.cache_dir, args.dtype, args.device_map, args.trust_remote_code
            )

        model_outputs: List[ModelOutput] = []
        for sample in test_samples:
            out_text = ""
            traj_pts: List[List[float]] = []

            if spec.model_type == "local":
                assert model is not None and processor is not None
                prompt_text, images = _build_fewshot_prompt(
                    train_examples,
                    sample,
                    args.system,
                    args.coord_hint,
                    args.min_points,
                    args.use_sharegpt_format,
                )
                local_messages = _build_local_messages(
                    train_examples,
                    sample,
                    args.system,
                    args.coord_hint,
                    args.min_points,
                )
                out_text = _infer_local(
                    model,
                    processor,
                    images,
                    prompt_text,
                    args.max_new_tokens,
                    args.temperature,
                    0.9,
                    messages=local_messages,
                )
            else:
                messages = _build_openrouter_messages(
                    train_examples,
                    sample,
                    args.system,
                    args.coord_hint,
                    args.min_points,
                )
                out_text = _infer_openrouter(
                    spec.model_id,
                    args.openrouter_api_key,
                    messages,
                    args.max_new_tokens,
                    args.temperature,
                )

            traj_pts = _extract_traj_xyz(out_text)
            valid = len(traj_pts) >= args.min_points
            model_outputs.append(ModelOutput(spec.name, out_text, traj_pts, valid))

        results_by_model[spec.name] = model_outputs

        if spec.model_type == "local" and model is not None:
            del model
            torch.cuda.empty_cache()

    manifest_path = args.out_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(test_samples):
            outputs = [results_by_model[spec.name][i] for spec in specs]
            composite_path = ""

            if args.save_overlays:
                cam_img = Image.open(sample.image_path).convert("RGB")
                comp = _make_image_plus_plots_composite(
                    camera_image=cam_img,
                    sample=sample,
                    outputs=outputs,
                    specs=specs,
                    panel_width=int(args.panel_width),
                )
                png_path = args.out_dir / f"{i+1:03d}_{sample.key}.png"
                comp.save(png_path)
                composite_path = str(png_path)

            result = SampleResult(
                key=sample.key,
                image_path=str(sample.image_path),
                past_points=sample.past_points,
                gt_points=sample.gt_points,
                target_dist=sample.target_dist,
                outputs=outputs,
                composite_path=composite_path,
            )
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    print(f"[DONE] {len(test_samples)} samples -> {manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
