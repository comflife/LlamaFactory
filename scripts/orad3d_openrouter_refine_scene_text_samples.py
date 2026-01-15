#!/usr/bin/env python3
"""
Refine ORAD-3D scene text using OpenRouter (OpenAI SDK compatible).

This script builds a model-conditioning image that includes:
    - the front-facing camera image
    - a side GT trajectory XY plot
    - a bottom GT Z-trend plot

Example:
    python scripts/orad3d_openrouter_refine_scene_text_samples.py \
        --model google/gemini-3-flash-preview \
        --num-samples 20

Environment (.env):
    OPENROUTER_API_KEY=...
    OPENROUTER_HTTP_REFERER=https://your.domain
    OPENROUTER_APP_TITLE=orad3d-xai-refine
"""

from __future__ import annotations


import argparse
import base64
import io
import json
import os
import random
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from openai import APIError, RateLimitError, Timeout, APIConnectionError


def _strip_optional_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def load_env_file(env_path: Path, *, override: bool = False) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_optional_quotes(value)
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


@dataclass(frozen=True)
class Sample:
    seq: str
    ts: str
    image_path: str
    scene_path: str
    gt_path: Optional[str]
    gt_key: str
    gt_points: Optional[List[List[float]]]
    original_text: str
    refined_text: str
    response_json: Optional[Dict[str, Any]]


def _iter_sequence_dirs(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.iterdir() if p.is_dir() and not p.name.endswith(".zip")], key=lambda p: p.name)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _resolve_image(seq_dir: Path, ts: str, image_folder: str) -> Optional[Path]:
    if image_folder == "image_data":
        p = seq_dir / "image_data" / f"{ts}.png"
        return p if p.exists() else None
    if image_folder == "gt_image":
        p = seq_dir / "gt_image" / f"{ts}_fillcolor.png"
        return p if p.exists() else None
    return None


def _encode_image_as_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _encode_pil_image_as_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _safe_choice_message_content(choice: Dict[str, Any]) -> str:
    content = choice.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part.get("text"), str))
    return ""


def _is_refined_too_short(text: str, min_words: int) -> bool:
    words = [w for w in text.strip().split() if w]
    return len(words) < min_words


SHORT_REFINE_ERROR = "Refined text too short or empty."


def call_xai_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    retries: int,
    retry_sleep_s: float,
) -> Dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.model_dump()
        except (RateLimitError, APIError, Timeout, APIConnectionError) as e:
            if attempt >= retries:
                raise RuntimeError(f"xAI API error after {retries+1} attempts: {e}")
            time.sleep(retry_sleep_s * (2 ** attempt))
    return {}


def request_refinement(
    *,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    retries: int,
    retry_sleep_s: float,
    min_words: int,
) -> Tuple[str, Dict[str, Any]]:
    last_response: Dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            response_json = call_xai_chat(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                retries=0,
                retry_sleep_s=retry_sleep_s,
            )
            last_response = response_json
            choices = response_json.get("choices", [])
            refined = _safe_choice_message_content(choices[0]) if choices else ""
            if refined and not _is_refined_too_short(refined, min_words):
                return refined, response_json
            raise RuntimeError(SHORT_REFINE_ERROR)
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(retry_sleep_s * (2 ** attempt))
    return "", last_response


def build_prompt(original_text: str) -> Tuple[str, str]:
    system = (
        "You are a careful autonomous-driving dataset annotator. "
        "You must be factual, concise, and avoid speculation. "
        "Do not invent objects that are not visible."
    )

    user = (
        "Task: Refine the scene text so it helps off-road trajectory generation.\n"
        "You will receive ONE composite image that contains:\n"
        "(1) a front-facing driving camera image,\n"
        "(2) a side panel showing the XY path direction/curvature,\n"
        "(3) a bottom panel showing the elevation profile.\n\n"
        "Goal: produce a better scene description that explains WHY the shown trajectory makes sense. "
        "Use the extra panels as reference to infer the path and elevation, "
        "but do not mention panels, plots, graphs, or GT in the output. "
        "Use the original description only if it is consistent, "
        "but always rewrite in your own words (never return it verbatim).\n\n"
        "Hard constraints:\n"
        "- No arrows or arrow-like symbols (do NOT use '->' or '=>').\n"
        "- Plain ASCII only.\n"
        "- Mention only what can be inferred from the image and the provided visuals.\n"
        "- Do NOT use the words 'z', 'z-trend', or 'z trend'.\n"
        "- Do NOT mention plots/graphs/GT/panels; do NOT say 'plot shows' or 'GT shows'.\n"
        "- If attribution is needed, use 'the image shows', not 'the plot shows'.\n"
        "- Avoid hedging like 'mostly straight' or 'nearly straight'; if it is straight, say straight.\n\n"
        "MUST include:\n"
        "- Trajectory direction/shape and magnitude: left/right/straight + gentle vs sharp; "
        "if multi-segment, summarize the sequence. Decide left vs right by the XY view, not the camera view; "
        "be accurate about whether the path bends left or right. "
        "Judge left/right magnitude from the XY view; "
        "if lateral offset is large, say clearly/strongly left or right, not slightly. "
        "Only use straight if the XY view stays near vertical with minimal lateral drift.\n"
        "- Path-affecting factors ahead: obstacles/risks that influence the trajectory and whether it avoids them.\n"
        "- Elevation behavior: mostly flat vs uphill vs downhill vs bumpy/undulating (no numbers).\n"
        "Tip: use commas and semicolons to pack information, but keep it readable.\n\n"
        f"Original description (to refine):\n{original_text}"
    )
    return system, user


def _load_gt_points_from_local_path(*, seq_dir: Path, ts: str, gt_key: str) -> Tuple[Optional[Path], Optional[List[List[float]]]]:
    """Loads GT trajectory points from ORAD-3D local_path/<ts>.json.

    Returns (json_path, points) where points are [[x,y,z], ...].
    """

    local_json = seq_dir / "local_path" / f"{ts}.json"
    if not local_json.is_file():
        return None, None

    try:
        obj = json.loads(local_json.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return local_json, None

    raw_pts = obj.get(gt_key)
    if not isinstance(raw_pts, list):
        return local_json, None

    pts: List[List[float]] = []
    for p in raw_pts:
        if not isinstance(p, list) or len(p) < 2:
            continue
        try:
            x = float(p[0])
            y = float(p[1])
            z = float(p[2]) if len(p) > 2 else 0.0
            pts.append([x, y, z])
        except Exception:
            continue

    if len(pts) < 2:
        return local_json, None
    return local_json, pts


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
        pts.append((float(px), float(py)))
    return pts


def render_gt_plot_panel(
    *,
    width: int,
    height: int,
    header: str,
    gt_points: List[List[float]],
    forward_axis: str,
    flip_lateral: bool,
) -> Image.Image:
    """Renders a side plot panel for GT trajectory (forward vs lateral).

    The plot is intentionally simple and designed for conditioning the model.
    """

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    pad = 12
    header_h = 26
    draw.rectangle([(0, 0), (width, header_h)], fill=(245, 245, 245))
    draw.text((pad, 6), header, fill=(0, 0, 0), font=font)

    if len(gt_points) < 2:
        draw.text((pad, header_h + 10), "(no GT)", fill=(0, 0, 0), font=font)
        return img

    plot_box = (pad, header_h + pad, width - pad, height - pad)
    x0, y0, x1, y1 = plot_box
    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)

    fwd, lat = _points_forward_lateral(gt_points, forward_axis=forward_axis, flip_lateral=flip_lateral)
    max_fwd = max(fwd, default=0.0)
    max_lat = max((abs(v) for v in lat), default=0.0)
    max_fwd = max(max_fwd, 1e-3)
    max_lat = max(max_lat, 1e-3)

    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    margin = 10
    usable_w = max(1, w - 2 * margin)
    usable_h = max(1, h - 2 * margin)
    scale = min(usable_h / max_fwd, (usable_w / 2.0) / max_lat)
    scale = max(0.5, min(float(scale), 200.0))

    cx = x0 + w / 2.0
    bottom = y0 + h - margin
    draw.line([(cx, y0 + margin), (cx, y0 + h - margin)], fill=(220, 220, 220), width=1)
    draw.line([(x0 + margin, bottom), (x0 + w - margin, bottom)], fill=(220, 220, 220), width=1)

    pts = _plot_points_in_box(fwd, lat, box=plot_box, scale=scale)
    if len(pts) >= 2:
        for a, b in zip(pts[:-1], pts[1:]):
            draw.line([a, b], fill=(0, 160, 0), width=3)
        for p in pts:
            r = 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(0, 160, 0))

    info = f"GT N={len(gt_points)}"
    draw.text((pad, header_h + 2), info, fill=(0, 120, 0), font=font)
    return img


def render_gt_z_trend_panel(
    *,
    size: Tuple[int, int],
    gt_points: List[List[float]],
) -> Image.Image:
    """Renders a bottom panel showing GT Z trend (relative to first point)."""

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
    draw.text((12, 4), "GT Z trend (relative)", fill=(0, 0, 0), font=font)

    if len(gt_points) < 2:
        draw.text((12, 16), "(no GT)", fill=(0, 0, 0), font=font)
        return img

    z0 = float(gt_points[0][2]) if len(gt_points[0]) > 2 else 0.0
    zs: List[float] = []
    for p in gt_points:
        if not isinstance(p, list) or len(p) < 3:
            zs.append(0.0)
        else:
            zs.append(float(p[2]) - z0)

    z_min = min(zs) if zs else -1.0
    z_max = max(zs) if zs else 1.0
    if abs(z_max - z_min) < 1e-6:
        z_min -= 1.0
        z_max += 1.0

    usable_w = max(1, x1 - x0)
    usable_h = max(1, y1 - y0)
    n = len(zs)

    if n == 1:
        xs = [x0 + usable_w / 2.0]
    else:
        xs = [x0 + (i * usable_w / (n - 1)) for i in range(n)]
    ys = [y1 - ((z - z_min) / (z_max - z_min)) * usable_h for z in zs]
    pts = list(zip(xs, ys))

    if z_min <= 0.0 <= z_max:
        y_zero = y1 - ((0.0 - z_min) / (z_max - z_min)) * usable_h
        draw.line([(x0, y_zero), (x1, y_zero)], fill=(220, 220, 220), width=1)

    if len(pts) >= 2:
        for a, b in zip(pts[:-1], pts[1:]):
            draw.line([a, b], fill=(0, 160, 0), width=2)
        for p in pts:
            r = 2
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(0, 160, 0))

    draw.text((x0 + 6, y0 + 6), f"GT N={len(zs)}", fill=(0, 120, 0), font=font)
    return img


def render_text_panel(
    *,
    width: int,
    height: int,
    header: str,
    original_text: str,
    refined_text: str,
) -> Image.Image:
    # Normalize text to ASCII to avoid Unicode encoding issues with PIL
    header = unicodedata.normalize("NFKD", header).encode("ascii", "ignore").decode("ascii")
    original_text = unicodedata.normalize("NFKD", original_text).encode("ascii", "ignore").decode("ascii")
    refined_text = unicodedata.normalize("NFKD", refined_text).encode("ascii", "ignore").decode("ascii")

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    x = 12
    y = 10
    line_gap = 4

    def draw_wrapped(title: str, text: str, y0: int) -> int:
        draw.text((x, y0), title, fill=(0, 0, 0), font=font)
        y0 += 14 + line_gap
        max_chars = max(20, (width - 2 * x) // 6)
        words = (text or "").replace("\n", " ").split()
        if not words:
            draw.text((x, y0), "(empty)", fill=(0, 0, 0), font=font)
            return y0 + 14 + line_gap

        line: List[str] = []
        for w in words:
            candidate = " ".join(line + [w])
            if len(candidate) <= max_chars:
                line.append(w)
                continue
            draw.text((x, y0), " ".join(line), fill=(0, 0, 0), font=font)
            y0 += 14 + line_gap
            line = [w]
        if line:
            draw.text((x, y0), " ".join(line), fill=(0, 0, 0), font=font)
            y0 += 14 + line_gap
        return y0 + 6

    draw.rectangle([(0, 0), (width, 26)], fill=(245, 245, 245))
    draw.text((x, 6), header, fill=(0, 0, 0), font=font)
    y = 32

    y = draw_wrapped("Original:", original_text, y)
    draw_wrapped("Refined:", refined_text, y)

    return img


def make_composite(image_path: Path, original_text: str, refined_text: str, header: str) -> Image.Image:
    base = Image.open(image_path).convert("RGB")
    panel_w = max(520, base.size[0] // 2)
    panel = render_text_panel(width=panel_w, height=base.size[1], header=header, original_text=original_text, refined_text=refined_text)

    out = Image.new("RGB", (base.size[0] + panel.size[0], base.size[1]), (255, 255, 255))
    out.paste(base, (0, 0))
    out.paste(panel, (base.size[0], 0))
    return out


def make_model_input_with_gt_plot(
    image_path: Path,
    *,
    gt_points: List[List[float]],
    header: str,
    forward_axis: str,
    flip_lateral: bool,
) -> Image.Image:
    base = Image.open(image_path).convert("RGB")
    panel_w = max(420, base.size[0] // 2)
    plot_panel = render_gt_plot_panel(
        width=panel_w,
        height=base.size[1],
        header=header,
        gt_points=gt_points,
        forward_axis=forward_axis,
        flip_lateral=flip_lateral,
    )

    top = Image.new("RGB", (base.size[0] + plot_panel.size[0], base.size[1]), (255, 255, 255))
    top.paste(base, (0, 0))
    top.paste(plot_panel, (base.size[0], 0))

    z_panel_h = max(140, base.size[1] // 4)
    z_panel = render_gt_z_trend_panel(size=(top.size[0], z_panel_h), gt_points=gt_points)

    out = Image.new("RGB", (top.size[0], top.size[1] + z_panel.size[1]), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(z_panel, (0, top.size[1]))
    return out


def gather_pairs(
    split_dir: Path,
    image_folder: str,
    max_scan: Optional[int],
    *,
    gt_key: str,
    require_gt: bool,
) -> List[Tuple[str, str, Path, Path, Optional[Path], Optional[List[List[float]]]]]:
    pairs: List[Tuple[str, str, Path, Path, Optional[Path], Optional[List[List[float]]]]] = []

    for seq_dir in _iter_sequence_dirs(split_dir):
        scene_dir = seq_dir / "scene_data"
        if not scene_dir.is_dir():
            continue

        for scene_path in sorted(scene_dir.glob("*.txt"), key=lambda p: p.name):
            ts = scene_path.stem
            img = _resolve_image(seq_dir, ts=ts, image_folder=image_folder)
            if img is None:
                continue
            gt_path, gt_points = _load_gt_points_from_local_path(seq_dir=seq_dir, ts=ts, gt_key=gt_key)
            if require_gt and gt_points is None:
                continue
            pairs.append((seq_dir.name, ts, img, scene_path, gt_path, gt_points))
            if max_scan is not None and len(pairs) >= max_scan:
                return pairs

    return pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine ORAD-3D scene text using xAI API.")
    ap.add_argument("--orad-root", type=Path, default=Path("/home/work/datasets/bg/ORAD-3D"))
    ap.add_argument("--split", type=str, default="training", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])
    ap.add_argument("--out-dir", type=Path, default=Path("/home/work/byounggun/LlamaFactory/orad3d_openrouter_refine_samples"))
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scan", type=int, default=None)

    ap.add_argument("--gt-key", type=str, default="trajectory_ins")
    ap.add_argument(
        "--require-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip samples without GT trajectory in local_path.",
    )
    ap.add_argument("--forward-axis", choices=["x", "y"], default="y")
    ap.add_argument("--flip-lateral", action="store_true")

    ap.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    ap.add_argument("--model", type=str, default="openai/gpt-5.2")
    ap.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    ap.add_argument(
        "--http-referer-env",
        type=str,
        default="OPENROUTER_HTTP_REFERER",
        help="Env var name for OpenRouter HTTP-Referer header (optional).",
    )
    ap.add_argument(
        "--app-title-env",
        type=str,
        default="OPENROUTER_APP_TITLE",
        help="Env var name for OpenRouter X-Title header (optional).",
    )
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--env-override", action="store_true")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--min-refined-words",
        type=int,
        default=8,
        help="Retry if refined text has fewer words than this.",
    )
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-sleep", type=float, default=1.0)

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    split_dir = args.orad_root / args.split

    repo_root = Path(__file__).resolve().parents[1]
    env_path = args.env_file or (repo_root / ".env" if (repo_root / ".env").exists() else Path.cwd() / ".env")
    load_env_file(env_path, override=args.env_override)

    pairs = gather_pairs(
        split_dir=split_dir,
        image_folder=args.image_folder,
        max_scan=args.max_scan,
        gt_key=str(args.gt_key),
        require_gt=bool(args.require_gt),
    )
    if not pairs:
        raise SystemExit(f"No pairs found under: {split_dir}")

    rng = random.Random(args.seed)
    n = min(args.num_samples, len(pairs))
    chosen = rng.sample(pairs, k=n) if len(pairs) > n else pairs

    client = None
    if not args.dry_run:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(f"Missing {args.api_key_env} environment variable.")
        default_headers: Dict[str, str] = {}
        http_referer = os.environ.get(str(args.http_referer_env), "").strip()
        app_title = os.environ.get(str(args.app_title_env), "").strip()
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if app_title:
            default_headers["X-Title"] = app_title

        client = OpenAI(
            base_url=args.base_url,
            api_key=api_key,
            timeout=args.timeout,
            default_headers=default_headers or None,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "samples.jsonl"
    samples: List[Sample] = []

    for i, (seq, ts, img_path, scene_path, gt_path, gt_points) in enumerate(chosen):
        original = _read_text(scene_path)
        refined = original
        response_json: Optional[Dict[str, Any]] = None

        if gt_points is None:
            gt_path2, gt_points2 = _load_gt_points_from_local_path(seq_dir=scene_path.parents[1], ts=ts, gt_key=str(args.gt_key))
            gt_path = gt_path or gt_path2
            gt_points = gt_points2
        if bool(args.require_gt) and gt_points is None:
            continue

        model_input_header = f"GT trajectory plots ({args.gt_key})"
        model_input_image = (
            make_model_input_with_gt_plot(
                img_path,
                gt_points=gt_points or [],
                header=model_input_header,
                forward_axis=str(args.forward_axis),
                flip_lateral=bool(args.flip_lateral),
            )
            if gt_points is not None
            else Image.open(img_path).convert("RGB")
        )

        if not args.dry_run and client:
            system, user = build_prompt(original)
            data_url = _encode_pil_image_as_data_url(model_input_image)
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ]
            try:
                refined, response_json = request_refinement(
                    client=client,
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    retries=args.retries,
                    retry_sleep_s=args.retry_sleep,
                    min_words=args.min_refined_words,
                )
            except Exception as e:
                err = str(e)
                response_json = {"error": err}
                if err == SHORT_REFINE_ERROR:
                    refined = ""
                    print(f"[WARN] {seq} {ts}: refined text too short; keeping empty.")
                elif not args.continue_on_error:
                    raise

        header = f"{args.split} / {seq} / {ts}"
        # Save an inspection composite: (image+GT plot) + (text panel)
        panel_w = max(520, model_input_image.size[0] // 2)
        text_panel = render_text_panel(
            width=panel_w,
            height=model_input_image.size[1],
            header=header,
            original_text=original,
            refined_text=refined,
        )
        composite = Image.new(
            "RGB",
            (model_input_image.size[0] + text_panel.size[0], model_input_image.size[1]),
            (255, 255, 255),
        )
        composite.paste(model_input_image, (0, 0))
        composite.paste(text_panel, (model_input_image.size[0], 0))
        out_img = args.out_dir / f"{i:02d}_{args.split}_{seq}_{ts}.png"
        composite.save(out_img)

        samples.append(
            Sample(
                seq=seq,
                ts=ts,
                image_path=str(img_path),
                scene_path=str(scene_path),
                gt_path=str(gt_path) if gt_path is not None else None,
                gt_key=str(args.gt_key),
                gt_points=gt_points,
                original_text=original,
                refined_text=refined,
                response_json=response_json,
            )
        )

        print(f"[OK] {i+1}/{n}: {out_img.name}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    print(f"[DONE] {len(samples)} samples saved to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
