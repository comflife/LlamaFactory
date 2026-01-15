#!/usr/bin/env python3
"""Refine ORAD-3D scene text using OpenRouter + GT plots and save as scene_data_refine.

This script scans ORAD-3D splits (training/validation/testing) and for each sequence folder:
- reads original scene text from: scene_data/<timestamp>.txt
- loads the corresponding image (image_data/<timestamp>.png or gt_image/<timestamp>_fillcolor.png)
- loads GT trajectory points from: local_path/<timestamp>.json
- builds a composite image (camera + XY path panel + elevation profile panel)
- calls OpenRouter (OpenAI-compatible) chat completion with image + original text
- writes refined text to: scene_data_refine/<timestamp>.txt

python scripts/orad3d_openrouter_refine_scene_text_all_splits_with_gt.py --model google/gemini-3-flash-preview --split training
python scripts/orad3d_openrouter_refine_scene_text_all_splits_with_gt.py --model google/gemini-3-flash-preview --split testing --overwrite


"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from openai import APIConnectionError, APIError, OpenAI, RateLimitError, Timeout


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


def _iter_sequence_dirs(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.iterdir() if p.is_dir() and not p.name.endswith(".zip")], key=lambda p: p.name)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _resolve_image(seq_dir: Path, ts: str, image_folder: str) -> Optional[Path]:
    if image_folder == "image_data":
        p = seq_dir / "image_data" / f"{ts}.png"
        return p if p.exists() else None
    if image_folder == "gt_image":
        p = seq_dir / "gt_image" / f"{ts}_fillcolor.png"
        return p if p.exists() else None
    return None


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
        return "\n".join(part.get("text", "") for part in content if isinstance(part.get("text"), str)).strip()
    return ""


def _is_refined_too_short(text: str, min_words: int) -> bool:
    words = [w for w in text.strip().split() if w]
    return len(words) < min_words


SHORT_REFINE_ERROR = "Refined text too short or empty."


def call_xai_chat(
    *,
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
        except (RateLimitError, APIError, Timeout, APIConnectionError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"OpenRouter API error after {retries + 1} attempts: {exc}")
            time.sleep(retry_sleep_s * (2**attempt))
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
            time.sleep(retry_sleep_s * (2**attempt))
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


def _load_gt_points_from_local_path(
    *,
    seq_dir: Path,
    ts: str,
    gt_key: str,
) -> Tuple[Optional[Path], Optional[List[List[float]]]]:
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


def _iter_scene_pairs(
    *,
    seq_dir: Path,
    image_folder: str,
    gt_key: str,
) -> Iterable[Tuple[str, Path, Optional[Path], Optional[Path], Optional[List[List[float]]]]]:
    scene_dir = seq_dir / "scene_data"
    if not scene_dir.is_dir():
        return []

    pairs: List[Tuple[str, Path, Optional[Path], Optional[Path], Optional[List[List[float]]]]] = []
    for scene_path in sorted(scene_dir.glob("*.txt"), key=lambda p: p.name):
        ts = scene_path.stem
        img_path = _resolve_image(seq_dir, ts=ts, image_folder=image_folder)
        gt_path, gt_points = _load_gt_points_from_local_path(seq_dir=seq_dir, ts=ts, gt_key=gt_key)
        pairs.append((ts, scene_path, img_path, gt_path, gt_points))
    return pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Refine ORAD-3D scene text using OpenRouter + GT plots and save into scene_data_refine."
    )
    ap.add_argument("--orad-root", type=Path, default=Path("/home/work/datasets/bg/ORAD-3D"))
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["training", "validation", "testing"],
        choices=["training", "validation", "testing"],
    )
    ap.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["training", "validation", "testing"],
        help="Single split to process (overrides --splits).",
    )
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])

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

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing scene_data_refine/<ts>.txt")
    ap.add_argument("--max-seqs", type=int, default=None, help="Limit number of sequences per split")
    ap.add_argument("--max-scenes", type=int, default=None, help="Limit number of scene txts per sequence")

    ap.add_argument("--dry-run", action="store_true", help="Scan and print planned outputs without calling API")
    ap.add_argument("--continue-on-error", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    splits = [args.split] if args.split else list(args.splits)

    repo_root = Path(__file__).resolve().parents[1]
    env_path = args.env_file or (repo_root / ".env" if (repo_root / ".env").exists() else Path.cwd() / ".env")
    load_env_file(env_path, override=args.env_override)

    client: Optional[OpenAI] = None
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

    total_done = 0
    total_skipped = 0
    total_errors = 0

    for split in splits:
        split_dir = args.orad_root / split
        if not split_dir.is_dir():
            raise SystemExit(f"Split dir not found: {split_dir}")

        seq_dirs = _iter_sequence_dirs(split_dir)
        if args.max_seqs is not None:
            seq_dirs = seq_dirs[: args.max_seqs]

        print(f"[SPLIT] {split}: {len(seq_dirs)} sequences")

        for seq_idx, seq_dir in enumerate(seq_dirs, start=1):
            pairs = list(_iter_scene_pairs(seq_dir=seq_dir, image_folder=args.image_folder, gt_key=str(args.gt_key)))
            if not pairs:
                continue
            if args.max_scenes is not None:
                pairs = pairs[: args.max_scenes]

            out_dir = seq_dir / "scene_data_refine"
            out_dir.mkdir(parents=True, exist_ok=True)

            for ts, scene_path, img_path, gt_path, gt_points in pairs:
                out_path = out_dir / f"{ts}.txt"
                if out_path.exists() and not args.overwrite:
                    total_skipped += 1
                    continue

                if img_path is None:
                    total_skipped += 1
                    print(
                        f"[SKIP] missing image for {split}/{seq_dir.name}/{ts} (image_folder={args.image_folder})"
                    )
                    continue

                if bool(args.require_gt) and gt_points is None:
                    total_skipped += 1
                    print(f"[SKIP] missing GT for {split}/{seq_dir.name}/{ts} (gt_key={args.gt_key})")
                    continue

                original = _read_text(scene_path)
                refined = original

                try:
                    if args.dry_run:
                        pass
                    else:
                        if client is None:
                            raise RuntimeError("Client not initialized.")

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
                        refined, _ = request_refinement(
                            client=client,
                            model=args.model,
                            messages=messages,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            retries=args.retries,
                            retry_sleep_s=args.retry_sleep,
                            min_words=args.min_refined_words,
                        )

                    _write_text(out_path, refined)
                    total_done += 1

                    if args.dry_run:
                        print(f"[DRY] {split}/{seq_dir.name}/{ts} -> {out_path}")
                    else:
                        print(f"[OK] {split}/{seq_dir.name}/{ts} -> {out_path}")

                except Exception as exc:
                    total_errors += 1
                    err = str(exc)
                    if err == SHORT_REFINE_ERROR:
                        refined = ""
                        _write_text(out_path, refined)
                        print(f"[WARN] {split}/{seq_dir.name}/{ts}: refined text too short; wrote empty.")
                        continue
                    print(f"[ERR] {split}/{seq_dir.name}/{ts}: {exc}")
                    if not args.continue_on_error:
                        raise

            print(f"[SEQ] {split} {seq_idx}/{len(seq_dirs)}: {seq_dir.name} done")

    print(f"[DONE] wrote={total_done} skipped={total_skipped} errors={total_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
