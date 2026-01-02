#!/usr/bin/env python3
"""Refine ORAD-3D scene text using xAI API (OpenAI SDK version)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
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


def _safe_choice_message_content(choice: Dict[str, Any]) -> str:
    content = choice.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part.get("text"), str))
    return ""


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


def build_prompt(original_text: str) -> Tuple[str, str]:
    system = (
        "You are a careful autonomous-driving dataset annotator. "
        "You must be factual and avoid speculation."
    )
    user = (
        "I am about to drive autonomously using this front-facing camera image. "
        "Please verify whether the following language description correctly matches the scene. "
        "If it is incorrect, rewrite it to be accurate. "
        "Keep it concise and in English. "
        "Return only the corrected description (no extra commentary).\n\n"
        f"Original description:\n{original_text}"
    )
    return system, user


def render_text_panel(
    *,
    width: int,
    height: int,
    header: str,
    original_text: str,
    refined_text: str,
) -> Image.Image:
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


def gather_pairs(split_dir: Path, image_folder: str, max_scan: Optional[int]) -> List[Tuple[str, str, Path, Path]]:
    pairs: List[Tuple[str, str, Path, Path]] = []

    for seq_dir in _iter_sequence_dirs(split_dir):
        scene_dir = seq_dir / "scene_data"
        if not scene_dir.is_dir():
            continue

        for scene_path in sorted(scene_dir.glob("*.txt"), key=lambda p: p.name):
            ts = scene_path.stem
            img = _resolve_image(seq_dir, ts=ts, image_folder=image_folder)
            if img is None:
                continue
            pairs.append((seq_dir.name, ts, img, scene_path))
            if max_scan is not None and len(pairs) >= max_scan:
                return pairs

    return pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine ORAD-3D scene text using xAI API.")
    ap.add_argument("--orad-root", type=Path, default=Path("/data3/ORAD-3D"))
    ap.add_argument("--split", type=str, default="training", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])
    ap.add_argument("--out-dir", type=Path, default=Path("/home/byounggun/LlamaFactory/orad3d_xai_refine_samples"))
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scan", type=int, default=None)

    ap.add_argument("--base-url", type=str, default="https://api.x.ai/v1")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--api-key-env", type=str, default="XAI_API_KEY")
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--env-override", action="store_true")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
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

    pairs = gather_pairs(split_dir=split_dir, image_folder=args.image_folder, max_scan=args.max_scan)
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
        client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "samples.jsonl"
    samples: List[Sample] = []

    for i, (seq, ts, img_path, scene_path) in enumerate(chosen):
        original = _read_text(scene_path)
        refined = original
        response_json: Optional[Dict[str, Any]] = None

        if not args.dry_run and client:
            system, user = build_prompt(original)
            data_url = _encode_image_as_data_url(img_path)
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
                response_json = call_xai_chat(
                    client=client,
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    retries=args.retries,
                    retry_sleep_s=args.retry_sleep,
                )
                choices = response_json.get("choices", [])
                if choices:
                    refined = _safe_choice_message_content(choices[0]) or refined
            except Exception as e:
                response_json = {"error": str(e)}
                if not args.continue_on_error:
                    raise

        header = f"{args.split} / {seq} / {ts}"
        composite = make_composite(img_path, original_text=original, refined_text=refined, header=header)
        out_img = args.out_dir / f"{i:02d}_{args.split}_{seq}_{ts}.png"
        composite.save(out_img)

        samples.append(Sample(seq=seq, ts=ts, image_path=str(img_path), scene_path=str(scene_path),
                             original_text=original, refined_text=refined, response_json=response_json))

        print(f"[OK] {i+1}/{n}: {out_img.name}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    print(f"[DONE] {len(samples)} samples saved to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())