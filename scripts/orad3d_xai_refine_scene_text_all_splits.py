#!/usr/bin/env python3
"""Refine ORAD-3D scene text using xAI API and save as scene_data_refine.

This script scans ORAD-3D splits (training/validation/testing) and for each sequence folder:
- reads original scene text from: scene_data/<timestamp>.txt
- optionally loads the corresponding image (image_data/<timestamp>.png or gt_image/<timestamp>_fillcolor.png)
- calls xAI (OpenAI-compatible) chat completion with image + original text
- writes refined text to: scene_data_refine/<timestamp>.txt

It preserves the original naming/layout so downstream code can treat it like a parallel scene_text source.

Example:
    python scripts/orad3d_xai_refine_scene_text_all_splits.py --model grok-4-1-fast-reasoning
    # If scene_data_refine/<ts>.txt exists, it will be skipped unless you pass --overwrite.
"""

from __future__ import annotations

import argparse
import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import APIConnectionError, APIError, OpenAI, RateLimitError, Timeout


def _strip_optional_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    return value


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
    return sorted(
        [p for p in split_dir.iterdir() if p.is_dir() and not p.name.endswith(".zip")], key=lambda p: p.name
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _resolve_image(seq_dir: Path, ts: str, image_folder: str) -> Optional[Path]:
    if image_folder == "image_data":
        candidate = seq_dir / "image_data" / f"{ts}.png"
        return candidate if candidate.exists() else None
    if image_folder == "gt_image":
        candidate = seq_dir / "gt_image" / f"{ts}_fillcolor.png"
        return candidate if candidate.exists() else None
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
        parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


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
                raise RuntimeError(f"xAI API error after {retries + 1} attempts: {exc}")
            time.sleep(retry_sleep_s * (2**attempt))

    return {}


def build_prompt(original_text: str) -> Tuple[str, str]:
    system = "You are a careful autonomous-driving dataset annotator. You must be factual and avoid speculation."
    user = (
        "You are an expert off-road autonomous driving AI generating Chain-of-Causation (CoC) reasoning for trajectory planning. "
        "Analyze this front-facing camera image for off-road terrain navigation. "
        "Focus on traversable areas, obstacles, slopes, gaps. "
        "Output ONLY in this exact format (English, concise):\n"
        "Decision: [explicit driving action, e.g., 'steer left around rock', 'accelerate through flat gap', 'slow climb steep slope']\n"
        "because [core visual cause, e.g., 'narrow traversable gap left']\n"
        "as [short causal chain, e.g., 'right blocked by boulders \u2192 left gap safer traction \u2192 gentle curve trajectory'].\n\n"
        f"Original description:\n{original_text}"
    )
    return system, user


def _iter_scene_pairs(
    *,
    seq_dir: Path,
    image_folder: str,
) -> Iterable[Tuple[str, Path, Optional[Path]]]:
    scene_dir = seq_dir / "scene_data"
    if not scene_dir.is_dir():
        return []

    pairs: List[Tuple[str, Path, Optional[Path]]] = []
    for scene_path in sorted(scene_dir.glob("*.txt"), key=lambda p: p.name):
        ts = scene_path.stem
        img_path = _resolve_image(seq_dir, ts=ts, image_folder=image_folder)
        pairs.append((ts, scene_path, img_path))
    return pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine ORAD-3D scene text and save into scene_data_refine.")
    ap.add_argument("--orad-root", type=Path, default=Path("/data3/ORAD-3D"))
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["training", "validation", "testing"],
        choices=["training", "validation", "testing"],
    )
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])

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

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing scene_data_refine/<ts>.txt")
    ap.add_argument("--max-seqs", type=int, default=None, help="Limit number of sequences per split")
    ap.add_argument("--max-scenes", type=int, default=None, help="Limit number of scene txts per sequence")

    ap.add_argument("--dry-run", action="store_true", help="Scan and print planned outputs without calling API")
    ap.add_argument("--continue-on-error", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_path = args.env_file or (repo_root / ".env" if (repo_root / ".env").exists() else Path.cwd() / ".env")
    load_env_file(env_path, override=args.env_override)

    client: Optional[OpenAI] = None
    if not args.dry_run:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(f"Missing {args.api_key_env} environment variable.")
        client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)

    total_done = 0
    total_skipped = 0
    total_errors = 0

    for split in args.splits:
        split_dir = args.orad_root / split
        if not split_dir.is_dir():
            raise SystemExit(f"Split dir not found: {split_dir}")

        seq_dirs = _iter_sequence_dirs(split_dir)
        if args.max_seqs is not None:
            seq_dirs = seq_dirs[: args.max_seqs]

        print(f"[SPLIT] {split}: {len(seq_dirs)} sequences")

        for seq_idx, seq_dir in enumerate(seq_dirs, start=1):
            pairs = list(_iter_scene_pairs(seq_dir=seq_dir, image_folder=args.image_folder))
            if not pairs:
                continue
            if args.max_scenes is not None:
                pairs = pairs[: args.max_scenes]

            out_dir = seq_dir / "scene_data_refine"
            out_dir.mkdir(parents=True, exist_ok=True)

            for ts, scene_path, img_path in pairs:
                out_path = out_dir / f"{ts}.txt"
                if out_path.exists() and not args.overwrite:
                    total_skipped += 1
                    continue

                original = _read_text(scene_path)
                refined = original

                try:
                    if args.dry_run:
                        pass
                    else:
                        if client is None:
                            raise RuntimeError("Client not initialized.")
                        if img_path is None:
                            raise RuntimeError(
                                f"Missing image for {split}/{seq_dir.name}/{ts} (image_folder={args.image_folder})"
                            )

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

                    _write_text(out_path, refined)
                    total_done += 1

                    if args.dry_run:
                        print(f"[DRY] {split}/{seq_dir.name}/{ts} -> {out_path}")
                    else:
                        print(f"[OK] {split}/{seq_dir.name}/{ts} -> {out_path}")

                except Exception as exc:
                    total_errors += 1
                    print(f"[ERR] {split}/{seq_dir.name}/{ts}: {exc}")
                    if not args.continue_on_error:
                        raise

            print(f"[SEQ] {split} {seq_idx}/{len(seq_dirs)}: {seq_dir.name} done")

    print(f"[DONE] wrote={total_done} skipped={total_skipped} errors={total_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
