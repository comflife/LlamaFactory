#!/usr/bin/env python3
"""Export ORAD-3D VLM predictions with GT trajectories for later evaluation.

This script runs inference for multiple adapters and saves per-sample JSONL rows:
  - image path + metadata
  - GT trajectory (if available; samples without GT are skipped)
  - per-adapter predicted trajectories + raw model text

Example:
python scripts/orad3d_adapter_infer_save.py \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --adapter sft_refine=/home/work/datasets/bg/byounggun/saves/orad3d/qwen3-vl-2b/lora/sft_v2_refine/checkpoint-3132 \
  --orad-root /home/work/datasets/bg/ORAD-3D \
  --split testing --image-folder image_data \
  --out-dir /home/work/byounggun/LlamaFactory/orad3d_export \
  --use-sharegpt-format --temperature 1e-6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)


_TRAJ_TOKEN_RE = re.compile(r"<\s*trajectory\s*>", re.IGNORECASE)
_POINT_RE = re.compile(
    r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]"
)

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    path: str


@dataclass(frozen=True)
class SampleItem:
    key: str
    image_path: Path
    gt_points: List[List[float]]
    meta: Dict[str, Any]
    gt_source: Optional[str]


@dataclass
class ModelOutput:
    name: str
    adapter_path: str
    output_text: str
    trajectory_points: List[List[float]]
    valid: bool


def _normalize_path_str(p: str) -> str:
    return str(p).replace("\\", "/").lstrip("./")


def _extract_text_from_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join([p for p in parts if p])
    return str(content)


def _extract_trajectory_section(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TRAJ_TOKEN_RE.search(text)
    if not m:
        return None
    return text[m.end() :].strip() or ""


def _extract_trajectory_points(text: str) -> List[List[float]]:
    pts: List[List[float]] = []
    for m in _POINT_RE.finditer(text):
        try:
            x = float(m.group(1))
            y = float(m.group(2))
            z = float(m.group(3))
            pts.append([x, y, z])
        except Exception:
            continue
    return pts


def _clean_output_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("<tool_call>", "").replace("</tool_call>", "").replace("<tool_call/>", "")
    return cleaned.strip()


def _candidate_image_keys(img_path: Path, *, orad_root: Optional[Path], meta: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    p = img_path
    keys.append(_normalize_path_str(str(p)))
    keys.append(_normalize_path_str(str(p.resolve())))

    if orad_root is not None:
        try:
            rel = p.resolve().relative_to(orad_root.resolve())
            keys.append(_normalize_path_str(str(rel)))
        except Exception:
            pass

    split = str(meta.get("split") or "").strip()
    seq = str(meta.get("sequence") or "").strip()
    ts = str(meta.get("timestamp") or "").strip()
    if split and seq and ts:
        keys.append(_normalize_path_str(f"{split}/{seq}/image_data/{ts}.png"))
        keys.append(_normalize_path_str(f"{split}/{seq}/gt_image/{ts}.png"))

    keys.append(_normalize_path_str(p.name))
    out: List[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _load_gt_trajectories_for_items(
    gt_jsonl: Path,
    *,
    wanted_keys: set[str],
) -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}
    if not gt_jsonl.is_file() or not wanted_keys:
        return out

    with gt_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            images = obj.get("images")
            if not isinstance(images, list) or not images:
                continue

            matched_img_keys: List[str] = []
            for img in images:
                if not isinstance(img, str):
                    continue
                k = _normalize_path_str(img)
                if k in wanted_keys:
                    matched_img_keys.append(k)
                bn = _normalize_path_str(Path(k).name)
                if bn in wanted_keys:
                    matched_img_keys.append(bn)

            if not matched_img_keys:
                continue

            messages = obj.get("messages")
            if not isinstance(messages, list) or not messages:
                continue

            assistant_text = ""
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    assistant_text = _extract_text_from_message_content(m.get("content"))
                    break

            traj_section = _extract_trajectory_section(assistant_text)
            if traj_section is None:
                continue
            pts = _extract_trajectory_points(traj_section)
            if len(pts) < 2:
                continue

            for k in matched_img_keys:
                out.setdefault(k, pts)

    return out


def _infer_local_path_from_image(img_path: Path) -> Optional[Path]:
    parent = img_path.parent
    if parent.name in ("image_data", "gt_image"):
        local_dir = parent.parent / "local_path"
        return local_dir / f"{img_path.stem}.json"
    return None


def _infer_local_path_from_orad_root(img_path: Path, orad_root: Optional[Path]) -> Optional[Path]:
    if orad_root is None:
        return None
    try:
        rel = img_path.resolve().relative_to(orad_root.resolve())
    except Exception:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    split, seq = parts[0], parts[1]
    return orad_root / split / seq / "local_path" / f"{img_path.stem}.json"


def _load_gt_from_local_path(
    *,
    img_path: Path,
    orad_root: Optional[Path],
    meta: Dict[str, Any],
    gt_key: str,
) -> Optional[Tuple[List[List[float]], str]]:
    candidates: List[Path] = []

    direct = _infer_local_path_from_image(img_path)
    if direct is not None:
        candidates.append(direct)

    via_root = _infer_local_path_from_orad_root(img_path, orad_root)
    if via_root is not None:
        candidates.append(via_root)

    split = str(meta.get("split") or "").strip()
    seq = str(meta.get("sequence") or "").strip()
    ts = str(meta.get("timestamp") or "").strip()
    if orad_root is not None and split and seq and ts:
        candidates.append(orad_root / split / seq / "local_path" / f"{ts}.json")

    local_json = next((p for p in candidates if p.is_file()), None)
    if local_json is None:
        return None

    try:
        obj = json.loads(local_json.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    pts = obj.get(gt_key)
    if not isinstance(pts, list):
        return None

    out: List[List[float]] = []
    for p in pts:
        if not isinstance(p, list) or len(p) < 2:
            continue
        try:
            out.append([float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0])
        except Exception:
            continue
    if len(out) < 2:
        return None
    return out, str(local_json)


def _maybe_set_cache_env(cache_dir: Optional[str]) -> None:
    if not cache_dir:
        return

    try:
        from llamafactory.model.loader import _maybe_set_hf_cache_env  # type: ignore

        _maybe_set_hf_cache_env(cache_dir)
        return
    except Exception:
        pass

    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))


def _parse_dtype(value: str) -> torch.dtype:
    v = value.strip().lower()
    if v in ("auto", ""):
        return torch.float16
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16"):
        return torch.float16
    if v in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _load_model_and_processor(
    *,
    base_model: str,
    adapter: str,
    cache_dir: Optional[str],
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
) -> Tuple[torch.nn.Module, Any]:
    _maybe_set_cache_env(cache_dir)

    tokenizer = None
    processor = None

    adapter_is_dir = False
    try:
        adapter_is_dir = os.path.isdir(adapter)
    except Exception:
        adapter_is_dir = False

    if adapter_is_dir:
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
        except Exception:
            tokenizer = None
        try:
            processor = AutoProcessor.from_pretrained(adapter, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
        except Exception:
            processor = None

    if processor is None:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
        except Exception:
            tokenizer = None

    if tokenizer is not None and hasattr(processor, "tokenizer"):
        try:
            processor.tokenizer = tokenizer  # type: ignore[attr-defined]
        except Exception:
            pass

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code, cache_dir=cache_dir)

    model = None
    torch_dtype: Any = "auto" if dtype.strip().lower() == "auto" else _parse_dtype(dtype)

    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            try:
                model = cls.from_pretrained(
                    base_model,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_dir,
                    dtype=torch_dtype,
                    device_map=device_map,
                )
            except TypeError:
                model = cls.from_pretrained(
                    base_model,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(f"Failed to load base model: {base_model} (model_type={getattr(cfg, 'model_type', None)})")

    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def _build_messages(prompt_text: str, system_text: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if (system_text or "").strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text.strip()}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (prompt_text or "").strip()},
            ],
        }
    )
    return messages


def _maybe_prefix_image_token(prompt_text: str, *, use_sharegpt_format: bool) -> str:
    txt = (prompt_text or "").strip()
    if not use_sharegpt_format:
        return txt
    if txt.lower().startswith("<image>"):
        return txt
    return f"<image>\n{txt}".strip()


def _prepare_inputs(
    processor: Any,
    *,
    image: Image.Image,
    system_text: str,
    prompt_text: str,
    use_sharegpt_format: bool,
) -> Tuple[Dict[str, torch.Tensor], int]:
    prompt_text = _maybe_prefix_image_token(prompt_text, use_sharegpt_format=use_sharegpt_format)
    messages = _build_messages(prompt_text, system_text)

    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif hasattr(getattr(processor, "tokenizer", None), "apply_chat_template"):
        chat_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        chat_text = f"<image>\n{prompt_text}"

    inputs = processor(text=[chat_text], images=[image], return_tensors="pt", padding=True)
    input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    return inputs, input_len


def _iter_orad_pairs(
    *,
    orad_root: Path,
    split: str,
    image_folder: str,
    max_scan: Optional[int],
) -> List[Tuple[str, str, Path]]:
    split_dir = orad_root / split
    if not split_dir.is_dir():
        raise SystemExit(f"Split dir not found: {split_dir}")

    pairs: List[Tuple[str, str, Path]] = []
    seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and not p.name.endswith(".zip")], key=lambda p: p.name)

    for seq_dir in seq_dirs:
        img_dir = seq_dir / image_folder
        if not img_dir.is_dir():
            continue

        for img_path in sorted(img_dir.glob("*.png"), key=lambda p: p.name):
            ts = img_path.stem
            pairs.append((seq_dir.name, ts, img_path))
            if max_scan is not None and len(pairs) >= max_scan:
                return pairs

    return pairs


def _parse_num_samples(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in ("all", "full"):
        return None
    try:
        parsed = int(raw)
    except Exception as exc:
        raise SystemExit(f"Invalid --num-samples value: {value}") from exc
    if parsed <= 0:
        return None
    return parsed


def _sample_items(
    *,
    args: argparse.Namespace,
    gt_map: Dict[str, List[List[float]]],
) -> List[SampleItem]:
    items: List[SampleItem] = []

    if args.image:
        for p in args.image:
            img_path = Path(p)
            meta = {"source": "image"}
            key = img_path.stem
            gt_points = None
            gt_source = None
            gt_local = _load_gt_from_local_path(
                img_path=img_path, orad_root=args.orad_root, meta=meta, gt_key=args.gt_key
            )
            if gt_local is not None:
                gt_points, gt_source = gt_local
            if gt_points is None:
                for cand in _candidate_image_keys(img_path, orad_root=args.orad_root, meta=meta):
                    if cand in gt_map:
                        gt_points = gt_map[cand]
                        gt_source = str(args.gt_jsonl) if args.gt_jsonl else None
                        break
            if gt_points is None:
                continue
            items.append(SampleItem(key=key, image_path=img_path, gt_points=gt_points, meta=meta, gt_source=gt_source))
        return items

    if args.orad_root is None:
        raise SystemExit("Provide either --image (repeatable) or --orad-root for ORAD-3D sampling.")

    pairs = _iter_orad_pairs(
        orad_root=args.orad_root,
        split=args.split,
        image_folder=args.image_folder,
        max_scan=args.max_scan,
    )
    if not pairs:
        raise SystemExit("No ORAD-3D images found for the given split/image-folder.")

    num_samples = _parse_num_samples(args.num_samples)

    rng = random.Random(args.seed)
    if num_samples is not None:
        rng.shuffle(pairs)

    for seq, ts, img_path in pairs:
        key = f"{args.split}_{seq}_{ts}"
        meta = {"source": "orad3d", "split": args.split, "sequence": seq, "timestamp": ts}
        gt_points = None
        gt_source = None
        gt_local = _load_gt_from_local_path(
            img_path=img_path, orad_root=args.orad_root, meta=meta, gt_key=args.gt_key
        )
        if gt_local is not None:
            gt_points, gt_source = gt_local
        if gt_points is None:
            for cand in _candidate_image_keys(img_path, orad_root=args.orad_root, meta=meta):
                if cand in gt_map:
                    gt_points = gt_map[cand]
                    gt_source = str(args.gt_jsonl) if args.gt_jsonl else None
                    break
        if gt_points is None:
            continue
        items.append(SampleItem(key=key, image_path=img_path, gt_points=gt_points, meta=meta, gt_source=gt_source))
        if num_samples is not None and len(items) >= num_samples:
            break

    return items


def _run_inference_for_adapter(
    *,
    adapter: AdapterSpec,
    items: Sequence[SampleItem],
    args: argparse.Namespace,
) -> Dict[str, ModelOutput]:
    model, processor = _load_model_and_processor(
        base_model=args.base_model,
        adapter=adapter.path,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )

    results: Dict[str, ModelOutput] = {}
    skipped_f = None
    if bool(args.debug_save_skipped):
        skipped_path = args.out_dir / f"skipped_outputs_{adapter.name}.jsonl"
        skipped_f = skipped_path.open("w", encoding="utf-8")

    for idx, item in enumerate(items, start=1):
        try:
            image = Image.open(item.image_path).convert("RGB")
        except Exception:
            results[item.key] = ModelOutput(
                name=adapter.name,
                adapter_path=adapter.path,
                output_text="",
                trajectory_points=[],
                valid=False,
            )
            continue

        try:
            inputs, input_len = _prepare_inputs(
                processor,
                image=image,
                system_text=args.system,
                prompt_text=args.prompt,
                use_sharegpt_format=bool(args.use_sharegpt_format),
            )
            if torch.cuda.is_available():
                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to("cuda")

            gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(args.max_new_tokens)}
            if float(args.temperature) > 0:
                gen_kwargs.update({"do_sample": True, "temperature": float(args.temperature), "top_p": float(args.top_p)})
            else:
                gen_kwargs.update({"do_sample": False})

            with torch.inference_mode():
                out_ids = model.generate(**inputs, **gen_kwargs)

            skip_special = bool(args.skip_special_tokens)
            try:
                full_text = processor.batch_decode(out_ids, skip_special_tokens=skip_special)[0]
            except Exception:
                full_text = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=skip_special)[0]  # type: ignore

            gen_ids = out_ids
            if input_len > 0 and isinstance(out_ids, torch.Tensor) and out_ids.ndim == 2 and out_ids.shape[1] > input_len:
                gen_ids = out_ids[:, input_len:]

            try:
                out_text = processor.batch_decode(gen_ids, skip_special_tokens=skip_special)[0]
            except Exception:
                out_text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=skip_special)[0]  # type: ignore

            out_text = _clean_output_text((out_text or "").strip())
            full_text = _clean_output_text((full_text or "").strip())
            traj_section = _extract_trajectory_section(out_text)

            traj_points: List[List[float]] = []
            valid = False
            if traj_section is not None:
                traj_points = _extract_trajectory_points(traj_section)
                valid = len(traj_points) >= 2

        except Exception:
            out_text = ""
            full_text = ""
            traj_points = []
            valid = False

        if not valid and bool(args.debug_print_skipped):
            head = (out_text[:220] + ("..." if len(out_text) > 220 else "")).replace("\n", "\\n")
            print(f"[SKIP] {adapter.name} {idx}/{len(items)} {item.key}: {head}")

        if not valid and skipped_f is not None:
            skipped_f.write(
                json.dumps(
                    {
                        "key": item.key,
                        "image_path": str(item.image_path),
                        "output_text": out_text,
                        "full_text": full_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        results[item.key] = ModelOutput(
            name=adapter.name,
            adapter_path=adapter.path,
            output_text=out_text,
            trajectory_points=traj_points,
            valid=valid,
        )

    if skipped_f is not None:
        skipped_f.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _parse_adapter_specs(values: Sequence[str]) -> List[AdapterSpec]:
    if not values:
        raise SystemExit("Provide --adapter name=path (repeatable) for at least 2 models.")
    specs: List[AdapterSpec] = []
    seen: set[str] = set()
    for idx, raw in enumerate(values):
        name = raw
        path = raw
        if "=" in raw:
            name, path = raw.split("=", 1)
        name = (name or "").strip() or f"model{idx + 1}"
        path = (path or "").strip()
        if not path:
            raise SystemExit(f"Invalid adapter spec: {raw}")
        if not Path(path).exists():
            raise SystemExit(f"Adapter not found: {path}")
        base_name = name
        suffix = 2
        while name in seen:
            name = f"{base_name}_{suffix}"
            suffix += 1
        seen.add(name)
        specs.append(AdapterSpec(name=name, path=path))
    return specs


def _write_run_config(args: argparse.Namespace, adapters: Sequence[AdapterSpec]) -> None:
    payload = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            payload[k] = str(v)
        else:
            payload[k] = v
    payload["adapters"] = [asdict(spec) for spec in adapters]
    out_path = args.out_dir / "run_config.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export ORAD-3D LoRA adapter predictions for later evaluation.")
    ap.add_argument("--base-model", type=str, default=_DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="Adapter spec as name=path (repeatable).",
    )
    ap.add_argument("--cache-dir", type=str, default="/home/work/byounggun/.cache/hf")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument(
        "--system",
        type=str,
        default=(
            "You are an off-road autonomous driving agent. "
            "Given an input camera image, describe the scene and provide a safe drivable trajectory. "
            "Output the trajectory after a <trajectory> token as a comma-separated list of [x,y,z] points."
        ),
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="I am seeing an off-road driving image. Please generate a safe drivable trajectory for my vehicle to follow.",
    )
    ap.add_argument("--use-sharegpt-format", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--skip-special-tokens", action="store_true")

    ap.add_argument("--debug-save-skipped", action="store_true")
    ap.add_argument("--debug-print-skipped", action="store_true")

    ap.add_argument("--image", action="append", default=None)
    ap.add_argument("--orad-root", type=Path, default=None)
    ap.add_argument("--gt-jsonl", type=Path, default=None)
    ap.add_argument("--gt-key", type=str, default="trajectory_ins")
    ap.add_argument("--split", type=str, default="testing", choices=["training", "validation", "testing"])
    ap.add_argument("--image-folder", type=str, default="image_data", choices=["image_data", "gt_image"])
    ap.add_argument("--num-samples", type=str, default="all", help="Number of samples (use all for full split).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scan", type=int, default=None)
    ap.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Keep samples even if some models have no parsed trajectory.",
    )

    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--output-name", type=str, default="predictions.jsonl")
    ap.add_argument("--skip-run-config", action="store_true", help="Do not write run_config.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adapters = _parse_adapter_specs(args.adapter)
    if len(adapters) < 1:
        raise SystemExit("Provide at least one adapter.")

    gt_map: Dict[str, List[List[float]]] = {}
    if args.gt_jsonl is not None:
        wanted: set[str] = set()
        if args.image:
            for p in args.image:
                img_path = Path(p)
                wanted.update(_candidate_image_keys(img_path, orad_root=args.orad_root, meta={"source": "image"}))
        elif args.orad_root is not None:
            pairs = _iter_orad_pairs(
                orad_root=args.orad_root,
                split=args.split,
                image_folder=args.image_folder,
                max_scan=args.max_scan,
            )
            for seq, ts, img_path in pairs:
                meta = {"source": "orad3d", "split": args.split, "sequence": seq, "timestamp": ts}
                wanted.update(_candidate_image_keys(img_path, orad_root=args.orad_root, meta=meta))
        gt_map = _load_gt_trajectories_for_items(args.gt_jsonl, wanted_keys=wanted)

    items = _sample_items(args=args, gt_map=gt_map)
    if not items:
        raise SystemExit("No samples with valid GT trajectories found.")

    if not bool(args.skip_run_config):
        _write_run_config(args, adapters)

    results_by_model: Dict[str, Dict[str, ModelOutput]] = {}
    for adapter in adapters:
        print(f"[LOAD] {adapter.name} -> {adapter.path}")
        results_by_model[adapter.name] = _run_inference_for_adapter(adapter=adapter, items=items, args=args)

    out_path = args.out_dir / args.output_name
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            outputs: List[ModelOutput] = []
            for adapter in adapters:
                out = results_by_model.get(adapter.name, {}).get(item.key)
                if out is None:
                    out = ModelOutput(
                        name=adapter.name,
                        adapter_path=adapter.path,
                        output_text="",
                        trajectory_points=[],
                        valid=False,
                    )
                outputs.append(out)

            if not args.allow_missing_models:
                if any(not o.valid for o in outputs):
                    continue

            row = {
                "key": item.key,
                "image_path": str(item.image_path),
                "meta": item.meta,
                "gt_trajectory": item.gt_points,
                "gt_source": item.gt_source,
                "outputs": [asdict(o) for o in outputs],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[DONE] wrote {kept} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
