#!/usr/bin/env python3

"""Verify whether <trajectory> survives into training labels.

This script loads the training YAML config, reads a few raw ShareGPT samples from the
dataset JSONL, then runs the exact LLaMAFactory ShareGPT converter + SFT dataset processor
to obtain `input_ids`/`labels` after truncation (cutoff_len).

It reports whether the assistant labels still contain `<trajectory>`. This is useful when
the `<trajectory>` section is placed near the end of the assistant response and may be
truncated away by `cutoff_len` (especially for VLMs where image tokens consume a lot of
budget).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields

from omegaconf import OmegaConf

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.model_args import ModelArguments
from llamafactory.model import load_tokenizer


def _load_config(path: str) -> dict:
	cfg = OmegaConf.load(path)
	cfg = OmegaConf.to_container(cfg, resolve=True)
	if not isinstance(cfg, dict):
		raise ValueError(f"Config at {path} must be a mapping, got: {type(cfg)}")
	return cfg


def _dataclass_from_config(dataclass_cls, cfg: dict):
	allowed = {f.name for f in fields(dataclass_cls)}
	kwargs = {k: v for k, v in cfg.items() if k in allowed}
	return dataclass_cls(**kwargs)


def _iter_jsonl(path: str, start: int, limit: int):
	with open(path, "r", encoding="utf-8") as f:
		for _ in range(start):
			line = f.readline()
			if not line:
				return

		count = 0
		while limit is None or count < limit:
			line = f.readline()
			if not line:
				return

			line = line.strip()
			if not line:
				continue

			yield json.loads(line)
			count += 1


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		default="examples/train_lora/orad3d_qwen3vl_2b_lora_sft.yaml",
		help="Training YAML config used for preprocessing (cutoff_len, template, media_dir, etc).",
	)
	parser.add_argument("--start", type=int, default=0, help="Skip this many JSONL lines before sampling.")
	parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to inspect.")
	parser.add_argument(
		"--show-label-chars",
		type=int,
		default=500,
		help="If >0, show the tail of decoded labels (after removing IGNORE_INDEX).",
	)
	parser.add_argument(
		"--show-raw-chars",
		type=int,
		default=300,
		help="If >0, show the tail of the raw assistant content.",
	)
	args = parser.parse_args()

	cfg = _load_config(args.config)
	# NOTE: We intentionally do NOT call `get_train_args()` here.
	# `get_train_args()` enforces distributed launch for training runs, but this script only needs
	# tokenization + preprocessing, which is safe to run in a single process.
	model_args = _dataclass_from_config(ModelArguments, cfg)
	data_args = _dataclass_from_config(DataArguments, cfg)

	stage = cfg.get("stage", "sft")
	if stage != "sft":
		raise ValueError(f"This verifier is for SFT. Got stage={stage!r}.")

	if data_args.dataset is None:
		dataset_names = []
	elif isinstance(data_args.dataset, list):
		dataset_names = [str(x).strip() for x in data_args.dataset if str(x).strip()]
	else:
		dataset_names = [x.strip() for x in str(data_args.dataset).split(",") if x.strip()]
	if len(dataset_names) != 1:
		raise ValueError(f"Expected exactly 1 dataset, got: {dataset_names}")

	dataset_attr = get_dataset_list(dataset_names, data_args.dataset_dir)[0]
	if dataset_attr.load_from != "file":
		raise ValueError(
			f"This verifier currently supports load_from='file'. Got {dataset_attr.load_from!r} for {dataset_attr}."
		)

	dataset_path = dataset_attr.dataset_name
	tokenizer_module = load_tokenizer(model_args)
	tokenizer = tokenizer_module["tokenizer"]
	processor = tokenizer_module["processor"]
	template = get_template_and_fix_tokenizer(tokenizer, data_args)

	converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
	if data_args.packing:
		dataset_processor = PackedSupervisedDatasetProcessor(template, tokenizer, processor, data_args)
	else:
		dataset_processor = SupervisedDatasetProcessor(template, tokenizer, processor, data_args)

	image_pad_id = None
	try:
		image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
	except Exception:
		image_pad_id = None

	summary_cfg = {
		"config": args.config,
		"dataset": dataset_names[0],
		"dataset_path": dataset_path,
		"cutoff_len": data_args.cutoff_len,
		"template": data_args.template,
		"media_dir": data_args.media_dir,
		"image_max_pixels": getattr(model_args, "image_max_pixels", None),
		"video_max_pixels": getattr(model_args, "video_max_pixels", None),
		"add_special_tokens": getattr(model_args, "add_special_tokens", None),
	}
	print("=== Verifier config ===")
	print(json.dumps(summary_cfg, ensure_ascii=False, indent=2))

	found_raw = 0
	found_labels = 0
	truncated_suspects = 0

	for i, raw in enumerate(_iter_jsonl(dataset_path, start=args.start, limit=args.num_samples)):
		converted = converter(raw)
		response = converted.get("_response") or []
		assistant_text = response[-1].get("content", "") if response else ""

		raw_has = "<trajectory>" in assistant_text
		if raw_has:
			found_raw += 1

		batch = {
			"_prompt": [converted.get("_prompt")],
			"_response": [converted.get("_response")],
			"_system": [converted.get("_system")],
			"_tools": [converted.get("_tools")],
			"_images": [converted.get("_images")],
			"_videos": [converted.get("_videos")],
			"_audios": [converted.get("_audios")],
		}

		processed = dataset_processor.preprocess_dataset(batch)
		if not processed.get("input_ids"):
			print(f"[{i}] dropped during preprocess (invalid example).")
			continue

		input_ids = processed["input_ids"][0]
		labels = processed["labels"][0]
		valid_labels = [x for x in labels if x != IGNORE_INDEX]
		decoded_labels = tokenizer.decode(valid_labels, skip_special_tokens=False)

		label_has = "<trajectory>" in decoded_labels
		if label_has:
			found_labels += 1

		num_image_tokens = None
		if image_pad_id is not None and isinstance(image_pad_id, int) and image_pad_id >= 0:
			num_image_tokens = sum(1 for t in input_ids if t == image_pad_id)

		is_truncated = raw_has and (not label_has)
		if is_truncated:
			truncated_suspects += 1

		print("\n--- sample {} ---".format(i))
		print(
			json.dumps(
				{
					"seq_len": len(input_ids),
					"valid_label_len": len(valid_labels),
					"num_image_pad_tokens": num_image_tokens,
					"raw_has_trajectory": raw_has,
					"labels_has_trajectory": label_has,
					"suspect_truncation": is_truncated,
				},
				ensure_ascii=False,
				indent=2,
			)
		)

		if args.show_raw_chars and args.show_raw_chars > 0:
			tail = assistant_text[-args.show_raw_chars :]
			print(f"raw_assistant_tail({args.show_raw_chars} chars):\n{tail}")

		if args.show_label_chars and args.show_label_chars > 0:
			tail = decoded_labels[-args.show_label_chars :]
			print(f"decoded_labels_tail({args.show_label_chars} chars):\n{tail}")

	print("\n=== Summary ===")
	print(
		json.dumps(
			{
				"checked": args.num_samples,
				"raw_has_trajectory": found_raw,
				"labels_has_trajectory": found_labels,
				"suspect_truncation": truncated_suspects,
			},
			ensure_ascii=False,
			indent=2,
		)
	)

	if truncated_suspects > 0:
		print(
			"\n[Recommendation] Some samples likely lost `<trajectory>` in labels due to truncation. "
			"For Qwen3-VL, image tokens can be large. Consider increasing `cutoff_len` (e.g., 4096/8192), "
			"reducing `image_max_pixels`, or moving `<trajectory>` earlier in the assistant response."
		)


if __name__ == "__main__":
	main()
