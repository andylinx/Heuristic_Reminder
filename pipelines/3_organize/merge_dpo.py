#!/usr/bin/env python3
"""Merge curated safety and benign preferences into ms-swift DPO JSONL."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def normalize_images(images: Any) -> List[str]:
    if not images:
        return []
    if isinstance(images, str):
        return [images]
    if not isinstance(images, Sequence):
        raise ValueError(f"Expected images to be a string or list, got {type(images).__name__}")
    return [str(image) for image in images if image]


def resolve_images(images: Any, data_base_path: str) -> List[str]:
    resolved = []
    for image_path in normalize_images(images):
        if os.path.isabs(image_path) or not data_base_path:
            resolved.append(image_path)
        else:
            resolved.append(os.path.join(data_base_path, image_path))
    return resolved


def add_image_placeholders(question: str, image_count: int) -> str:
    """Ensure the prompt contains exactly one ms-swift placeholder per image."""
    question = str(question).strip()
    placeholder_count = question.count("<image>")
    if placeholder_count == image_count:
        return question
    if placeholder_count == 0:
        prefix = "\n".join("<image>" for _ in range(image_count))
        return f"{prefix}\n{question}" if prefix else question
    raise ValueError(
        f"Prompt has {placeholder_count} <image> placeholders for {image_count} images"
    )


def build_record(item: Dict[str, Any], images: List[str]) -> Dict[str, Any]:
    question = item.get("question")
    accepted = item.get("accepted", item.get("full_response", ""))
    rejected = item.get("rejected", item.get("rejected_response", ""))
    if not question or not isinstance(accepted, str) or not accepted.strip():
        raise ValueError("Preference item is missing a non-empty question or accepted response")
    if not isinstance(rejected, str) or not rejected.strip():
        raise ValueError("Preference item is missing a non-empty rejected response")

    return {
        "messages": [
            {
                "role": "user",
                "content": add_image_placeholders(str(question), len(images)),
            },
            {"role": "assistant", "content": accepted},
        ],
        "images": images,
        "rejected_response": rejected,
    }


def iter_preferences(
    datasets: Iterable[List[Dict[str, Any]]],
    data_base_path: str,
    require_think: bool,
    skip_missing_images: bool,
) -> tuple[List[Dict[str, Any]], List[str], int]:
    records: List[Dict[str, Any]] = []
    missing_images: List[str] = []
    skipped_no_think = 0

    for dataset in datasets:
        for item in dataset:
            accepted = item.get("accepted", item.get("full_response", ""))
            rejected = item.get("rejected", item.get("rejected_response", ""))
            if require_think and (
                "</think>" not in str(accepted) or "</think>" not in str(rejected)
            ):
                skipped_no_think += 1
                continue

            images = resolve_images(item.get("images", []), data_base_path)
            item_missing = [path for path in images if not os.path.exists(path)]
            if item_missing:
                missing_images.extend(item_missing)
                if skip_missing_images:
                    continue
            records.append(build_record(item, images))

    return records, missing_images, skipped_no_think


def merge_dpo_data(
    safety_file: str,
    benign_file: str,
    output_file: str,
    data_base_path: str = "",
    seed: int = 42,
    require_think: bool = True,
    skip_missing_images: bool = False,
) -> str:
    records, missing_images, skipped_no_think = iter_preferences(
        datasets=[load_json(safety_file), load_json(benign_file)],
        data_base_path=data_base_path,
        require_think=require_think,
        skip_missing_images=skip_missing_images,
    )

    if missing_images and not skip_missing_images:
        preview = "\n".join(f"  - {path}" for path in sorted(set(missing_images))[:10])
        raise FileNotFoundError(
            f"Found {len(missing_images)} missing image references. "
            "Fix --data-base-path or pass --skip-missing-images:\n"
            f"{preview}"
        )
    if not records:
        raise ValueError("No valid DPO records were produced")

    random.Random(seed).shuffle(records)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")

    print(f"Merged {len(records)} DPO records into {output_file}")
    if skipped_no_think:
        print(f"Skipped {skipped_no_think} records without complete <think> traces")
    if missing_images:
        print(f"Skipped {len(missing_images)} missing image references")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge safety and benign preference JSON into ms-swift DPO JSONL"
    )
    parser.add_argument("--safety", required=True, help="Safety preference JSON file")
    parser.add_argument("--benign", required=True, help="Benign preference JSON file")
    parser.add_argument("--output", required=True, help="Output DPO JSONL file")
    parser.add_argument(
        "--data-base-path",
        default="",
        help="Optional base path used to resolve relative image paths",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--allow-no-think",
        action="store_true",
        help="Keep pairs that do not contain </think> in both responses",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip records with missing images instead of failing",
    )
    args = parser.parse_args()

    merge_dpo_data(
        safety_file=args.safety,
        benign_file=args.benign,
        output_file=args.output,
        data_base_path=args.data_base_path,
        seed=args.seed,
        require_think=not args.allow_no_think,
        skip_missing_images=args.skip_missing_images,
    )


if __name__ == "__main__":
    main()
