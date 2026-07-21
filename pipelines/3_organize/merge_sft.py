#!/usr/bin/env python3
"""Merge curated safety and benign data into a JSONL file for SFT."""

import argparse
import json
import os
import random
from typing import Any, Dict, List


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_images(images: Any, data_base_path: str) -> List[str]:
    if isinstance(images, str):
        images = [images]
    resolved = []
    for image_path in images or []:
        if os.path.isabs(image_path) or not data_base_path:
            resolved.append(image_path)
        else:
            resolved.append(os.path.join(data_base_path, image_path))
    return resolved


def add_image_placeholders(question: str, image_count: int) -> str:
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


def build_record(question: str, response: str, images: List[str]) -> Dict:
    return {
        "messages": [
            {"role": "user", "content": add_image_placeholders(question, len(images))},
            {"role": "assistant", "content": response},
        ],
        "images": images,
    }


def merge_sft_data(
    safety_file: str,
    benign_file: str,
    output_file: str,
    data_base_path: str = "",
    seed: int = 42,
) -> str:
    safety_data = load_json(safety_file)
    benign_data = load_json(benign_file)

    merged_records: List[Dict] = []
    missing_images: List[str] = []

    for item in safety_data:
        response = item.get("accepted", "")
        if "</think>" not in response:
            continue
        image_paths = resolve_images(item.get("images", []), data_base_path)
        if any(not os.path.exists(path) for path in image_paths):
            missing_images.extend(path for path in image_paths if not os.path.exists(path))
            continue
        merged_records.append(build_record(item["question"], response, image_paths))

    for item in benign_data:
        response = item.get("accepted", item.get("full_response", ""))
        if "</think>" not in response:
            continue
        image_paths = resolve_images(item.get("images", []), data_base_path)
        if any(not os.path.exists(path) for path in image_paths):
            missing_images.extend(path for path in image_paths if not os.path.exists(path))
            continue
        merged_records.append(build_record(item["question"], response, image_paths))

    random.Random(seed).shuffle(merged_records)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as handle:
        for item in merged_records:
            json.dump(item, handle, ensure_ascii=False)
            handle.write("\n")

    print(f"Merged {len(merged_records)} records into {output_file}")
    if missing_images:
        print(f"Skipped {len(missing_images)} missing image references")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge safety and benign data into JSONL for SFT")
    parser.add_argument("--safety", required=True, help="Safety preference JSON file")
    parser.add_argument("--benign", required=True, help="Benign preference JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--data-base-path",
        default="",
        help="Optional base path used to resolve relative image paths",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    merge_sft_data(
        safety_file=args.safety,
        benign_file=args.benign,
        output_file=args.output,
        data_base_path=args.data_base_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
