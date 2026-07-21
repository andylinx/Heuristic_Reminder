#!/usr/bin/env python3
"""Wrapper that documents how to run the attention extraction step."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: run the standalone attention extraction script"
    )
    parser.add_argument("--input", "-i", help="Input JSON file from Step 2")
    parser.add_argument("--output", "-o", help="Output JSON file for attention scores")
    parser.add_argument("--model-path", help="Path to the local Qwen2.5-VL model")
    parser.add_argument("--device", default="cuda:0", help="Torch device passed to calc_step_only.py")
    args = parser.parse_args()

    print(
        "\n".join(
            [
                "Use the standalone attention script directly:",
                "",
                "  python attn_calc/calc_step_only.py \\",
                f"    --model-path {args.model_path or '<model_path>'} \\",
                f"    --question-file {args.input or '<input_json>'} \\",
                f"    --output-file {args.output or '<output_json>'} \\",
                f"    --device {args.device}",
                "",
                "The input file should come from benign Step 2 and contain image paths plus questions.",
            ]
        )
    )


if __name__ == "__main__":
    main()
