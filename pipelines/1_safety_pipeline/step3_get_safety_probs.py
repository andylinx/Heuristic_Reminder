#!/usr/bin/env python3
"""
Step 3: Calculate Step-Level Safety Probabilities

This script computes progressive unsafe probabilities for each reasoning step.
It splits the thinking process into segments and evaluates safety at each step.

Input: JSON file with unsafe responses (from Step 2)
Output: JSON file with segment-level safety probabilities
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from models import LlamaGuard, DEFAULT_MODEL_PATH
from utils import load_json, save_json


# Regex to extract thinking content
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_think(text: str) -> Optional[str]:
    """Extract content inside <think>...</think> tags."""
    if not isinstance(text, str):
        return None
    
    # If </think> exists but <think> doesn't, add <think> at the beginning
    if "</think>" in text and "<think>" not in text:
        text = "<think>" + text
    
    m = THINK_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def split_think_into_segments(think_text: str) -> List[str]:
    """
    Split thinking content into segments by double newlines.
    
    Args:
        think_text: Thinking content string
        
    Returns:
        List of thinking segments
    """
    parts = [p.strip() for p in think_text.split("\n\n")]
    return [p for p in parts if p]


def compute_progressive_safety_probs(
    input_file: str,
    output_file: str,
    guard_model_path: str = DEFAULT_MODEL_PATH,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> str:
    """
    Compute progressive safety probabilities for each thinking segment.
    
    Args:
        input_file: Path to unsafe responses JSON (from Step 2)
        output_file: Path to save segment probabilities
        guard_model_path: Path to Llama Guard model
        start_idx: Start processing from this index
        end_idx: End processing at this index (None = process all)
        
    Returns:
        Path to output file
    """
    print(f"Loading unsafe items from {input_file}...")
    all_data = load_json(input_file)
    
    # Filter by indices
    if end_idx is None:
        end_idx = len(all_data)
    data = all_data[start_idx:end_idx]
    
    print(f"Processing {len(data)} items (indices {start_idx}-{end_idx-1})")
    
    print(f"Initializing Llama Guard from {guard_model_path}...")
    llama_guard = LlamaGuard(guard_model_path)
    
    results = []
    
    print("Computing progressive safety probabilities...")
    for idx, item in enumerate(tqdm(data, desc="Processing items")):
        question = item.get("question", "")
        response = item.get("response", "")
        
        # Extract thinking content
        think_text = extract_think(response)
        if not think_text:
            print(f"Warning: No <think> block found in item, skipping")
            continue
        
        # Split into segments
        segments = split_think_into_segments(think_text)
        if not segments:
            print(f"Warning: No segments found, skipping")
            continue
        
        # Compute progressive probabilities
        segment_probs = []
        accumulated_thinking = []
        
        for seg_idx, segment in enumerate(segments):
            accumulated_thinking.append(segment)
            accumulated_text = "\n\n".join(accumulated_thinking)
            
            # Get unsafe probability for accumulated thinking
            try:
                image_path = None
                if item.get("image_paths") and len(item["image_paths"]) > 0:
                    image_path = item["image_paths"][0]
                
                unsafe_prob = llama_guard.get_unsafe_probability(
                    accumulated_text,
                    image_path
                )
                segment_probs.append(unsafe_prob)
                
            except Exception as e:
                print(f"Error computing probability for segment {seg_idx}: {e}")
                segment_probs.append(0.0)
        
        # Create output item
        output_item = {
            "sample_id": item.get("index", item.get("sample_id")),
            "index": start_idx + idx,
            "question": question,
            "images": item.get("images", []),
            "original_response": response,
            "segment_probs": segment_probs,
            "category": item.get("category", ""),
            "sub_category": item.get("sub_category", ""),
            "guard_label": item.get("guard_label", ""),
            "guard_categories": item.get("guard_categories", []),
            "source": item.get("source", "")
        }
        
        results.append(output_item)
    
    print(f"\nCompleted! Processed {len(results)} items with safety probabilities")
    
    # Save results
    save_json(results, output_file)
    print(f"Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Calculate step-level safety probabilities"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with unsafe responses (from Step 2)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for segment probabilities"
    )
    parser.add_argument(
        "--guard-model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to Llama Guard model"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index"
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index (default: process all)"
    )
    
    args = parser.parse_args()
    
    compute_progressive_safety_probs(
        input_file=args.input,
        output_file=args.output,
        guard_model_path=args.guard_model_path,
        start_idx=args.start,
        end_idx=args.end
    )


if __name__ == "__main__":
    main()
