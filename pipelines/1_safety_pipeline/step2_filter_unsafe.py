#!/usr/bin/env python3
"""
Step 2: Filter Unsafe Responses

This script uses Llama Guard to identify and filter responses that contain unsafe content.
Only unsafe responses will be kept for further processing.

Input: JSON file with model responses (from Step 1)
Output: JSON file with only unsafe responses
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from models import LlamaGuard, DEFAULT_MODEL_PATH, parse_guard_output, extract_response_after_think
from utils import load_json, save_json


def filter_unsafe_responses(
    input_file: str,
    output_file: str,
    guard_model_path: str = DEFAULT_MODEL_PATH
) -> str:
    """
    Filter responses to keep only unsafe ones.
    
    Args:
        input_file: Path to responses JSON from Step 1
        output_file: Path to save filtered unsafe responses
        guard_model_path: Path to Llama Guard model
        
    Returns:
        Path to output file
    """
    print(f"Loading responses from {input_file}...")
    data = load_json(input_file)
    
    print(f"Initializing Llama Guard from {guard_model_path}...")
    llama_guard = LlamaGuard(guard_model_path)
    
    unsafe_items = []
    
    print("Filtering for unsafe responses...")
    for item in tqdm(data, desc="Evaluating safety"):
        response = item.get("response", "")
        
        # Extract actual response (remove thinking tags if present)
        processed_response = extract_response_after_think(str(response))
        
        # Get guard evaluation
        try:
            label, categories, raw_response = llama_guard.classify_text(processed_response)
            
            item["guard_label"] = label
            item["guard_categories"] = categories
            item["guard_raw_response"] = raw_response
            
            # Keep only unsafe responses
            if label == "unsafe":
                unsafe_items.append(item)
                
        except Exception as e:
            print(f"Error evaluating item: {e}")
            continue
    
    print(f"\nFiltering complete:")
    print(f"  Total responses: {len(data)}")
    print(f"  Unsafe responses: {len(unsafe_items)} ({len(unsafe_items)/len(data)*100:.1f}%)")
    
    # Save unsafe items
    save_json(unsafe_items, output_file)
    print(f"Unsafe responses saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Filter unsafe responses using Llama Guard"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with responses (from Step 1)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for unsafe responses"
    )
    parser.add_argument(
        "--guard-model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to Llama Guard model"
    )
    
    args = parser.parse_args()
    
    filter_unsafe_responses(
        input_file=args.input,
        output_file=args.output,
        guard_model_path=args.guard_model_path
    )


if __name__ == "__main__":
    main()
