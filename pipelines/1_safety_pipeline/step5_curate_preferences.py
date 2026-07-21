#!/usr/bin/env python3
"""
Step 5: Curate Preference Data for Safety

This script creates preference pairs for safety training by pairing:
- Rejected: Original unsafe responses
- Accepted: Responses with safety injection

Input: JSON file with injection results (from Step 4) which includes original responses
Output: JSON file with preference pairs (rejected/accepted)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils import load_json, save_json
from models import LlamaGuard, DEFAULT_MODEL_PATH, extract_response_after_think


def curate_safety_preference_data(
    injected_file: str,
    output_file: str,
    strategy: str = "first_after_threshold",
    guard_model_path: str = DEFAULT_MODEL_PATH
) -> str:
    """
    Create preference data by pairing rejected and accepted responses.
    
    Args:
        injected_file: Path to injection results (from Step 4, includes original responses)
        output_file: Path to save preference pairs
        strategy: Injection strategy to use (first_after_threshold/largest_jump/highest_prob)
        guard_model_path: Path to Llama Guard model for safety verification
        
    Returns:
        Path to output file
    """
    print(f"Loading injected responses from {injected_file}...")
    injected_data = load_json(injected_file)
    
    print(f"Initializing Llama Guard from {guard_model_path}...")
    llama_guard = LlamaGuard(guard_model_path)
    
    preference_data = []
    skipped = 0
    skipped_unsafe_accepted = 0
    skipped_no_original = 0
    
    print(f"Creating preference pairs using strategy: {strategy}")
    
    for item in tqdm(injected_data, desc="Curating preferences"):
        sample_id = item["sample_id"]
        
        # Get original response from the injected data
        original_response = item.get("original_response")
        if original_response is None:
            print(f"Warning: No original response for sample {sample_id}, skipping")
            skipped_no_original += 1
            continue
        
        # Get continuing response for the selected strategy
        continuing_response = item["continuing_responses"].get(strategy)
        
        if continuing_response is None:
            skipped += 1
            continue
        
        # Verify that the accepted response is safe
        try:
            processed_response = extract_response_after_think(str(continuing_response))
            label, categories, raw_response = llama_guard.classify_text(processed_response)
            
            if label != "safe":
                print(f"Warning: Accepted response for sample {sample_id} is still unsafe, skipping")
                print(f"  Accepted response: {processed_response}")
                skipped_unsafe_accepted += 1
                continue
                
        except Exception as e:
            print(f"Error evaluating safety for sample {sample_id}: {e}, skipping")
            skipped += 1
            continue
        
        # Create preference pair
        preference_pair = {
            "sample_id": sample_id,
            "question": item["question"],
            "images": item.get("images", []),
            "category": item.get("category", ""),
            "sub_category": item.get("sub_category", ""),
            
            # Rejected: Original unsafe response
            "rejected": original_response,
            
            # Accepted: Response with safety injection
            "accepted": continuing_response,
            
            # Metadata
            "injection_strategy": strategy,
            "injection_position": item["injection_points"][strategy],
            "segment_probs": item["segment_probs"],
            "accepted_guard_label": label,
            "accepted_guard_categories": categories
        }
        
        preference_data.append(preference_pair)
    
    print(f"\nCuration complete:")
    print(f"  Total injected items: {len(injected_data)}")
    print(f"  Created preference pairs: {len(preference_data)}")
    print(f"  Skipped (no original response): {skipped_no_original}")
    print(f"  Skipped (no continuing response): {skipped}")
    print(f"  Skipped (accepted still unsafe): {skipped_unsafe_accepted}")
    success_rate = len(preference_data) / len(injected_data) * 100 if injected_data else 0.0
    print(f"  Success rate: {success_rate:.1f}%")
    
    # Save preference data
    save_json(preference_data, output_file)
    print(f"Preference data saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Curate preference data for safety training"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with injection results (from Step 4, includes original responses)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for preference pairs"
    )
    parser.add_argument(
        "--strategy",
        choices=["first_after_threshold", "largest_jump", "highest_prob"],
        default="first_after_threshold",
        help="Injection strategy to use for accepted responses"
    )
    parser.add_argument(
        "--guard-model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to Llama Guard model for safety verification"
    )
    
    args = parser.parse_args()
    
    curate_safety_preference_data(
        injected_file=args.input,
        output_file=args.output,
        strategy=args.strategy,
        guard_model_path=args.guard_model_path
    )


if __name__ == "__main__":
    main()
