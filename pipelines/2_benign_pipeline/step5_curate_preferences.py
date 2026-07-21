#!/usr/bin/env python3
"""
Step 5: Curate Preference Data for Benign Training

This script creates preference pairs for benign training by pairing:
- Rejected: Original responses (possibly with attention drift)
- Accepted: Responses with image-recall injection

Input: JSON file with injection results (from Step 4) and original responses (from Step 2)
Output: JSON file with preference pairs (rejected/accepted)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils import load_json, save_json


def curate_benign_preference_data(
    injected_file: str,
    original_file: str,
    output_file: str
) -> str:
    """
    Create preference data by pairing rejected and accepted responses.
    
    Args:
        injected_file: Path to injection results (from Step 4)
        original_file: Path to original responses (from Step 2)
        output_file: Path to save preference pairs
        
    Returns:
        Path to output file
    """
    print(f"Loading injected responses from {injected_file}...")
    injected_data = load_json(injected_file)
    
    print(f"Loading original responses from {original_file}...")
    original_data = load_json(original_file)
    
    # Index original data by sample_id
    original_by_id = {}
    for item in original_data:
        sample_id = item.get("index", item.get("sample_id"))
        if sample_id is not None:
            original_by_id[sample_id] = item
    
    print(f"Indexed {len(original_by_id)} original responses")
    
    preference_data = []
    skipped = 0
    
    print("Creating preference pairs...")
    
    for item in injected_data:
        sample_id = item["sample_id"]
        
        # Get original response
        if sample_id not in original_by_id:
            print(f"Warning: No original response for sample {sample_id}, skipping")
            skipped += 1
            continue
        
        original_item = original_by_id[sample_id]
        original_response = original_item.get("responses", [""])[0]  # Take first response
        
        # Get injection results
        injection_results = item.get("injection_results", [])
        
        if not injection_results:
            print(f"Warning: No injection results for sample {sample_id}, skipping")
            skipped += 1
            continue
        
        # Create one preference pair per injection point
        for result in injection_results:
            continuing_response = result.get("continuing_response")
            
            if continuing_response is None:
                continue
            
            # Create preference pair
            preference_pair = {
                "sample_id": sample_id,
                "question": item["question"],
                "images": item.get("images", []),
                "source": original_item.get("source", ""),
                
                # Rejected: Original response (may have attention drift)
                "rejected": original_response,
                
                # Accepted: Response with image-recall injection
                "accepted": continuing_response,
                
                # Metadata
                "injection_position": result["injection_position"],
                "injection_strategy": item.get("injection_strategy", "unknown")
            }
            
            preference_data.append(preference_pair)
    
    print(f"\nCuration complete:")
    print(f"  Total injected items: {len(injected_data)}")
    print(f"  Total original items: {len(original_data)}")
    print(f"  Created preference pairs: {len(preference_data)}")
    print(f"  Skipped items: {skipped}")
    
    # Save preference data
    save_json(preference_data, output_file)
    print(f"Preference data saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Curate preference data for benign training"
    )
    parser.add_argument(
        "--injected", "-j",
        required=True,
        help="Input JSON file with injection results (from Step 4)"
    )
    parser.add_argument(
        "--original", "-r",
        required=True,
        help="Input JSON file with original responses (from Step 2)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for preference pairs"
    )
    
    args = parser.parse_args()
    
    curate_benign_preference_data(
        injected_file=args.injected,
        original_file=args.original,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
