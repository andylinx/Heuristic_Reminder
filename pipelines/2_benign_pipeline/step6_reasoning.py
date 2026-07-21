#!/usr/bin/env python3
"""
Step 5: Rewrite Thinking Process

This script uses a vLLM-served model to rewrite thinking processes that contain
thoughts where the model seems to know the answer beforehand. It preserves genuine
reasoning while removing or rewriting suspicious pre-knowledge.

Input: JSON file with preference data
Output: JSON file with rewritten thinking processes
"""

import os
import sys
import json
import argparse
import re
from typing import Optional, Dict, Any, List
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from models import VLLMClient
from utils import load_json, save_json, save_progressive, load_progress


REWRITE_PROMPT = """You are a reasoning process editor. Your task is to review and rewrite a thinking process to ensure it represents genuine step-by-step reasoning without unnecessary repetition.

**Your Goal:**
Keep as much of the original thinking as possible, but identify and fix two main issues:
1. Remove or condense repetitive thoughts that restate the same ideas
2. Rewrite thoughts where the model appears to know the answer beforehand instead of reasoning through it naturally

**What to Remove or Condense:**
1. Repeated statements that say the same thing in different words
2. Redundant restatements of previous conclusions
3. Unnecessary repetition of intermediate results
4. Circular reasoning where the same logic is restated multiple times
5. Verbatim repetition of earlier thoughts

**What to Look For (Pre-knowledge Issues):**
1. Statements that jump to conclusions without showing the reasoning steps
2. Thoughts that reference the final answer before the reasoning is complete
3. Sudden insights without logical progression from prior thoughts

**What to Keep:**
1. Genuine exploratory thinking and hypothesis formation
2. Step-by-step logical deductions (stated once, clearly)
3. Calculations and verification steps
4. Natural corrections and reconsidering of approaches (but not repeated corrections)
5. Uncertainty and consideration of alternatives (without repeating the same alternatives)

**Instructions:**
- Read the original thinking carefully
- Remove repetitive content while preserving the logical flow
- Condense redundant statements into concise expressions
- Only rewrite portions that show pre-knowledge of the answer
- Ensure the rewritten thinking flows naturally and progresses logically
- Maintain the same level of detail in non-problematic sections
- The rewritten thinking should reach the same conclusion but through clearer, non-repetitive reasoning

**Original Thinking:**
{original_thinking}

Now, please output the rewritten thinking without any additional commentary.
"""


def extract_thinking(text: str) -> Optional[str]:
    """Extract content inside <think>...</think> tags."""
    if not isinstance(text, str):
        return None
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract content after </think> tag."""
    if not isinstance(text, str):
        return None
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def rewrite_thinking(
    client: VLLMClient,
    original_thinking: str,
    model_name: Optional[str] = None,
    max_tokens: int = 20480,
    temperature: float = 0.3
) -> Optional[str]:
    """
    Use vLLM model to rewrite thinking process.
    
    Args:
        client: VLLMClient instance
        original_thinking: Original thinking content (without tags)
        model_name: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more conservative)
        
    Returns:
        Rewritten thinking content or None if failed
    """
    try:
        prompt = REWRITE_PROMPT.format(original_thinking=original_thinking)
        
        result = client.generate_response(
            prompt=prompt,
            image_paths=None,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
        response = result["response"]
        
        return response.strip()
        
    except Exception as e:
        print(f"Error during rewriting: {e}")
        return None


def process_preference_data(
    input_file: str,
    output_file: str,
    vllm_url: str = "http://localhost:8000",
    model_name: Optional[str] = None,
    rewrite_field: str = "accepted",
    max_tokens: int = 20480,
    temperature: float = 0.3,
    save_interval: int = 10,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> str:
    """
    Process preference data and rewrite thinking processes.
    
    Args:
        input_file: Path to input JSON with preference data
        output_file: Path to save rewritten data
        vllm_url: vLLM server URL
        model_name: Model name
        rewrite_field: Which field to rewrite ("accepted" or "rejected")
        max_tokens: Maximum tokens for rewriting
        temperature: Sampling temperature
        save_interval: Save progress every N items
        start_idx: Start processing from this index
        end_idx: End processing at this index (None = process all)
        
    Returns:
        Path to output file
    """
    print(f"Loading data from {input_file}...")
    all_data = load_json(input_file)
    
    # Filter by indices
    if end_idx is None:
        end_idx = len(all_data)
    data = all_data[start_idx:end_idx]
    
    print(f"Processing {len(data)} items (indices {start_idx}-{end_idx-1})")
    print(f"Connecting to vLLM at {vllm_url}...")
    client = VLLMClient(vllm_url)
    
    # Check for existing progress
    processed_items, resume_idx = load_progress(output_file)
    if processed_items:
        print(f"Resuming from index {resume_idx} ({len(processed_items)} items already processed)")
    else:
        processed_items = []
        resume_idx = 0
    
    skipped_no_think = 0
    skipped_error = 0
    rewritten_count = 0
    
    print(f"Rewriting thinking processes in '{rewrite_field}' field...")
    
    for idx in tqdm(range(resume_idx, len(data)), desc="Rewriting", initial=resume_idx, total=len(data)):
        item = data[idx]
        
        # Get the field to rewrite
        original_response = item.get(rewrite_field, "")
        
        if not original_response:
            print(f"Warning: No '{rewrite_field}' field in item {idx}, skipping")
            skipped_error += 1
            continue
        
        # Check for </think> tag
        if "</think>" not in original_response:
            print(f"Warning: No </think> tag in item {idx}, skipping")
            skipped_no_think += 1
            continue
        
        # Extract thinking and answer
        original_thinking = extract_thinking(original_response)
        answer_part = extract_answer(original_response)
        
        if not original_thinking:
            print(f"Warning: Could not extract thinking from item {idx}, skipping")
            skipped_error += 1
            continue
        
        # Rewrite thinking
        rewritten_thinking = rewrite_thinking(
            client=client,
            original_thinking=original_thinking,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if rewritten_thinking is None:
            print(f"Warning: Failed to rewrite item {idx}, keeping original")
            output_item = item.copy()
            output_item["rewritten"] = False
            output_item["rewrite_error"] = True
            processed_items.append(output_item)
            skipped_error += 1
            continue
        
        # Reconstruct response with rewritten thinking
        rewritten_response = f"<think>\n{rewritten_thinking}\n</think>\n{answer_part}"
        
        # Create output item
        output_item = item.copy()
        output_item[f"{rewrite_field}_original"] = original_response
        output_item[rewrite_field] = rewritten_response
        output_item["rewritten"] = True
        output_item["absolute_index"] = start_idx + idx
        
        processed_items.append(output_item)
        rewritten_count += 1
        
        # Progressive save
        if (idx + 1) % save_interval == 0:
            save_progressive(processed_items, output_file, idx, is_final=False)
    
    # Final save
    save_progressive(processed_items, output_file, len(data) - 1, is_final=True)
    
    print(f"\nCompleted!")
    print(f"  Total items: {len(data)}")
    print(f"  Successfully rewritten: {rewritten_count}")
    print(f"  Skipped (no </think>): {skipped_no_think}")
    print(f"  Skipped (errors): {skipped_error}")
    print(f"Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Rewrite thinking processes to remove pre-knowledge"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with preference data"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for rewritten data"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL for rewriting model"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name (default: auto-detect from the vLLM server)"
    )
    parser.add_argument(
        "--field",
        choices=["accepted", "rejected"],
        default="accepted",
        help="Which field to rewrite (default: accepted)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20480,
        help="Maximum tokens for rewriting"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (lower = more conservative)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save progress every N items"
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
    
    process_preference_data(
        input_file=args.input,
        output_file=args.output,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        rewrite_field=args.field,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        save_interval=args.save_interval,
        start_idx=args.start,
        end_idx=args.end
    )


if __name__ == "__main__":
    main()
