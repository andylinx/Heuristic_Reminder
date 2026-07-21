#!/usr/bin/env python3
"""
Step 7: Refine Answer to Match Thinking

This script uses a vLLM-served model to refine answers to ensure they are coherent
with the thinking process. It makes minimal modifications to align the answer with
the reasoning while preserving the original response as much as possible.

Input: JSON file with rewritten thinking data
Output: JSON file with refined answers
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


REFINE_ANSWER_PROMPT = """You are an answer refinement specialist. Your task is to review a thinking process and its corresponding answer, then refine the answer to ensure it is coherent with the thinking while making the LEAST modifications possible.

**Your Goal:**
Make minimal changes to the answer to ensure:
1. The answer logically follows from the thinking process
2. Key conclusions from the thinking are reflected in the answer
3. The tone and style remain consistent
4. No contradictions exist between thinking and answer

**What to Check:**
1. **Logical Coherence**: Does the answer align with the conclusions in the thinking?
2. **Completeness**: Does the answer address what the thinking analyzed?
3. **Consistency**: Are there any contradictions between thinking and answer?
4. **Clarity**: Is the answer clear and well-structured based on the reasoning?

**What to Modify (MINIMALLY):**
1. Fix any contradictions between thinking and answer
2. Adjust the answer if it doesn't reflect the thinking's conclusion
3. Add brief clarifications if the thinking reveals important nuances not in the answer
4. Improve structure if the thinking suggests a better organization

**What to PRESERVE:**
1. The original wording and phrasing as much as possible
2. The answer's length (don't make it significantly longer or shorter)
3. The writing style and tone
4. Specific examples, numbers, or facts mentioned
5. The core content of the answer

**Critical Rule:**
If the answer is already coherent with the thinking, return it UNCHANGED or with only minimal adjustments. Do not rewrite unnecessarily.

**Thinking Process:**
{thinking}

**Current Answer:**
{answer}

Now, please output the refined answer without any additional commentary. Remember: make the LEAST modifications necessary.
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


def refine_answer(
    client: VLLMClient,
    thinking: str,
    answer: str,
    model_name: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.2
) -> Optional[str]:
    """
    Use vLLM model to refine answer to match thinking.
    
    Args:
        client: VLLMClient instance
        thinking: Thinking process content
        answer: Original answer content
        model_name: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more conservative)
        
    Returns:
        Refined answer content or None if failed
    """
    try:
        prompt = REFINE_ANSWER_PROMPT.format(thinking=thinking, answer=answer)
        
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
        print(f"Error during answer refinement: {e}")
        return None


def process_answer_refinement(
    input_file: str,
    output_file: str,
    vllm_url: str = "http://localhost:8000",
    model_name: Optional[str] = None,
    refine_field: str = "accepted",
    max_tokens: int = 8192,
    temperature: float = 0.2,
    save_interval: int = 200,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> str:
    """
    Process data and refine answers to match thinking.
    
    Args:
        input_file: Path to input JSON with rewritten thinking
        output_file: Path to save refined data
        vllm_url: vLLM server URL
        model_name: Model name
        refine_field: Which field to refine ("accepted" or "rejected")
        max_tokens: Maximum tokens for refinement
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
    refined_count = 0
    
    print(f"Refining answers in '{refine_field}' field...")
    
    for idx in tqdm(range(resume_idx, len(data)), desc="Refining Answers", initial=resume_idx, total=len(data)):
        item = data[idx]
        
        # Get the field to refine
        full_response = item.get(refine_field, "")
        
        if not full_response:
            print(f"Warning: No '{refine_field}' field in item {idx}, skipping")
            skipped_error += 1
            continue
        
        # Check for </think> tag
        if "</think>" not in full_response:
            print(f"Warning: No </think> tag in item {idx}, skipping")
            skipped_no_think += 1
            continue
        
        # Extract thinking and answer
        thinking = extract_thinking(full_response)
        answer = extract_answer(full_response)
        
        if not thinking or not answer:
            print(f"Warning: Could not extract thinking or answer from item {idx}, skipping")
            skipped_error += 1
            continue
        
        # Refine answer
        refined_answer = refine_answer(
            client=client,
            thinking=thinking,
            answer=answer,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if refined_answer is None:
            print(f"Warning: Failed to refine item {idx}, keeping original")
            output_item = item.copy()
            output_item["answer_refined"] = False
            output_item["refine_error"] = True
            processed_items.append(output_item)
            skipped_error += 1
            continue
        
        # Reconstruct response with refined answer
        refined_response = f"<think>\n{thinking}\n</think>\n{refined_answer}"
        
        # Create output item
        output_item = item.copy()
        output_item[f"{refine_field}_before_answer_refine"] = full_response
        output_item[refine_field] = refined_response
        output_item["answer_refined"] = True
        output_item["absolute_index"] = start_idx + idx
        
        processed_items.append(output_item)
        refined_count += 1
        
        # Progressive save
        if (idx + 1) % save_interval == 0:
            save_progressive(processed_items, output_file, idx, is_final=False)
    
    # Final save
    save_progressive(processed_items, output_file, len(data) - 1, is_final=True)
    
    print(f"\nCompleted!")
    print(f"  Total items: {len(data)}")
    print(f"  Successfully refined: {refined_count}")
    print(f"  Skipped (no </think>): {skipped_no_think}")
    print(f"  Skipped (errors): {skipped_error}")
    print(f"Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 7: Refine answers to match thinking with minimal modifications"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with rewritten thinking"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for refined answers"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL for refinement model"
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
        help="Which field to refine (default: accepted)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for refinement"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (lower = more conservative)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
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
    
    process_answer_refinement(
        input_file=args.input,
        output_file=args.output,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        refine_field=args.field,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        save_interval=args.save_interval,
        start_idx=args.start,
        end_idx=args.end
    )


if __name__ == "__main__":
    main()
