#!/usr/bin/env python3
"""
Step 1: Generate Model Responses for Safety Data

This script generates responses from a VLM on safety-related questions.
The responses will later be filtered for unsafe content.

Input: Safety dataset JSON (e.g., MM-SafetyBench, JailBreakV)
Output: JSON file with model responses
"""

import os
import sys
import argparse
from tqdm import tqdm
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from models import VLLMClient
from utils import load_json, save_progressive, load_progress, resolve_image_paths


def process_single_item(
    client,
    item: Dict[str, Any],
    idx: int,
    start_idx: int,
    data_base_path: str,
    model_name: Optional[str],
    max_tokens: int
) -> Optional[Dict[str, Any]]:
    """
    Process a single item to generate response.
    
    Args:
        client: VLLMClient instance
        item: Data item to process
        idx: Index in the batch
        start_idx: Global start index
        data_base_path: Base path for image files
        model_name: Model name for identification
        max_tokens: Maximum tokens to generate
        
    Returns:
        Output item dict or None if error
    """
    question = item.get("question", "")
    images = item.get("images", [])
    
    image_paths = resolve_image_paths(images, data_base_path)
    
    try:
        # Generate response using vLLM
        result = client.generate_response(
            prompt=question,
            image_paths=image_paths,
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        # Don't strip() to preserve special tokens like <think>
        response_text = result["response"]
        
        # Create output item
        output_item = {
            "index": start_idx + idx,
            "question": question,
            "images": images,
            "image_paths": image_paths,
            "response": response_text,
            "source": item.get("source", "unknown"),
            "category": item.get("category", ""),
            "sub_category": item.get("sub_category", "")
        }
        
        return output_item
        
    except Exception as e:
        print(f"Error processing item {idx}: {e}")
        return None


def generate_responses(
    client,
    input_file: str,
    output_file: str,
    data_base_path: str,
    model_name: Optional[str] = None,
    max_tokens: int = 10240,
    save_interval: int = 100,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    num_threads: int = 8
) -> str:
    """
    Generate responses for safety dataset using multi-threading.
    
    Args:
        client: VLLMClient or HuggingFaceClient instance
        input_file: Path to input JSON with questions
        output_file: Path to save responses
        data_base_path: Base path for image files
        model_name: Model name for identification
        max_tokens: Maximum tokens to generate
        save_interval: Save progress every N items
        start_idx: Start processing from this index
        end_idx: End processing at this index (None = process all)
        num_threads: Number of threads for parallel processing
        
    Returns:
        Path to output file
    """
    # Load input data
    print(f"Loading data from {input_file}...")
    all_data = load_json(input_file)
    
    # Handle different data structures
    if isinstance(all_data, dict) and "results" in all_data:
        all_data = all_data["results"]
    
    # Filter by indices
    if end_idx is None:
        end_idx = len(all_data)
    data = all_data[start_idx:end_idx]

    model_name = client.resolve_model(model_name)
    print(f"Using model: {model_name}")
    
    print(f"Processing {len(data)} items (indices {start_idx}-{end_idx-1}) with {num_threads} threads")
    
    # Check for existing progress
    processed_items, resume_idx = load_progress(output_file)
    if processed_items:
        print(f"Resuming from index {resume_idx} ({len(processed_items)} items already processed)")
    else:
        processed_items = []
        resume_idx = 0
    
    # Thread-safe lock for updating processed_items
    lock = threading.Lock()
    processed_count = len(processed_items)
    
    # Create items to process (only unprocessed ones)
    items_to_process = [(idx, data[idx]) for idx in range(resume_idx, len(data))]
    
    # Progress bar
    pbar = tqdm(total=len(items_to_process), desc="Generating responses", initial=0)
    
    # Process items in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                process_single_item,
                client,
                item,
                idx,
                start_idx,
                data_base_path,
                model_name,
                max_tokens
            ): idx 
            for idx, item in items_to_process
        }
        
        # Process completed tasks
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                output_item = future.result()
                if output_item:
                    with lock:
                        processed_items.append(output_item)
                        processed_count += 1
                        
                        # Progressive save
                        if processed_count % save_interval == 0:
                            # Sort by index before saving to maintain order
                            sorted_items = sorted(processed_items, key=lambda x: x["index"])
                            save_progressive(sorted_items, output_file, idx, is_final=False)
                            
            except Exception as e:
                print(f"Error processing future for item {idx}: {e}")
            
            pbar.update(1)
    
    pbar.close()
    
    # Sort by index before final save to maintain order
    processed_items = sorted(processed_items, key=lambda x: x["index"])
    
    # Final save
    save_progressive(processed_items, output_file, len(data) - 1, is_final=True)
    
    print(f"\nCompleted! Processed {len(processed_items)} items")
    print(f"Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate model responses for safety data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with safety questions"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for responses"
    )
    parser.add_argument(
        "--data-base-path",
        default="",
        help="Optional base path used to resolve relative image paths"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name (default: auto-detect from the vLLM server)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10240,
        help="Maximum tokens to generate"
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
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for parallel processing (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Create vLLM client
    print(f"Using vLLM client at: {args.vllm_url}")
    client = VLLMClient(args.vllm_url)
    
    # Generate responses
    generate_responses(
        client=client,
        input_file=args.input,
        output_file=args.output,
        data_base_path=args.data_base_path,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        save_interval=args.save_interval,
        start_idx=args.start,
        end_idx=args.end,
        num_threads=args.num_threads
    )


if __name__ == "__main__":
    main()
