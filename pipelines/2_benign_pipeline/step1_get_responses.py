#!/usr/bin/env python3
"""
Step 1: Generate Model Responses for Benign Data

This script generates multiple responses from a VLM on benign questions
(e.g., visual reasoning, illusion detection). Multiple samples per question
help identify questions that are not trivially easy.

Input: Benign dataset JSON (e.g., Illusion dataset)
Output: JSON file with multiple model responses per question
"""

import os
import sys
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional, Any

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from models import VLLMClient
from utils import load_json, save_progressive, load_progress, resolve_image_paths


def generate_multiple_responses(
    client: VLLMClient,
    question: str,
    image_paths: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    num_samples: int = 3,
    max_tokens: int = 8192
) -> List[str]:
    """
    Generate multiple diverse responses for a single question.
    
    Args:
        client: VLLMClient instance
        question: Question text
        image_paths: List of image paths
        num_samples: Number of responses to generate
        max_tokens: Maximum tokens per response
        
    Returns:
        List of generated responses
    """
    responses = []
    
    # Different sampling configs for diversity
    sampling_configs = [
        {"temperature": 0.3, "top_p": 0.8},
        {"temperature": 0.7, "top_p": 0.9},
        {"temperature": 1.0, "top_p": 0.95},
    ]
    
    for i in range(num_samples):
        try:
            config = sampling_configs[i % len(sampling_configs)]
            
            result = client.generate_response(
                prompt=question,
                image_paths=image_paths,
                model=model_name,
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=max_tokens,
            )
            
            response = result["response"].strip()
            responses.append(response)
            
        except Exception as e:
            print(f"Error generating response {i+1}: {e}")
            responses.append("")
    
    return responses


def generate_benign_responses(
    input_file: str,
    output_file: str,
    vllm_url: str = "http://localhost:8000",
    data_base_path: str = "",
    model_name: Optional[str] = None,
    num_samples: int = 3,
    max_tokens: int = 8192,
    save_interval: int = 100,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> str:
    """
    Generate multiple responses for benign dataset.
    
    Args:
        input_file: Path to input JSON with questions
        output_file: Path to save responses
        vllm_url: vLLM server URL
        data_base_path: Base path for image files
        num_samples: Number of responses per question
        max_tokens: Maximum tokens per response
        save_interval: Save progress every N items
        start_idx: Start processing from this index
        end_idx: End processing at this index (None = process all)
        
    Returns:
        Path to output file
    """
    print(f"Loading data from {input_file}...")
    all_data = load_json(input_file)
    
    # Initialize client
    client = VLLMClient(vllm_url)

    # Filter by indices
    if end_idx is None:
        end_idx = len(all_data)
    data = all_data[start_idx:end_idx]

    model_name = client.resolve_model(model_name)
    print(f"Using model: {model_name}")
    
    print(f"Processing {len(data)} items (indices {start_idx}-{end_idx-1})")
    print(f"Generating {num_samples} responses per question")
    
    # Check for existing progress
    processed_items, resume_idx = load_progress(output_file)
    if processed_items:
        print(f"Resuming from index {resume_idx} ({len(processed_items)} items already processed)")
    else:
        processed_items = []
        resume_idx = 0
    
    # Process items
    for idx in tqdm(range(resume_idx, len(data)), desc="Generating responses", initial=resume_idx):
        item = data[idx]
        
        question = item.get("question", "")
        ground_truth = item.get("answer", "")
        images = item.get("images", [])
        
        # Convert to absolute paths
        if images:
            if isinstance(images, str):
                images = [images]
            image_paths = resolve_image_paths(images, data_base_path)
        else:
            image_paths = None
        
        try:
            # Generate multiple responses
            responses = generate_multiple_responses(
                client=client,
                question=question,
                image_paths=image_paths,
                model_name=model_name,
                num_samples=num_samples,
                max_tokens=max_tokens
            )
            
            # Create output item
            output_item = {
                "index": start_idx + idx,
                "qid": item.get("qid", ""),
                "question": question,
                "ground_truth": ground_truth,
                "responses": responses,
                "images": images,
                "image_paths": image_paths,
                "source": item.get("source", "")
            }
            
            processed_items.append(output_item)
            
            # Progressive save
            if (idx + 1) % save_interval == 0:
                save_progressive(processed_items, output_file, idx, is_final=False)
                
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue
    
    # Final save
    save_progressive(processed_items, output_file, len(data) - 1, is_final=True)
    
    print(f"\nCompleted! Processed {len(processed_items)} items")
    print(f"Total responses generated: {len(processed_items) * num_samples}")
    print(f"Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate multiple responses for benign data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with benign questions"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for responses"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL"
    )
    parser.add_argument(
        "--data-base-path",
        default="",
        help="Optional base path used to resolve relative image paths"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name (default: auto-detect from the vLLM server)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of responses per question"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens per response"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
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
    
    generate_benign_responses(
        input_file=args.input,
        output_file=args.output,
        vllm_url=args.vllm_url,
        data_base_path=args.data_base_path,
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        save_interval=args.save_interval,
        start_idx=args.start,
        end_idx=args.end
    )


if __name__ == "__main__":
    main()
