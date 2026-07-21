#!/usr/bin/env python3
"""
Step 2: Filter Out Simple Questions

This script evaluates the correctness of multiple responses per question
and filters out questions where all responses are correct (too simple).
Keeps only challenging questions where at least one response is incorrect.

Input: JSON file with multiple responses (from Step 1)
Output: JSON file with only non-trivial questions
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from models import VLLMClient
from utils import load_json, save_json


def evaluate_single_answer(
    client: VLLMClient,
    question: str,
    predicted_answer: str,
    ground_truth: str
) -> bool:
    """
    Evaluate a single answer using the model as a judge.
    
    Args:
        client: VLLMClient instance
        question: Original question
        predicted_answer: Answer to evaluate
        ground_truth: Correct answer
        
    Returns:
        True if answer is correct, False otherwise
    """
    eval_prompt = f"""You are an expert evaluator. Please determine if the predicted answer is correct based on the question and ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted_answer}

Please analyze whether the predicted answer matches the ground truth answer. Consider:
1. If it's a multiple choice question, check if the final answer choice (A, B, C, D, etc.) matches
2. For other questions, check if the key content or number is correct even if the format differs

Respond with only "CORRECT" or "INCORRECT" followed by a brief explanation."""

    try:
        result = client.generate_response(
            prompt=eval_prompt,
            image_paths=None,  # Text-only evaluation
            max_tokens=1024,
            top_p=0.1  # Deterministic evaluation
        )
        
        evaluation = result["response"].strip().upper()
        return evaluation.startswith("CORRECT")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        # Fallback to simple string matching
        return str(ground_truth).strip().lower() in predicted_answer.lower()


def evaluate_responses_parallel(
    client: VLLMClient,
    question: str,
    responses: List[str],
    ground_truth: str,
    max_workers: int = 3
) -> List[bool]:
    """
    Evaluate multiple responses in parallel.
    
    Args:
        client: VLLMClient instance
        question: Original question
        responses: List of responses to evaluate
        ground_truth: Correct answer
        max_workers: Number of parallel workers
        
    Returns:
        List of boolean correctness values
    """
    evaluations = [False] * len(responses)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_single_answer,
                client,
                question,
                response,
                ground_truth
            ): idx
            for idx, response in enumerate(responses)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                evaluations[idx] = future.result()
            except Exception as e:
                print(f"Error evaluating response {idx}: {e}")
                evaluations[idx] = False
    
    return evaluations


def filter_simple_questions(
    input_file: str,
    output_file: str,
    vllm_url: str = "http://localhost:8000",
    max_workers: int = 3
) -> str:
    """
    Filter out questions where all responses are correct.
    
    Args:
        input_file: Path to responses JSON (from Step 1)
        output_file: Path to save filtered questions
        vllm_url: vLLM server URL for evaluation
        max_workers: Number of parallel workers
        
    Returns:
        Path to output file
    """
    print(f"Loading responses from {input_file}...")
    data = load_json(input_file)
    
    print(f"Connecting to vLLM at {vllm_url} for evaluation...")
    client = VLLMClient(vllm_url)
    
    filtered_data = []
    pass_rate_stats = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print("Evaluating responses and filtering...")
    for item in tqdm(data, desc="Evaluating"):
        question = item["question"]
        ground_truth = item["ground_truth"]
        responses = item["responses"]
        
        # Evaluate all responses
        evaluations = evaluate_responses_parallel(
            client=client,
            question=question,
            responses=responses,
            ground_truth=ground_truth,
            max_workers=max_workers
        )
        
        correct_count = sum(evaluations)
        
        # Add evaluation results to item
        item["evaluations"] = evaluations
        item["correct_count"] = correct_count
        item["pass_rate"] = correct_count / len(responses)
        
        pass_rate_stats[correct_count] = pass_rate_stats.get(correct_count, 0) + 1
        
        # Keep only if not all correct (not too simple)
        if correct_count < len(responses):
            filtered_data.append(item)
    
    print(f"\nFiltering complete:")
    print(f"  Total questions: {len(data)}")
    print(f"\nPass rate distribution:")
    responses_per_question = len(data[0]['responses']) if data else 0
    for correct_count in sorted(pass_rate_stats.keys()):
        count = pass_rate_stats[correct_count]
        percentage = count / len(data) * 100 if len(data) > 0 else 0
        print(f"    {correct_count}/{responses_per_question} correct: {count:4d} items ({percentage:5.1f}%)")

    filtered_ratio = len(filtered_data) / len(data) * 100 if data else 0.0
    print(f"\n  Filtered (not all correct): {len(filtered_data)} ({filtered_ratio:.1f}%)")
    
    # Save filtered data
    save_json(filtered_data, output_file)
    print(f"Filtered questions saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Filter out simple questions (all responses correct)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with responses (from Step 1)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for filtered questions"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL for evaluation"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Number of parallel evaluation workers"
    )
    
    args = parser.parse_args()
    
    filter_simple_questions(
        input_file=args.input,
        output_file=args.output,
        vllm_url=args.vllm_url,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
