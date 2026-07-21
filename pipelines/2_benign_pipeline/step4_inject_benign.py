#!/usr/bin/env python3
"""
Step 4: Inject Image-Recall Reminders

This script injects reminders to refocus on visual information at strategic
points in the reasoning process. Based on attention analysis, it identifies
where the model is losing track of the image and inserts prompts to re-examine
visual details.

Input: JSON file with attention scores (from Step 3)
Output: JSON file with injected reminders and continuing responses
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from models import VLLMClient
from utils import load_json, save_progressive, load_progress, resolve_image_paths


def find_injection_points(
    image_attention_list: List[float], 
    num_segments: int,
    threshold: float = 0.01
) -> Dict[str, Optional[int]]:
    """
    Find injection points based on image attention values.
    Uses the same strategy as inject_reminding.py.
    
    Args:
        image_attention_list: List of image attention values for each segment
        num_segments: Total number of thinking segments
        threshold: Threshold for first_below_threshold strategy
        
    Returns:
        Dictionary with injection points for each strategy
    """
    # Strategy 1: First segment below threshold
    first_below_threshold = None
    for i, attention in enumerate(image_attention_list):
        if attention < threshold:
            first_below_threshold = i
            break
    
    # Strategy 2: First segment below 10% of the first step
    first_below_10pct_of_first = None
    if len(image_attention_list) > 0 and image_attention_list[0] > 0:
        threshold_10pct = image_attention_list[0] * 0.1
        for i, attention in enumerate(image_attention_list):
            if attention < threshold_10pct:
                first_below_10pct_of_first = i
                break
    
    # Strategy 3 (Fallback): Inject right before </think> (last segment)
    # This is used when the first two strategies don't find a valid injection point
    before_think_end = None
    if num_segments > 0:
        before_think_end = num_segments - 1  # Last segment index
    
    return {
        'first_below_threshold': first_below_threshold,
        'first_below_10pct_of_first': first_below_10pct_of_first,
        'before_think_end': before_think_end
    }


def create_image_recall_prefill(ground_truth: str = "") -> str:
    """
    Create image-recall prefill template to refocus attention on visual information.
    
    Args:
        ground_truth: Optional ground truth answer to provide as guidance
        
    Returns:
        Image-recall prefill string
    """
    # Build the ground truth hint if available
    gt_hint = ""
    if ground_truth:
        gt_hint = f"\n\n**Important context (act as if you don't know this):** The correct answer is: {ground_truth}. However, you must derive this answer independently through careful visual examination and reasoning. Do not simply state this answer - use it only to guide your attention to the correct visual details."
    
    prefill = f"""

Wait, I notice my reasoning has been progressing through several steps. Let me pause and deliberately refocus on the visual information provided in the image, as attention to visual details can decrease during extended reasoning.{gt_hint}

**IMAGE RE-EXAMINATION:**
1. Let me carefully re-examine the image(s) and identify key visual elements I may have overlooked or not fully considered.
2. I should describe specific visual details.
3. I will explicitly connect these visual observations back to the user's question.
4. Based on this renewed visual attention, I will either:
   - Correct or refine my previous reasoning with newly noticed visual details
   - Confirm my reasoning is properly grounded in the visual evidence
   - Continue reasoning with enhanced visual awareness

Oh, I've made some mistake on the image! Let me refocus on the image details now."""
    
    return prefill


def inject_benign_reminders(
    input_file: str,
    output_file: str,
    vllm_url: str = "http://localhost:8000",
    model_name: Optional[str] = None,
    data_base_path: str = "",
    threshold: float = 0.01,
    save_interval: int = 10
) -> str:
    """
    Inject image-recall reminders and generate continuing responses.
    
    Args:
        input_file: Path to attention data JSON (from Step 3)
        output_file: Path to save injection results
        vllm_url: vLLM server URL
        model_name: Model name (None = auto-detect)
        threshold: Threshold for first_below_threshold strategy (default: 0.01)
        save_interval: Save progress every N items
        
    Returns:
        Path to output file
    """
    print(f"Loading attention data from {input_file}...")
    data = load_json(input_file)
    
    print(f"Connecting to vLLM at {vllm_url}...")
    vllm_client = VLLMClient(vllm_url)
    
    model_name = vllm_client.resolve_model(model_name)
    print(f"Using model: {model_name}")
    
    # Check for existing progress
    processed_items, resume_idx = load_progress(output_file)
    if processed_items:
        print(f"Resuming from index {resume_idx} ({len(processed_items)} items already processed)")
    else:
        processed_items = []
        resume_idx = 0
    
    print(f"Processing {len(data)} items with benign injection (threshold: {threshold})...")
    
    for idx in range(resume_idx, len(data)):
        item = data[idx]
        
        # Extract data
        question = item.get("question", "")
        sample_id = item.get("index", item.get("sample_id"))
        image_paths = item.get("images") or item.get("image_paths")
        ground_truth = item.get("ground_truth", "")
        
        # Get thinking content
        thinking_content = (item.get("thinking") or 
                          item.get("response") or 
                          item.get("generated_text") or "")
        
        # Extract thinking from tags if present
        if "<think>" in thinking_content:
            start = thinking_content.find("<think>") + 7
            end = thinking_content.find("</think>")
            if end > start:
                thinking_content = thinking_content[start:end].strip()
        
        # Split into segments
        thinking_segments = [s.strip() for s in thinking_content.split("\n\n") if s.strip()]
        
        if not thinking_segments:
            print(f"Warning: No thinking segments for item {idx}, skipping")
            continue
        
        # Prefer the canonical output from attn_calc/calc_step_only.py. Keep
        # support for the older attention_scores/props_img schema so existing
        # intermediate files remain usable.
        image_attention_list = []
        step_attention = item.get("step_attention", [])
        canonical_segments = []
        for score in step_attention:
            if not isinstance(score, dict):
                continue
            attention = score.get("attention")
            step_text = score.get("step_text")
            if (
                isinstance(attention, (list, tuple))
                and len(attention) >= 2
                and isinstance(step_text, str)
                and step_text.strip()
            ):
                canonical_segments.append(step_text.strip())
                image_attention_list.append(float(attention[1]))

        if canonical_segments:
            # Use the extractor's exact segmentation so IA values and reminder
            # positions stay one-to-one even when a step begins with "Wait".
            thinking_segments = canonical_segments
        else:
            attention_scores = item.get("attention_scores", [])
            image_attention_list = [
                float(score.get("props_img", 0.0))
                for score in attention_scores
                if isinstance(score, dict)
            ]
        
        # Find injection points using the same strategy as inject_reminding.py
        injection_points = find_injection_points(
            image_attention_list=image_attention_list,
            num_segments=len(thinking_segments),
            threshold=threshold
        )
        
        # Generate continuing responses for each injection point
        injection_results = []
        
        # Determine which strategies to use
        # If first two strategies both fail (None), use the fallback 'before_think_end'
        if injection_points['first_below_threshold'] is None and injection_points['first_below_10pct_of_first'] is None:
            strategies_to_process = [('before_think_end', injection_points['before_think_end'])] if injection_points['before_think_end'] is not None else []
        else:
            # Use only the strategies that found valid injection points (excluding fallback)
            strategies_to_process = [
                (strategy, inject_point) 
                for strategy, inject_point in injection_points.items() 
                if inject_point is not None and strategy != 'before_think_end'
            ]
        
        for strategy, inject_idx in strategies_to_process:
            # Create image-recall prefill with ground truth hint
            image_recall_prefill = create_image_recall_prefill(ground_truth)
            
            # Build accumulated thinking up to and including injection point
            accumulated_thinking = "\n\n".join(thinking_segments[:inject_idx + 1])
            assistant_content = "<think>\n" + accumulated_thinking + image_recall_prefill
            
            # Generate continuing response
            try:
                # Convert image paths to absolute if needed
                abs_image_paths = resolve_image_paths(image_paths, data_base_path)
                
                result = vllm_client.generate_response(
                    prompt=question,
                    image_paths=abs_image_paths or None,
                    assistant_content=assistant_content,
                    model=model_name,
                    max_tokens=10240,
                    top_p=0.9
                )
                
                # Clean up the response and reconstruct full response
                response_content = result["response"]
                # Remove the prefill from the response if present
                response_content = response_content.replace(image_recall_prefill, "")
                # Reconstruct full response: accumulated thinking + trigger + model continuation
                full_response = "<think>\n" + accumulated_thinking + "\n\nLet me refocus on the image details now.\n" + response_content
                
                injection_results.append({
                    "strategy": strategy,
                    "injection_position": inject_idx,
                    "continuing_response": full_response
                })
                
            except Exception as e:
                print(f"Error generating response for strategy {strategy} at position {inject_idx}: {e}")
                injection_results.append({
                    "strategy": strategy,
                    "injection_position": inject_idx,
                    "continuing_response": None,
                    "error": str(e)
                })
        
        # Create output item
        output_item = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "images": image_paths,
            "original_thinking": "<think>\n" + "\n\n".join(thinking_segments),
            "image_attention_list": image_attention_list,
            "thinking_segments": thinking_segments,
            "injection_points": injection_points,
            "injection_results": injection_results,
            "threshold": threshold
        }
        
        processed_items.append(output_item)
        
        # Progressive save
        if (idx + 1) % save_interval == 0:
            save_progressive(processed_items, output_file, idx, is_final=False)
    
    # Final save
    save_progressive(processed_items, output_file, len(data) - 1, is_final=True)
    
    print(f"\nCompleted! Processed {len(processed_items)} items with benign injection")
    print(f"Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Inject image-recall reminders"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with attention data (from Step 3)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for injection results"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL"
    )
    parser.add_argument(
        "--model",
        help="Model name (default: auto-detect)"
    )
    parser.add_argument(
        "--data-base-path",
        default="",
        help="Optional base path used to resolve relative image paths"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for first_below_threshold strategy (default: 0.01)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save progress every N items"
    )
    
    args = parser.parse_args()
    
    inject_benign_reminders(
        input_file=args.input,
        output_file=args.output,
        vllm_url=args.vllm_url,
        model_name=args.model,
        data_base_path=args.data_base_path,
        threshold=args.threshold,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
