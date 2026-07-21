import math
import torch
import os
import json
import gc
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def cleanup_gpu_memory():
    """Aggressive GPU memory cleanup utility function."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def print_gpu_memory_usage(prefix="", device=None):
    """Print current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"{prefix}GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB (Device: {device})")


def attention_compute(attention, sys_len, img_len):
    attention = torch.mean(attention, dim=1)
    attention = attention.squeeze(0).numpy()
    props_sys = attention[-1][:sys_len].sum()
    props_img = attention[-1][sys_len:sys_len+img_len].sum()
    props_txt = attention[-1][sys_len+img_len:].sum()
    return [props_sys, props_img, props_txt]


def attention_compute_per_token(attention, sys_len, img_len, token_idx):
    """Compute attention distribution for a specific generated token"""
    attention = torch.mean(attention, dim=1)  # Average over heads
    attention = attention.squeeze(0).to(torch.float32).cpu().numpy()  # Remove batch dimension, convert to float32, and move to CPU
    
    # Get attention weights for the specific token
    token_attention = attention[token_idx]
    
    props_sys = token_attention[:sys_len].sum()
    props_img = token_attention[sys_len:sys_len+img_len].sum()
    props_txt = token_attention[sys_len+img_len:].sum()
    
    return [props_sys, props_img, props_txt]


def run_attention_analysis(
    model_path,
    question_file,
    output_file,
    start: int = 0,
    end: int = None,
    device: str = "cuda:0",
):
    """Run attention analysis with memory-optimized settings.

    Args:
        model_path: HF path to model.
        question_file: JSON file with a list of items having keys: index, image_path, question.
        output_file: Path to save JSON results.
        start: Start index for sample selection.
        end: End index for sample selection (None means to the end).
        device: Device to use for computation.
    """

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=device,
            cache_implementation="static",
        ).eval()
    except TypeError:
        # Fallback for older transformers versions
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=device,
        ).eval()

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, padding_side='left', use_fast=True
    )

    all_questions = json.load(open(question_file, "r"))
    questions = all_questions[start:end] if end is not None else all_questions[start:]

    results = []
    save_interval = 50  # Save every 50 samples

    for sample_idx, line in enumerate(tqdm(questions)):
        idx = line.get("id", sample_idx)
        
        # Handle both single image_path and multiple image_paths
        if "image_paths" in line:
            image_files = line["image_paths"] if isinstance(line["image_paths"], list) else [line["image_paths"]]
        elif "image_path" in line:
            image_files = [line["image_path"]] if isinstance(line["image_path"], str) else line["image_path"]
        else:
            print(f"Skip question {idx}: No image_path or image_paths found.")
            continue
            
        qs = line["question"]

        # Reasoning Model - build content list with all images
        content = []
        for img_file in image_files:
            content.append({"type": "image", "image": img_file})
        content.append({"type": "text", "text": qs})
        
        messages_query = [{
            "role": "user",
            "content": content,
        }]
   
        image_inputs, _ = process_vision_info(messages_query)
        text_query = processor.apply_chat_template(messages_query, tokenize=False, add_generation_prompt=False)
        inputs = processor(
            text=[text_query],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Calculate input breakdown like in original_attn.py
        input_ids = inputs['input_ids'][0].tolist()
        try:
            vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            pos = input_ids.index(vision_start_token_id) + 1
            pos_end = input_ids.index(vision_end_token_id)
        except ValueError:
            print(f"Skip question {idx}: Vision tokens not found.")
            continue

        sys_len = pos
        img_len = pos_end - pos
        txt_len = len(input_ids) - pos_end - 1
        
        # print(f"Sample {idx} - Input breakdown: System={sys_len}, Image={img_len}, Text={txt_len}")

        
        try:
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True,
                do_sample=False,  # Use greedy decoding for consistency
            )
            
            input_length = inputs['input_ids'].shape[1]
            generated_length = outputs.sequences.shape[1] - input_length
                
        except torch.cuda.OutOfMemoryError:
            # Aggressive cleanup on OOM
            if 'outputs' in locals():
                del outputs
            if 'inputs' in locals():
                del inputs
            if 'image_inputs' in locals():
                del image_inputs
            if 'text_query' in locals():
                del text_query
            cleanup_gpu_memory()
            print(f"Skip question {idx}: OOM during generation.")
            continue

        # Decode the generated text and get attention data
        generated_sequence = outputs.sequences[0]
        generated_tokens = generated_sequence[input_length:]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if len(outputs.attentions) > 0:
            # Split generated text into reasoning steps by \n\n first
            initial_steps = [step.strip() for step in generated_text.split("\n\n") if step.strip()]
            
            if not initial_steps:
                # If no \n\n found, treat entire text as one step
                initial_steps = [generated_text.strip()] if generated_text.strip() else []
            
            # Further split steps that don't start with "Wait" by splitting on "Wait"
            reasoning_steps = []
            for step in initial_steps:
                if not step.startswith("Wait"):
                    # Split on "Wait" and keep the delimiter
                    parts = step.split("Wait")
                    if len(parts) > 1:
                        # First part (before first "Wait")
                        if parts[0].strip():
                            reasoning_steps.append(parts[0].strip())
                        # Remaining parts with "Wait" prepended
                        for part in parts[1:]:
                            if part.strip():
                                reasoning_steps.append("Wait" + part.strip())
                    else:
                        # No "Wait" found, keep as is
                        reasoning_steps.append(step)
                else:
                    # Starts with "Wait", keep as is
                    reasoning_steps.append(step)
            
            print(f"Found {len(reasoning_steps)} reasoning steps (after splitting on 'Wait')")
            
            step_attention_data = []
            
            if reasoning_steps:
                # Process attention for each generated token like original_attn.py
                all_attentions = outputs.attentions  # List of tuples, one per generated token
                num_generated_tokens = len(all_attentions)
                num_layers = len(all_attentions[0]) if num_generated_tokens > 0 else 0
                
                # Get per-token attention with immediate layer averaging (memory efficient)
                per_token_attention = []
                for token_idx in range(num_generated_tokens):
                    # Immediately average across layers for each token to save memory
                    layer_attention_sums = np.zeros(3)  # [sys, img, txt]
                    
                    for layer_idx in range(num_layers):
                        # Get attention for this layer and this token
                        layer_attention = all_attentions[token_idx][layer_idx]  # Shape: [batch, heads, seq_len, seq_len]
                        
                        # Calculate attention distribution using same method as original_attn.py
                        attention = torch.mean(layer_attention, dim=1)  # Average over heads
                        attention = attention.squeeze(0).to(torch.float32).cpu().numpy()  # Remove batch dimension, convert to float32, and move to CPU
                        
                        # Get attention weights for the last generated token
                        token_attention = attention[-1]
                        
                        props_sys = float(token_attention[:sys_len].sum())
                        props_img = float(token_attention[sys_len:sys_len+img_len].sum())
                        props_txt = float(token_attention[sys_len+img_len:].sum())
                        
                        # Accumulate across layers
                        layer_attention_sums += np.array([props_sys, props_img, props_txt])
                        
                        # Clean up immediately
                        del layer_attention, attention, token_attention
                    
                    # Average across layers and store only the final result
                    token_avg_attention = layer_attention_sums / num_layers
                    per_token_attention.append(token_avg_attention)
                
                # Clean up attention data immediately
                del outputs.attentions
                torch.cuda.empty_cache()
                
                # Now map tokens to steps and compute average attention per step
                current_token_pos = 0
                
                for step_idx, step_text in enumerate(reasoning_steps):
                    # Tokenize this step to find how many tokens it contains
                    step_tokens = processor.tokenizer.encode(step_text, add_special_tokens=False)
                    step_length = len(step_tokens)
                    
                    # Get tokens for this step
                    step_start = current_token_pos
                    step_end = min(current_token_pos + step_length, num_generated_tokens)
                    
                    if step_start < num_generated_tokens:
                        # Collect attention for tokens in this step
                        step_token_attentions = []
                        for token_pos in range(step_start, step_end):
                            if token_pos < len(per_token_attention):
                                # per_token_attention is already averaged across layers
                                step_token_attentions.append(per_token_attention[token_pos])
                        
                        if step_token_attentions:
                            # Average attention across all tokens in this step
                            step_avg_attention = np.mean(step_token_attentions, axis=0)
                            step_attention_data.append({
                                'step_index': step_idx,
                                'step_text': step_text,
                                'step_tokens': step_length,
                                'attention': [float(x) for x in step_avg_attention]  # [sys, img, txt]
                            })
                            
                            # Print step attention for debugging
                            total_attn = sum(step_avg_attention)
                            if total_attn > 0:
                                normalized_attn = [x/total_attn for x in step_avg_attention]
                                # print(f"Step {step_idx+1}: Sys={normalized_attn[0]:.3f}, Img={normalized_attn[1]:.3f}, Text={normalized_attn[2]:.3f}")
                    
                    current_token_pos += step_length
                
                # Clean up remaining data
                del per_token_attention
                torch.cuda.empty_cache()
            
            # Store original JSON item + new data
            result_data = line.copy()  # Copy original JSON item
            result_data.update({
                'generated_text': generated_text,
                'reasoning_steps': reasoning_steps,
                'step_attention': step_attention_data
            })
        else:
            result_data = line.copy()  # Copy original JSON item
            result_data.update({
                'generated_text': generated_text,
                'reasoning_steps': [],
                'step_attention': []
            })
        
        results.append(result_data)

        # Progressive save every 50 samples
        if (sample_idx + 1) % save_interval == 0:
            temp_file = output_file.replace('.json', f'_temp_{sample_idx + 1}.json')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Temporary save: {sample_idx + 1} samples saved to {temp_file}")

        # Comprehensive cleanup of all temporary variables
        del outputs, inputs, image_inputs, text_query, generated_sequence, generated_tokens, generated_text, result_data
        if 'messages_query' in locals():
            del messages_query
        if 'input_ids' in locals():
            del input_ids
        
        # Force garbage collection and GPU memory cleanup
        cleanup_gpu_memory()
        
        # Additional aggressive GPU memory cleanup every 10 samples
        if (sample_idx + 1) % 10 == 0:
            cleanup_gpu_memory()

    # Final save as JSON format
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Final save: {len(results)} samples saved to {output_file}")
    
    # Clean up temporary files
    import glob
    temp_pattern = output_file.replace('.json', '_temp_*.json')
    temp_files = glob.glob(temp_pattern)
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Cleaned up temporary file: {temp_file}")
        except:
            pass


def prepare_attention_data(path):
    """Prepare attention data for visualization from step_attention arrays."""
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract all individual step attention values
    all_step_attention = []
    for result in results:
        if isinstance(result, dict) and 'step_attention' in result:
            step_attn_list = result['step_attention']
            if step_attn_list:  # Not empty
                # step_attn_list is list of step attention dicts
                for step_data in step_attn_list:
                    if 'attention' in step_data:
                        all_step_attention.append(step_data['attention'])  # attention is [3] array
    
    if all_step_attention:
        # Convert to numpy array - now all elements are [3] so it's uniform
        all_step_attention = np.array(all_step_attention)  # Shape: [total_steps, 3]
        # Average across all steps from all samples
        avg_attention = np.mean(all_step_attention, axis=0)  # Shape: [3]
        # Reshape for plotting compatibility: [3, 1]
        return avg_attention.reshape(-1, 1)
    
    return None  # Fallback


def print_results_summary(path):
    """Print a brief summary of results."""
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    total_samples = len(results)
    total_steps = 0
    
    for i, result in enumerate(results):
        if isinstance(result, dict):
            if 'step_attention' in result:
                num_steps = len(result['step_attention'])
                total_steps += num_steps
    
    avg_steps = total_steps / total_samples if total_samples > 0 else 0
    print(f"Processed {total_samples} samples:")
    print(f"  - Average {avg_steps:.1f} reasoning steps per sample")
    print(f"  - Total: {total_steps} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output-file", "--answers-file", dest="output_file", type=str, required=True, help="Path to save JSON result file")
    parser.add_argument("--start", type=int, default=0, help="Start index for sample selection")
    parser.add_argument("--end", type=int, default=None, help="End index for sample selection (None means to the end)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation")

    args = parser.parse_args()

    # Run attention analysis
    run_attention_analysis(
        model_path=args.model_path,
        question_file=args.question_file,
        output_file=args.output_file,
        start=args.start,
        end=args.end,
        device=args.device,
    )
    
    # Print brief summary
    print_results_summary(args.output_file)
