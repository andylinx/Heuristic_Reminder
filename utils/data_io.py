"""Data I/O and path utilities for the release pipelines."""

import os
import json
from typing import Dict, Any, Iterable, List, Optional, Tuple, Union


def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data (dict, list, etc.)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the JSON file
        indent: JSON indentation level (default: 2)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def resolve_image_paths(
    image_paths: Optional[Union[str, Iterable[str]]],
    data_base_path: str = ""
) -> List[str]:
    """Resolve zero, one, or many image paths against an optional base path."""
    if not image_paths:
        return []

    if isinstance(image_paths, str):
        items = [image_paths]
    else:
        items = list(image_paths)

    resolved_paths = []
    for image_path in items:
        if not image_path:
            continue
        if os.path.isabs(image_path) or not data_base_path:
            resolved_paths.append(image_path)
        else:
            resolved_paths.append(os.path.join(data_base_path, image_path))
    return resolved_paths


def save_progressive(data: List[Dict[str, Any]], 
                    filepath: str, 
                    current_index: int,
                    is_final: bool = False) -> None:
    """
    Save progressive results during processing.
    
    Args:
        data: List of processed items
        filepath: Output file path
        current_index: Current processing index
        is_final: Whether this is the final save
    """
    base_name, ext = os.path.splitext(filepath)
    
    if is_final:
        # Final save - use original filename
        save_json(data, filepath)
        
        # Clean up temporary progress files
        temp_pattern = f"{base_name}_progress_"
        output_dir = os.path.dirname(filepath) or "."
        
        for filename in os.listdir(output_dir):
            if filename.startswith(os.path.basename(temp_pattern)):
                try:
                    os.remove(os.path.join(output_dir, filename))
                except Exception:
                    pass
    else:
        # Progressive save - use temporary filename
        temp_filepath = f"{base_name}_progress_{current_index:06d}{ext}"
        save_data = {
            "results": data,
            "last_processed_index": current_index,
            "total_processed": len(data)
        }
        save_json(save_data, temp_filepath)


def load_progress(filepath: str) -> Tuple[Optional[List[Dict[str, Any]]], int]:
    """
    Load existing progress from temporary files.
    
    Args:
        filepath: Output file path to check for progress files
        
    Returns:
        Tuple of (processed_data, start_index)
        - processed_data: List of already processed items (None if no progress found)
        - start_index: Index to resume from (0 if no progress found)
    """
    base_name, ext = os.path.splitext(filepath)
    temp_pattern = f"{base_name}_progress_"
    
    output_dir = os.path.dirname(filepath) or "."
    
    # Find all progress files
    progress_files = []
    for filename in os.listdir(output_dir):
        if filename.startswith(os.path.basename(temp_pattern)) and filename.endswith(ext):
            try:
                # Extract index from filename
                index_str = filename.replace(os.path.basename(temp_pattern), "").replace(ext, "")
                index = int(index_str)
                progress_files.append((index, os.path.join(output_dir, filename)))
            except ValueError:
                continue
    
    if not progress_files:
        return None, 0
    
    # Get the latest progress file
    latest_index, latest_file = max(progress_files)
    
    try:
        progress_data = load_json(latest_file)
        
        if isinstance(progress_data, dict) and "results" in progress_data:
            results = progress_data["results"]
            last_index = progress_data.get("last_processed_index", len(results) - 1)
            return results, last_index + 1
        elif isinstance(progress_data, list):
            return progress_data, len(progress_data)
        else:
            return None, 0
            
    except Exception as e:
        print(f"Warning: Could not load progress from {latest_file}: {e}")
        return None, 0


def validate_json_structure(data: Any, required_fields: List[str], 
                           data_type: type = dict) -> bool:
    """
    Validate JSON data structure.
    
    Args:
        data: Data to validate
        required_fields: List of required field names
        data_type: Expected data type (dict or list)
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, data_type):
        raise ValueError(f"Expected data type {data_type.__name__}, got {type(data).__name__}")
    
    if data_type == dict:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
    elif data_type == list and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, dict):
            missing_fields = [field for field in required_fields if field not in first_item]
            if missing_fields:
                raise ValueError(f"First item missing required fields: {missing_fields}")
    
    return True
