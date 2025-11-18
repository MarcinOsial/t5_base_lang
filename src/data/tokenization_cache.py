"""
Tokenization cache module for pre-tokenizing datasets to avoid repeated tokenization in workers.

This module provides functions to tokenize entire datasets once and cache the results,
eliminating the bottleneck of tokenizing in each DataLoader worker process.
"""

import os
import pickle
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("root")


def _tokenize_single_example(datapoint, tokenizer, max_seq_len):
    """
    Tokenize a single datapoint.
    
    Returns numpy arrays instead of PyTorch tensors to avoid mmap issues
    with multiprocessing 'spawn' method. Tensors will be converted back
    in PytorchDataset.__getitem__.
    
    Args:
        datapoint: Dictionary with 'input' and 'target'/'answer_choices' fields
        tokenizer: HuggingFace tokenizer instance
        max_seq_len: Maximum sequence length
        
    Returns:
        Tokenized datapoint dictionary with numpy arrays (not tensors)
    """
    import numpy as np
    
    # Tokenize input
    input_dict = tokenizer(
        datapoint["input"], 
        return_tensors="pt", 
        truncation=True,
        max_length=max_seq_len
    )
    
    # Convert to numpy arrays for easier pickle/multiprocessing
    tokenized = {
        "input_ids": input_dict["input_ids"][0].cpu().numpy(),
        "input_mask": input_dict["attention_mask"][0].cpu().numpy(),
    }
    
    # Tokenize answer choices if present (evaluation)
    if "answer_choices" in datapoint:
        all_choices_ids = []
        all_choices_masks = []
        for choice in datapoint["answer_choices"]:
            choice_dict = tokenizer(
                choice, 
                return_tensors="pt", 
                truncation=True,
                max_length=max_seq_len
            )
            # Convert to numpy arrays
            all_choices_ids.append(choice_dict["input_ids"][0].cpu().numpy())
            all_choices_masks.append(choice_dict["attention_mask"][0].cpu().numpy())
        
        tokenized["all_choices_ids"] = all_choices_ids
        # Use 'all_choices_mask' (singular) to match T5Wrapper.py expectation
        tokenized["all_choices_mask"] = all_choices_masks
    else:
        # Tokenize target (training)
        if "target" in datapoint:
            target_dict = tokenizer(
                datapoint["target"], 
                return_tensors="pt", 
                truncation=True,
                max_length=max_seq_len
            )
            # Convert to numpy arrays
            tokenized["target_ids"] = target_dict["input_ids"][0].cpu().numpy()
            tokenized["target_mask"] = target_dict["attention_mask"][0].cpu().numpy()
    
    # Preserve other fields (lbl, idx, etc.)
    for key in ["lbl", "idx"]:
        if key in datapoint:
            tokenized[key] = datapoint[key]
    
    return tokenized


def _get_tokenization_cache_path(
    dataset_name: str,
    split: str,
    template_idx: int,
    tokenizer_name: str,
    max_seq_len: int,
    cache_dir: Optional[Path] = None
) -> Path:
    """
    Generate cache path for tokenized dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (train/validation/test)
        template_idx: Template index used
        tokenizer_name: Name/identifier of tokenizer
        max_seq_len: Maximum sequence length
        cache_dir: Base cache directory. If None, uses .cache/tokenized/ in project root
        
    Returns:
        Path to cache file
    """
    if cache_dir is None:
        project_root = Path(__file__).parent.parent.parent
        cache_dir = project_root / ".cache" / "tokenized"
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache filename from parameters
    cache_filename = f"{dataset_name}_{split}_template{template_idx}_{tokenizer_name}_maxlen{max_seq_len}.pkl"
    return cache_dir / cache_filename


def _compute_tokenization_hash(dataset: List[Dict[str, Any]], tokenizer_name: str, max_seq_len: int) -> str:
    """
    Compute hash for tokenization cache validation.
    
    Args:
        dataset: List of datapoints
        tokenizer_name: Name/identifier of tokenizer
        max_seq_len: Maximum sequence length
        
    Returns:
        Hash string for cache validation
    """
    if not dataset:
        return "empty"
    
    # Create hash from dataset characteristics and tokenizer config
    hash_data = {
        "count": len(dataset),
        "first_input": dataset[0].get("input", "")[:100] if dataset else "",  # First 100 chars
        "last_input": dataset[-1].get("input", "")[:100] if dataset else "",
        "tokenizer": tokenizer_name,
        "max_seq_len": max_seq_len,
    }
    hash_str = json.dumps(hash_data, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()


def _migrate_old_cache_format(tokenized_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Migrate old cache format to new format.
    
    Old format used 'all_choices_masks' (with 's'), new format uses 'all_choices_mask' (without 's').
    This function converts old cache entries to the new format.
    
    Args:
        tokenized_dataset: List of tokenized datapoints (may contain old format)
        
    Returns:
        List of tokenized datapoints with new format (all_choices_mask instead of all_choices_masks)
    """
    migrated_count = 0
    for datapoint in tokenized_dataset:
        # Check if old format key exists
        if "all_choices_masks" in datapoint and "all_choices_mask" not in datapoint:
            # Migrate: rename old key to new key
            datapoint["all_choices_mask"] = datapoint.pop("all_choices_masks")
            migrated_count += 1
    
    if migrated_count > 0:
        logger.info(f"Migrated {migrated_count} examples from old cache format (all_choices_masks -> all_choices_mask)")
    
    return tokenized_dataset


def _load_tokenization_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load tokenized dataset from disk cache.
    
    Automatically migrates old cache format (all_choices_masks) to new format (all_choices_mask).
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Cached data dict if valid, None otherwise (with migrated format)
    """
    if not cache_path.exists():
        return None
    
    try:
        logger.info(f"Loading tokenized dataset from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Migrate old cache format to new format if needed
        needs_save = False
        if "tokenized_dataset" in cached_data:
            # Check if any examples need migration (have old key)
            has_old_format = any("all_choices_masks" in dp for dp in cached_data["tokenized_dataset"])
            if has_old_format:
                # Migration needed - perform it
                cached_data["tokenized_dataset"] = _migrate_old_cache_format(cached_data["tokenized_dataset"])
                # Save migrated cache back to disk to avoid re-migration on next load
                needs_save = True
        
        # Save migrated cache back to disk to avoid re-migration on next load
        if needs_save:
            logger.info(f"Saving migrated cache back to disk: {cache_path}")
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cached_data, f)
                logger.info(f"Migrated cache saved successfully")
            except Exception as save_error:
                logger.warning(f"Failed to save migrated cache: {save_error} (will re-migrate on next load)")
        
        logger.info(f"Tokenization cache loaded successfully: {len(cached_data.get('tokenized_dataset', []))} examples")
        return cached_data
    except Exception as e:
        logger.warning(f"Failed to load tokenization cache from {cache_path}: {e}")
        return None


def _save_tokenization_cache(
    cache_path: Path,
    tokenized_dataset: List[Dict[str, Any]],
    dataset_hash: str,
    dataset_name: str
):
    """
    Save tokenized dataset to disk cache.
    
    Args:
        cache_path: Path to cache file
        tokenized_dataset: List of tokenized datapoints
        dataset_hash: Hash of original dataset for validation
        dataset_name: Name of the dataset
    """
    try:
        cache_data = {
            "tokenized_dataset": tokenized_dataset,
            "dataset_hash": dataset_hash,
            "dataset_name": dataset_name,
        }
        logger.info(f"Saving tokenized dataset to cache: {cache_path} ({len(tokenized_dataset)} examples)")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Tokenization cache saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save tokenization cache to {cache_path}: {e}")


def pre_tokenize_dataset(
    dataset: List[Dict[str, Any]],
    tokenizer: Any,
    dataset_name: str,
    split: str,
    template_idx: int,
    max_seq_len: int = 128,
    num_workers: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Pre-tokenize entire dataset with disk caching and multiprocessing support.
    
    This function tokenizes the entire dataset once in the main process (or with multiprocessing),
    avoiding repeated tokenization in DataLoader workers. Results are cached to disk.
    
    Args:
        dataset: List of datapoints with 'input' and 'target'/'answer_choices' fields (strings)
        tokenizer: HuggingFace tokenizer instance
        dataset_name: Name of the dataset (for cache naming)
        split: Dataset split (train/validation/test)
        template_idx: Template index used
        max_seq_len: Maximum sequence length for tokenization
        num_workers: Number of workers for multiprocessing. If None, uses CPU count - 1
        cache_dir: Base cache directory. If None, uses .cache/tokenized/ in project root
        use_cache: Whether to use disk cache (default: True)
        
    Returns:
        List of tokenized datapoints with 'input_ids', 'input_mask', 'target_ids', etc.
    """
    # Get tokenizer identifier for cache
    tokenizer_name = getattr(tokenizer, "name_or_path", "unknown")
    if hasattr(tokenizer, "vocab_size"):
        tokenizer_name = f"{tokenizer_name}_vocab{tokenizer.vocab_size}"
    
    # Compute hash for cache validation
    dataset_hash = _compute_tokenization_hash(dataset, tokenizer_name, max_seq_len)
    
    # Check disk cache
    if use_cache:
        cache_path = _get_tokenization_cache_path(
            dataset_name, split, template_idx, tokenizer_name, max_seq_len, cache_dir
        )
        cached_data = _load_tokenization_cache(cache_path)
        
        if cached_data is not None and cached_data.get("dataset_hash") == dataset_hash:
            # Cache hit - return cached tokenized dataset
            logger.info(f"Using tokenized dataset from cache (hash match)")
            return cached_data["tokenized_dataset"]
    
    # Cache miss or cache disabled - tokenize from scratch
    logger.info(f"Pre-tokenizing dataset from scratch ({len(dataset)} examples)")
    
    # Tokenize sequentially (tokenizer is not pickleable, so multiprocessing is not straightforward)
    # Sequential tokenization is fast enough for pre-processing step (happens once, cached to disk)
    logger.info(f"Tokenizing dataset sequentially ({len(dataset)} examples)")
    tokenized_dataset = []
    for idx, datapoint in enumerate(dataset):
        if (idx + 1) % 1000 == 0:
            logger.info(f"Tokenized {idx + 1}/{len(dataset)} examples...")
        tokenized = _tokenize_single_example(datapoint, tokenizer, max_seq_len)
        tokenized_dataset.append(tokenized)
    
    logger.info(f"Tokenization completed: {len(tokenized_dataset)} examples processed")
    
    # Save to disk cache
    if use_cache:
        _save_tokenization_cache(cache_path, tokenized_dataset, dataset_hash, dataset_name)
    
    return tokenized_dataset

