#!/usr/bin/env python3
"""
Test script to verify winogrande dataset loading solution.
This abstracts the solution from the main codebase to test it independently.
"""

import os
import shutil
import glob
from datasets import load_dataset

# Set HF_HOME to match training.py configuration
os.environ["HF_HOME"] = os.path.join("/raid/NFS_SHARE/home/marcin.osial/ties-merging/.cache/huggingface/")

def clean_winogrande_cache(cache_base_dir=None):
    """Clean corrupted winogrande cache files and directories."""
    if cache_base_dir is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_base_dir = os.path.join(hf_home, "datasets")
        else:
            cache_base_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    
    if not os.path.exists(cache_base_dir):
        print(f"Cache directory does not exist: {cache_base_dir}")
        return
    
    print(f"Cleaning winogrande cache from: {cache_base_dir}")
    removed_count = 0
    
    # Remove parquet folders
    parquet_dir = os.path.join(cache_base_dir, "parquet")
    if os.path.exists(parquet_dir):
        winogrande_patterns = [
            os.path.join(parquet_dir, "winogrande_debiased-*"),
            os.path.join(parquet_dir, "winogrande_xl-*"),
            os.path.join(parquet_dir, "winogrande_*-*"),
        ]
        for pattern in winogrande_patterns:
            for cache_dir in glob.glob(pattern):
                if os.path.isdir(cache_dir):
                    try:
                        shutil.rmtree(cache_dir)
                        print(f"Removed cache directory: {cache_dir}")
                        removed_count += 1
                    except Exception as e:
                        print(f"Failed to remove {cache_dir}: {e}")
    
    # Remove lock files
    lock_patterns = [
        os.path.join(cache_base_dir, "*winogrande*.lock"),
        os.path.join(cache_base_dir, "*winogrande*_builder.lock"),
        os.path.join(cache_base_dir, "*winogrande*_incomplete_info.lock"),
    ]
    for pattern in lock_patterns:
        for lock_file in glob.glob(pattern):
            if os.path.isfile(lock_file):
                try:
                    os.remove(lock_file)
                    print(f"Removed lock file: {lock_file}")
                    removed_count += 1
                except Exception as e:
                    print(f"Failed to remove {lock_file}: {e}")
    
    print(f"Cache cleanup completed. Removed {removed_count} items.\n")

def test_load_approach_old(split="train"):
    """Old approach: load single split (causes NonMatchingSplitsSizesError)"""
    print(f"=== Testing OLD approach: loading split='{split}' ===")
    try:
        # This is the problematic approach
        data = load_dataset("winogrande", "winogrande_debiased", split=split)
        print(f"✓ Successfully loaded split '{split}'")
        print(f"  Size: {len(data)} examples")
        return True
    except Exception as e:
        print(f"✗ Error loading split '{split}': {e}")
        return False

def test_load_approach_new(split="train"):
    """New approach: load full dataset, then select split"""
    print(f"=== Testing NEW approach: loading full dataset, then selecting split='{split}' ===")
    try:
        # Load entire dataset (all splits) first
        full_dataset = load_dataset("winogrande", "winogrande_debiased")
        print(f"✓ Successfully loaded full dataset")
        print(f"  Available splits: {list(full_dataset.keys())}")
        
        # Select the specific split we need
        data = full_dataset[split]
        print(f"✓ Successfully selected split '{split}'")
        print(f"  Size: {len(data)} examples")
        
        # Verify all splits have correct sizes
        print(f"\n  Split sizes verification:")
        print(f"    train: {len(full_dataset['train'])} (expected: 9248)")
        print(f"    test: {len(full_dataset['test'])} (expected: 1767)")
        print(f"    validation: {len(full_dataset['validation'])} (expected: 1267)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Winogrande Dataset Loading Solutions")
    print("=" * 80)
    print()
    
    # Clean cache first
    print("Step 1: Cleaning cache...")
    clean_winogrande_cache()
    
    # Test old approach (should fail)
    print("\nStep 2: Testing OLD approach (single split loading)...")
    old_success = test_load_approach_old("train")
    
    # Clean cache again before new test
    print("\nStep 3: Cleaning cache again...")
    clean_winogrande_cache()
    
    # Test new approach (should succeed)
    print("\nStep 4: Testing NEW approach (full dataset loading)...")
    new_success = test_load_approach_new("train")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Old approach (single split): {'✓ SUCCESS' if old_success else '✗ FAILED'}")
    print(f"New approach (full dataset): {'✓ SUCCESS' if new_success else '✗ FAILED'}")
    print("=" * 80)

