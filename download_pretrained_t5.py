#!/usr/bin/env python3
"""
Script to download and save pretrained T5-base model.
This ensures the model is cached and saves it as pretrained.pt for merging purposes.
"""

import os
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set HF_HOME to match training.py configuration
os.environ["HF_HOME"] = os.path.join(
    "/raid/NFS_SHARE/home/marcin.osial/ties-merging/.cache/huggingface/"
)

print("=" * 80)
print("Downloading pretrained T5-base model from HuggingFace Hub...")
print("=" * 80)

# Model name from config
pretrained_model_name = "t5-base"

# Download model and tokenizer (will be cached automatically)
print(f"\n1. Loading model: {pretrained_model_name}")
print("   (This will download from HuggingFace Hub if not cached)")
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

print(f"\n2. Model loaded successfully!")
print(f"   Model type: {type(model)}")
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Save model state_dict as pretrained.pt
pretrained_path = "/raid/NFS_SHARE/home/marcin.osial/ties-merging/models/pretrained.pt"
print(f"\n3. Saving pretrained model state_dict to: {pretrained_path}")

# Get state_dict
pretrained_state_dict = model.state_dict()

# Save to file
torch.save(pretrained_state_dict, pretrained_path)
print(f"   ✓ Saved successfully! File size: {os.path.getsize(pretrained_path) / (1024**3):.2f} GB")

# Verify file exists
if os.path.exists(pretrained_path):
    print(f"\n4. Verification: File exists at {pretrained_path}")
    print("   ✓ Pretrained model ready for merging!")
else:
    print(f"\n4. ERROR: File not found at {pretrained_path}")
    sys.exit(1)

print("\n" + "=" * 80)
print("SUCCESS: Pretrained T5-base model downloaded and saved!")
print("=" * 80)
print(f"\nModel cached at: {os.environ['HF_HOME']}")
print(f"Pretrained checkpoint saved at: {pretrained_path}")
print("\nYou can now proceed with training fine-tuned models.")

