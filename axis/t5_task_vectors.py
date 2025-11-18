"""Task vectors for T5 models.

This module provides T5TaskVector class for computing task vectors
from T5-base pretrained and finetuned checkpoints.
"""

import torch
from transformers import AutoModelForSeq2SeqLM

from axis.task_vectors import NonLinearTaskVector


class T5TaskVector(NonLinearTaskVector):
    """Task vector for T5 models.
    
    Handles loading T5 models from HuggingFace (pretrained) and
    from .pt checkpoint files (finetuned).
    """
    
    def __init__(
        self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, use_half=True
    ):
        """Initialize a T5TaskVector.
        
        Args:
            pretrained_checkpoint: Either "t5-base" (HuggingFace model name) or path to .pt file
            finetuned_checkpoint: Path to finetuned checkpoint .pt file
            vector: Dictionary containing the task vector directly
            use_half: Whether to convert to half precision (float16)
        """
        super().__init__(
            pretrained_checkpoint=pretrained_checkpoint,
            finetuned_checkpoint=finetuned_checkpoint,
            vector=vector,
            use_half=use_half
        )
    
    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model.
        
        Args:
            checkpoint: Either "t5-base" (HuggingFace) or path to .pt file
            
        Returns:
            Model with state_dict loaded (normalized to have transformer. prefix)
        """
        # If checkpoint is a HuggingFace model name (e.g., "t5-base")
        # Check if it's a model name (not a file path)
        # File paths typically contain '/' and end with '.pt' or are absolute paths
        is_huggingface_model = (
            isinstance(checkpoint, str) and 
            not checkpoint.endswith('.pt') and 
            (checkpoint == "t5-base" or checkpoint.startswith("google/t5") or 
             (checkpoint.startswith("t5-") and '/' not in checkpoint))
        )
        
        if is_huggingface_model:
            # Load from HuggingFace
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            # Normalize state_dict to have transformer. prefix to match finetuned checkpoints
            base_state_dict = model.state_dict()
            normalized_state_dict = {}
            for key, value in base_state_dict.items():
                # Add transformer. prefix if not present
                if not key.startswith("transformer."):
                    normalized_key = f"transformer.{key}"
                else:
                    normalized_key = key
                normalized_state_dict[normalized_key] = value
            
            # Wrap in StateDictWrapper to match finetuned format
            class StateDictWrapper:
                def __init__(self, state_dict):
                    self._state_dict = state_dict
                
                def state_dict(self):
                    return self._state_dict
            
            return StateDictWrapper(normalized_state_dict)
        else:
            # Load from .pt file (state_dict format)
            if isinstance(checkpoint, str):
                state_dict = torch.load(checkpoint, map_location="cpu")
            else:
                state_dict = checkpoint
            
            # If it's already a state_dict (dict of tensors), we need to wrap it
            # For T5, checkpoints are typically state_dicts, not full models
            if isinstance(state_dict, dict) and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
                # Remove _orig_mod. prefix if present (from torch.compile)
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("_orig_mod."):
                        new_key = key[len("_orig_mod."):]
                        cleaned_state_dict[new_key] = value
                    else:
                        cleaned_state_dict[key] = value
                
                # It's a state_dict - create a dummy model wrapper
                class StateDictWrapper:
                    def __init__(self, state_dict):
                        self._state_dict = state_dict
                    
                    def state_dict(self):
                        return self._state_dict
                
                return StateDictWrapper(cleaned_state_dict)
            else:
                # It's a full model object
                return state_dict
    
    def _cast_to_same_type(self, other):
        """Cast other task vector to same type as self."""
        if isinstance(other, T5TaskVector):
            return other
        else:
            # Convert to T5TaskVector
            return T5TaskVector(vector=other.vector, use_half=not self.vector[list(self.vector.keys())[0]].dtype == torch.float32)

