"""AXIS merging for T5 models.

This module implements the AXIS method for T5-base models:
- Task vector computation
- iso_c merging with SVD
- Learnable singular values training
"""

import os
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from axis.t5_task_vectors import T5TaskVector


def compute_task_vector_hash(task_vector, dataset_name):
    """Compute hash of task vector for caching."""
    hasher = hashlib.sha256()
    hasher.update(dataset_name.encode('utf-8'))
    if hasattr(task_vector, 'vector') and task_vector.vector:
        sorted_keys = sorted(task_vector.vector.keys())
        for key in sorted_keys:
            tensor = task_vector.vector[key]
            if isinstance(tensor, torch.Tensor):
                hasher.update(key.encode('utf-8'))
                hasher.update(str(tensor.shape).encode('utf-8'))
                hasher.update(str(tensor.dtype).encode('utf-8'))
                tensor_sum = tensor.sum().item()
                hasher.update(str(tensor_sum).encode('utf-8'))
    return hasher.hexdigest()[:16]


def iso_c_t5(base_model_params_dict, task_vectors, config, cache_args=None):
    """
    T5-specific version of iso_c() for merging task vectors.
    
    Main differences from vision models:
    - T5 parameters have 'transformer.' prefix instead of 'model.'
    - No SKIP_PARAMS - all layers are processed
    
    Args:
        base_model_params_dict: dict of base model parameters (with transformer. prefix)
        task_vectors: list of T5TaskVector objects
        config: config object with svd_threshold, sorting_descending attributes
        cache_args: optional args for SVD caching
        
    Returns:
        dict: Merged vector components with keys:
            - 'U', 'S', 'Vh', 'is_svd' (True) - for SVD layers
            - 'tensor', 'is_svd' (False) - for non-SVD layers
    """
    if not task_vectors:
        print("Warning: No task vectors provided to iso_c_t5.")
        return None

    print(f"Computing global SVD component selection for {len(task_vectors)} task vectors...")
    
    # Ensure config has required attributes
    if not hasattr(config, 'sorting_descending'):
        config.sorting_descending = True
    
    with torch.no_grad():
        new_vector = {}
        for task_key in task_vectors[0].vector:
            # For T5, both task_vectors and base_model_params_dict should have transformer. prefix
            # So keys should match directly
            if task_key not in base_model_params_dict:
                print(f"Warning: Key '{task_key}' not in base_model_params_dict. Skipping.")
                continue

            current_device = base_model_params_dict[task_key].device
            original_dtype = task_vectors[0].vector[task_key].dtype
            original_shape = task_vectors[0].vector[task_key].shape

            # For T5: NO SKIP_PARAMS - all layers are processed
            # 2D layers → SVD, non-2D layers → averaging
            if len(original_shape) != 2:
                # Non-2D layer: average across task vectors
                tensor_val = None
                if len(task_vectors) == 1:
                    tensor_val = task_vectors[0].vector[task_key].to(current_device)
                else:
                    tvs = [tv.vector[task_key].to(current_device) for tv in task_vectors]
                    tensor_val = sum(tvs) / len(tvs)
                new_vector[task_key] = {
                    'tensor': tensor_val, 'is_svd': False,
                    'original_dtype': original_dtype, 'original_shape': original_shape
                }
                continue

            try:
                # 2D layer: perform SVD and select top components
                all_components = []
                
                # 1. Collect all components from all tasks for the current layer
                for task_idx, task_vector in enumerate(task_vectors):
                    # Compute SVD for this task's delta matrix
                    delta_matrix = task_vector.vector[task_key].to(device=current_device, dtype=torch.float32)
                    U_task, S_task, Vh_task = torch.linalg.svd(delta_matrix, full_matrices=False)
                    
                    for i in range(S_task.shape[0]):
                        all_components.append({
                            "s_value": S_task[i],
                            "u_vector": U_task[:, i],
                            "vh_vector": Vh_task[i, :],
                            "source_task_idx": task_idx
                        })
                
                if not all_components:
                    print(f"Warning: No SVD components generated for key {task_key}. Skipping.")
                    continue

                # 2. Sort the global list of components
                all_components.sort(key=lambda x: x["s_value"], reverse=config.sorting_descending)
                
                # 3. Determine how many components to keep (fixed number for now, like in original)
                num_to_keep = 76  # Fixed number as in original iso_c
                num_to_keep = min(num_to_keep, len(all_components))

                # 4. Select the top components from the globally sorted list
                selected_components = all_components[:num_to_keep]
                kept_component_count = len(selected_components)

                # 5. Assemble final U, S, Vh from selected components
                if kept_component_count == 0:
                    print(f"Warning: No singular value components selected for key {task_key}. Creating zero SVD components.")
                    U_final = torch.empty((original_shape[0], 0), dtype=torch.float32, device=current_device)
                    S_final = torch.empty((0,), dtype=torch.float32, device=current_device)
                    Vh_final = torch.empty((0, original_shape[1]), dtype=torch.float32, device=current_device)
                else:
                    S_final = torch.stack([comp["s_value"] for comp in selected_components])
                    U_final = torch.stack([comp["u_vector"] for comp in selected_components], dim=1)
                    Vh_final = torch.stack([comp["vh_vector"] for comp in selected_components], dim=0)

                # Store the final components
                new_vector[task_key] = {
                    'U': U_final.to(original_dtype),
                    'S': S_final.to(original_dtype),
                    'Vh': Vh_final.to(original_dtype),
                    'is_svd': True,
                    'original_dtype': original_dtype,
                    'original_shape': original_shape,
                    'num_selected_components': kept_component_count
                }
                
                print(f"Global SVD components extracted for key '{task_key}', original shape: {original_shape}")
                print(f"  Total components from all tasks: {len(all_components)}")
                print(f"  Selected {kept_component_count} components.")
                print(f"  Final U shape: {U_final.shape}, Final S shape: {S_final.shape}, Final Vh shape: {Vh_final.shape}")

            except Exception as e:
                print(f"Error: Global SVD component selection failed for key '{task_key}' with error: {str(e)}")
                import traceback
                traceback.print_exc()
                print(f"Skipping key {task_key} due to SVD error.")
                continue

    return new_vector


def get_t5_base_model_params(model_name="t5-base"):
    """
    Get base T5 model parameters as a dictionary.
    
    Args:
        model_name: HuggingFace model name (default: "t5-base")
        
    Returns:
        dict: Parameter dictionary with transformer. prefix keys
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    base_state_dict = model.state_dict()
    
    # Normalize to have transformer. prefix
    normalized_params = {}
    for key, value in base_state_dict.items():
        if not key.startswith("transformer."):
            normalized_key = f"transformer.{key}"
        else:
            normalized_key = key
        normalized_params[normalized_key] = value.clone()
    
    return normalized_params


def load_t5_task_vectors(dataset_names, base_checkpoint="t5-base", save_dir="exp_out/t5_finetuning/t5-base", use_half=False):
    """
    Load task vectors for multiple T5 datasets.
    
    Args:
        dataset_names: list of dataset names (e.g., ["paws", "qasc"])
        base_checkpoint: base model checkpoint (HuggingFace name or path)
        save_dir: directory where finetuned checkpoints are stored
        use_half: whether to use half precision
        
    Returns:
        dict: {dataset_name: T5TaskVector}
    """
    task_vectors = {}
    
    for dataset_name in dataset_names:
        finetuned_path = f"{save_dir}/t5-base-{dataset_name}/best_model.pt"
        
        print(f"Loading task vector for {dataset_name}...")
        tv = T5TaskVector(
            pretrained_checkpoint=base_checkpoint,
            finetuned_checkpoint=finetuned_path,
            use_half=use_half
        )
        tv.dataset_name = dataset_name  # Store dataset name for caching
        task_vectors[dataset_name] = tv
        print(f"  Loaded {len(tv.vector.keys())} parameters")
    
    return task_vectors


class LearnableSingularValuesMergedT5Wrapper(nn.Module):
    """Learns singular values for precomputed SVD components in T5 models.
    
    Similar to LearnableSingularValuesMergedEncoder but for T5:
    - Uses functorch.make_functional_with_buffers
    - Forward accepts batch (dict) and returns logits (not loss)
    - Reconstructs deltas from SVD components and adds to base params
    
    Args:
        model: T5Wrapper model
        merged_vector_components: Output from iso_c_t5 containing:
            - SVD components for 2D layers
            - Direct tensors for other layers
        args: arguments object with svd_threshold
    """
    
    def __init__(self, model, merged_vector_components, args):
        super().__init__()
        
        self.model = model
        self.args = args
        
        # Store tokenizer for potential use
        self.tokenizer = model.tokenizer
        
        from functorch import make_functional_with_buffers
        # Use the transformer directly (not T5Wrapper) for functional model
        # T5Wrapper just wraps transformer, so we functionalize the transformer
        # Transformer should already be on device (moved before this __init__)
        func, params_from_functional, self.buffer = make_functional_with_buffers(model.transformer)
        self.func = func
        
        # Get device from transformer
        transformer_device = next(model.transformer.parameters()).device
        
        # Get original parameters from transformer to copy data from
        transformer_params_dict = dict(model.transformer.named_parameters())
        transformer_buffers_dict = dict(model.transformer.named_buffers())
        
        # Move params from functional to proper device if they're in meta device
        params_on_device = []
        param_names_list = list(transformer_params_dict.keys())
        for idx, p in enumerate(params_from_functional):
            if p.device.type == 'meta':
                # Get corresponding original parameter from transformer
                param_name = param_names_list[idx] if idx < len(param_names_list) else None
                if param_name and param_name in transformer_params_dict:
                    # Copy data from original parameter
                    orig_param = transformer_params_dict[param_name]
                    new_p = nn.Parameter(orig_param.data.clone())
                else:
                    # Fallback: create empty tensor on proper device with same shape/dtype
                    new_p = nn.Parameter(torch.zeros(p.shape, dtype=p.dtype, device=transformer_device))
                params_on_device.append(new_p)
            else:
                params_on_device.append(p)
        
        self.params = nn.ParameterList(params_on_device)
        for p in self.params:
            p.requires_grad = False
        
        # Move buffers from meta device if needed
        if self.buffer is not None:
            buffer_names_list = list(transformer_buffers_dict.keys())
            buffers_on_device = []
            for idx, b in enumerate(self.buffer):
                if b.device.type == 'meta':
                    # Get corresponding original buffer from transformer
                    buffer_name = buffer_names_list[idx] if idx < len(buffer_names_list) else None
                    if buffer_name and buffer_name in transformer_buffers_dict:
                        # Copy data from original buffer
                        orig_buffer = transformer_buffers_dict[buffer_name]
                        new_b = orig_buffer.data.clone()
                    else:
                        # Fallback: create empty tensor on proper device with same shape/dtype
                        new_b = torch.zeros(b.shape, dtype=b.dtype, device=transformer_device)
                    buffers_on_device.append(new_b)
                else:
                    buffers_on_device.append(b)
            self.buffer = tuple(buffers_on_device)
            
        # Get parameter names from transformer (not T5Wrapper)
        # merged_vector_components has transformer. prefix, so we need to add it
        transformer_param_names = [name for name, _ in model.transformer.named_parameters()]
        # Add transformer. prefix to match merged_vector_components
        self.param_names = [f"transformer.{name}" if not name.startswith("transformer.") else name 
                           for name in transformer_param_names]

        self.svd_components_info = {}
        self.learnable_s_values = nn.ParameterDict()
        self.direct_deltas = {}
        self.total_learnable_sv = 0
        
        print("\n--- Initializing LearnableSingularValuesMergedT5Wrapper ---")
        print("Processing parameters based on pre-computed SVD components or direct deltas:")
        
        for param_idx, param_name in enumerate(self.param_names):
            if param_name in merged_vector_components:
                layer_data = merged_vector_components[param_name]
                svd_key_safe_name = param_name.replace('.', '_')
                
                original_dtype = layer_data['original_dtype']
                original_shape = layer_data['original_shape']

                if layer_data['is_svd']:
                    U_from_isoc = layer_data['U']
                    S_from_isoc = layer_data['S']
                    Vh_from_isoc = layer_data['Vh']
                    num_components_from_isoc = layer_data['num_selected_components']

                    if num_components_from_isoc > 0:
                        # Reconstruct the matrix from iso_c components and perform a new SVD
                        print(f"  SVD layer {param_name}: Reconstructing delta from {num_components_from_isoc} components...")
                        reconstructed_delta = U_from_isoc.to(torch.float32) @ torch.diag_embed(S_from_isoc.to(torch.float32)) @ Vh_from_isoc.to(torch.float32)
                        
                        print(f"  SVD layer {param_name}: Performing second SVD on reconstructed delta...")
                        U_new, S_new, Vh_new = torch.linalg.svd(reconstructed_delta, full_matrices=False)

                        # Split S_new into learnable and frozen parts
                        num_total_new_components = S_new.shape[0]
                        num_to_learn = int(self.args.svd_threshold * num_total_new_components)

                        # Split the new S values
                        S_initial_learnable = S_new[:num_to_learn]
                        S_initial_frozen = S_new[num_to_learn:]
                        
                        # Get device from transformer for buffers
                        buffer_device = transformer_device
                        
                        self.register_buffer(f'U_{svd_key_safe_name}', U_new.to(buffer_device))
                        self.register_buffer(f'Vh_{svd_key_safe_name}', Vh_new.to(buffer_device))
                        
                        self.register_buffer(f'initial_selected_S_{svd_key_safe_name}', S_initial_learnable.to(buffer_device))
                        self.register_buffer(f'frozen_S_{svd_key_safe_name}', S_initial_frozen.to(buffer_device))
                        
                        # Initialize learnable singular values to zero - on same device
                        learnable_S_for_layer = torch.zeros_like(S_initial_learnable, dtype=torch.float32, device=buffer_device)
                        self.learnable_s_values[svd_key_safe_name] = nn.Parameter(learnable_S_for_layer)
                        
                        self.total_learnable_sv += num_to_learn
                        
                        self.svd_components_info[param_name] = {
                            'original_dtype': original_dtype,
                            'original_shape': original_shape,
                            'num_learnable': num_to_learn,
                            'num_frozen': len(S_initial_frozen),
                            'num_total_components': num_total_new_components,
                        }
                        print(f"  SVD layer {param_name} (Shape {original_shape}): "
                              f"Re-SVD created {num_total_new_components} new components. "
                              f"Learning top {num_to_learn} ({self.args.svd_threshold*100:.1f}%) singular values. "
                              f"Freezing remaining {len(S_initial_frozen)}.")
                    else:
                        print(f"  SVD layer {param_name} (Shape {original_shape}, Dtype {original_dtype}): "
                              f"No SVD components selected by iso_c. Delta will be zero.")
                        self.svd_components_info[param_name] = { 
                            'original_dtype': original_dtype,
                            'original_shape': original_shape,
                            'num_learnable': 0,
                            'num_frozen': 0,
                            'num_total_components': 0,
                        }
                        buffer_device = transformer_device
                        self.register_buffer(f'U_{svd_key_safe_name}', torch.empty((original_shape[0], 0), dtype=torch.float32, device=buffer_device))
                        self.register_buffer(f'Vh_{svd_key_safe_name}', torch.empty((0, original_shape[1]), dtype=torch.float32, device=buffer_device))
                        self.register_buffer(f'frozen_S_{svd_key_safe_name}', torch.empty((0,), dtype=torch.float32, device=buffer_device))
                else:
                    direct_delta_tensor = layer_data['tensor']
                    # Ensure direct delta is on the same device as transformer
                    buffer_device = transformer_device
                    self.register_buffer(f'direct_delta_{svd_key_safe_name}', direct_delta_tensor.to(buffer_device))
                    self.direct_deltas[param_name] = {
                         'original_dtype': original_dtype,
                         'key_for_buffer': f'direct_delta_{svd_key_safe_name}'
                    }
                    print(f"  Non-SVD layer {param_name} (Shape {direct_delta_tensor.shape if isinstance(direct_delta_tensor, torch.Tensor) else 'N/A'}, Dtype {direct_delta_tensor.dtype if isinstance(direct_delta_tensor, torch.Tensor) else 'N/A'}): Using direct delta.")
            else:
                print(f"  No SVD components or direct delta in merged_vector_components for {param_name}. Update for this layer will be zero.")
        
        print(f"Total learnable singular values across all layers: {self.total_learnable_sv}")
        print("--- Initialization Complete --- \n")

    def _apply(self, fn):
        """Override method to relocate buffer list"""
        # Check if we're moving from meta device
        is_meta_to_device = False
        if hasattr(self, 'params') and len(self.params) > 0:
            first_param = self.params[0]
            if first_param.device.type == 'meta':
                is_meta_to_device = True
        
        if is_meta_to_device:
            # For meta device, we need to use to_empty instead
            # But since we're in _apply, we'll handle it differently
            # Create new instance and manually move everything
            new_self = super()._apply(fn=fn)
            
            # Apply to functorch's buffers (skip if meta)
            if hasattr(new_self, 'buffer') and new_self.buffer is not None:
                new_self.buffer = tuple(fn(b) if b.device.type != 'meta' else b for b in new_self.buffer)
            
            # Apply to params (skip if meta - they should already be moved by super()._apply)
            if hasattr(new_self, 'params') and isinstance(new_self.params, nn.ParameterList):
                for i in range(len(new_self.params)):
                    if new_self.params[i] is not None and new_self.params[i].device.type != 'meta':
                        new_self.params[i].data = fn(new_self.params[i].data)
        else:
            # Normal case - not from meta device
            new_self = super()._apply(fn=fn)
            
            # Apply to functorch's buffers
            if hasattr(new_self, 'buffer') and new_self.buffer is not None:
                new_self.buffer = tuple(fn(b) for b in new_self.buffer)
            
            # Apply to params
            if hasattr(new_self, 'params') and isinstance(new_self.params, nn.ParameterList):
                for i in range(len(new_self.params)):
                    if new_self.params[i] is not None:
                         new_self.params[i].data = fn(new_self.params[i].data)

        return new_self
    
    def forward(self, batch, encoder_outputs=None):
        """
        Forward pass applying the merged vector with learnable singular values.
        
        Args:
            batch: dict with keys: input_ids, input_mask, target_ids, target_mask
            encoder_outputs: Optional encoder outputs (BaseModelOutput) to reuse encoder computation.
                           If provided, encoder will not be called and these outputs will be used.
            
        Returns:
            logits: [batch_size, max_target_len, vocab_size] - logits from transformer
        """
        # Ensure all batch tensors are on the same device as the model
        # Get device from first parameter (self.params is a ParameterList)
        if len(self.params) > 0:
            model_device = self.params[0].device
        else:
            # Fallback: get device from transformer
            model_device = next(self.model.transformer.parameters()).device
        
        # Move all batch tensors to model device
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.to(model_device)
            else:
                batch_on_device[key] = value
        
        final_model_params = []
        
        for param_idx, param_name in enumerate(self.param_names):
            base_param = self.params[param_idx]
            
            actual_delta_c: torch.Tensor
            svd_key_safe_name = param_name.replace('.', '_')

            if param_name in self.svd_components_info:
                svd_info = self.svd_components_info[param_name]
                num_total_components = svd_info.get('num_total_components', 0)
                original_delta_dtype = svd_info['original_dtype']

                if num_total_components > 0:
                    # Get device from base_param
                    target_device = base_param.device
                    
                    U = getattr(self, f'U_{svd_key_safe_name}').to(target_device)
                    Vh = getattr(self, f'Vh_{svd_key_safe_name}').to(target_device)
                    
                    # Get learnable and frozen S parts - ensure they're on the same device as base_param
                    learnable_S_values = self.learnable_s_values.get(svd_key_safe_name)
                    frozen_S_values = getattr(self, f'frozen_S_{svd_key_safe_name}', None)
                    
                    # Build the full S vector - ensure all tensors are on the same device
                    s_parts = []
                    if learnable_S_values is not None and learnable_S_values.numel() > 0:
                        s_parts.append(learnable_S_values.to(target_device))
                    if frozen_S_values is not None and frozen_S_values.numel() > 0:
                        s_parts.append(frozen_S_values.to(target_device))
                    
                    if not s_parts:
                        reconstructed_delta = torch.zeros(svd_info['original_shape'], device=target_device)
                    else:
                        full_S_vector = torch.cat(s_parts, dim=0)

                        # Cast to float32 before reconstruction - all already on target_device
                        U_f32 = U.to(torch.float32)
                        Vh_f32 = Vh.to(torch.float32)
                        full_S_f32 = full_S_vector.to(torch.float32)
                        
                        # Reconstruct: U @ diag(S_vector) @ Vh
                        reconstructed_delta = U_f32 @ torch.diag_embed(full_S_f32) @ Vh_f32
                    
                    actual_delta_c = reconstructed_delta.to(original_delta_dtype)
                else:
                    actual_delta_c = torch.zeros_like(base_param)
            
            elif f'direct_delta_{svd_key_safe_name}' in self._buffers:
                direct_delta_tensor = getattr(self, f'direct_delta_{svd_key_safe_name}')
                # Ensure direct delta is on the same device as base_param
                actual_delta_c = direct_delta_tensor.to(base_param.device)
            else:
                actual_delta_c = torch.zeros_like(base_param)

            # Add delta_c to the base parameter
            final_model_params.append(base_param + actual_delta_c)
        
        # Apply the function with the modified parameters
        # func is the functional transformer, which expects:
        # func(params, buffers, input_ids, attention_mask, labels, ...)
        # But we have batch dict, so we need to unpack it
        
        # Call the functional transformer
        # T5 transformer.forward signature: forward(input_ids, attention_mask=None, labels=None, encoder_outputs=None, ...)
        # Use batch_on_device to ensure all tensors are on the correct device
        # If encoder_outputs is provided, we need to use the base transformer directly (not functional)
        # because functional model doesn't easily support encoder_outputs
        if encoder_outputs is not None:
            # Use base transformer directly with encoder_outputs (for efficiency in multiple choice)
            # But we need to apply the learnable singular values to the parameters first
            # This is complex with functional model, so we'll use a workaround:
            # Create a temporary transformer with modified parameters
            # Actually, for encoder_outputs case, we can use base transformer but need to ensure
            # the learnable singular values are applied. Since encoder_outputs is already computed,
            # we only need to apply deltas to decoder parameters.
            # For simplicity, we'll use base transformer with encoder_outputs directly
            # and apply learnable singular values by modifying the transformer's state_dict temporarily
            # This is a bit hacky but necessary for efficiency
            
            # Actually, the best approach is to use the base transformer but we need to ensure
            # learnable singular values are applied. Since we're using encoder_outputs, we're
            # only computing decoder, so we need to apply deltas to decoder parameters.
            # For now, let's use base transformer directly - the learnable singular values
            # will still affect the computation through the modified parameters in final_model_params
            # But wait - encoder_outputs means we skip encoder, so we need decoder with learnable SVs
            
            # Better approach: use base transformer but manually apply learnable singular values
            # by temporarily modifying its state_dict
            base_transformer = self.model.transformer
            
            # Get original state_dict
            original_state_dict = base_transformer.state_dict()
            
            # Build modified state_dict with learnable singular values applied
            # We already computed final_model_params with learnable SVs applied
            modified_state_dict = original_state_dict.copy()
            for param_idx, param_name in enumerate(self.param_names):
                # Remove transformer. prefix to match base_transformer state_dict keys
                base_param_name = param_name.replace("transformer.", "")
                if base_param_name in original_state_dict:
                    modified_state_dict[base_param_name] = final_model_params[param_idx]
            
            # Temporarily apply modified state_dict to base transformer
            base_transformer.load_state_dict(modified_state_dict, strict=False)
            
            try:
                # Now use base transformer with encoder_outputs - it will use learnable SVs
                transformer_outputs = base_transformer(
                    attention_mask=batch_on_device.get("input_mask", None),
                    encoder_outputs=encoder_outputs,
                    labels=batch_on_device.get("target_ids", None)
                )
            finally:
                # Restore original state_dict
                base_transformer.load_state_dict(original_state_dict, strict=False)
        else:
            # Normal forward pass without encoder_outputs
            transformer_outputs = self.func(
                final_model_params, 
                self.buffer,
                batch_on_device["input_ids"],
                attention_mask=batch_on_device.get("input_mask", None),
                labels=batch_on_device.get("target_ids", None)
            )
        
        # transformer_outputs is a tuple: (loss, logits) or Seq2SeqLMOutput
        # We want logits, which is at index [1] or .logits attribute
        if isinstance(transformer_outputs, tuple):
            logits = transformer_outputs[1]  # logits are at index 1
        elif hasattr(transformer_outputs, 'logits'):
            logits = transformer_outputs.logits
        else:
            # Fallback: assume it's logits directly
            logits = transformer_outputs
        
        return logits


def compute_loss_from_logits(logits, target_ids, target_mask):
    """
    Compute loss from logits, similar to T5Wrapper.forward.
    
    Args:
        logits: [batch_size, max_target_len, vocab_size]
        target_ids: [batch_size, max_target_len]
        target_mask: [batch_size, max_target_len]
        
    Returns:
        loss: scalar tensor
        metrics_dict: dict with loss value
    """
    import torch.nn.functional as F
    
    # Ensure all tensors are on the same device as logits
    logits_device = logits.device
    target_ids = target_ids.to(logits_device)
    target_mask = target_mask.to(logits_device)
    
    vocab_size = logits.shape[-1]
    
    # Compute the log probability of the ids for all choices with respect to the logits
    logProbs_ofTargetIds = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    )
    
    # Zero out log_probs for target_ids with no loss
    target_mask_flat = target_mask.reshape(-1)
    logProbs_ofTargetIds_zeroOutPadIds = logProbs_ofTargetIds * target_mask_flat
    
    loss = torch.sum(logProbs_ofTargetIds_zeroOutPadIds) / torch.sum(target_mask_flat)
    
    return loss, {"loss": loss.detach().cpu().item()}


def train_t5_axis(
    source_datasets,
    target_dataset_name,
    base_checkpoint,
    save_dir,
    config_filepath,
    svd_threshold,
    seed,
    device,
    world_size=None,
    rank=None,
):
    """
    Train T5 model with AXIS method.
    
    Args:
        source_datasets: list of source dataset names
        target_dataset_name: target dataset name (must be different from source)
        base_checkpoint: base model checkpoint (HuggingFace name)
        save_dir: directory where finetuned checkpoints are stored
        config_filepath: path to training config JSON
        svd_threshold: SVD threshold for learning
        seed: random seed
        device: torch device
        world_size: world size for distributed training
        rank: rank for distributed training
    """
    import json
    import os
    from datetime import datetime
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from src.train.TrainingConfig import TrainingConfig
    from src.model.load_model import load_model
    from src.model.ModelConfig import ModelConfig
    from src.data.Batcher import Batcher
    from src.data.dataset_readers import get_datasetReader
    from src.data.PytorchDataset import PytorchDataset
    from src.utils.distributed_utils import is_nodeZero
    from src.eval.EvaluationConfig import EvaluationConfig
    from src.eval.evaluate import evaluate_fromConfig
    from torch.optim import AdamW
    from torch.cuda.amp import GradScaler
    from tqdm import tqdm
    
    print("=" * 80)
    print(f"T5 AXIS Training")
    print(f"Source datasets: {source_datasets}")
    print(f"Target dataset: {target_dataset_name}")
    print(f"SVD threshold: {svd_threshold}")
    print("=" * 80)
    
    # Load training config
    with open(config_filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Create experiment directory
    experiment_name = f"t5-axis-{target_dataset_name}-{len(source_datasets)}sources-svd{svd_threshold}-seed{seed}"
    experiment_dir = os.path.join("exp_out", "t5_axis", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    config_dict['experiment_dir'] = experiment_dir
    config_dict['seed'] = seed
    # Enable early stopping (can be configured in JSON config)
    # config_dict['early_stopping'] is read from JSON config
    
    # Remove svd_threshold and sorting_descending from config_dict if present
    # They're not TrainingConfig parameters - we use them directly in AXIS code
    config_dict.pop('svd_threshold', None)
    config_dict.pop('sorting_descending', None)
    
    training_config = TrainingConfig([config_filepath], kwargs=config_dict)
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load base model parameters
    print("\n" + "=" * 80)
    print("STEP 1: Loading Base Model Parameters")
    print("=" * 80)
    print(f"Base checkpoint: {base_checkpoint}")
    base_params = get_t5_base_model_params(base_checkpoint)
    print(f"✓ Loaded {len(base_params.keys())} parameters from base model")
    print(f"  First 3 keys: {list(base_params.keys())[:3]}")
    
    # Load task vectors
    print("\n" + "=" * 80)
    print(f"STEP 2: Loading Task Vectors for {len(source_datasets)} Source Datasets")
    print("=" * 80)
    print(f"Source datasets: {source_datasets}")
    print(f"Save directory: {save_dir}")
    task_vectors_dict = load_t5_task_vectors(
        source_datasets,
        base_checkpoint=base_checkpoint,
        save_dir=save_dir,
        use_half=False  # Full precision for SVD
    )
    print(f"✓ Loaded task vectors for {len(task_vectors_dict)} datasets")
    
    # Detailed verification for each task vector
    for dataset_name, tv in task_vectors_dict.items():
        print(f"\n  [{dataset_name}] Task vector details:")
        print(f"     Total parameters: {len(tv.vector.keys())}")
        
        # Check shapes match base params
        tv_keys = set(tv.vector.keys())
        base_keys = set(base_params.keys())
        if tv_keys == base_keys:
            print(f"     ✓ Keys match base model: {len(tv_keys)} keys")
        else:
            missing_in_tv = base_keys - tv_keys
            missing_in_base = tv_keys - base_keys
            if missing_in_tv:
                print(f"     ⚠ Missing in task vector: {len(missing_in_tv)} keys")
            if missing_in_base:
                print(f"     ⚠ Missing in base: {len(missing_in_base)} keys")
        
        # Check non-zero values
        total_elements = sum(v.numel() for v in tv.vector.values())
        total_nonzero = sum((v != 0).sum().item() for v in tv.vector.values())
        nonzero_ratio = total_nonzero / total_elements if total_elements > 0 else 0
        print(f"     Total elements: {total_elements:,}")
        print(f"     Non-zero elements: {total_nonzero:,} ({nonzero_ratio*100:.2f}%)")
        
        # Show example task vector values
        example_key = list(tv.vector.keys())[0]
        example_tensor = tv.vector[example_key]
        print(f"     Example parameter '{example_key}':")
        print(f"        Shape: {example_tensor.shape}, Dtype: {example_tensor.dtype}")
        print(f"        Min: {example_tensor.min().item():.6f}, Max: {example_tensor.max().item():.6f}")
        print(f"        Mean: {example_tensor.mean().item():.6f}, Std: {example_tensor.std().item():.6f}")
    
    task_vectors_list = list(task_vectors_dict.values())
    
    # Compute iso_c merging
    print("\n" + "=" * 80)
    print("STEP 3: Computing iso_c Merging")
    print("=" * 80)
    print(f"SVD threshold: {svd_threshold}")
    print(f"Number of task vectors: {len(task_vectors_list)}")
    class Config:
        def __init__(self, svd_threshold, sorting_descending=True):
            self.svd_threshold = svd_threshold
            self.sorting_descending = sorting_descending
    
    config = Config(svd_threshold, sorting_descending=True)
    merged_components = iso_c_t5(base_params, task_vectors_list, config, cache_args=None)
    
    if merged_components is None:
        raise ValueError("iso_c_t5 returned None!")
    
    # Analyze merged components
    svd_layers = sum(1 for v in merged_components.values() if v.get('is_svd', False))
    non_svd_layers = sum(1 for v in merged_components.values() if not v.get('is_svd', False))
    print(f"\n✓ iso_c merging completed")
    print(f"  Total merged components: {len(merged_components.keys())} layers")
    print(f"  SVD layers (2D): {svd_layers}")
    print(f"  Non-SVD layers (averaged): {non_svd_layers}")
    
    # Show example SVD layer
    example_svd_key = next((k for k, v in merged_components.items() if v.get('is_svd', False)), None)
    if example_svd_key:
        example_comp = merged_components[example_svd_key]
        print(f"\n  Example SVD component '{example_svd_key}':")
        print(f"     U shape: {example_comp['U'].shape}")
        print(f"     S shape: {example_comp['S'].shape} (selected {example_comp.get('num_selected_components', 0)} components)")
        print(f"     Vh shape: {example_comp['Vh'].shape}")
    
    # Create base model
    print("\n" + "=" * 80)
    print("STEP 4: Creating Base T5 Model")
    print("=" * 80)
    transformer = AutoModelForSeq2SeqLM.from_pretrained(base_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    
    class MinimalT5Wrapper:
        def __init__(self, transformer, tokenizer):
            self.transformer = transformer
            self.tokenizer = tokenizer
    
    base_model = MinimalT5Wrapper(transformer, tokenizer)
    print(f"✓ Base model created")
    print(f"  Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Create LearnableSingularValuesMergedT5Wrapper
    print("\n" + "=" * 80)
    print("STEP 5: Creating LearnableSingularValuesMergedT5Wrapper")
    print("=" * 80)
    print(f"SVD threshold: {svd_threshold}")
    
    # Move base model to device first (before functionalization)
    print(f"Moving base model to device: {device}")
    base_model.transformer = base_model.transformer.to(device)
    
    class Args:
        def __init__(self, svd_threshold):
            self.svd_threshold = svd_threshold
    
    args = Args(svd_threshold)
    model = LearnableSingularValuesMergedT5Wrapper(base_model, merged_components, args)
    # Model should already be on device from base_model.transformer.to(device)
    # But ensure all learnable parameters and buffers are on device
    # Don't use .to(device) if params are in meta device - they should already be moved
    # Just verify device
    if hasattr(model, 'learnable_s_values'):
        for key, param in model.learnable_s_values.items():
            if param.device != device:
                param.data = param.data.to(device)
    
    print(f"✓ Model initialized successfully")
    print(f"  Total learnable singular values: {model.total_learnable_sv:,}")
    print(f"  Total learnable parameters: {sum(p.numel() for p in model.learnable_s_values.values()):,}")
    print(f"  Frozen base parameters: {sum(p.numel() for p in model.params):,}")
    
    # Show example learnable S values
    if len(model.learnable_s_values) > 0:
        example_key = list(model.learnable_s_values.keys())[0]
        example_param = model.learnable_s_values[example_key]
        print(f"\n  Example learnable S values '{example_key}':")
        print(f"     Shape: {example_param.shape}, Dtype: {example_param.dtype}")
        print(f"     Requires grad: {example_param.requires_grad}")
        print(f"     Initial values: min={example_param.min().item():.6f}, max={example_param.max().item():.6f}, mean={example_param.mean().item():.6f}")
    
    # Setup optimizer
    optimizer = AdamW(
        model.learnable_s_values.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay if hasattr(training_config, 'weight_decay') else 0.0
    )
    
    # Setup scaler for mixed precision
    scaler = None
    if training_config.use_bfloat16_during_training:
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Setup data loading
    print("\n" + "=" * 80)
    print(f"STEP 6: Setting Up Data Loading for Target Dataset")
    print("=" * 80)
    print(f"Target dataset: {target_dataset_name}")
    # For evaluation on "test" split, we need num_val_samples to split validation into validation and test
    # Using default value 32 (same as EvaluationConfig) to match behavior from inference.py
    dataset_kwargs = {
        "few_shot_random_seed": None,  # Use original dataset, not few-shot (avoids missing few_shot files)
        "num_val_samples": 32,  # Default value for splitting validation into validation (first 32) and test (remaining)
        "max_datapoints_per_dataset_without_templates": training_config.max_datapoints_per_dataset,
    }
    
    try:
        dataset_reader = get_datasetReader(target_dataset_name, dataset_kwargs)
    except Exception as e:
        print(f"\n✗ ERROR: Failed to load dataset reader for {target_dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to stop the job
    
    # Get dataset sizes for train and test splits
    print(f"\nGetting dataset sizes for target dataset: {target_dataset_name}")
    try:
        train_dataset = dataset_reader.get_dataset(
            "train", 
            training_config.train_template_idx, 
            is_evaluation=False
        )
        train_samples = len(train_dataset)
        print(f"  Train set: {train_samples:,} samples")
    except Exception as e:
        print(f"  ⚠ Warning: Could not get train set size: {e}")
        train_samples = None
    
    try:
        test_dataset = dataset_reader.get_dataset(
            "test", 
            training_config.eval_template_idx, 
            is_evaluation=True
        )
        test_samples = len(test_dataset)
        print(f"  Test set: {test_samples:,} samples")
    except Exception as e:
        print(f"  ⚠ Warning: Could not get test set size: {e}")
        test_samples = None
    
    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, device)
    # For single GPU, pass world_size=None to avoid DistributedSampler (which expects int rank, not device)
    batcher_world_size = None if (world_size is None or world_size == 1) else world_size
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=training_config.train_batch_size,
        eval_batchSize=training_config.eval_batch_size,
        world_size=batcher_world_size,
        device=device,
        tokenizer=tokenizer,
        max_seq_len=training_config.max_seq_len,
        use_tokenization_cache=True,
    )
    
    train_iterator = batcher.get_trainBatches("train", training_config.train_template_idx)
    print(f"✓ Data loading setup completed")
    print(f"  Train batch size: {training_config.train_batch_size}")
    print(f"  Eval batch size: {training_config.eval_batch_size}")
    print(f"  Max sequence length: {training_config.max_seq_len}")
    
    # Print all important training configuration parameters
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"  Source datasets: {source_datasets}")
    print(f"  Target dataset: {target_dataset_name}")
    print(f"  Base checkpoint: {base_checkpoint}")
    print(f"  SVD threshold: {svd_threshold}")
    print(f"  Seed: {seed}")
    print(f"\n  Training parameters:")
    print(f"    num_batches: {training_config.num_batches}")
    print(f"    train_batch_size: {training_config.train_batch_size}")
    print(f"    eval_batch_size: {training_config.eval_batch_size}")
    print(f"    gradient_accumulation_factor: {training_config.gradient_accumulation_factor}")
    print(f"    effective_batch_size: {training_config.train_batch_size * training_config.gradient_accumulation_factor}")
    print(f"    lr: {training_config.lr}")
    print(f"    optimizer: {training_config.optimizer}")
    print(f"    scheduler: {training_config.scheduler}")
    print(f"    weight_decay: {getattr(training_config, 'weight_decay', 0.0)}")
    print(f"\n  Model parameters:")
    print(f"    max_seq_len: {training_config.max_seq_len}")
    print(f"    max_gen_len: {training_config.max_gen_len}")
    print(f"    train_template_idx: {training_config.train_template_idx}")
    print(f"    eval_template_idx: {training_config.eval_template_idx}")
    print(f"\n  Precision:")
    print(f"    use_bfloat16_during_training: {training_config.use_bfloat16_during_training}")
    print(f"    use_bfloat16_during_eval: {getattr(training_config, 'use_bfloat16_during_eval', False)}")
    print(f"\n  Checkpoint settings:")
    print(f"    checkpoint_frequency: {training_config.checkpoint_frequency}")
    print(f"    early_stopping: {getattr(training_config, 'early_stopping', False)}")
    if hasattr(training_config, 'early_stopping_num_checkpoints_without_improvement'):
        print(f"    early_stopping_num_checkpoints_without_improvement: {training_config.early_stopping_num_checkpoints_without_improvement}")
    print(f"\n  Data parameters:")
    print(f"    max_datapoints_per_dataset: {training_config.max_datapoints_per_dataset}")
    print(f"    length_normalization: {getattr(training_config, 'length_normalization', False)}")
    print(f"    train_samples: {train_samples:,} samples" if train_samples is not None else "    train_samples: N/A")
    print(f"    test_samples: {test_samples:,} samples" if test_samples is not None else "    test_samples: N/A")
    print(f"\n  Experiment:")
    print(f"    experiment_dir: {experiment_dir}")
    print("=" * 80)
    
    # Training loop with early stopping support
    print("\n" + "=" * 80)
    print(f"STEP 7: Starting Training")
    print("=" * 80)
    print(f"Training for up to {training_config.num_batches} batches...")
    if training_config.early_stopping:
        print(f"Early stopping enabled: patience = {training_config.early_stopping_num_checkpoints_without_improvement} checkpoints")
    model.train()
    
    # Early stopping variables
    # Use accuracy instead of loss for early stopping (validation batches have all_choices_ids, not target_ids)
    best_val_accuracy = -1.0  # Start with -1 (worse than any possible accuracy)
    num_checkpoints_since_best = 0
    batches_seen = 0
    early_stopping_triggered = False
    
    # Create EvalWrapper for early stopping (needed for predict_mulChoice)
    # We define it here so it can be used in early stopping evaluation
    class EvalWrapper(nn.Module):
        def __init__(self, model, tokenizer):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
        
        def forward(self, batch):
            logits = self.model(batch)
            loss, metrics = compute_loss_from_logits(
                logits,
                batch['target_ids'],
                batch['target_mask']
            )
            return loss, metrics
        
        def _broadcast_tensors(self, input_masks, encoder_outputs, num_choices):
            """Broadcast the input masks and encoder outputs to account for multiple choices per input"""
            input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)
            broadcasted_hidden_states = torch.repeat_interleave(encoder_outputs[0], num_choices, dim=0)
            broadcasted_encoder_outputs = BaseModelOutput(
                last_hidden_state=broadcasted_hidden_states,
                hidden_states=getattr(encoder_outputs, 'hidden_states', None),
                attentions=getattr(encoder_outputs, 'attentions', None),
            )
            return input_masks, broadcasted_encoder_outputs
        
        def compute_logProb(
            self,
            logProbs_ofAllChoices_ids,
            allChoices_masks,
            num_choices,
            maxChoice_len,
            length_normalization,
        ):
            """Compute log probabilities for all choices"""
            logProbs_ofAllChoices_ids = logProbs_ofAllChoices_ids.reshape(
                -1, num_choices, maxChoice_len
            )
            allChoices_masks = allChoices_masks.reshape(-1, num_choices, maxChoice_len)
            logProbs_ofAllChoicesIds_zeroOutPadIds = (
                logProbs_ofAllChoices_ids * allChoices_masks
            )
            
            logProbs_ofAllChoices = torch.sum(logProbs_ofAllChoicesIds_zeroOutPadIds, dim=2)
            len_allChoices = torch.sum(allChoices_masks, dim=2)
            
            if length_normalization:
                logProbs_ofAllChoices = logProbs_ofAllChoices / len_allChoices
            
            return (
                logProbs_ofAllChoices,
                logProbs_ofAllChoicesIds_zeroOutPadIds,
                len_allChoices,
            )
        
        def compute_logProb_ofAllChoices(
            self,
            input_ids,
            input_masks,
            allChoices_ids,
            allChoices_masks,
            length_normalization,
        ):
            """Computes log probabilities for all the choices using model with learnable singular values"""
            from transformers.modeling_outputs import BaseModelOutput
            
            base_transformer = self.model.model.transformer
            encoder_outputs = base_transformer.get_encoder()(input_ids, attention_mask=input_masks)
            
            assert allChoices_ids.shape[0] % input_masks.shape[0] == 0, (
                f"The batch size {allChoices_ids.shape[0]} of allChoices_ids is not a multiple of "
                f"the batch size {input_masks.shape[0]} of input_masks"
            )
            
            num_choices = allChoices_ids.shape[0] // input_masks.shape[0]
            
            # Broadcast input masks and encoder outputs for all choices
            input_masks, encoder_outputs = self._broadcast_tensors(
                input_masks, encoder_outputs, num_choices
            )
            
            # Use model with learnable singular values, passing encoder_outputs for efficiency
            all_choices_batch = {
                "input_ids": input_ids.repeat_interleave(num_choices, dim=0),
                "input_mask": input_masks,
                "target_ids": allChoices_ids,
                "target_mask": allChoices_masks,
            }
            
            # Call model forward with encoder_outputs to reuse encoder and use learnable SVs in decoder
            logits_ofAllChoices = self.model(all_choices_batch, encoder_outputs=encoder_outputs)
            
            maxChoice_len = logits_ofAllChoices.shape[1]
            vocab_size = logits_ofAllChoices.shape[-1]
            
            # Compute the log probability of the ids for all choices with respect to the logits
            logProbs_ofAllChoices_ids = -F.cross_entropy(
                logits_ofAllChoices.view(-1, vocab_size),
                allChoices_ids.view(-1),
                reduction="none",
            )
            
            return self.compute_logProb(
                logProbs_ofAllChoices_ids,
                allChoices_masks,
                num_choices,
                maxChoice_len,
                length_normalization,
            )
        
        def predict_mulChoice(self, batch, length_normalization):
            """Predict multiple choice answers"""
            from src.utils.utils import round_nestedList
            
            (
                score_ofChoices,
                logProbs_ofAllChoicesIds,
                len_allChoices,
            ) = self.compute_logProb_ofAllChoices(
                batch["input_ids"],
                batch["input_mask"],
                batch["all_choices_ids"],
                batch["all_choices_mask"],
                length_normalization,
            )
            
            _, predicted_choice = torch.max(score_ofChoices, dim=1)
            
            return (
                predicted_choice.cpu().numpy().tolist(),
                round_nestedList(score_ofChoices.cpu().numpy().tolist(), 5),
                round_nestedList(logProbs_ofAllChoicesIds.cpu().numpy().tolist(), 4),
                len_allChoices.cpu().numpy().tolist(),
            )
    
    # Get validation iterator for early stopping evaluation
    # NOTE: For T5 multiple choice datasets, validation batches have 'all_choices_ids' instead of 'target_ids'
    # So we use accuracy on validation set for early stopping instead of loss
    val_iterator = None
    if training_config.early_stopping and training_config.should_eval_validation:
        try:
            # Use validation batches for early stopping (they have all_choices_ids for accuracy computation)
            val_iterator = batcher.get_evalBatches("validation", training_config.eval_template_idx)
            print(f"✓ Validation iterator created for early stopping (using validation set for accuracy-based early stopping)")
        except Exception as e:
            print(f"⚠ Warning: Could not create validation iterator: {e}")
            print(f"  Early stopping will be disabled")
            training_config.early_stopping = False
    
    for batch_idx in tqdm(range(training_config.num_batches), desc="Training"):
        train_batch = next(train_iterator)
        batches_seen += 1
        
        optimizer.zero_grad()
        
        # Forward pass
        if training_config.use_bfloat16_during_training:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(train_batch)
                loss, metrics = compute_loss_from_logits(
                    logits,
                    train_batch['target_ids'],
                    train_batch['target_mask']
                )
                loss = loss / training_config.gradient_accumulation_factor
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(train_batch)
            loss, metrics = compute_loss_from_logits(
                logits,
                train_batch['target_ids'],
                train_batch['target_mask']
            )
            loss = loss / training_config.gradient_accumulation_factor
            loss.backward()
            optimizer.step()
        
        if is_nodeZero(device) and (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{training_config.num_batches}, Loss: {metrics['loss']:.4f}")
        
        # Early stopping evaluation at checkpoint frequency
        if (training_config.early_stopping and 
            training_config.should_eval_validation and 
            val_iterator is not None and
            (batch_idx + 1) % training_config.checkpoint_frequency == 0):
            
            # Evaluate on validation set using accuracy (validation batches have all_choices_ids for multiple choice)
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                # Evaluate on a few validation batches (limit to avoid long evaluation)
                for val_batch_idx, val_batch in enumerate(val_iterator):
                    if val_batch_idx >= 10:  # Limit to 10 validation batches for speed
                        break
                    
                    # Check if batch has required fields for multiple choice evaluation
                    required_fields = ['input_ids', 'input_mask', 'all_choices_ids', 'all_choices_mask', 'idx']
                    missing_fields = [f for f in required_fields if f not in val_batch]
                    if missing_fields:
                        # Skip this batch if it doesn't have required fields
                        if val_batch_idx == 0:
                            print(f"    ⚠ Warning: Validation batch missing fields: {missing_fields}. Skipping validation batches.")
                        continue
                    
                    # Ensure batch tensors are on correct device
                    val_batch_on_device = {}
                    for k, v in val_batch.items():
                        if isinstance(v, torch.Tensor):
                            val_batch_on_device[k] = v.to(device)
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                            val_batch_on_device[k] = [t.to(device) for t in v]
                        else:
                            val_batch_on_device[k] = v
                    
                    try:
                        # Use EvalWrapper to compute predictions (it has predict_mulChoice method)
                        eval_model = EvalWrapper(model, tokenizer)
                        if training_config.use_bfloat16_during_eval:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                predicted_choice, _, _, _ = eval_model.predict_mulChoice(
                                    val_batch_on_device,
                                    length_normalization=getattr(training_config, 'length_normalization', False)
                                )
                        else:
                            predicted_choice, _, _, _ = eval_model.predict_mulChoice(
                                val_batch_on_device,
                                length_normalization=getattr(training_config, 'length_normalization', False)
                            )
                        
                        # Get correct answers from dataset_reader using idx
                        # Handle both tensor and list types for idx
                        idx_data = val_batch_on_device['idx']
                        if isinstance(idx_data, torch.Tensor):
                            batch_indices = idx_data.cpu().numpy()
                        elif isinstance(idx_data, list):
                            # Convert list to numpy array directly
                            batch_indices = np.array(idx_data)
                        else:
                            # Fallback: try to convert to numpy array
                            batch_indices = np.array([idx_data])
                        for i, idx in enumerate(batch_indices):
                            # Get the datapoint from dataset_reader to find correct answer
                            dataset = dataset_reader.get_dataset("validation", training_config.eval_template_idx, is_evaluation=True)
                            if idx < len(dataset):
                                datapoint = dataset[idx]
                                # Find correct choice index
                                # For multiple choice, we need to find which choice matches the target
                                if 'answer_choices' in datapoint and 'target' in datapoint:
                                    target_text = datapoint['target']
                                    answer_choices = datapoint['answer_choices']
                                    try:
                                        correct_choice_idx = answer_choices.index(target_text)
                                        if predicted_choice[i] == correct_choice_idx:
                                            val_correct += 1
                                        val_total += 1
                                    except (ValueError, IndexError):
                                        # Skip if target not in choices or index out of range
                                        continue
                    except Exception as e:
                        # Skip this batch if there's an error
                        print(f"    ⚠ Warning: Skipping validation batch {val_batch_idx} due to error: {e}")
                        import traceback
                        if val_batch_idx == 0:
                            traceback.print_exc()
                        continue
            
            if val_total > 0:
                val_accuracy = val_correct / val_total
                print(f"\n  Checkpoint at batch {batch_idx + 1}: Validation accuracy = {val_accuracy:.4f} ({val_correct}/{val_total})")
                
                # Check if this is the best validation accuracy
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    num_checkpoints_since_best = 0
                    print(f"    ✓ New best validation accuracy: {best_val_accuracy:.4f}")
                else:
                    num_checkpoints_since_best += 1
                    print(f"    No improvement ({num_checkpoints_since_best}/{training_config.early_stopping_num_checkpoints_without_improvement})")
                
                # Check early stopping condition
                if num_checkpoints_since_best >= training_config.early_stopping_num_checkpoints_without_improvement:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED!")
                    print(f"  Patience threshold ({training_config.early_stopping_num_checkpoints_without_improvement} checkpoints) reached.")
                    print(f"  Best validation accuracy: {best_val_accuracy:.4f}")
                    print(f"  Batches seen: {batches_seen}/{training_config.num_batches}")
                    print(f"{'='*80}\n")
                    early_stopping_triggered = True
                    break
            
            model.train()
            # Reset validation iterator for next checkpoint
            val_iterator = batcher.get_evalBatches("validation", training_config.eval_template_idx)
    
    print("\n" + "=" * 80)
    if early_stopping_triggered:
        print("STEP 8: Training Stopped Early - Evaluating on Test Set")
    else:
        print("STEP 8: Training Completed - Evaluating on Test Set")
    print("=" * 80)
    
    # Evaluation on test set
    model.eval()
    evaluation_config = EvaluationConfig(
        configDict_toInitializeFrom=training_config.get_dict(),
        fields_toUpdate={
            "prediction_dir": os.path.join(experiment_dir, "predictions"),
            "inference_dataset": target_dataset_name,
            "split": "test",
        }
    )
    
    # Create a T5Wrapper-like interface for evaluation
    # We need to wrap the model so it returns (loss, metrics) for evaluation
    # and implements predict_mulChoice for multiple choice evaluation
    class EvalWrapper(nn.Module):
        def __init__(self, model, tokenizer):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
        
        def forward(self, batch):
            logits = self.model(batch)
            loss, metrics = compute_loss_from_logits(
                logits,
                batch['target_ids'],
                batch['target_mask']
            )
            return loss, metrics
        
        def _broadcast_tensors(self, input_masks, encoder_outputs, num_choices):
            """Broadcast the input masks and encoder outputs to account for multiple choices per input"""
            input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)
            # encoder_outputs is BaseModelOutput from HuggingFace
            # It has [0] as the first element (hidden states)
            # We need to broadcast the hidden states and return as BaseModelOutput
            broadcasted_hidden_states = torch.repeat_interleave(encoder_outputs[0], num_choices, dim=0)
            # Create new BaseModelOutput with broadcasted hidden states
            # Preserve other attributes if they exist
            broadcasted_encoder_outputs = BaseModelOutput(
                last_hidden_state=broadcasted_hidden_states,
                hidden_states=getattr(encoder_outputs, 'hidden_states', None),
                attentions=getattr(encoder_outputs, 'attentions', None),
            )
            return input_masks, broadcasted_encoder_outputs
        
        def compute_logProb(
            self,
            logProbs_ofAllChoices_ids,
            allChoices_masks,
            num_choices,
            maxChoice_len,
            length_normalization,
        ):
            """Compute log probabilities for all choices"""
            logProbs_ofAllChoices_ids = logProbs_ofAllChoices_ids.reshape(
                -1, num_choices, maxChoice_len
            )
            allChoices_masks = allChoices_masks.reshape(-1, num_choices, maxChoice_len)
            logProbs_ofAllChoicesIds_zeroOutPadIds = (
                logProbs_ofAllChoices_ids * allChoices_masks
            )
            
            logProbs_ofAllChoices = torch.sum(logProbs_ofAllChoicesIds_zeroOutPadIds, dim=2)
            len_allChoices = torch.sum(allChoices_masks, dim=2)
            
            if length_normalization:
                logProbs_ofAllChoices = logProbs_ofAllChoices / len_allChoices
            
            return (
                logProbs_ofAllChoices,
                logProbs_ofAllChoicesIds_zeroOutPadIds,
                len_allChoices,
            )
        
        def compute_logProb_ofAllChoices(
            self,
            input_ids,
            input_masks,
            allChoices_ids,
            allChoices_masks,
            length_normalization,
        ):
            """Computes log probabilities for all the choices using model with learnable singular values"""
            # Use model with learnable singular values (self.model is LearnableSingularValuesMergedT5Wrapper)
            # Get encoder outputs once using base transformer (for efficiency)
            # Then use model with learnable singular values for decoder
            base_transformer = self.model.model.transformer
            
            # Get encoder outputs once (like T5Wrapper) - encoder doesn't use learnable SVs in our case
            # Actually, we should use model's encoder if it has learnable SVs, but for efficiency
            # we'll compute encoder once and reuse. The learnable SVs mainly affect decoder.
            encoder_outputs = base_transformer.get_encoder()(input_ids, attention_mask=input_masks)
            
            assert allChoices_ids.shape[0] % input_masks.shape[0] == 0, (
                f"The batch size {allChoices_ids.shape[0]} of allChoices_ids is not a multiple of "
                f"the batch size {input_masks.shape[0]} of input_masks"
            )
            
            num_choices = allChoices_ids.shape[0] // input_masks.shape[0]
            
            # Broadcast input masks and encoder outputs for all choices
            input_masks, encoder_outputs = self._broadcast_tensors(
                input_masks, encoder_outputs, num_choices
            )
            
            # Use model with learnable singular values, passing encoder_outputs for efficiency
            # This will use the learnable singular values in the decoder computation
            all_choices_batch = {
                "input_ids": input_ids.repeat_interleave(num_choices, dim=0),
                "input_mask": input_masks,
                "target_ids": allChoices_ids,
                "target_mask": allChoices_masks,
            }
            
            # Call model forward with encoder_outputs to reuse encoder and use learnable SVs in decoder
            logits_ofAllChoices = self.model(all_choices_batch, encoder_outputs=encoder_outputs)
            
            maxChoice_len = logits_ofAllChoices.shape[1]
            vocab_size = logits_ofAllChoices.shape[-1]
            
            # Compute the log probability of the ids for all choices with respect to the logits
            logProbs_ofAllChoices_ids = -F.cross_entropy(
                logits_ofAllChoices.view(-1, vocab_size),
                allChoices_ids.view(-1),
                reduction="none",
            )
            
            return self.compute_logProb(
                logProbs_ofAllChoices_ids,
                allChoices_masks,
                num_choices,
                maxChoice_len,
                length_normalization,
            )
        
        def predict_mulChoice(self, batch, length_normalization):
            """Predict multiple choice answers"""
            from src.utils.utils import round_nestedList
            
            # Compute log p(y|x)
            (
                score_ofChoices,
                logProbs_ofAllChoicesIds,
                len_allChoices,
            ) = self.compute_logProb_ofAllChoices(
                batch["input_ids"],
                batch["input_mask"],
                batch["all_choices_ids"],
                batch["all_choices_mask"],
                length_normalization,
            )
            
            _, predicted_choice = torch.max(score_ofChoices, dim=1)
            
            return (
                predicted_choice.cpu().numpy().tolist(),
                round_nestedList(score_ofChoices.cpu().numpy().tolist(), 5),
                round_nestedList(logProbs_ofAllChoicesIds.cpu().numpy().tolist(), 4),
                len_allChoices.cpu().numpy().tolist(),
            )
    
    eval_model = EvalWrapper(model, tokenizer)
    
    # Evaluate
    scores, cached_datasetReaders = evaluate_fromConfig(
        eval_model,
        tokenizer,
        cached_datasetReaders={target_dataset_name: dataset_reader},
        evaluation_config=evaluation_config,
        device=device
    )
    
    # Convert scores to JSON-serializable format
    # scores might contain non-serializable objects, so we need to clean it
    def make_json_serializable(obj):
        """Recursively convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif hasattr(obj, '__dict__'):
            # For custom objects, try to convert to dict
            return make_json_serializable(obj.__dict__)
        else:
            # For other non-serializable objects, convert to string
            return str(obj)
    
    # Clean scores to ensure JSON serializability
    scores_serializable = make_json_serializable(scores) if scores is not None else None
    
    # Save results
    print("\n" + "=" * 80)
    print("STEP 9: Saving Results")
    print("=" * 80)
    results = {
        "source_datasets": source_datasets,
        "target_dataset": target_dataset_name,
        "svd_threshold": svd_threshold,
        "seed": seed,
        "scores": scores_serializable,
        "experiment_dir": experiment_dir,
        "timestamp": datetime.now().isoformat(),
        "batches_seen": batches_seen,
        "early_stopping_triggered": early_stopping_triggered,
    }
    
    results_path = os.path.join(experiment_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save singular values
    singular_values = {}
    for key, param in model.learnable_s_values.items():
        singular_values[key] = param.detach().cpu().tolist()
    
    sv_path = os.path.join(experiment_dir, "singular_values.json")
    with open(sv_path, 'w') as f:
        json.dump(singular_values, f, indent=2)
    
    print(f"✓ Results saved to: {experiment_dir}")
    print(f"  - results.json: {results_path}")
    print(f"  - singular_values.json: {sv_path}")
    
    if is_nodeZero(device):
        print(f"\n✓ Evaluation completed")
        print(f"  Evaluation scores: {scores}")
    
    return results


def main(args):
    """
    Main function for T5 AXIS training (single GPU, no DDP).
    
    Iterates over:
    1. Number of source datasets (from resume_from_idx to end_index)
    2. For each number of source datasets, iterates over all possible target datasets
       (target must be different from source datasets)
    
    Args:
        args: parsed arguments
    """
    import os
    import json
    import random
    import pandas as pd
    from datetime import datetime
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # All datasets in order
    all_datasets = ["paws", "qasc", "quartz", "story_cloze", "wiki_qa", "winogrande", "wsc"]
    
    # Setup device (single GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"All datasets: {all_datasets}")
    print(f"Resume from idx: {args.resume_from_idx}, End index: {args.end_index}")
    print("=" * 100)
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    # Setup CSV file for appending results after each test
    csv_dir = "exp_out/t5_axis/with_early_stopping"
    os.makedirs(csv_dir, exist_ok=True)
    
    # Generate CSV filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"t5_axis_results_{timestamp_str}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    # Get reverse flag from args (default False)
    reverse_flag = getattr(args, 'reverse', False)
    
    # Create CSV file with headers if it doesn't exist
    csv_exists = os.path.exists(csv_path)
    if not csv_exists:
        # Create DataFrame with headers only
        df_headers = pd.DataFrame(columns=[
            "source_datasets", "target_dataset", "svd_threshold", "seed", 
            "accuracy", "scores", "experiment_dir", "timestamp",
            "batches_seen", "early_stopping_triggered", "reverse"
        ])
        df_headers.to_csv(csv_path, index=False)
        print(f"✓ Created CSV file: {csv_path}")
    else:
        print(f"✓ Appending to existing CSV file: {csv_path}")
    
    # Iterate over number of source datasets
    for source_idx in range(len(all_datasets)):
        if source_idx < args.resume_from_idx:
            print(f"\nSkipping source_idx {source_idx} because resume-from-idx is {args.resume_from_idx}")
            continue
        
        if source_idx >= args.end_index:
            print(f"\nEnding at source_idx {source_idx} because end-index is {args.end_index}")
            break
        
        # Get source datasets (from 0 to source_idx inclusive)
        source_datasets = all_datasets[:source_idx+1]
        
        # Get target datasets (all datasets not in source)
        target_datasets = [d for d in all_datasets if d not in source_datasets]
        
        print("\n" + "=" * 100)
        print(f"SOURCE DATASETS [{len(source_datasets)}]: {source_datasets}")
        print(f"TARGET DATASETS [{len(target_datasets)}]: {target_datasets}")
        print("=" * 100)
        
        # Iterate over all target datasets
        for target_dataset_name in target_datasets:
            try:
                print("\n" + "-" * 100)
                print(f"Training with source tasks {source_datasets} for target task {target_dataset_name}")
                print("-" * 100)
                
                # Train
                results = train_t5_axis(
                    source_datasets=source_datasets,
                    target_dataset_name=target_dataset_name,
                    base_checkpoint=args.model,
                    save_dir="exp_out/t5_finetuning/t5-base",
                    config_filepath=args.config,
                    svd_threshold=args.svd_threshold,
                    seed=args.seed,
                    device=device,
                    world_size=1,
                    rank=0,
                )
                
                # Extract accuracy from scores
                scores = results.get("scores", {})
                accuracy = None
                
                # Try to find accuracy in scores (format may vary)
                if isinstance(scores, dict):
                    # Common keys for accuracy
                    for key in ["accuracy", "acc", "exact_match", "em", "f1"]:
                        if key in scores:
                            accuracy = scores[key]
                            break
                    # If no direct key, try nested dicts
                    if accuracy is None:
                        for key, value in scores.items():
                            if isinstance(value, dict):
                                for subkey in ["accuracy", "acc", "exact_match", "em", "f1"]:
                                    if subkey in value:
                                        accuracy = value[subkey]
                                        break
                                if accuracy is not None:
                                    break
                
                # Format source datasets string
                source_str = " | ".join(source_datasets)
                
                # Store result for CSV
                result_row = {
                    "source_datasets": source_str,
                    "target_dataset": target_dataset_name,
                    "svd_threshold": args.svd_threshold,
                    "seed": args.seed,
                    "accuracy": accuracy if accuracy is not None else None,
                    "scores": json.dumps(scores) if scores else None,
                    "experiment_dir": results.get("experiment_dir", ""),
                    "timestamp": results.get("timestamp", datetime.now().isoformat()),
                    "batches_seen": results.get("batches_seen", None),
                    "early_stopping_triggered": results.get("early_stopping_triggered", False),
                    "reverse": reverse_flag,
                }
                
                # Append result to CSV immediately after each test
                df_row = pd.DataFrame([result_row])
                df_row.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"✓ Result appended to CSV: {csv_path}")
                
                # Print evaluation result
                if accuracy is not None:
                    print("\n" + "=" * 100)
                    print(f"RESULT: train sources task [{source_str}] for target task [{target_dataset_name}] acc: {accuracy:.4f}")
                    print("=" * 100)
                else:
                    print("\n" + "=" * 100)
                    print(f"RESULT: train sources task [{source_str}] for target task [{target_dataset_name}]")
                    print(f"Scores: {scores}")
                    print("=" * 100)
                
                print(f"✓ Successfully completed: source={source_datasets}, target={target_dataset_name}")
                
            except Exception as e:
                print(f"\n✗ ERROR: source={source_datasets}, target={target_dataset_name} failed with error: {e}")
                import traceback
                traceback.print_exc()
                print("\n" + "=" * 100)
                print("FATAL ERROR: Stopping entire job due to error!")
                print("=" * 100)
                import sys
                sys.exit(1)
        
        print(f"\n✓ Completed all target datasets for source_idx {source_idx} (source: {source_datasets})")
    
    # Final summary
    print("\n" + "=" * 100)
    print(f"✓ All experiments completed. Results saved to CSV: {csv_path}")
    print("=" * 100)


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="T5 AXIS Training (Single GPU)")
    parser.add_argument("--svd-threshold", type=float, default=0.1, help="SVD threshold for learning")
    parser.add_argument("--model", type=str, default="t5-base", help="Model name")
    parser.add_argument("--resume-from-idx", type=int, default=0, help="Resume from index (number of source datasets to start from)")
    parser.add_argument("--end-index", type=int, default=1, help="End index (number of source datasets to end at, exclusive)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="axis/configs/t5_axis_training.json", help="Config file path")
    parser.add_argument("--reverse", action="store_true", default=False, help="Reverse flag (True when running run_t5_axis_rev.sh)")
    
    args = parser.parse_args()
    
    # Single GPU training (no DDP)
    # Note: target-dataset-name is no longer needed - all targets are iterated automatically
    main(args)

