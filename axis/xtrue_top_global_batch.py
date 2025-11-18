"""Given an objective, learn the coefficients and singular values 
on SVD components of merged task vectors for a dataset.

Features:
- Uses iso_c to precompute and select SVD components (U, S, Vh)
- Learns selected singular values and layer-wise coefficients
- Maintains original parameter dtype and shape information
"""

import os
import time
import json
import torch
import torchvision
import sys
import argparse

# Remove the wandb environment variables that disable functionality
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_DISABLE_SERVICE"] = "true"

# Redirect wandb stderr to /dev/null before import
old_stderr = sys.stderr
try:
    with open('/dev/null', 'w') as devnull:
        sys.stderr = devnull
        import wandb
    sys.stderr = old_stderr
    print(f"[DEBUG] wandb imported successfully (version: {wandb.__version__})")
except Exception as e:
    sys.stderr = old_stderr
    print(f"[ERROR] Failed to import wandb: {e}")
    print("[ERROR] This is a critical error - wandb is required for logging.")
    print("[ERROR] Please update wandb: pip install --upgrade wandb")
    import traceback
    traceback.print_exc()
    raise

import random
import numpy as np
import socket
import subprocess
import platform
import psutil
import hashlib
from datetime import datetime, timedelta

from torch.cuda.amp import GradScaler

def supports_bfloat16():
    """
    Sprawdza, czy GPU obsługuje bfloat16 (Ampere+ architektura).
    bfloat16 jest dostępny na GPU z compute capability >= 8.0 (A100, RTX 30xx, RTX 40xx, H100, etc.)
    
    Returns:
        bool: True jeśli GPU obsługuje bfloat16, False w przeciwnym razie
    """
    if not torch.cuda.is_available():
        return False
    
    # Sprawdź compute capability
    device = torch.cuda.current_device()
    compute_capability = torch.cuda.get_device_capability(device)
    major_version = compute_capability[0]
    
    # Ampere (8.0) i nowsze obsługują bfloat16
    return major_version >= 8

from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.composition import WeightedImageEncoder, WeightedLinearizedModel
from src.composition import TopValuesTaskBasedSVDWeightedImageEncoder
from src.composition import MergedTaskVectorImageEncoder
from torch import nn

from src.utils import cosine_lr
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    # Jeżeli x jest None lub p jest None, zwróć 0
    if x is None or p is None:
        return 0
    # Dla SVDWeightedImageEncoder, regularyzacja jest na wartościach osobliwych, nie na współczynnikach
    if not x.requires_grad:
        return 0
    return gamma * torch.norm(x, p=p, dim=0).mean()

def set_seed(seed: int) -> None:
    """
    Set random seed for all possible random number generators for reproducibility.
    
    Args:
        seed: The random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For completely deterministic results, set the following flags
    # Note: This may slow down training
    # Changed to False/True for performance - cuDNN benchmark finds fastest algorithms
    torch.backends.cudnn.deterministic = False  # Set to False for speed
    torch.backends.cudnn.benchmark = True       # Enable benchmark mode for faster operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if is_main_process():
        print(f"Random seed set to {seed}")
        print(f"cuDNN benchmark enabled for faster training (deterministic=False)")

def create_experiment_commit(args=None):
    """
    Create a git commit at the beginning of an experiment run.
    This ensures we have a record of the exact code state when the experiment was started.
    
    Args:
        args: Optional experiment arguments to include in the commit message
        
    Returns:
        str: The commit hash or None if commit failed
    """
    try:
        # Check if there are any changes to commit
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Get current commit hash even if there are no changes
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        current_commit = hash_result.stdout.strip()
        
        if not status_result.stdout.strip():
            print("No changes to commit. Code is already in a clean state.")
            print(f"Current commit: {current_commit}")
            return current_commit
        
        # Get hostname and timestamp for the commit message
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create commit message with experiment details if provided
        commit_message = f"Experiment snapshot: {timestamp} on {hostname}"
        
        if args is not None:
            # Add key experiment parameters to the commit message
            exp_details = []
            
            if hasattr(args, 'model'):
                exp_details.append(f"model={args.model}")
            
            if hasattr(args, 'target_dataset'):
                exp_details.append(f"dataset={args.target_dataset}")
            elif hasattr(args, 'datasets') and args.datasets:
                exp_details.append(f"datasets={args.datasets[0] if len(args.datasets) == 1 else f'{len(args.datasets)} datasets'}")
            
            if hasattr(args, 'epochs'):
                exp_details.append(f"epochs={args.epochs}")
                
            if hasattr(args, 'lr'):
                exp_details.append(f"lr={args.lr}")
                
            if exp_details:
                commit_message += f" - {', '.join(exp_details)}"
        
        # Add all changes
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit changes
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract commit hash
        commit_hash = None
        for line in commit_result.stdout.split('\n'):
            if line.startswith('['):
                parts = line.split()
                if len(parts) > 1:
                    commit_hash = parts[1]
                    break
        
        print(f"Created experiment snapshot commit: {commit_hash}")
        print(f"Commit message: {commit_message}")
        
        return commit_hash
    
    except subprocess.SubprocessError as e:
        print(f"Warning: Failed to create experiment commit: {e}")
        return None

def is_port_available(port):
    """
    Check if a port is available for use by attempting to bind to it.
    
    Args:
        port: The port number to check
        
    Returns:
        bool: True if the port is available, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False

def find_available_port(port_list):
    """
    Find the first available port from a list of ports.
    
    Args:
        port_list: List of port numbers to check
        
    Returns:
        int: First available port from the list or None if none available
    """
    import socket
    import time
    import random
    
    # Shuffle the port list to avoid always trying the same ports first
    port_list = list(port_list)  # Create a copy to avoid modifying the original
    random.shuffle(port_list)
    
    for port in port_list:
        # Check with TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
            tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                tcp_socket.bind(('', port))
                # Double check by waiting a moment and trying again
                time.sleep(0.1)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as second_check:
                    second_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        second_check.bind(('', port))
                        print(f"Port {port} is confirmed available")
                        return port
                    except OSError:
                        print(f"Port {port} failed second availability check")
                        continue
            except OSError as e:
                print(f"Port {port} is not available: {e}")
                continue
    
    print("No available ports found in the provided list")
    return None

def compute_task_vector_hash(task_vector, dataset_name):
    """
    Oblicza hash zawartości task_vector dla identyfikacji cache.
    Hashuje klucze, kształty i sumy wartości tensorów dla szybkości.
    
    Args:
        task_vector: TaskVector object z atrybutem .vector (dict)
        dataset_name: str - nazwa datasetu
        
    Returns:
        str: Hash w formacie hex (np. "a1b2c3d4...")
    """
    hasher = hashlib.sha256()
    
    # Hash dataset name
    hasher.update(dataset_name.encode('utf-8'))
    
    # Hash keys and tensor metadata (shape, dtype, sum for speed)
    if hasattr(task_vector, 'vector') and task_vector.vector:
        # Sort keys for deterministic hash
        sorted_keys = sorted(task_vector.vector.keys())
        for key in sorted_keys:
            tensor = task_vector.vector[key]
            if isinstance(tensor, torch.Tensor):
                # Hash key, shape, dtype, and sum of values (fast approximation)
                hasher.update(key.encode('utf-8'))
                hasher.update(str(tensor.shape).encode('utf-8'))
                hasher.update(str(tensor.dtype).encode('utf-8'))
                # Use sum as fast content hash (not perfect but fast)
                tensor_sum = tensor.sum().item()
                hasher.update(str(tensor_sum).encode('utf-8'))
    
    return hasher.hexdigest()[:16]  # Use first 16 chars for shorter filenames

def get_svd_cache_path(args, dataset_name, task_vector_hash):
    """
    Generuje ścieżkę do pliku cache.
    
    Args:
        args: arguments object z atrybutem model (np. "ViT-B-16") i save (ścieżka)
        dataset_name: str - nazwa datasetu
        task_vector_hash: str - hash zawartości task_vector
        
    Returns:
        str: Ścieżka np. "{args.save}/svd_cache/{model}_{dataset_name}_{hash}.pt"
    """
    cache_dir = os.path.join(args.save, "svd_cache")
    os.makedirs(cache_dir, exist_ok=True)
    model_name = args.model.replace("-", "_")  # Replace dashes for filesystem compatibility
    cache_filename = f"{model_name}_{dataset_name}_{task_vector_hash}.pt"
    return os.path.join(cache_dir, cache_filename)

def save_svd_components_to_cache(cache_path, svd_components_dict):
    """
    Zapisuje komponenty SVD do pliku .pt.
    
    Args:
        cache_path: str - ścieżka do pliku
        svd_components_dict: dict - {layer_key: {'U': tensor, 'S': tensor, 'Vh': tensor, 'shape': tuple, 'dtype': str}}
    """
    try:
        # Convert tensors to CPU before saving
        cache_data = {}
        for layer_key, components in svd_components_dict.items():
            cache_data[layer_key] = {
                'U': components['U'].cpu(),
                'S': components['S'].cpu(),
                'Vh': components['Vh'].cpu(),
                'shape': components.get('shape', components['U'].shape),
                'dtype': str(components.get('dtype', components['U'].dtype))
            }
        torch.save(cache_data, cache_path)
        print(f"[SVD CACHE] Saved SVD components to {cache_path}")
    except Exception as e:
        print(f"[SVD CACHE ERROR] Failed to save cache to {cache_path}: {e}")

def load_svd_components_from_cache(cache_path):
    """
    Ładuje komponenty SVD z pliku .pt.
    
    Args:
        cache_path: str - ścieżka do pliku cache
        
    Returns:
        dict lub None jeśli plik nie istnieje lub jest uszkodzony
        Format: {layer_key: {'U': tensor, 'S': tensor, 'Vh': tensor, 'shape': tuple, 'dtype': str}}
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
        print(f"[SVD CACHE] Loaded SVD components from {cache_path}")
        return cache_data
    except Exception as e:
        print(f"[SVD CACHE ERROR] Failed to load cache from {cache_path}: {e}")
        print(f"[SVD CACHE] Will recompute SVD components")
        return None

def iso_c(base_model_params_dict, task_vectors, config, cache_args=None):
    """
    Merges task vectors by collecting all SVD components from all tasks for each layer,
    globally sorting them by singular value, and selecting a subset based on a threshold
    relative to a single task's component count.

    For each 2D parameter:
    1. For each task, perform SVD on the task's delta matrix.
    2. Collect all singular components (U, S, Vh) from all tasks into a single list for the layer.
    3. Sort this global list of components based on their singular values.
    4. Select the top N components, where N is determined by `config.svd_threshold`
       applied to the number of components from a *single* task.
    5. The selected U, S, and Vh components are returned directly, without a second SVD.
    Non-2D layers' deltas are averaged.
    
    Returns:
        dict: Layer components with keys:
            - 'U', 'S', 'Vh', 'is_svd' (True) - for SVD layers
            - 'tensor', 'is_svd' (False) - for non-SVD layers
            - Metadata: original_dtype, original_shape, num_selected_components
    """
    if not task_vectors:
        print("Warning: No task vectors provided to iso_c.")
        return None

    print(f"Computing global SVD component selection for {len(task_vectors)} task vectors...")
    
    with torch.no_grad():
        new_vector = {}
        for task_key in task_vectors[0].vector:
            # Boilerplate to get device, dtype, shape
            base_key = task_key
            if task_key.startswith("model."):
                base_key = task_key[len("model."):]

            if base_key not in base_model_params_dict:
                print(f"Warning: Base key '{base_key}' (from task key '{task_key}') not in base_model_params_dict. Skipping.")
                continue

            current_device = base_model_params_dict[base_key].device
            original_dtype = task_vectors[0].vector[task_key].dtype
            original_shape = task_vectors[0].vector[task_key].shape

            # Skip non-2D layers
            SKIP_PARAMS = [
                "positional_embedding", "visual.positional_embedding",
                "visual.proj", "token_embedding", "visual.proj", "text_projection", "token_embedding.weight",
                "model.positional_embedding", "model.visual.positional_embedding", "model.token_embedding.weight",
                "model.visual.proj", "model.token_embedding", "model.visual.proj", "model.text_projection"
            ]

            if len(original_shape) != 2 or "text_projection" in task_key or task_key in SKIP_PARAMS:
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
                # --- New logic: Global component selection ---
                all_components = []
                # 1. Collect all components from all tasks for the current layer
                for task_idx, task_vector in enumerate(task_vectors):
                    # Try to load SVD components from disk cache if available
                    U_task, S_task, Vh_task = None, None, None
                    if cache_args is not None:
                        # Get dataset name for this task vector
                        dataset_name = None
                        if hasattr(task_vector, 'dataset_name'):
                            dataset_name = task_vector.dataset_name
                        elif task_idx < len(task_vectors) and hasattr(task_vectors[task_idx], 'dataset_name'):
                            dataset_name = task_vectors[task_idx].dataset_name
                        
                        if dataset_name:
                            # Compute hash and get cache path
                            task_vector_hash = compute_task_vector_hash(task_vector, dataset_name)
                            cache_path = get_svd_cache_path(cache_args, dataset_name, task_vector_hash)
                            
                            # Try to load from disk
                            cached_components = load_svd_components_from_cache(cache_path)
                            if cached_components is not None and task_key in cached_components:
                                # Load from disk cache
                                components = cached_components[task_key]
                                U_task = components['U'].to(device=current_device)
                                S_task = components['S'].to(device=current_device)
                                Vh_task = components['Vh'].to(device=current_device)
                    
                    if U_task is None or S_task is None or Vh_task is None:
                        # Compute SVD (not in cache or cache not available)
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
                
                # 3. Determine how many components to keep.
                # Threshold is relative to the number of components in a *single* task.
                # max_singular_values_per_task = min(original_shape)
                # num_to_keep_old = int(config.svd_threshold * max_singular_values_per_task)
                
                # New logic: Keep a fixed number of components (e.g., 768), but not more than available.
                num_to_keep = 76
                num_to_keep = min(num_to_keep, len(all_components))

                # Ensure at least one component is kept if threshold > 0 and rounding caused 0
                if num_to_keep == 0 and config.svd_threshold > 0:
                    print(f"Warning for {task_key}: num_to_keep is 0 with svd_threshold {config.svd_threshold}. Model will handle 0 components.")

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
                print(f"  Selected {kept_component_count} components (fixed selection).")
                print(f"  Final U shape: {U_final.shape}, Final S shape: {S_final.shape}, Final Vh shape: {Vh_final.shape}")

            except Exception as e:
                print(f"Error: Global SVD component selection failed for key '{task_key}' with error: {str(e)}")
                import traceback
                traceback.print_exc()
                print(f"Skipping key {task_key} due to SVD error.")
                continue

    return new_vector

def precompute_svd_components_for_all_tasks(task_vectors_dict, base_model_params_dict, args):
    """
    Oblicza i cache'uje komponenty SVD dla wszystkich task_vectors przed pętlą main.
    Dla każdego task_vector:
    1. Oblicza hash zawartości
    2. Sprawdza cache na dysku
    3. Jeśli nie ma: oblicza SVD dla wszystkich warstw 2D
    4. Zapisuje do cache na dysku
    
    NOTE: Komponenty NIE są trzymane w pamięci - tylko na dysku.
    iso_c() będzie ładować je z dysku gdy potrzebne.
    
    Args:
        task_vectors_dict: dict {dataset_name: TaskVector}
        base_model_params_dict: dict - parametry base modelu
        args: arguments object
        
    Returns:
        None (cache jest tylko na dysku, nie w pamięci)
    """
    if not is_main_process():
        # Only main process handles cache
        return
    
    print("\n" + "="*80)
    print("[SVD CACHE] Pre-computing SVD components for all task vectors...")
    print("[SVD CACHE] Cache will be stored on disk only (not in memory)")
    print("="*80)
    
    print(f"[SVD CACHE] Processing {len(task_vectors_dict)} task vectors")
    
    # Process each task vector
    for dataset_name, task_vector in task_vectors_dict.items():
        print(f"\n[SVD CACHE] Processing task vector: {dataset_name}")
        
        # Compute hash
        task_vector_hash = compute_task_vector_hash(task_vector, dataset_name)
        print(f"[SVD CACHE] Hash for {dataset_name}: {task_vector_hash}")
        
        # Get cache path
        cache_path = get_svd_cache_path(args, dataset_name, task_vector_hash)
        
        # Try to load from cache
        cached_components = load_svd_components_from_cache(cache_path)
        
        if cached_components is not None:
            # Cache found on disk - no need to load into memory
            # iso_c() will load it from disk when needed
            print(f"[SVD CACHE] Cached SVD components found on disk for {dataset_name} (will be loaded on-demand)")
        else:
            # Cache not found - compute SVD for all 2D layers
            print(f"[SVD CACHE] Computing SVD components for {dataset_name} (not in cache)")
            
            if not hasattr(task_vector, 'vector') or not task_vector.vector:
                print(f"[SVD CACHE] Warning: Task vector {dataset_name} has no vector attribute, skipping")
                continue
            
            components_to_save = {}
            
            # Get reference for keys
            reference_keys = list(task_vector.vector.keys())
            
            for layer_key in reference_keys:
                # Determine base_key
                base_key = layer_key
                if layer_key.startswith("model."):
                    base_key = layer_key[len("model."):]
                
                # Check if key exists in base model
                if base_key not in base_model_params_dict:
                    continue
                
                # Get shape and dtype
                original_shape = task_vector.vector[layer_key].shape
                original_dtype = task_vector.vector[layer_key].dtype
                current_device = base_model_params_dict[base_key].device
                
                # Skip non-2D layers (same logic as in iso_c)
                SKIP_PARAMS = [
                    "positional_embedding", "visual.positional_embedding",
                    "visual.proj", "token_embedding", "visual.proj", "text_projection", "token_embedding.weight",
                    "model.positional_embedding", "model.visual.positional_embedding", "model.token_embedding.weight",
                    "model.visual.proj", "model.token_embedding", "model.visual.proj", "model.text_projection"
                ]
                
                if len(original_shape) != 2 or "text_projection" in layer_key or layer_key in SKIP_PARAMS:
                    continue
                
                # Compute SVD for this layer
                try:
                    delta_matrix = task_vector.vector[layer_key].to(device=current_device, dtype=torch.float32)
                    U_task, S_task, Vh_task = torch.linalg.svd(delta_matrix, full_matrices=False)
                    
                    # Store for saving to disk only (not in memory)
                    components_to_save[layer_key] = {
                        'U': U_task.cpu(),
                        'S': S_task.cpu(),
                        'Vh': Vh_task.cpu(),
                        'shape': original_shape,
                        'dtype': str(original_dtype)
                    }
                except Exception as e:
                    print(f"[SVD CACHE] Error computing SVD for {layer_key} in {dataset_name}: {e}")
                    continue
            
            # Save all components to disk
            if components_to_save:
                save_svd_components_to_cache(cache_path, components_to_save)
                print(f"[SVD CACHE] Saved {len(components_to_save)} SVD components to disk for {dataset_name}")
            else:
                print(f"[SVD CACHE] Warning: No SVD components computed for {dataset_name}")
    
    print("\n" + "="*80)
    print("[SVD CACHE] Pre-computation complete. Components are on disk and will be loaded on-demand.")
    print("="*80 + "\n")

def create_task_vector_from_merged(merged_vector, task_vectors, args):
    """(Note: Primarily for compatibility - expects reconstructed tensors)
    Creates TaskVector from traditional merged vector format.
    May require reconstruction from components for new iso_c outputs.
    """
    # Determine the type of task vector to create based on the original vectors
    if args.finetuning_mode == "linear":
        return LinearizedTaskVector(vector=merged_vector, use_half=not args.no_use_half)
    else:
        return NonLinearTaskVector(vector=merged_vector, use_half=not args.no_use_half)

class LearnableSingularValuesMergedEncoder(nn.Module):
    """Learns singular values for precomputed SVD components.
    
    Uses components from iso_c instead of performing new SVD:
    - Stores U/Vh as buffers
    - Initializes selected S values to 0.0 and makes them learnable
    - Handles non-SVD layers through direct delta tensors
    
    Args:
        merged_vector_components: Output from iso_c containing:
            - SVD components for 2D layers
            - Direct tensors for other layers
    """
    def __init__(self, model, merged_vector_components, args):
        super().__init__()
        
        self.model = model
        self.args = args
        
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir
        
        from functorch import make_functional_with_buffers
        func, params_from_functional, self.buffer = make_functional_with_buffers(model)
        self.func = lambda p, b, x: func(p, b, x)
        self.params = nn.ParameterList(params_from_functional)
        for p in self.params:
            p.requires_grad = False
            
        self.param_names = [name for name, _ in model.named_parameters()]

        self.svd_components_info = {}
        self.learnable_s_values = nn.ParameterDict()
        self.direct_deltas = {}
        self.total_learnable_sv = 0
        
        print("\n--- Initializing LearnableSingularValuesMergedEncoder ---")
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
                        # --- NEW STEP 1: Reconstruct the matrix from iso_c components and perform a new SVD ---
                        print(f"  SVD layer {param_name}: Reconstructing delta from {num_components_from_isoc} components...")
                        reconstructed_delta = U_from_isoc.to(torch.float32) @ torch.diag_embed(S_from_isoc.to(torch.float32)) @ Vh_from_isoc.to(torch.float32)
                        
                        print(f"  SVD layer {param_name}: Performing second SVD on reconstructed delta...")
                        U_new, S_new, Vh_new = torch.linalg.svd(reconstructed_delta, full_matrices=False)
                        # The new components (U_new, S_new, Vh_new) now form the basis for this layer.

                        # --- NEW LOGIC on NEW components: Split S_new into learnable and frozen parts ---
                        num_total_new_components = S_new.shape[0]
                        num_to_learn = int(self.args.svd_threshold * num_total_new_components)

                        # Split the new S values
                        S_initial_learnable = S_new[:num_to_learn]
                        S_initial_frozen = S_new[num_to_learn:]
                        
                        self.register_buffer(f'U_{svd_key_safe_name}', U_new)
                        self.register_buffer(f'Vh_{svd_key_safe_name}', Vh_new)
                        
                        self.register_buffer(f'initial_selected_S_{svd_key_safe_name}', S_initial_learnable)
                        self.register_buffer(f'frozen_S_{svd_key_safe_name}', S_initial_frozen)
                        
                        # Initialize learnable singular values to zero (not from initial values)
                        learnable_S_for_layer = torch.zeros_like(S_initial_learnable, dtype=torch.float32)
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
                        device = self.params[param_idx].device
                        self.register_buffer(f'U_{svd_key_safe_name}', torch.empty((original_shape[0], 0), dtype=torch.float32, device=device))
                        self.register_buffer(f'Vh_{svd_key_safe_name}', torch.empty((0, original_shape[1]), dtype=torch.float32, device=device))
                        self.register_buffer(f'frozen_S_{svd_key_safe_name}', torch.empty((0,), dtype=torch.float32, device=device))
                else:
                    direct_delta_tensor = layer_data['tensor']
                    self.register_buffer(f'direct_delta_{svd_key_safe_name}', direct_delta_tensor)
                    self.direct_deltas[param_name] = {
                         'original_dtype': original_dtype,
                         'key_for_buffer': f'direct_delta_{svd_key_safe_name}'
                    }
                    print(f"  Non-SVD layer {param_name} (Shape {direct_delta_tensor.shape if isinstance(direct_delta_tensor, torch.Tensor) else 'N/A'}, Dtype {direct_delta_tensor.dtype if isinstance(direct_delta_tensor, torch.Tensor) else 'N/A'}): Using direct delta.")
            else:
                print(f"  No SVD components or direct delta in merged_vector_components for {param_name}. Update for this layer will be zero.")
        
        print(f"Total learnable singular values across all layers: {self.total_learnable_sv}")
        print("--- Initialization Complete --- \n")
        
        # Add a dummy parameter to avoid DDP issues with models with no parameters
        # self.dummy_param = nn.Parameter(torch.zeros(1))

    def _apply(self, fn):
        """Override method to relocate buffer list"""
        # Apply to nn.Module's parameters and registered buffers first
        # This handles the learnable S-values and various buffers for SVD components (U, Vh, initial S) and direct deltas.
        new_self = super()._apply(fn=fn)
        
        # Apply to functorch's buffers
        if hasattr(new_self, 'buffer') and new_self.buffer is not None:
            new_self.buffer = tuple(fn(b) for b in new_self.buffer)

        # Apply to other tensors we manage: self.merged_vector (dictionary of tensors)
        # and self.params (ParameterList of frozen base parameters)
        # merged_vector is no longer a direct attribute holding tensors in the same way.
        # self.direct_deltas now just holds metadata, the tensors are buffers.
        # if hasattr(new_self, 'merged_vector') and new_self.merged_vector is not None:
        #     new_merged_vector_dict = {}
        #     for k, v_tensor in new_self.merged_vector.items():
        #         if isinstance(v_tensor, torch.Tensor):
        #             new_merged_vector_dict[k] = fn(v_tensor)
        #         else:
        #             new_merged_vector_dict[k] = v_tensor # Keep non-tensors as is
        #     new_self.merged_vector = new_merged_vector_dict
        
        if hasattr(new_self, 'params') and isinstance(new_self.params, nn.ParameterList):
            for i in range(len(new_self.params)):
                if new_self.params[i] is not None: # Should always be a tensor
                     new_self.params[i].data = fn(new_self.params[i].data)


        return new_self
    
    def train(self, mode=True):
        super().train(mode)

    def forward(self, x):
        """Forward pass applying the merged vector with learnable singular values to the model parameters."""
        final_model_params = [] # These will be passed to self.func

        for param_idx, param_name in enumerate(self.param_names):
            base_param = self.params[param_idx] # Current base weight
            
            actual_delta_c: torch.Tensor
            svd_key_safe_name = param_name.replace('.', '_')

            if param_name in self.svd_components_info:
                svd_info = self.svd_components_info[param_name]
                num_total_components = svd_info.get('num_total_components', 0)
                original_delta_dtype = svd_info['original_dtype']

                if num_total_components > 0:
                    U = getattr(self, f'U_{svd_key_safe_name}')
                    Vh = getattr(self, f'Vh_{svd_key_safe_name}')
                    
                    # Get learnable and frozen S parts
                    learnable_S_values = self.learnable_s_values.get(svd_key_safe_name)
                    frozen_S_values = getattr(self, f'frozen_S_{svd_key_safe_name}', None)
                    
                    # Build the full S vector by concatenating learnable and frozen parts
                    s_parts = []
                    if learnable_S_values is not None and learnable_S_values.numel() > 0:
                        s_parts.append(learnable_S_values)
                    if frozen_S_values is not None and frozen_S_values.numel() > 0:
                        s_parts.append(frozen_S_values)
                    
                    if not s_parts:
                        reconstructed_delta = torch.zeros(svd_info['original_shape'], device=base_param.device)
                    else:
                        full_S_vector = torch.cat(s_parts, dim=0)

                        # Explicitly cast to float32 before reconstruction
                        U_f32 = U.to(torch.float32)
                        Vh_f32 = Vh.to(torch.float32)
                        full_S_f32 = full_S_vector.to(torch.float32)
                        
                        # Reconstruct: U @ diag(S_vector) @ Vh
                        reconstructed_delta = U_f32 @ torch.diag_embed(full_S_f32) @ Vh_f32
                    
                    actual_delta_c = reconstructed_delta.to(original_delta_dtype) # Convert back to original delta's dtype
                else:
                    # No learnable SVD components for this layer (e.g. iso_c selected 0)
                    actual_delta_c = torch.zeros_like(base_param)
            
            elif f'direct_delta_{svd_key_safe_name}' in self._buffers: # Check if buffer exists
                # Non-SVD layer, use direct delta from buffer
                direct_delta_tensor = getattr(self, f'direct_delta_{svd_key_safe_name}')
                actual_delta_c = direct_delta_tensor # Already has correct dtype from buffer
            else:
                    # No SVD components and no direct delta found for this parameter.
                    # This implies it wasn't in merged_vector_components from iso_c.
                    # Delta is effectively zero.
                    actual_delta_c = torch.zeros_like(base_param)

            # Add delta_c to the base parameter
            final_model_params.append(base_param + actual_delta_c)
        
        # Apply the function with the modified parameters
        return self.func(final_model_params, self.buffer, x)

def main(rank, args):
    # First set up distributed processing
    args.rank = rank
    
    # Track if distributed is initialized
    distributed_initialized = False
    
    try:
        # Use a different port to avoid "Address already in use" errors
        # Define a list of available ports to choose from
        available_ports = list(range(29520, 29590))
        # Use the port from args if specified, otherwise find first available port
        if hasattr(args, 'port') and args.port is not None and args.port > 0:
            selected_port = args.port
            print(f"Using user-specified port {selected_port} for distributed training")
        else:
            # Try to find an available port
            selected_port = find_available_port(available_ports)
            if selected_port is None:
                print("Warning: No available ports found. Using a random port which may cause issues.")
                selected_port = random.choice(available_ports)
                print(f"Selected random port {selected_port} - this may cause issues if already in use")
            else:
                print(f"Found available port {selected_port} for distributed training")

        args.port = selected_port
        
        # Initialize distributed processing
        setup_ddp(args.rank, args.world_size, port=selected_port)
        distributed_initialized = True
        print("no_use_half", args.no_use_half)
        # NOTE: Removed automatic no_use_half=True setting to prevent CUDA OOM errors
        # User should explicitly set --no-use-half if needed for SVD operations
        
        # Then set the random seed for reproducibility
        if args.seed is not None:
            set_seed(args.seed)
        
        # Load the individual task vectors.
        pool = [
            "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
            "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech101", "Caltech256",
            "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
        ]
        # task_vectors = {}
        # for dataset in pool:
        #     if args.finetuning_mode == "linear":
        #         pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        #         finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        #         task_vectors[dataset] = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        #     else:
        #         pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        #         finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        #         task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

        args.target_dataset = args.target_dataset_name + "Val"
        if args.target_dataset_name in args.datasets:
            args.datasets.remove(args.target_dataset_name)
        original_datasets = args.datasets.copy()  # Make a copy to avoid modifying the original list
        
        # Pre-compute SVD components for all task vectors before the main loop
        # This creates disk cache that will be loaded on-demand in iso_c()
        # Create base_model_params_dict for all processes (needed for train() in each iteration)
        base_model_params_dict = None
        if args.finetuning_mode == "linear":
            if is_main_process():
                print("Warning: Linear mode - skipping SVD precomputation")
        else:
            # Create base model to get params (needed for all processes, not just main)
            base_image_encoder = ImageEncoder(args)
            base_model_params_dict = {
                name: tensor.clone() 
                for name, tensor in base_image_encoder.model.state_dict().items()
            }
            
            if is_main_process():
                # Load all task vectors once for precomputation (only in main process)
                all_task_vectors_dict = {}
                for dataset in original_datasets:
                    if args.finetuning_mode == "linear":
                        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
                        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
                        all_task_vectors_dict[dataset] = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
                    else:
                        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
                        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
                        all_task_vectors_dict[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
                
                # Precompute SVD components for all tasks (saves to disk only)
                precompute_svd_components_for_all_tasks(all_task_vectors_dict, base_model_params_dict, args)
                del all_task_vectors_dict  # Free memory
            
            # Free base_image_encoder but keep base_model_params_dict for train()
            del base_image_encoder
        
        # Loop over each dataset up to the full set
        for idx, dataset in enumerate(original_datasets):

            if idx < args.resume_from_idx:
                print(f"Skipping iteration {idx+1} because resume-from-idx is {args.resume_from_idx}")
                continue

            if idx >= args.end_index:
                print(f"Ending iteration {idx+1} because end-index is {args.end_index}")
                break

            try:
                # For each iteration, use datasets from 0 to idx (inclusive)
                args.datasets = original_datasets[:idx+1]

                # Print information about task vector precision mode
                if is_main_process():
                    precision_mode = "half precision (float16)" if not args.no_use_half else "full precision (float32)"
                    print(f"Creating task vectors using {precision_mode}")
                    print(f"SVD learning mode: learning top {args.svd_threshold*100:.1f}% singular values for each layer")
                    if args.svd_threshold > 0 and args.no_use_half:
                        print(f"Using SVD thresholding (in iso_c) with threshold {args.svd_threshold}")
                        if args.keep_top_values:
                            print(f"Mode: KEEP top {args.svd_threshold*100}% singular values, ZERO OUT the rest")
                        else:
                            print(f"Mode: ZERO OUT top {args.svd_threshold*100}% singular values, KEEP the rest")
                    elif args.svd_threshold > 0 and not args.no_use_half:
                        print(f"WARNING: SVD thresholding requires full precision. Use --no-use-half flag for SVD operations.")

                task_vectors = {}
                for dataset in args.datasets:
                    if args.finetuning_mode == "linear":
                        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
                        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
                        task_vectors[dataset] = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
                    else:
                        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
                        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
                        task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

            
                print("=" * 100)
                print(f"Learning SVD-merged task vector singular values on {args.target_dataset} with {len(args.datasets)} datasets")
                print(f"Datasets being used: {args.datasets}")
                print("=" * 100)

                # Print all command line arguments
                print("\nCommand line arguments:")
                print("-" * 50)
                for arg, value in vars(args).items():
                    print(f"{arg}: {value}")
                print("-" * 50)
                print()
                # Pass base_model_params_dict to train() to avoid recreating it in each iteration
                train(task_vectors, args, base_model_params_dict=base_model_params_dict)
                print(f"Successfully completed iteration {idx+1} with {len(args.datasets)} datasets")
                
            except Exception as e:
                print(f"ERROR: Iteration {idx+1} with {len(args.datasets)} datasets failed with error: {e}")
                import traceback
                traceback.print_exc()
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory error detected - stopping the entire job!")
                    if distributed_initialized:
                        cleanup_ddp()
                    import sys
                    sys.exit(1)
            
    finally:
        # Only cleanup distributed processing at the very end
        if distributed_initialized:
            try:
                cleanup_ddp()
            except Exception as e:
                print(f"Warning: Error during distributed cleanup: {e}")
    
    # Restore the original datasets after all iterations are complete
    args.datasets = original_datasets


def train(task_vectors, args, base_model_params_dict=None):
    """
    Train model with learnable SVD components.
    
    Args:
        task_vectors: dict {dataset_name: TaskVector}
        args: arguments object
        base_model_params_dict: dict - base model parameters (if None, will be created)
    """
    # Set seed for this process to ensure deterministic behavior

    # Track the experiment start time
    if is_main_process():
        experiment_start_time = datetime.now()
        formatted_start_time = experiment_start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[TIMING] Experiment for dataset {args.target_dataset} with {len(args.datasets)} started at: {formatted_start_time}")
    
    # Format subsample value for filename/run name
    subsample_str = f"s{int(args.subsample * 100)}" if isinstance(args.subsample, float) else f"s{args.subsample}"

    # Initialize comp_acc dictionary to avoid undefined variable error
    comp_acc = {}
    

    if args.seed is not None:
        set_seed(args.seed + args.rank)  # Add rank to avoid identical random values across processes

    target_dataset = args.target_dataset
    
    # Get experiment commit hash from environment variable
    experiment_commit = os.environ.get('EXPERIMENT_COMMIT_HASH', 'unknown')
    
    # Remove the task vector for the target task
    task_vectors_list = [v for k, v in task_vectors.items() if target_dataset.replace("Val", "") != k]
    num_source_tasks = len(task_vectors_list)

    # --- Create base_image_encoder to get its parameters *before* calling iso_c ---
    # This is also done before wandb.init as per original comment structure
    if args.finetuning_mode == "linear":
        # Linear mode still not implemented for this approach
        if is_main_process():
            print("Error: Linear mode with singular value learning not implemented.")
            if wandb.run is not None:
                wandb.finish(quiet=True)
        return
    else:
        # Use provided base_model_params_dict if available, otherwise create base_image_encoder
        if base_model_params_dict is None:
            # Create the base encoder first
            base_image_encoder = ImageEncoder(args)
            # Prepare base_model_params_dict for iso_c
            base_model_params_dict = {
                name: tensor.clone() 
                for name, tensor in base_image_encoder.model.state_dict().items()
            }
        else:
            # Use provided base_model_params_dict (created in main() before the loop)
            # Still need base_image_encoder for later use
            base_image_encoder = ImageEncoder(args)
    
    # Use iso_c to merge the task vectors
    if is_main_process():
        print("\n" + "="*50)
        print("ISO-C LEARN MODE: Merging task vectors with SVD-based component selection")
        print(f"SVD performed on (base_param + task_delta) for each task.")
        print(f"Learning top {args.svd_threshold*100:.1f}% singular values for each merged layer")
        print("="*50 + "\n")
    
    # Merge the task vectors into a single vector description
    # Pass args for cache loading (iso_c will load from disk on-demand)
    merged_vector_components = iso_c(base_model_params_dict, task_vectors_list, args, cache_args=args)
    
    if merged_vector_components is None:
        if is_main_process():
            print("Error: Failed to merge task vectors with SVD.")
            # Ensure wandb is finished even if initialization failed
            if wandb.run is not None:
                wandb.finish(quiet=True)
        return
    
    # --- Create image encoder wrapper *after* SVD merging ---
    # base_image_encoder was created earlier (if not in linear mode)
    # The LearnableSingularValuesMergedEncoder is created within the same 'else' block 
    # as base_image_encoder to ensure base_image_encoder is defined.
    if args.finetuning_mode == "linear":
        # This case is already handled and returns above. Redundant check, but safe.
        if is_main_process():
            print("Error: Linear mode (should have been caught earlier).")
        return
    else:
        # base_image_encoder is already defined from above
        # Then create the LearnableSingularValuesMergedEncoder wrapper
        # Pass the components directly from iso_c
        image_encoder = LearnableSingularValuesMergedEncoder(
            base_image_encoder, 
            merged_vector_components, # Use the output from the corrected iso_c call
            args
        )
    # ------------------------------------------------------------------------

    # Initialize wandb in the main process (after distributed setup and encoder creation)
    if is_main_process():
        # Collect system information
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_count": torch.cuda.device_count(),
                "gpu_model": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            }
        # Use WANDB_API_KEY environment variable if possible, or fall back to hardcoded key
        # Better to set this in your environment or via command line: export WANDB_API_KEY=your_api_key
        api_key = os.environ.get("WANDB_API_KEY", "66d9e2b16753e25dd022a685b96fadf363a8f58b")
        wandb.login(key=api_key)
        
        # Add more descriptive run name with SVD info
        threshold_pct = int(args.svd_threshold * 100)
        # Removed fixed_values from run name, using threshold percentage
        run_name = f"{args.model}_{target_dataset}-isoc-learn-svd-top{threshold_pct}pct-e{args.epochs}-nbdts{len(args.datasets)}"
        
        # Add SVD thresholding info (from iso_c) to run name if enabled
        if hasattr(args, 'svd_threshold') and args.svd_threshold > 0:
            iso_c_threshold_pct = int(args.svd_threshold * 100)
            iso_c_zeroing_mode = "keepTop" if args.keep_top_values else "zeroTop"
            # Distinguish iso_c thresholding mode in name if different from learning threshold (though currently they use the same arg)
            run_name += f"-iso_c_{iso_c_zeroing_mode}{iso_c_threshold_pct}pct"
        
        # Add subsample info to run name
        run_name += f"-{subsample_str}"
        
        wandb.init(
            project="atlas5",
            entity="osialm",
            name=run_name,
            config={
                # Basic experiment config
                "type": "learn_coef",
                "genre": "isoc_svd_components_learning", # Updated genre
                "kind": f"axis_{int(args.svd_threshold*100)}pct_batch_fix", # Updated kind
                "iter_fixed": 3,
                "model": args.model,
                "save": args.save,
                "target_dataset": target_dataset,
                "used_datasets": args.datasets,
                "partition": args.partition,
                "#nb_dts": len(args.datasets),
                "port": args.port,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size * args.num_grad_accumulation,
                "finetuning_mode": args.finetuning_mode,
                "lp_reg": args.lp_reg,
                "seed": args.seed,
                "weight_decay": args.wd,
                "git_commit": experiment_commit,
                "subsample": args.subsample,
                "iso_c_svd_threshold": args.svd_threshold, # Threshold used during iso_c merging
                "iso_c_keep_top_values": args.keep_top_values, # Mode for iso_c merging
                "learnable_sv_percentage": args.svd_threshold * 100, # Percentage of SVs learned
                # Updated total_learnable_values and approach description
                "total_learnable_singular_values": image_encoder.total_learnable_sv if hasattr(image_encoder, 'total_learnable_sv') else 0, 
                "learning_approach": "iso_c_selected_svd_components_with_learnable_S", # Updated approach
                
                # System information
                "system": {
                    "gpu": gpu_info,
                    "cpu_count": psutil.cpu_count(),
                    "cpu_physical_count": psutil.cpu_count(logical=False),
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                },
                
                # Runtime info
                "runtime": {
                    "pytorch_version": torch.__version__,
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                    "python_version": sys.version.split()[0],
                },
                
                # Job details
                "job_id": os.environ.get('SLURM_JOB_ID', 'unknown'),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                
                # Training details
                "grad_clip_value": 1.0,
                "num_grad_accumulation": args.num_grad_accumulation,
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.num_grad_accumulation * args.world_size,
                "mixed_precision": True,  # Using torch.autocast
                
                # Model details
                "port_selection_method": "auto_find_available",
                "sorting_descending": args.sorting_descending,
            }
        )
        
        # Print WandB URL
        if wandb.run:
            print(f"WandB Run URL: {wandb.run.url}")
            print("-" * 50 + "\n")
        
        # Log git commit hash to make it easier to find experiments
        wandb.log({"git_commit": experiment_commit})
    
        # Log initial fixed singular values to wandb
        initial_selected_sv_log = {}
        for param_name in image_encoder.param_names:
            svd_key_safe_name = param_name.replace('.', '_')
            # New buffer name for initial S values selected by iso_c
            initial_s_buffer_name = f'initial_selected_S_{svd_key_safe_name}'
            if hasattr(image_encoder, initial_s_buffer_name):
                initial_s_values = getattr(image_encoder, initial_s_buffer_name)
                if initial_s_values.numel() > 0:
                    # Log first few and basic stats for initial selected S values
                    initial_selected_sv_log[f"initial_selected_sv_{svd_key_safe_name}_count"] = initial_s_values.numel()
                    initial_selected_sv_log[f"initial_selected_sv_{svd_key_safe_name}_mean"] = initial_s_values.mean().item()
                    initial_selected_sv_log[f"initial_selected_sv_{svd_key_safe_name}_max"] = initial_s_values.max().item()
                    for i, val in enumerate(initial_s_values[:5].tolist()): # Log first 5 values
                        initial_selected_sv_log[f"initial_selected_sv_{svd_key_safe_name}_idx{i}"] = val
        if initial_selected_sv_log:
            wandb.log(initial_selected_sv_log)
    
    ckpdir = os.path.join(args.save, target_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    classification_head = get_classification_head(args, target_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.to(device='cuda')
    
    # Determine mixed precision dtype and scaler usage based on GPU capability
    if supports_bfloat16() and not args.no_use_half:
        mixed_precision_dtype = torch.bfloat16
        use_scaler = False  # bfloat16 doesn't need GradScaler
        if is_main_process():
            print(f"[MIXED PRECISION] Using bfloat16 (GPU supports it, no GradScaler needed)")
    elif not args.no_use_half:
        mixed_precision_dtype = torch.float16
        use_scaler = True  # float16 needs GradScaler
        if is_main_process():
            print(f"[MIXED PRECISION] Using float16 (with GradScaler)")
    else:
        mixed_precision_dtype = torch.float32
        use_scaler = False  # float32 doesn't need scaler
        if is_main_process():
            print(f"[MIXED PRECISION] Using float32 (full precision)")

    # Use more aggressive random crop with horizontal flip
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ), torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + model.train_preprocess.transforms[-3:])
    dataset = get_dataset(
        target_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(data_loader)

    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    
    # Try to compile model with torch.compile if available (PyTorch 2.0+)
    compiled_model = None
    if hasattr(torch, 'compile'):
        try:
            if is_main_process():
                print("[TORCH.COMPILE] Attempting to compile model with mode='reduce-overhead'...")
            compiled_model = torch.compile(model, mode='reduce-overhead')
            if is_main_process():
                print("[TORCH.COMPILE] Model compiled successfully")
        except Exception as e:
            if is_main_process():
                print(f"[TORCH.COMPILE] Warning: Failed to compile model: {e}")
                print("[TORCH.COMPILE] Continuing with uncompiled model")
            compiled_model = None
    
    # Use compiled model if available, otherwise use original
    model_to_wrap = compiled_model if compiled_model is not None else model
    
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model_to_wrap,
        device_ids=[args.rank],
        find_unused_parameters=True,
        output_device=args.rank,
        gradient_as_bucket_view=True,  # Optimize gradient communication
        static_graph=True,  # Assume graph doesn't change between iterations
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # Do not use warm up
    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # Get SLURM job ID from environment variables (use 'unknown' if not available)
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
    if "unknown" in slurm_job_id:
        slurm_job_id = time.strftime("%Y%m%d-%H%M%S")
    
    # Create a suffix for filenames that includes SVD thresholding info if enabled
    zeroing_mode = "keepTop" if args.keep_top_values else "zeroTop"
    svd_suffix = f"-{zeroing_mode}{int(args.svd_threshold * 100)}pct"
    
    # Define paths for saving results, including subsample info and learn percentage
    learn_sv_suffix = f"-learnTop{int(args.svd_threshold * 100)}pct"
    log_path = os.path.join('/raid/NFS_SHARE/home/marcin.osial/atlas/results', f"slurm-{slurm_job_id}-{target_dataset}_isoc_learned_svd{learn_sv_suffix}{svd_suffix}_{subsample_str}.json")
    print("log_path", log_path)

    scaler = GradScaler() if use_scaler else None
    
    # Update wandb config with actual dtype and scaler info
    if is_main_process() and wandb.run is not None:
        wandb.config.update({
            "mixed_precision_dtype": str(mixed_precision_dtype),
            "use_grad_scaler": use_scaler,
        })
    
    if is_main_process():
        # print(f"=> Zero-shot accuracy on {target_dataset}:\t{100*args.zs_acc[target_dataset]:.2f}%.")
        # Log zero-shot accuracy to wandb
        wandb.log({
            "target_dataset": str(target_dataset),  # Convert to string to ensure JSON serializable
            # "zero_shot_accuracy": 100 * args.zs_acc[target_dataset]
        })
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                comp_acc = json.load(f)
        else:
            comp_acc = {}

    # best_model_state = None # Removed as per request
    # best_acc = args.zs_acc[target_dataset] # best_acc is no longer updated based on validation

    # Calculate and log number of trainable parameters more accurately
    num_trainable_sv = 0
    if hasattr(image_encoder, 'total_learnable_sv'):
        num_trainable_sv = image_encoder.total_learnable_sv
        
    num_trainable_params = num_trainable_sv
    
    # Get total parameters from ddp_model, which includes all params (base + learnable)
    # For accurate trainable params, we sum num_trainable_sv and num_trainable_multipliers as calculated
    # ddp_model.parameters() will also include frozen base_params of image_encoder
    
    # Total params in the DDP wrapped model (includes everything)
    total_params_in_ddp_model = sum(p.numel() for p in ddp_model.parameters())
    
    if is_main_process():
        print(f"\nModel Parameter Stats:")
        # total_params_base_model = sum(p.numel() for p in model.image_encoder.model.parameters()) # Approx base model
        # print(f"Total parameters in base model (approx): {total_params_base_model:,}")
        print(f"Total parameters in DDP model (includes frozen base + learnable): {total_params_in_ddp_model:,}")
        print(f"Trainable singular values: {num_trainable_sv:,}")
        print(f"Total trainable parameters: {num_trainable_params:,}")
        
        # Percentage trainable relative to a hypothetical model of just the learnable parts isn't very meaningful.
        # More meaningful is % of the full model size that is being tuned.
        # However, `total_params_in_ddp_model` includes the frozen base params.
        # Let's assume `total_params` for percentage calculation should be based on effective parameters.
        # For now, use num_trainable_params / (num_trainable_params + num_frozen_base_params) if available,
        # or simply log num_trainable_params if a clean "total model params" is hard to define without double counting.
        
        # Let's log the absolute number of trainable parameters clearly.
        # The percentage can be tricky. If we consider the "original model" size for the denominator:
        original_model_params_count = sum(p.numel() for p in image_encoder.params) # These are the frozen functional params
        # print(f"Number of frozen base parameters in functional model: {original_model_params_count:,}")
        
        percentage_trainable_vs_base = 0
        if original_model_params_count > 0 : # Avoid division by zero
             # This reflects how much of the original model's parameter count is being made "plastic" or modified.
             # This isn't quite right as trainable params are *additional* or *modifying*.
             # A better reflection is trainable_params / (original_model_params + trainable_params),
             # but that can be misleading if trainable_params replace some original ones.
             # Simplest: trainable / (total_params_in_ddp_model) if ddp_model contains everything.
             pass

        if total_params_in_ddp_model > 0:
            print(f"Percentage trainable of DDP model: {100 * num_trainable_params / total_params_in_ddp_model:.4f}%\n")

        wandb.config.update({
            "total_parameters_ddp_model": total_params_in_ddp_model,
            "trainable_singular_values": num_trainable_sv,
            "trainable_parameters": num_trainable_params,
            "trainable_parameters_percentage_of_ddp": (100 * num_trainable_params / total_params_in_ddp_model) if total_params_in_ddp_model > 0 else 0
        })
        
        # Also log as metrics for tracking over time
        wandb.log({
            "total_parameters_ddp_model": total_params_in_ddp_model,
            "trainable_singular_values": num_trainable_sv,
            "trainable_parameters_total": num_trainable_params, # Renamed for clarity
            "trainable_parameters_percentage_of_ddp": (100 * num_trainable_params / total_params_in_ddp_model) if total_params_in_ddp_model > 0 else 0
        })

    for epoch in range(args.epochs):
        # Track epoch metrics
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_total_correct = 0
        epoch_total_samples = 0
        
        ddp_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device='cuda', non_blocking=True)
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=mixed_precision_dtype):
                logits = ddp_model(inputs)
                labels = batch["labels"].to(device='cuda', non_blocking=True)
                loss = loss_fn(logits, labels)
                
                # Calculate training accuracy for the current batch
                preds = logits.argmax(dim=-1)
                correct_in_batch = (preds == labels).sum().item()
                total_in_batch = labels.size(0)
                batch_training_accuracy = correct_in_batch / total_in_batch

                # Apply regularization on singular values
                sv_reg = 0.0
                for param_name, param in ddp_model.module.image_encoder.learnable_s_values.items():
                    sv_reg += torch.norm(param, p=2)
                sv_reg = args.lp_reg * sv_reg if args.lp_reg is not None else 0
                loss = loss + sv_reg / args.num_grad_accumulation
                
                loss = loss / args.num_grad_accumulation
                
                # Track loss for epoch average
                if is_main_process():
                    epoch_loss += loss.item() * args.num_grad_accumulation
                    epoch_steps += 1

            # Backward pass with or without scaler depending on dtype
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"           # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",   # noqa: E501
                    flush=True,
                )
                print(f"Batch Training Accuracy: {batch_training_accuracy*100:.2f}%") # Print batch accuracy
                
                # Log batch metrics to wandb
                batch_log = {
                    "dataset": str(target_dataset),  # Convert to string to ensure JSON serializable
                    "epoch": epoch,
                    "batch": i,
                    "batch_loss": loss.item() * args.num_grad_accumulation,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "batch_time": batch_time,
                    "data_time": data_time,
                    "global_step": step,
                    "singular_value_reg": sv_reg.item() if hasattr(sv_reg, 'item') else sv_reg,
                    "batch_training_accuracy": batch_training_accuracy * 100,
                }
                
                wandb.log(batch_log)

            # Accumulate correct predictions and samples for epoch accuracy (for the current DDP process)
            # Ensure these are properly handled for DDP reduction later
            # We need to sum `correct_in_batch` and `total_in_batch` from each batch within this DDP process.
            # Then, these sums will be all_reduced across DDP processes.
            # Initialize outside the `if (i + 1) % args.num_grad_accumulation == 0:` block if they are per-batch sums
            # Let's define running sums for the current DDP process for the epoch
            if 'process_epoch_correct' not in locals(): # Initialize if first batch of epoch for this process
                process_epoch_correct = 0
                process_epoch_samples = 0
            process_epoch_correct += correct_in_batch
            process_epoch_samples += total_in_batch

        # Evaluate after each epoch
        if is_main_process():
            # Aggregate epoch_total_correct and epoch_total_samples from all DDP processes
            # Create tensors for all_reduce
            # These sums are from `process_epoch_correct` and `process_epoch_samples` accumulated in each process
            
            # Sum correct predictions and total samples from all processes
            # We need to get the accumulated `process_epoch_correct` and `process_epoch_samples` from each process
            # For this, we'll create tensors on the current device and then all_reduce them.

            # We need to collect `process_epoch_correct` and `process_epoch_samples` from all processes.
            # Let's define `global_epoch_correct` and `global_epoch_samples` as tensors for all_reduce.
            # Ensure these are initialized before the loop if they are accumulated over batches for each process.
            # The accumulation `process_epoch_correct` and `process_epoch_samples` happened inside the batch loop.

            # Tensors for all_reduce. Must be on the same device.
            # `process_epoch_correct` and `process_epoch_samples` are scalars from the loop for the current process.
            # So we convert them to tensors.
            tensor_process_epoch_correct = torch.tensor(process_epoch_correct, dtype=torch.float32, device=args.rank)
            tensor_process_epoch_samples = torch.tensor(process_epoch_samples, dtype=torch.float32, device=args.rank)

            torch.distributed.all_reduce(tensor_process_epoch_correct, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(tensor_process_epoch_samples, op=torch.distributed.ReduceOp.SUM)

            global_epoch_correct = tensor_process_epoch_correct.item()
            global_epoch_samples = tensor_process_epoch_samples.item()

            epoch_training_accuracy = 0
            if global_epoch_samples > 0:
                epoch_training_accuracy = (global_epoch_correct / global_epoch_samples) * 100
            else:
                print("Warning: global_epoch_samples is 0. Cannot compute epoch training accuracy.")

            # Reset for next epoch for this process
            process_epoch_correct = 0
            process_epoch_samples = 0

            # Log average epoch loss
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            wandb.log({
                "target_dataset": str(target_dataset),  # Convert to string to ensure JSON serializable
                "epoch": epoch,
                "epoch_loss": avg_epoch_loss,
                "epoch_training_accuracy": epoch_training_accuracy, # Log epoch training accuracy
            })
            
            print(f"Epoch {epoch}: Average Loss: {avg_epoch_loss:.4f}, Epoch Training Accuracy: {epoch_training_accuracy:.2f}%")

            # Removed validation set evaluation during training to speed up calculations
            # Final evaluation on test set is performed after training completes
            
            # Log learnable singular values
            image_encoder = ddp_model.module.image_encoder
            epoch_log = {
                "dataset": str(target_dataset),  # Convert to string to ensure JSON serializable
                "epoch": epoch,
            }
            
            for key, param in image_encoder.learnable_s_values.items():
                # For each parameter, log the top 5 singular values (or all if fewer than 5)
                values = param.detach().cpu().tolist()
                for i, val in enumerate(values[:5]): # key is already param_name.replace('.', '_')
                    epoch_log[f"sv_{key}_idx{i}_value"] = val 
                
                # Also log mean and max
                epoch_log[f"sv_{key}_mean"] = param.detach().mean().item()
                epoch_log[f"sv_{key}_max"] = param.detach().max().item()

            wandb.log(epoch_log)

    if is_main_process():
        # --- 1. Evaluate the model at its final training state (last epoch) ---
        print("\nEvaluating model at the end of training (last epoch state)...")
        image_encoder_last_state = ddp_model.module.image_encoder
        
        # Debug: Log sums of learnable parameters for the last state
        last_s_values_sum = 0.0
        if hasattr(image_encoder_last_state, 'learnable_s_values'):
            for param in image_encoder_last_state.learnable_s_values.values():
                last_s_values_sum += param.detach().sum().item()
        
        print(f"[DEBUG LAST STATE] Sum of learnable_s_values: {last_s_values_sum}")
        wandb.log({
            "debug_last_state_s_values_sum": last_s_values_sum
        })

        test_metrics_last_model = eval_single_dataset(image_encoder_last_state, target_dataset.replace("Val", ""), args)
        final_test_acc_last_model = test_metrics_last_model["top1"]
        
        print(f"Final test accuracy (last model state) on {target_dataset.replace('Val', '')}: {100 * final_test_acc_last_model:.2f}%")
        wandb.log({
            "dataset": str(target_dataset.replace('Val', '')), 
            "final_test_accuracy_last": 100 * final_test_acc_last_model,
        })

        target_dataset_test = target_dataset.replace("Val", "")
        
        comp_acc[target_dataset_test] = final_test_acc_last_model # Store the one from the last model state
        
        # Log final results (only last_model)
        wandb.log({
            "dataset": str(target_dataset_test), 
            "final_test_accuracy": 100 * final_test_acc_last_model, # On Test set from last model
            "git_commit": experiment_commit,
        })
        
        with open(log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)

        # Create a new path for the test accuracy file with SLURM job ID
        # Use the same suffix generation as model_path/log_path
        test_acc_path = os.path.join('/raid/NFS_SHARE/home/marcin.osial/atlas/results',
                                     f"slurm-{slurm_job_id}-{target_dataset_test}_isoc_learned_svd{learn_sv_suffix}{svd_suffix}_{subsample_str}_accuracy.json")
        
        # Prepare test accuracy data with additional information
        test_acc_data = {
            "target_dataset": str(target_dataset_test),  # Convert to string to ensure JSON serializable
            "number_of_used_datasets": len(args.datasets),  # Number of datasets used in this run
            "datasets_names": args.datasets,  # List of dataset names used
            "test_accuracy_last_model": float(final_test_acc_last_model), # Final test accuracy (from last model state)
            "test_metrics_last_model": test_metrics_last_model, # Metrics from last model state
            "git_commit": experiment_commit,  # Include the commit hash in results file
            "iso_c_svd_threshold": args.svd_threshold if hasattr(args, 'svd_threshold') else 0.0,
            "iso_c_keep_top_values": args.keep_top_values if hasattr(args, 'keep_top_values') else False,
            "iso_c_zeroing_mode": "keep_top" if (hasattr(args, 'keep_top_values') and args.keep_top_values) else "zero_top",
            "iso_c_zeroing_percentage": int(args.svd_threshold * 100) if hasattr(args, 'svd_threshold') else 0,
            "learnable_sv_percentage": args.svd_threshold * 100, # Percentage learned (actually, % selected by iso_c, all of which are learned)
            # The final_sv_values dictionary is added below
            # "learned_singular_values": {k: v.tolist() for k, v in image_encoder.learnable_s_values.items()},
            # Add initial selected singular values to the results file
            "initial_selected_singular_values_from_isoc": {}
        }
        
        # Add learned singular values with clear indices to the final results
        final_sv_values = {}
        for key, param in image_encoder.learnable_s_values.items():
            values = param.detach().cpu().tolist()
            # Store each singular value with its index for clearer data analysis
            sv_with_indices = {f"idx{i}": val for i, val in enumerate(values)}
            final_sv_values[key] = sv_with_indices 
            
            # Also log to wandb with clear indices
            for i, val in enumerate(values): # key is param_name.replace('.', '_')
                wandb.log({f"final_sv_{key}_idx{i}_value": val})
                
        # Add to test_acc_data
        test_acc_data["learned_singular_values"] = final_sv_values
        
        # Add initial fixed singular values to the final results file
        initial_selected_sv_data_for_json = {}
        for param_name in image_encoder.param_names:
            svd_key_safe_name = param_name.replace('.', '_')
            initial_s_buffer_name = f'initial_selected_S_{svd_key_safe_name}'
            if hasattr(image_encoder, initial_s_buffer_name):
                initial_s_values = getattr(image_encoder, initial_s_buffer_name)
                if initial_s_values.numel() > 0:
                    selected_sv_with_indices = {f"idx{i}": val for i, val in enumerate(initial_s_values.tolist())}
                    initial_selected_sv_data_for_json[f"initial_selected_sv_{svd_key_safe_name}"] = selected_sv_with_indices
        test_acc_data["initial_selected_singular_values_from_isoc"] = initial_selected_sv_data_for_json

        # Save or append the test accuracy data to the file
        if os.path.exists(test_acc_path):
            # Load existing data
            with open(test_acc_path, 'r') as f:
                existing_data = json.load(f)
                
            # Convert to list if single dict
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
                
            # Append new data
            existing_data.append(test_acc_data)
            
            # Save updated data
            with open(test_acc_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
        else:
            # Create new file with initial data
            with open(test_acc_path, 'w') as f:
                json.dump([test_acc_data], f, indent=4)
        print(f"Test accuracy saved to {test_acc_path}")
     
    # Calculate experiment duration
    experiment_end_time = datetime.now()
    experiment_duration = experiment_end_time - experiment_start_time
    formatted_end_time = experiment_end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Format duration as hours:minutes:seconds
    hours, remainder = divmod(experiment_duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    if is_main_process():
        print(f"\n[TIMING] Experiment for dataset {args.target_dataset} with {len(args.datasets)} completed at: {formatted_end_time}")
        print(f"[TIMING] Total duration: {formatted_duration} (H:M:S)")
        
        # Log timing information to wandb
        wandb.log({
            "experiment_start_time": formatted_start_time,
            "experiment_end_time": formatted_end_time,
            "experiment_duration_seconds": experiment_duration.total_seconds(),
            "experiment_duration_formatted": formatted_duration,
        })
    
    # Finish the wandb run before cleaning up distributed process
    if is_main_process():
        wandb.finish(quiet=True)


if __name__ == "__main__":
    # Define target datasets
    target_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
        "CIFAR10",
        "CIFAR100",
        "ImageNet",
        "STL10",
        "Food101",
        "Caltech101",
        "Caltech256",
        "FGVCAircraft",
        "Flowers102",
        "OxfordIIITPet",
        "CUB200",
        "PascalVOC",
        "Country211",
        "UCF101",
    ]

    # Parse command line arguments
    args = parse_arguments()
    
    # Add default attributes
    if not hasattr(args, 'no_commit'):
        args.no_commit = False
    
    # Print experiment tracking info
    print("=" * 80)
    print("AUTOMATIC EXPERIMENT TRACKING")
    print("-" * 50)
    print("This script automatically creates a git commit at the beginning of each run")
    print("to ensure that experiment code states are tracked.")
    print("Use --no-commit flag to skip this behavior if needed.")
    print("=" * 80)
    
    # Set default values
    args.datasets = target_datasets
    args.lr = 0.01
    args.epochs = 10
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    
    args.batch_size = 576 if args.model == "ViT-L-14" else 576
    args.num_grad_accumulation = 1 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    
    # Set the seed to 0 for deterministic runs
    if args.seed is None:
        args.seed = 0
    args.save = args.save + f"{args.model}"
    
    # Load zero-shot accuracies
    # with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
    #     args.zs_acc = json.load(f)
    
    # Create a git commit
    experiment_commit = "skipped-by-user"
    if not args.no_commit:
        experiment_commit = create_experiment_commit(args)
        print(f"Code state committed with hash: {experiment_commit}")
    else:
        print("Automatic git commit skipped (--no-commit flag used)")
    
    # NOTE: Removed automatic no_use_half=True setting to prevent CUDA OOM errors
    # User should explicitly set --no-use-half if needed for SVD operations
    
    # Set SVD-related flags
    args.isoc = True
    args.use_svd = True
    # Removed the default setting for args.fixed_values
    # if not hasattr(args, 'fixed_values'):
    #     args.fixed_values = 3  # Default number of top singular values to learn per source task
    
    # Set SVD thresholding parameters (used for both iso_c merging and % learned)
    # args.svd_threshold = 0.1 # Keep the top 10% of singular values
    args.keep_top_values = True  # False = zero out top values, True = keep top values
    args.sorting_descending = True

    
    print("\n" + "=" * 80)
    print("SVD LEARNING & THRESHOLDING CONFIGURATION:")
    print(f"- Learning top {args.svd_threshold * 100:.1f}% singular values for each layer")
    print(f"- iso_c Threshold: {args.svd_threshold * 100:.1f}% of singular values")
    print(f"- iso_c Mode: {'KEEP top values, ZERO OUT the rest' if args.keep_top_values else 'ZERO OUT top values, KEEP the rest'}")
    print("=" * 80 + "\n")
    
    # Store commit hash
    os.environ['EXPERIMENT_COMMIT_HASH'] = experiment_commit if experiment_commit else ""
    
    # Launch distributed training
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
