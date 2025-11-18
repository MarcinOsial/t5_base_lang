import os
import pickle
import random
import socket
import subprocess
import platform
import psutil
import sys
import time
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import torch
import tqdm
from src.datasets.common import maybe_dictionarize
from torch.utils.data.sampler import BatchSampler
import itertools

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier

def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)

def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu")
    if device is not None:
        model = model.to(device)
    return model

def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)

def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)

class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_optimal_coef(
    results,
    metric="avg_normalized_top1",
    minimize=False,
    control_metric=None,
    control_metric_threshold=0.0,
):
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        print(f"Scaling coef: {scaling_coef}")
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef


def nonlinear_advantage(nonlinear_acc, linear_acc, num_classes):
    """Computes the normalized non-linear advantage of a finetuned model.

    The nonlinear_advantage is defined as:
        error_rate(linear_model) - error_rate(nonlinear_model) / (1 - 1 / num_classes)
    and takes values between [-1, 1]. A value of 0 indicates that the nonlinear
    model is no better than the linear one. Meanwhile, a value of 1 indicates
    that the nonlinear model is perfect and the linear trivial, and a value of
    -1 indicates the opposite.
    """
    return (nonlinear_acc - linear_acc) / (1.0 - 1.0 / num_classes)

class IndexWrapper(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
    def __getitem__(self, index):
        instance = self.dataset[index]
        if isinstance(instance, dict):
            instance["index"] = index
            return instance
        return *instance, index
    
    def __len__(self):
        return len(self.dataset)
    
def get_n_shots(dataset, shots, n_class, args):
    index_dataset = IndexWrapper(dataset)
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    targets = - torch.ones(len(dataset), dtype=torch.long)
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            batch = maybe_dictionarize(batch)
            targets[batch["index"]] = batch["labels"].to(targets.device)
            if i >= 1000:
                print("Too much data, breaking ...")
                break
            
    to_keep = torch.tensor([], dtype=torch.long)
    for c in range(n_class):
        cond = (targets == c)
        ids_c = torch.arange(len(targets))[cond]
        a = torch.randperm(len(ids_c))
        to_keep = torch.cat((to_keep, ids_c[a[-shots:]]))
        
    return to_keep

def get_preds(dataset, model, args):
    index_dataset = IndexWrapper(dataset)
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    all_preds = - torch.ones((len(dataset), model.module.classification_head.out_features))
    trusted = torch.zeros(len(dataset), dtype=torch.bool)
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            batch = maybe_dictionarize(batch)
            preds = model(batch["images"].cuda())
            all_preds[batch["index"]] = torch.nn.functional.softmax(preds, dim=-1).to(all_preds)
    return all_preds


class TIPWrapper(torch.nn.Module):
    def __init__(self, model, features_cache, labels):
        super().__init__()
        for p in model.parameters():
            p.requires_grad = False    
        self.model = model
        
        features_cache = features_cache.permute(1, 0).detach() #Just in case
        self.adapter = torch.nn.Linear(features_cache.shape[0], features_cache.shape[1], bias=False)
        self.adapter.weight.data = features_cache.t()
        self.beta_alpha = torch.nn.Parameter(torch.tensor([1.,2.]))
        self.labels = torch.nn.functional.one_hot(labels.long())
        print("Num classes", self.model.classification_head.weight.shape[0])

    def forward(self, x, tv_logits=None, feats=None):
        if tv_logits is None:
            tv_logits, feats = self.model(x, return_features=True)
        
        affinity = self.adapter(feats)
        cache_logits = ((-1) * (self.beta_alpha[0] - self.beta_alpha[0] * affinity)).exp() @ self.labels.to(affinity)
        logits = cache_logits * self.beta_alpha[1] + tv_logits
        return logits
    
class LPPWrapper(torch.nn.Module):
    def __init__(self, model, features_cache, labels, shots):
        super().__init__()
        for p in model.parameters():
            p.requires_grad = False
            
        self.model = model        
        from src.lpplusplus import init_lp
        self.adapter, self.alpha_vec, self.lr_alpha, self.lr_temp = init_lp(features_cache, labels, self.model.classification_head.weight.T / 100., shots)

    def forward(self, x, tv_logits=None, feats=None):
        if tv_logits is None:
            tv_logits, feats = self.model(x, return_features=True)
            
        vision_logits = self.adapter(feats)
        logits = vision_logits + torch.ones(feats.shape[0], 1).to(feats) @ self.alpha_vec.to(feats) * tv_logits / 100
        return logits
    
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler, epochs):
        self.sampler = sampler
        self.epochs = epochs

    def __iter__(self):
        for _ in range(self.epochs):
            yield from iter(self.sampler)

    def __len__(self):
        return self.epochs * len(self.sampler)

    
def iterate_once(iterable):
   
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class TwoStreamBatchSampler(BatchSampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.inter_batch_size = 3 * batch_size // 4
        self.batch_size = batch_size

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, 3*self.batch_size // 4),
                    grouper(secondary_iter,  self.batch_size // 4))
        )

    def __len__(self):
        return len(self.primary_indices) // self.inter_batch_size
    
class TwoAsymetricTransform:
    """Create two asymetrics transforms of the same image"""

    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2
 
    def __call__(self, x, *args, **kwargs):
        return [self.transform(x, *args, **kwargs), self.transform2(x, *args, **kwargs)]

# Import is_main_process from distributed utils if needed, or define it simply
# Assuming a simple check for now if distributed utils are not part of this module
def is_main_process() -> bool:
    """Checks if the current process is the main process."""
    # This is a placeholder. Replace with actual check from your distributed setup
    # e.g., return torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True
    return os.environ.get("RANK", "0") == "0"

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits.

    Args:
        x: Input tensor of logits (shape: [batch_size, num_classes]).

    Returns:
        Tensor containing the entropy for each sample (shape: [batch_size]).
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def lp_reg(x: Optional[torch.Tensor], p: Optional[Union[int, float, str]] = None, gamma: float = 0.5) -> Union[torch.Tensor, float]:
    """Calculates the Lp norm regularization term.

    Args:
        x: The tensor to regularize. If None or requires_grad is False, returns 0.
        p: The order of the norm (e.g., 1 for L1, 2 for L2). Defaults to None.
        gamma: The regularization strength coefficient. Defaults to 0.5.

    Returns:
        The calculated regularization term (scalar tensor) or 0.0.
    """
    # If x is None, p is None, or tensor doesn't require gradients, return 0
    if x is None or p is None or not x.requires_grad:
        return 0.0
    # Calculate the Lp norm across dimension 0 and take the mean
    return gamma * torch.norm(x, p=p, dim=0).mean()

def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across relevant libraries.

    Args:
        seed: The integer value to use as the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set deterministic options for CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    if is_main_process():
        print(f"Random seed set to {seed}")

def create_experiment_commit(args=None) -> Optional[str]:
    """Creates a git commit to snapshot the code state at the start of an experiment.

    Includes hostname, timestamp, and optional experiment arguments in the commit message.
    Adds all current changes before committing.

    Args:
        args: Optional argparse namespace or similar object containing experiment
              parameters to include in the commit message.

    Returns:
        The commit hash (str) if successful, otherwise None.
    """
    try:
        # Check if inside a git repository
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        # Check for uncommitted changes
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )

        # Get current commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        current_commit = hash_result.stdout.strip()

        if not status_result.stdout.strip():
            if is_main_process():
                print("No changes to commit. Current commit:", current_commit)
            return current_commit

        # Prepare commit message
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Experiment snapshot: {timestamp} on {hostname}"

        if args is not None:
            exp_details = []
            # Safely access common args attributes
            if hasattr(args, 'model'): exp_details.append(f"model={args.model}")
            if hasattr(args, 'target_dataset'): exp_details.append(f"dataset={args.target_dataset}")
            elif hasattr(args, 'datasets') and args.datasets: exp_details.append(f"datasets={args.datasets[0] if len(args.datasets) == 1 else f'{len(args.datasets)} datasets'}")
            if hasattr(args, 'epochs'): exp_details.append(f"epochs={args.epochs}")
            if hasattr(args, 'lr'): exp_details.append(f"lr={args.lr}")
            if hasattr(args, 'blockwise_coef') and args.blockwise_coef: exp_details.append("blockwise=True")

            if exp_details:
                commit_message += f" - {', '.join(exp_details)}"

        # Add and commit changes
        subprocess.run(["git", "add", "."], check=True)
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            check=True
        )

        # Extract new commit hash
        commit_hash = None
        output_lines = commit_result.stdout.split('\n')
        if output_lines and output_lines[0].startswith('['):
            parts = output_lines[0].split()
            if len(parts) > 1:
                # Handle potential branch name like '(root-commit)' or 'main'
                commit_hash = parts[1].strip(']')

        if is_main_process():
            print(f"Created experiment snapshot commit: {commit_hash}")
            print(f"Commit message: {commit_message}")

        return commit_hash

    except subprocess.CalledProcessError as e:
        if "not a git repository" in e.stderr.lower():
             if is_main_process():
                print("Warning: Not inside a git repository. Skipping experiment commit.")
        elif is_main_process():
            print(f"Warning: Failed to create experiment commit: {e}")
        return None
    except FileNotFoundError:
         if is_main_process():
            print("Warning: 'git' command not found. Skipping experiment commit.")
         return None
    except Exception as e: # Catch other potential errors
        if is_main_process():
            print(f"Warning: An unexpected error occurred during git commit: {e}")
        return None


def is_port_available(port: int) -> bool:
    """Checks if a TCP port is available by trying to bind to it.

    Args:
        port: The port number to check.

    Returns:
        True if the port is available, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # Try to bind to the port
            s.bind(('', port))
            # Set SO_REUSEADDR to allow immediate reuse if needed elsewhere
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return True
        except OSError:
            # Port is likely already in use
            return False

def find_available_port(port_list: List[int]) -> Optional[int]:
    """Finds the first available TCP port from a given list.

    Shuffles the list and performs a double check for reliability.

    Args:
        port_list: A list of port numbers to check.

    Returns:
        The first available port number found, or None if none are available.
    """
    port_list = list(port_list)  # Create a copy
    random.shuffle(port_list)

    for port in port_list:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
            tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                tcp_socket.bind(('', port))
                # Short delay and double check
                time.sleep(0.1)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as second_check:
                    second_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        second_check.bind(('', port))
                        if is_main_process():
                            print(f"Port {port} confirmed available.")
                        return port
                    except OSError:
                        if is_main_process():
                            print(f"Port {port} failed second check (became unavailable).")
                        continue # Try next port
            except OSError:
                if is_main_process():
                    print(f"Port {port} is not available.")
                continue # Try next port

    if is_main_process():
        print("No available ports found in the provided list.")
    return None

# Add other general utility functions here if needed
