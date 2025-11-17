import sys, os
import warnings

# Filter out noisy warnings from external libraries
# These warnings are from dependencies and don't affect functionality
warnings.filterwarnings("ignore", message=".*repr.*attribute.*Field.*function.*")
warnings.filterwarnings("ignore", message=".*frozen.*attribute.*Field.*function.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Using custom data configuration.*")
warnings.filterwarnings("ignore", message=".*Found cached dataset.*")
# Suppress all warnings from specific modules
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")
warnings.filterwarnings("ignore", module="promptsource.templates")
warnings.filterwarnings("ignore", module="datasets.builder")

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())
os.environ["HF_HOME"] = os.path.join("/raid/NFS_SHARE/home/marcin.osial/ties-merging/.cache/huggingface/")

# Configure torch inductor environment variables to prevent C++ compilation errors
# These settings help avoid issues like 'zuf0 was not declared in this scope'
# CRITICAL: Completely disable C++ codegen - use Triton only
# This must be set BEFORE importing torch
os.environ["TORCHINDUCTOR_DISABLE_ONLINE_SOFTMAX"] = "1"  # Disable problematic online softmax
os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"  # Use unique kernel names to avoid conflicts
# CRITICAL: Force Triton backend and disable C++ codegen completely
os.environ["TORCH_COMPILE_DEBUG"] = "0"  # Disable debug mode for better performance
# Disable C++ codegen entirely - this is the key setting to avoid 'zuf0' errors
# Use multiple environment variables to ensure C++ codegen is disabled
os.environ["TORCHINDUCTOR_USE_CPP_WRAPPER"] = "0"  # Disable C++ wrapper completely
os.environ["TORCHINDUCTOR_CPP"] = "0"  # Disable C++ codegen (alternative variable name)
os.environ["TORCHINDUCTOR_CPP_WRAPPER"] = "0"  # Disable C++ wrapper (another alternative)
# Force static shapes to avoid dynamic shape compilation issues
os.environ["TORCHDYNAMO_DISABLE"] = "0"  # Keep dynamo enabled

# Disable tokenizers parallelism to avoid warnings when using DataLoader with num_workers > 0
# This prevents "The current process just got forked" warnings when workers inherit tokenizer state
# Must be set BEFORE importing transformers or using tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set multiprocessing start method to 'spawn' BEFORE any imports that use multiprocessing
# This is critical for CUDA compatibility - must be set before creating any processes
import multiprocessing as mp
try:
    # Try to set start method if not already set
    current_method = mp.get_start_method(allow_none=True)
    if current_method is None:
        mp.set_start_method('spawn', force=False)
        print("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    elif current_method != 'spawn':
        print(f"WARNING: Multiprocessing start method is '{current_method}', not 'spawn'. "
              f"This may cause CUDA issues. Attempting to change to 'spawn'...")
        try:
            mp.set_start_method('spawn', force=True)
            print("Successfully changed multiprocessing start method to 'spawn'")
        except RuntimeError as e:
            print(f"Could not change start method: {e}. Continuing with '{current_method}'.")
except RuntimeError as e:
    # Start method already set - that's OK, we'll handle it in Batcher
    print(f"Multiprocessing start method already set: {mp.get_start_method(allow_none=True)}")

import torch
import argparse
import logging
from tqdm import tqdm

# CRITICAL: Monkey patch inductor to completely disable C++ codegen
# This prevents 'zuf0 was not declared' errors by forcing Triton-only backend
def _disable_cpp_codegen():
    """Completely disable C++ codegen in PyTorch Inductor to avoid compilation errors."""
    try:
        if hasattr(torch, '_inductor'):
            # Disable C++ codegen at the lowest level
            if hasattr(torch._inductor, 'config') and hasattr(torch._inductor.config, 'cpp'):
                if hasattr(torch._inductor.config.cpp, 'enabled'):
                    torch._inductor.config.cpp.enabled = False
                # Also try to patch the codegen function if it exists
                try:
                    # Try to disable C++ codegen by patching the wrapper
                    if hasattr(torch._inductor.config.cpp, 'use_cpp_wrapper'):
                        torch._inductor.config.cpp.use_cpp_wrapper = False
                except Exception:
                    pass
    except Exception as e:
        # If patching fails, log but continue
        pass

# Call immediately after torch import to disable C++ before any compilation
_disable_cpp_codegen()

from collections import OrderedDict

import torch.multiprocessing as torch_mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn

from src.model.Checkpointer import Checkpointer
from src.model.load_model import load_model
from src.model.ModelConfig import ModelConfig
from src.model.utils import get_parameterCount

from src.train.TrainingConfig import TrainingConfig
from src.train.utils import (
    load_checkpointToResumeFrom,
    construct_optimizer,
    construct_scheduler,
)

from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.MultiEvaluationConfig import MultiEvaluationConfig
from src.eval.scores import get_averageScores, concatenate_scores, extact_score
from src.eval.evaluate import evaluate_multipleConfigs

from src.utils.utils import (
    group_by,
    map_forDictionaries,
    deep_update,
    ParseKwargs,
    set_seeds,
    get_average,
)
from src.utils.distributed_utils import (
    reduce_gatheredOutput,
    is_nodeZero,
    is_distributedSetup,
)


from src.data.Batcher import Batcher
from src.data.dataset_readers import get_datasetReader
from src.data.dataset_mixtures import get_datasetMixtureReader, get_datasetMixture
from src.data.PytorchDataset import PytorchDataset

logger = logging.getLogger("root")



def evaluate_checkpoint(
    model,
    tokenizer,
    cached_datasetReaders,
    evaluation_config,
    inference_dataset_mixture,
    batch_idx,
    should_evalTrain,
    should_evalValidation,
    device,
    training_config=None,
):
    """

    Args:
        model:
        evaluation_batchers:
        evaluation_config:
        inference_dataset_mixture:
        batch_idx:
        should_evalTrain:
        should_evalValidation:
        device:

    Returns:

    """
    logger.info(f"Evaluating checkpoint")

    # Ensure prediction_dir is set (required for evaluation)
    if evaluation_config.prediction_dir is None:
        raise ValueError(
            f"evaluation_config.prediction_dir is None. This should be set from training_config.experiment_dir. "
            f"Please check that experiment_dir is properly set in TrainingConfig."
        )

    batch_predictionDir = os.path.join(
        evaluation_config.prediction_dir, f"batch_{batch_idx}"
    )

    fields_toIterateOver = []
    fields_toUpdate = {"prediction_dir": batch_predictionDir}

    """
    Compute arguments for evaluating various splits
    """
    splits_toEvaluate = []
    if should_evalTrain:
        splits_toEvaluate.append("train")
    if should_evalValidation:
        splits_toEvaluate.append("validation")

    if len(splits_toEvaluate) > 1:
        fields_toIterateOver.append("split")
        fields_toUpdate["split"] = splits_toEvaluate
    elif len(splits_toEvaluate) == 1:
        # If only one split, ensure it's set in fields_toUpdate to prevent None errors
        # This ensures split is properly set even if evaluation_config.split was None
        fields_toUpdate["split"] = splits_toEvaluate[0]

    if len(splits_toEvaluate) == 0:
        raise ValueError("No splits to evaluate")

    """
    Compute arguments for evaluating dataset mixture
    """
    if inference_dataset_mixture is not None:
        fields_toIterateOver.append("inference_dataset")
        fields_toUpdate["inference_dataset"] = get_datasetMixture(
            inference_dataset_mixture
        )
    elif evaluation_config.inference_dataset is None:
        # If inference_dataset_mixture is None and inference_dataset is None,
        # use train_dataset for evaluation (default behavior: evaluate on training dataset)
        train_dataset = None
        if training_config is not None and hasattr(training_config, 'train_dataset') and training_config.train_dataset is not None:
            train_dataset = training_config.train_dataset
        elif hasattr(evaluation_config, 'train_dataset') and evaluation_config.train_dataset is not None:
            train_dataset = evaluation_config.train_dataset
        
        if train_dataset is not None:
            evaluation_config.inference_dataset = train_dataset
            logger.info(f"inference_dataset was None, using train_dataset: {train_dataset}")
        else:
            raise ValueError(
                "Both inference_dataset_mixture and inference_dataset are None, "
                "and train_dataset is not available. Please set inference_dataset or inference_dataset_mixture."
            )

    # CRITICAL: Ensure prediction_dir is set in both evaluation_config and base_config_dict
    # This prevents NoneType errors in MultiEvaluationConfig.get_allConfigs()
    if evaluation_config.prediction_dir is None:
        evaluation_config.prediction_dir = batch_predictionDir
        logger.warning(f"evaluation_config.prediction_dir was None. Set to: {batch_predictionDir}")
    
    base_config_dict = evaluation_config.get_dict()
    if base_config_dict.get("prediction_dir") is None:
        # Fallback: use batch_predictionDir if prediction_dir is missing
        base_config_dict["prediction_dir"] = batch_predictionDir
        logger.warning(f"base_config_dict.prediction_dir was None. Set to: {batch_predictionDir}")
    
    # Ensure fields_toUpdate also has prediction_dir set (redundancy check)
    if fields_toUpdate.get("prediction_dir") is None:
        fields_toUpdate["prediction_dir"] = batch_predictionDir
        logger.warning(f"fields_toUpdate.prediction_dir was None. Set to: {batch_predictionDir}")
    
    # CRITICAL: Ensure split is set in base_config_dict to prevent NoneType errors
    # If split is None in base_config_dict, use the first split from splits_toEvaluate
    if base_config_dict.get("split") is None:
        if len(splits_toEvaluate) > 0:
            base_config_dict["split"] = splits_toEvaluate[0]
            logger.warning(f"base_config_dict.split was None. Set to: {splits_toEvaluate[0]}")
        else:
            # Fallback to default "validation" if splits_toEvaluate is empty (should not happen)
            base_config_dict["split"] = "validation"
            logger.warning(f"base_config_dict.split was None and splits_toEvaluate is empty. Set to default: 'validation'")
    
    # Final validation before creating MultiEvaluationConfig
    if base_config_dict.get("prediction_dir") is None or fields_toUpdate.get("prediction_dir") is None:
        raise ValueError(
            f"Cannot create MultiEvaluationConfig: prediction_dir is None. "
            f"base_config_dict['prediction_dir']={base_config_dict.get('prediction_dir')}, "
            f"fields_toUpdate['prediction_dir']={fields_toUpdate.get('prediction_dir')}, "
            f"batch_predictionDir={batch_predictionDir}"
        )
    
    multiEvaluation_config = MultiEvaluationConfig(
        fields_toIterateOver=fields_toIterateOver,
        values_toIterateOver=None,
        configDict_toInitializeFrom=base_config_dict,
        fields_toUpdate=fields_toUpdate,
    )
    
    # Final validation after creating MultiEvaluationConfig
    if multiEvaluation_config.prediction_dir is None:
        raise ValueError(
            f"MultiEvaluationConfig.prediction_dir is None after creation. "
            f"This will cause errors in getAndMake_specificPredictionDir. "
            f"base_config_dict had prediction_dir={base_config_dict.get('prediction_dir')}, "
            f"fields_toUpdate had prediction_dir={fields_toUpdate.get('prediction_dir')}"
        )

    multiple_configAndScores, cached_datasetReaders = evaluate_multipleConfigs(
        model, tokenizer, cached_datasetReaders, multiEvaluation_config, device
    )
    if is_nodeZero(device):
        groupScores_bySplit = group_by(
            multiple_configAndScores, lambda x: x["config"]["split"]
        )

        if inference_dataset_mixture is not None:
            averageScore_perSplit = map_forDictionaries(
                my_dict=groupScores_bySplit, map_fn=get_averageScores
            )
            # Since the scores to concatenate are of different datasets, the returned dictionary
            # will show the datset for each score
            concatenatedScores_perSplit = map_forDictionaries(
                my_dict=groupScores_bySplit, map_fn=concatenate_scores
            )
            checkpoint_scores = deep_update(
                concatenatedScores_perSplit, averageScore_perSplit
            )
        else:
            checkpoint_scores = map_forDictionaries(
                my_dict=groupScores_bySplit, map_fn=extact_score
            )

        if "validation" in checkpoint_scores:
            score_toSelectCheckpoint = checkpoint_scores["validation"]["average"]
        else:
            score_toSelectCheckpoint = checkpoint_scores["train"]["average"]

        checkpoint_scores["score_to_select_checkpoint"] = score_toSelectCheckpoint
    else:
        checkpoint_scores = None

    return checkpoint_scores, cached_datasetReaders


def train(device, world_size, training_config):

    if is_distributedSetup(training_config.world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12346"
        torch.cuda.set_device(device)
        dist.init_process_group(
            "nccl", rank=device, world_size=training_config.world_size
        )

    set_seeds(training_config.seed)

    # Ensure experiment_dir is set (required for prediction_dir)
    if training_config.experiment_dir is None:
        raise ValueError(
            "experiment_dir is None. This can happen if:\n"
            "1. experiment_dir is not set in config file\n"
            "2. TrainingConfig was created with create_expDir=False\n"
            "3. Resuming from checkpoint without setting experiment_dir\n"
            "Please set experiment_dir explicitly or ensure it's created during TrainingConfig initialization."
        )
    
    prediction_dir = os.path.join(training_config.experiment_dir, "predictions")
    evaluation_config = EvaluationConfig(
        configDict_toInitializeFrom=training_config.get_dict(),
        fields_toUpdate={"prediction_dir": prediction_dir},
    )

    model_config = ModelConfig(configDict_toInitializeFrom=training_config.get_dict())
    model, tokenizer, trainableParameter_regex, _ = load_model(
        model_config, cached_models={}, device=device
    )

    logger.info(
        f"Parameter count: {get_parameterCount(model, trainableParameter_regex)}"
    )
    
    optimizer = construct_optimizer(
        model,
        trainableParameter_regex,
        training_config.optimizer,
        training_config.lr,
        training_config.weight_decay,
    )

    scheduler = None
    if training_config.scheduler is not None:
        scheduler = construct_scheduler(
            optimizer,
            training_config.scheduler,
            training_config.num_batches,
            training_config.warmup_ratio,
        )

    if training_config.resume_checkpoint_filepath is not None:
        (
            model,
            optimizer,
            scheduler,
            batchIdx_toResumeFrom,
            current_bestScore,
        ) = load_checkpointToResumeFrom(
            training_config.resume_checkpoint_filepath, model, optimizer, scheduler
        )

    else:
        batchIdx_toResumeFrom = 0
        current_bestScore = 0

    if is_distributedSetup(training_config.world_size):
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    
    # Compile model for faster training (PyTorch 2.0+)
    # Must be done AFTER DDP wrapping (DDP + compile works together)
    # This can provide 20-30% speedup on modern GPUs
    model_compiled = False
    try:
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            # Clear torch inductor cache to avoid C++ compilation errors from corrupted cache
            # This fixes issues like 'zuf0 was not declared in this scope' errors
            import shutil
            import glob
            cache_patterns = [
                "/tmp/torchinductor_*",
                os.path.expanduser("~/.cache/torch/inductor_cache"),
            ]
            cache_cleared = False
            for pattern in cache_patterns:
                try:
                    for cache_dir in glob.glob(pattern):
                        if os.path.isdir(cache_dir):
                            shutil.rmtree(cache_dir)
                            logger.info(f"Cleared torch inductor cache: {cache_dir}")
                            cache_cleared = True
                except Exception as cache_err:
                    logger.debug(f"Could not clear cache {pattern}: {cache_err}")
            
            if cache_cleared:
                logger.info("Torch inductor cache cleared to prevent C++ compilation errors")
            
            # Configure torch.compile to capture scalar outputs (fixes graph breaks)
            # This prevents graph breaks from operations like loss.item() or loss.detach().cpu().item()
            if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
                torch._dynamo.config.capture_scalar_outputs = True
                # Force static shapes by default to avoid dynamic shape compilation bugs
                if hasattr(torch._dynamo.config, 'assume_static_by_default'):
                    torch._dynamo.config.assume_static_by_default = True
                logger.info("Configured torch._dynamo to capture scalar outputs and assume static shapes (prevents graph breaks)")
            
            # CRITICAL FIX: Use 'aot_eager' backend instead of 'inductor' to avoid C++ compilation errors
            # Problem: torch.compile(backend="inductor") succeeds at compile time but fails during first forward pass
            # with C++ compilation errors like 'zuf0 was not declared in this scope'. This is a known bug in
            # PyTorch Inductor C++ codegen. The 'aot_eager' backend compiles the model but doesn't use the
            # problematic C++ codegen, making it stable while still providing speedup over uncompiled model.
            logger.info("Compiling model with torch.compile using aot_eager backend (stable, no C++ codegen)...")
            
            try:
                model = torch.compile(
                    model,
                    fullgraph=False,  # Allow graph breaks for problematic operations
                    dynamic=False,  # Force static shapes to avoid compilation errors
                    backend="aot_eager",  # Use aot_eager backend - no C++ codegen, stable and reliable
                )
                model_compiled = True
                logger.info("Model compiled successfully with aot_eager backend (stable, no C++ codegen)")
            except Exception as compile_err:
                # If even aot_eager fails, continue without compilation
                logger.warning(f"Could not compile model with aot_eager backend: {compile_err}")
                logger.warning("Continuing without torch.compile. Model will run uncompiled (slower but stable).")
                model_compiled = False
    except Exception as e:
        logger.warning(f"Could not compile model (general error): {e}")
        # If compilation fails completely, continue without compilation - this is the safest fallback
        # The model will run without compilation, which is slower but guaranteed to work
        logger.warning("Continuing without torch.compile. Model will run uncompiled (slower but stable).")
        model_compiled = False

    dataset_kwargs = {
        "few_shot_random_seed": evaluation_config.few_shot_random_seed,
        "num_val_samples": evaluation_config.num_val_samples,
        "max_datapoints_per_dataset_without_templates": training_config.max_datapoints_per_dataset_without_templates,
    }
    if training_config.train_dataset_mixture is not None:

        dataset_reader, cached_datasetReaders = get_datasetMixtureReader(
            training_config.train_dataset_mixture,
            training_config.max_datapoints_per_dataset,
            dataset_kwargs,
        )

    else:
        dataset_reader = get_datasetReader(
            training_config.train_dataset, dataset_kwargs
        )
        cached_datasetReaders = {training_config.train_dataset: dataset_reader}

    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, device)
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=training_config.train_batch_size,
        eval_batchSize=evaluation_config.eval_batch_size,
        world_size=evaluation_config.world_size,
        device=device,
        tokenizer=tokenizer,  # Pass tokenizer for pre-tokenization cache
        max_seq_len=training_config.max_seq_len,  # Pass max_seq_len for tokenization
        use_tokenization_cache=True,  # Enable tokenization cache to eliminate bottleneck
    )

    train_iterator = batcher.get_trainBatches(
        "train", training_config.train_template_idx
    )

    if is_nodeZero(device):
        checkpointer = Checkpointer(
            trainableParameter_regex,
            training_config.experiment_dir,
            training_config.should_save_most_recent_state,
            training_config.should_save_every_checkpoint,
            training_config.world_size,
            training_config.should_save_to_gcp,
            training_config.gradient_accumulation_factor,
            current_bestScore,
        )

    if training_config.should_eval_at_beginning:
        logger.info(f"Evaluating before training")

        checkpoint_scores, cached_datasetReaders = evaluate_checkpoint(
            model,
            tokenizer,
            cached_datasetReaders,
            evaluation_config,
            inference_dataset_mixture=training_config.inference_dataset_mixture,
            batch_idx=0,
            should_evalTrain=training_config.should_eval_train,
            should_evalValidation=training_config.should_eval_validation,
            device=device,
            training_config=training_config,
        )

        if is_nodeZero(device):
            checkpointer.checkpoint(
                model, optimizer, scheduler, checkpoint_scores, 0, dont_saveModel=True
            )

    if training_config.use_bfloat16_during_training:
        # Use new API to avoid FutureWarning
        scaler = torch.amp.GradScaler('cuda', enabled=True)

    for i in tqdm(
        range(
            training_config.num_batches * training_config.gradient_accumulation_factor
        )
    ):
        batch_idx = i // (training_config.gradient_accumulation_factor)
        set_seeds(training_config.seed + batch_idx)

        if batch_idx <= batchIdx_toResumeFrom:
            continue

        model.train()

        train_batch = next(train_iterator)
        
        # Move batch to device (moved here from collate_fn to support multiprocessing with CUDA)
        # This allows using multiprocessing workers even with CUDA
        if device is not None:
            device_batch = {}
            for k, v in train_batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    # Handle nested lists of tensors (e.g., all_choices_ids, all_choices_mask)
                    device_batch[k] = [t.to(device) for t in v]
                else:
                    device_batch[k] = v
            train_batch = device_batch

        # Handle potential compilation errors during first forward pass
        # If compilation fails at runtime, disable it and use uncompiled model
        # Note: With aot_eager backend, these errors should be very rare
        if training_config.use_bfloat16_during_training:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                try:
                    loss, current_metrics = model(train_batch)
                except (RuntimeError, Exception) as e:
                    # Check if this is a compilation error from torch.compile (should be rare with aot_eager backend)
                    # This fallback handles any unexpected compilation errors during forward pass
                    if model_compiled and ("CppCompileError" in str(e) or "InductorError" in str(e) or 
                                         "InductorError" in str(type(e).__name__) or 
                                         "was not declared in this scope" in str(e) or
                                         "compilation" in str(e).lower()):
                        logger.error(f"Compilation error during forward pass: {e}")
                        logger.warning("Disabling torch.compile due to compilation error. Continuing with uncompiled model.")
                        # Unwrap compiled model - get the original model back
                        if hasattr(model, '_orig_mod'):
                            model = model._orig_mod
                        elif hasattr(model, 'module'):  # If wrapped in DDP
                            if hasattr(model.module, '_orig_mod'):
                                model.module = model.module._orig_mod
                        model_compiled = False
                        # Retry with uncompiled model
                        loss, current_metrics = model(train_batch)
                    else:
                        raise  # Re-raise if it's a different error
                loss = loss / training_config.gradient_accumulation_factor
            scaler.scale(loss).backward()
        else:
            try:
                loss, current_metrics = model(train_batch)
            except (RuntimeError, Exception) as e:
                # Check if this is a compilation error from torch.compile (should be rare with aot_eager backend)
                # This fallback handles any unexpected compilation errors during forward pass
                if model_compiled and ("CppCompileError" in str(e) or "InductorError" in str(e) or 
                                     "InductorError" in str(type(e).__name__) or 
                                     "was not declared in this scope" in str(e) or
                                     "compilation" in str(e).lower()):
                    logger.error(f"Compilation error during forward pass: {e}")
                    logger.warning("Disabling torch.compile due to compilation error. Continuing with uncompiled model.")
                    # Unwrap compiled model - get the original model back
                    if hasattr(model, '_orig_mod'):
                        model = model._orig_mod
                    elif hasattr(model, 'module'):  # If wrapped in DDP
                        if hasattr(model.module, '_orig_mod'):
                            model.module = model.module._orig_mod
                    model_compiled = False
                    # Retry with uncompiled model
                    loss, current_metrics = model(train_batch)
                else:
                    raise  # Re-raise if it's a different error
            loss = loss / training_config.gradient_accumulation_factor
            loss.backward()

        if is_distributedSetup(training_config.world_size):
            gathered_currentMetrics = [{}] * training_config.world_size
            dist.gather_object(
                current_metrics,
                gathered_currentMetrics if is_nodeZero(device) else None,
                dst=0,
            )

            if is_nodeZero(device):
                current_metrics = reduce_gatheredOutput(
                    gathered_currentMetrics, get_average
                )

        if is_nodeZero(device):
            checkpointer.update_runningSumOfMetrics(current_metrics)

        if (i + 1) % training_config.gradient_accumulation_factor == 0:
            # Clip norm of gradient
            if training_config.norm_to_clip_gradient is not None:
                # Unscale gradient if using bfloat16 so clipping can be correct magnitude
                if training_config.use_bfloat16_during_training:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.norm_to_clip_gradient
                )

            # Take a gradient step
            if training_config.use_bfloat16_during_training:
                if training_config.scheduler is None:
                    scaler.step(optimizer)
                else:
                    scaler.step(scheduler)
                scaler.update()
            else:
                optimizer.step()
                if training_config.scheduler is not None:
                    scheduler.step()

            # Reset optimizer
            optimizer.zero_grad()

            if (batch_idx + 1) % training_config.checkpoint_frequency == 0:
                checkpoint_scores, cached_datasetReaders = evaluate_checkpoint(
                    model,
                    tokenizer,
                    cached_datasetReaders,
                    evaluation_config,
                    inference_dataset_mixture=training_config.inference_dataset_mixture,
                    batch_idx=batch_idx,
                    should_evalTrain=training_config.should_eval_train,
                    should_evalValidation=training_config.should_eval_validation,
                    device=device,
                    training_config=training_config,
                )

                if is_nodeZero(device):
                    (
                        current_log,
                        numCheckpoints_sinceBestCheckpoint,
                    ) = checkpointer.checkpoint(
                        model, optimizer, scheduler, checkpoint_scores, batch_idx
                    )

                    logger.info(f"Finished {batch_idx} batches with log {current_log}")
                    if training_config.early_stopping:
                        # Early stopping mechanism with patience:
                        # - numCheckpoints_sinceBestCheckpoint is incremented at every checkpoint evaluation
                        # - It is reset to 0 whenever a new best model is found
                        # - If it reaches patience threshold (default: 5), training stops
                        # - This means: if no improvement for 5 consecutive checkpoints, stop training
                        if (
                            numCheckpoints_sinceBestCheckpoint
                            >= training_config.early_stopping_num_checkpoints_without_improvement
                        ):
                            # Early stopping triggered - model was already saved as best_model.pt
                            best_model_path = os.path.join(training_config.experiment_dir, "best_model.pt")
                            abs_best_model_path = os.path.abspath(best_model_path)
                            print(f"\n{'='*80}")
                            print(f"EARLY STOPPING TRIGGERED!")
                            print(f"  Patience threshold ({training_config.early_stopping_num_checkpoints_without_improvement} checkpoints) reached.")
                            print(f"  Best model saved at: {abs_best_model_path}")
                            print(f"  Experiment directory: {os.path.abspath(training_config.experiment_dir)}")
                            print(f"{'='*80}\n")
                            logger.info(f"Early stopping triggered. Best model saved at: {abs_best_model_path}")
                            if is_distributedSetup(training_config.world_size):
                                dist.destroy_process_group()
                            return

    if is_distributedSetup(training_config.world_size):
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_filepaths", action="store", type=str, nargs="*", required=True
    )
    parser.add_argument("-d", "--debug_mode", action="store_true")
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting training")

    training_config = TrainingConfig(args.config_filepaths, args.kwargs)

    if training_config.world_size is not None:
        torch_mp.spawn(
            train,
            args=(training_config.world_size, training_config),
            nprocs=training_config.world_size,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train(device, None, training_config)
