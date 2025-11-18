import os
import logging
import copy
import random
import datasets
import json
import pickle
import hashlib
import shutil
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from promptsource.templates import DatasetTemplates, Template
from datasets import load_dataset

logger = logging.getLogger("root")
# datasets.disable_progress_bar()


def _process_chunk_worker(args):
    """
    Global worker function for multiprocessing template application.
    This function must be at module level to be pickleable.
    
    Args:
        args: Tuple of (chunk_datapoints, start_idx, template_idx, num_templates, 
                       is_evaluation, all_templates)
              
    Returns:
        List of processed datapoints
    """
    import copy
    import random
    
    chunk_datapoints, start_idx, template_idx, num_templates, is_evaluation, all_templates = args
    
    chunk_results = []
    for local_idx, datapoint in enumerate(chunk_datapoints):
        datapoint_idx = start_idx + local_idx
        
        # Determine which template to use
        if template_idx >= 0:
            templateIdx_forDatapoint = template_idx
        elif template_idx == -1:
            templateIdx_forDatapoint = datapoint_idx % num_templates
        elif template_idx == -3:
            templateIdx_forDatapoint = random.randint(0, len(all_templates) - 1)
        else:
            raise ValueError(f"Invalid template index {template_idx}")
        
        template = all_templates[templateIdx_forDatapoint]
        new_datapoint = copy.deepcopy(datapoint)
        
        # For evaluation, add answer_choices if they exist
        if is_evaluation:
            answer_choices = template.get_answer_choices_list(datapoint)
            if answer_choices is not None:
                new_datapoint["answer_choices"] = answer_choices
        
        # Apply template to datapoint
        input_txt, target_txt = template.apply(datapoint)
        new_datapoint["input"] = input_txt
        
        # Add target (correct answer)
        if not is_evaluation or "answer_choices" not in new_datapoint:
            new_datapoint["target"] = target_txt
        
        chunk_results.append(new_datapoint)
    
    return chunk_results


class DatasetReader(object):
    """
    DatasetReader objects reads dataset and has all attributes specific to dataset
    
    Features:
    - Disk cache for processed templates to avoid reprocessing on each run
    - Multiprocessing for parallel template application
    - Automatic cache invalidation based on dataset/template hash
    """

    def __init__(self, dataset_stash, template_stash, cache_dir=None, num_workers=None, **kwargs):
        """
        Initialize DatasetReader with optional disk cache and multiprocessing support.
        
        Args:
            dataset_stash: Tuple identifying the dataset (e.g., ("paws", "labeled_final"))
            template_stash: Tuple identifying templates (e.g., ("paws", "labeled_final"))
            cache_dir: Directory for disk cache. If None, uses .cache/templates/ in project root
            num_workers: Number of parallel workers for template processing. If None, uses CPU count
            **kwargs: Additional keyword arguments (extracted from dataset_kwargs by subclasses)
        """
        self.dataset_stash = dataset_stash
        self.template_stash = template_stash

        # Extract cache_dir and num_workers from kwargs if not provided directly
        # This allows subclasses to pass them via dataset_kwargs
        if cache_dir is None:
            cache_dir = kwargs.pop('cache_dir', None)
        if num_workers is None:
            num_workers = kwargs.pop('num_workers', None)

        self.all_templates = self._get_datasetTemplates(None, None)

        self.cached_origData = {}
        self.cached_datasets = {}
        
        # Setup cache directory
        if cache_dir is None:
            # Default to .cache/templates/ in project root
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / ".cache" / "templates"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup multiprocessing workers
        # Limit to 16 workers max (aligned with SLURM --cpus-per-task=32)
        # This prevents excessive workers on cluster nodes with many CPUs
        if num_workers is None:
            num_workers = min(max(1, cpu_count() - 1), 16)  # Max 16 workers, leave CPUs free for main process
        self.num_workers = num_workers
        
        logger.info(f"DatasetReader initialized with cache_dir={self.cache_dir}, num_workers={self.num_workers}")

    def _get_origData(self, split):
        """
        Reads the original dataset split from huggingface. Converts the label to an int and returns the updated dataset.
        Args:
            split:

        Returns:

        """

        if self.few_shot_random_seed is not None:
            return self._read_few_shot_dataset(split, self.few_shot_random_seed)
        else:
            return self._read_origin_dataset(split)

    def _read_origin_dataset(self, split):
        """
        Reads the original dataset split from huggingface. Converts the label to an int and returns the updated dataset.
        Args:
            split:

        Returns:

        """
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            logger.info(f"\t\tLoading Full Data for {self.name}")
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )
            orig_data = []
            # converting label to int and caching the split of the dataset.
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                total_validation_samples = len(orig_data)
                logger.info(
                    f"[DATASET SPLIT INFO] {self.name.upper()}: "
                    f"Loaded {total_validation_samples} samples from HuggingFace validation split"
                )
                
                if len(orig_data) > self.num_val_samples:
                    orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )
                    logger.info(
                        f"[DATASET SPLIT INFO] {self.name.upper()}: "
                        f"Split into validation ({len(orig_val_data)} samples, first {self.num_val_samples}) "
                        f"and test ({len(orig_test_data)} samples, remaining {total_validation_samples - self.num_val_samples})"
                    )
                else:
                    print(f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}")
                    print("splitting equally")
                    num_val = len(orig_data)//2
                    orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                        orig_data, num_val
                    )
                    logger.info(
                        f"[DATASET SPLIT INFO] {self.name.upper()}: "
                        f"Split equally: validation ({len(orig_val_data)} samples) "
                        f"and test ({len(orig_test_data)} samples)"
                    )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        # Log which split is being returned and how many samples it contains
        returned_split = self.cached_origData[split]
        logger.info(
            f"[DATASET USAGE INFO] {self.name.upper()}: "
            f"Using split '{split}' with {len(returned_split)} samples for evaluation"
        )
        
        return returned_split

    def _read_few_shot_dataset(
        self,
        split,
        few_shot_random_seed,
    ):
        if split not in self.cached_origData:
            logger.info(
                f"\t\tLoading Few Shot Data for {self.name} with seed {few_shot_random_seed}"
            )
            file_path = os.path.join(
                "data",
                "few_shot",
                self.name,
                f"{few_shot_random_seed}_seed.jsonl",
            )
            if os.path.exists(file_path):
                with open(file_path, "r") as fin:
                    data = []
                    for idx, line in enumerate(fin.readlines()):
                        example = json.loads(line.strip("\n"))
                        example["lbl"] = int(example["label"])
                        data.append(example)
                    self.cached_origData[split] = data
            else:
                raise ValueError(f"Few shot dataset not found at {file_path}")

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        """
        Returns a list of all templates for the dataset with the given metrics and not in the list of templates to ignore.
        Args:
            templateNames_toIgnore:
            metrics_toUse: specify the metric to use so that we only include templates which
                           match the metric we want to use

        Returns:

        """
        all_templates = []

        # Get original templates from promptsource
        for template in DatasetTemplates(*self.template_stash).templates.values():
            # Filter out templates that
            # 1) are not designed for original task
            # 2) have different metrics than we want to use
            # 3) are ones that we want to ignore based on the name
            if template.metadata.original_task:
                should_ignoreTemplate = False

                for metric in template.metadata.metrics:
                    if metric not in metrics_toUse:
                        should_ignoreTemplate = True

                for template_name in templateNames_toIgnore:
                    if template.name == template_name:
                        should_ignoreTemplate = True

                if not should_ignoreTemplate:
                    all_templates.append(template)

        return all_templates

    def _process_single_datapoint(
        self, datapoint, datapoint_idx, template_idx, num_templates, is_evaluation
    ):
        """
        Process a single datapoint with template application.
        This is the core logic extracted for multiprocessing.
        
        Args:
            datapoint: Original datapoint dictionary
            datapoint_idx: Index of datapoint in dataset
            template_idx: Template index to use (or special value -1, -3)
            num_templates: Total number of templates
            is_evaluation: Whether this is for evaluation
            
        Returns:
            Processed datapoint dictionary
        """
        # Determine which template to use for this datapoint
        if template_idx >= 0:
            templateIdx_forDatapoint = template_idx
        elif template_idx == -1:
            templateIdx_forDatapoint = datapoint_idx % num_templates
        elif template_idx == -3:
            templateIdx_forDatapoint = random.randint(0, len(self.all_templates) - 1)
        else:
            raise ValueError(f"Invalid template index {template_idx}")
        
        template = self.all_templates[templateIdx_forDatapoint]
        new_datapoint = copy.deepcopy(datapoint)
        
        # For evaluation, add answer_choices if they exist
        if is_evaluation:
            answer_choices = template.get_answer_choices_list(datapoint)
            if answer_choices is not None:
                new_datapoint["answer_choices"] = answer_choices
        
        # Apply template to datapoint
        input_txt, target_txt = template.apply(datapoint)
        new_datapoint["input"] = input_txt
        
        # Add target (correct answer)
        if not is_evaluation or "answer_choices" not in new_datapoint:
            new_datapoint["target"] = target_txt
        
        return new_datapoint
    
    def _applyTemplate_toData(
        self, orig_data, num_templates, template_idx, is_evaluation
    ):
        """
        Apply templates to dataset with multiprocessing support.
        
        Uses multiprocessing Pool to parallelize template application across datapoints.
        For small datasets (<1000 examples), falls back to sequential processing.
        
        Args:
            orig_data: List of original datapoints (before template application)
            num_templates: Total number of available templates
            template_idx: Template index to use (>=0 for fixed, -1 for cycling, -3 for random)
            is_evaluation: Whether this is for evaluation (affects answer_choices)
            
        Returns:
            List of processed datapoints with 'input' and 'target' fields
        """
        num_examples = len(orig_data)
        
        # For small datasets, use sequential processing (overhead not worth it)
        if num_examples < 1000 or self.num_workers == 1:
            logger.debug(f"Using sequential processing for {num_examples} examples")
            dataset = []
            for datapoint_idx, datapoint in enumerate(orig_data):
                processed = self._process_single_datapoint(
                    datapoint, datapoint_idx, template_idx, num_templates, is_evaluation
                )
                dataset.append(processed)
            return dataset
        
        # Use multiprocessing for larger datasets
        logger.info(f"Applying templates with multiprocessing ({self.num_workers} workers) for {num_examples} examples")
        
        # Split data into chunks for workers
        chunk_size = max(1, num_examples // (self.num_workers * 4))  # 4 chunks per worker
        chunks = []
        for i in range(0, num_examples, chunk_size):
            chunk = orig_data[i:i + chunk_size]
            # Prepare arguments: (chunk_datapoints, start_idx, template_idx, num_templates, is_evaluation, all_templates)
            chunks.append((chunk, i, template_idx, num_templates, is_evaluation, self.all_templates))
        
        # Process chunks in parallel using global worker function
        with Pool(processes=self.num_workers) as pool:
            chunk_results = pool.map(_process_chunk_worker, chunks)
        
        # Flatten results
        dataset = []
        for chunk_result in chunk_results:
            dataset.extend(chunk_result)
        
        logger.info(f"Template application completed: {len(dataset)} examples processed")
        return dataset

    def _split_val_into_val_and_test(self, orig_data, num_val_samples=32, seed=42):
        """
        Splits the validation set into validation and test set. This is done by taking the first 1000 examples
        as the test set and the rest as the validation set.
        Args:
            orig_data:
            seed:

        Returns:

        """
        random.seed(seed)
        random.shuffle(orig_data)
        val_data = orig_data[:num_val_samples]
        test_data = orig_data[num_val_samples:]
        return val_data, test_data

    def _get_cache_path(self, split, template_idx, is_evaluation):
        """
        Generate cache file path for processed dataset.
        
        Args:
            split: Dataset split (train/validation/test)
            template_idx: Template index used
            is_evaluation: Whether this is for evaluation
            
        Returns:
            Path object for cache file
        """
        # Create cache filename based on dataset name, split, template_idx, and evaluation flag
        eval_suffix = "_eval" if is_evaluation else "_train"
        cache_filename = f"{self.name}_{split}_template{template_idx}{eval_suffix}.pkl"
        return self.cache_dir / cache_filename
    
    def _compute_dataset_hash(self, orig_data):
        """
        Compute hash of dataset for cache invalidation.
        Uses first/last examples and total count to detect changes.
        
        Args:
            orig_data: List of datapoints
            
        Returns:
            Hash string for cache validation
        """
        if not orig_data:
            return "empty"
        
        # Create hash from dataset characteristics
        hash_data = {
            "count": len(orig_data),
            "first_idx": orig_data[0].get("idx", 0) if orig_data else 0,
            "last_idx": orig_data[-1].get("idx", len(orig_data) - 1) if orig_data else 0,
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_path):
        """
        Load processed dataset from disk cache.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Cached dataset if valid, None otherwise
        """
        if not cache_path.exists():
            return None
        
        try:
            logger.info(f"Loading dataset from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            logger.info(f"Cache loaded successfully: {len(cached_data.get('dataset', []))} examples")
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path, dataset, dataset_hash):
        """
        Save processed dataset to disk cache.
        
        Args:
            cache_path: Path to cache file
            dataset: Processed dataset to cache
            dataset_hash: Hash of original data for validation
        """
        try:
            cache_data = {
                "dataset": dataset,
                "dataset_hash": dataset_hash,
                "dataset_name": self.name,
            }
            logger.info(f"Saving dataset to cache: {cache_path} ({len(dataset)} examples)")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")
    
    def get_dataset(
        self, split, template_idx, is_evaluation, max_samples_per_dataset=None
    ):
        """
        Create dataset that includes the template with disk cache and multiprocessing support.

        Args:
            split: Dataset split (train/validation/test)
            template_idx:
                if >=0, then we use the fixed template_idx across entire dataset
                if ==-1, then we use all template across entire the dataset, where different
                         datapoints can have different templates. A datapoint will always be
                         mapped to the same template though
                if ==-2, then we take the cross product of all templates and all datapoints.
                if ==-3, apply a random template to each datapoint.
            is_evaluation: whether the split is for evaluation (where it will have answer_choices)
                            or for training (where it will only have the target)
            max_samples_per_dataset: Maximum number of samples to return (after template application)
            
        Returns:
            dataset: List of processed datapoints with 'input' and 'target' fields
        """
        # Check in-memory cache first
        if (split, template_idx) not in self.cached_datasets:
            # Get num_templates early - needed for logging regardless of cache hit/miss
            num_templates = self.get_numTemplates()
            
            # Get original data
            orig_data = self._get_origData(split)
            total_examples = len(orig_data)
            orig_data = (
                orig_data[: self.max_datapoints_per_dataset_without_templates]
                if self.max_datapoints_per_dataset_without_templates
                and split.lower() == "train"
                else orig_data
            )
            logger.info(
                f"\tDataset:{self.name.upper()}\tSplit:{split}\tSelected Examples: {len(orig_data)}\tNum Total Example:{total_examples}"
            )
            logger.info(
                f"[EVALUATION INFO] {self.name.upper()}: "
                f"Will evaluate on {len(orig_data)} samples from split '{split}' "
                f"(out of {total_examples} total available for this split)"
            )
            
            # Compute hash for cache validation
            dataset_hash = self._compute_dataset_hash(orig_data)
            
            # Check disk cache
            cache_path = self._get_cache_path(split, template_idx, is_evaluation)
            cached_data = self._load_from_cache(cache_path)
            
            if cached_data is not None and cached_data.get("dataset_hash") == dataset_hash:
                # Cache hit - use cached dataset
                logger.info(f"Using cached dataset from disk (hash match)")
                dataset = cached_data["dataset"]
            else:
                # Cache miss or invalid - process from scratch
                logger.info(f"Processing dataset from scratch (cache miss or invalid)")

                # template_idx -2 means we do a cross product of each datapoint with each template
                if template_idx == -2:
                    dataset = []
                    for iterate_templateIdx in range(num_templates):
                        dataset.extend(
                            self._applyTemplate_toData(
                                orig_data, num_templates, iterate_templateIdx, is_evaluation
                            )
                        )
                # otherwise apply template to dataset
                else:
                    dataset = self._applyTemplate_toData(
                        orig_data, num_templates, template_idx, is_evaluation
                    )
                
                # Save to disk cache for future use
                self._save_to_cache(cache_path, dataset, dataset_hash)
            
            # Shuffle examples and select max_samples
            random.Random(4).shuffle(dataset)
            total_examples_with_templates = len(dataset)
            dataset = (
                dataset[:max_samples_per_dataset]
                if max_samples_per_dataset
                else dataset
            )
            logger.info(
                f"\tDataset:{self.name.upper()}\tSplit:{split}\tNum Selected Example with Templates:{len(dataset)}\tTemplate Idx:{template_idx}\tNum Templates:{num_templates}\tNum Examples with Template:{total_examples_with_templates}"
            )

            # Store in memory cache
            self.cached_datasets[(split, template_idx)] = dataset

        return self.cached_datasets[(split, template_idx)]

    def get_numTemplates(self):
        return len(self.all_templates)

    def get_metricsForDataset(self):
        return self.all_templates[0].metadata.metrics


class RTEReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "rte"), template_stash=("super_glue", "rte"),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "rte"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class HSwagReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("hellaswag",), template_stash=("hellaswag",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "h-swag"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        all_templates = super()._get_datasetTemplates(
            ["Randomized prompts template"], ["Accuracy"]
        )

        # Add each template from the several templates in the randomized prompt individually
        listOf_randomJinjas = [
            (
                "randomized prompt 1",
                "Can you pick the correct ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 2",
                "The task is to generate the ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 3",
                "How does this sentence end? {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 4",
                "From the list of endings described below, what ending makes the most sense for the sentence {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
        ]

        for name, jinja in listOf_randomJinjas:
            all_templates.append(
                Template(
                    name=name,
                    jinja=jinja,
                    reference="",
                    answer_choices='{{endings | join("|||")}}',
                )
            )

        return all_templates


class COPAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "copa"), template_stash=("super_glue", "copa"),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "copa"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates(
            [
                "…which may be caused by",
                "…What could happen next, C1 or C2?",
                "…As a result, C1 or C2?",
                "…why? C1 or C2",
            ],
            ["Accuracy"],
        )


class WiCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "wic"), template_stash=("super_glue", "wic"),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "wic"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class WinograndeReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("winogrande", "winogrande_debiased"),
            template_stash=("winogrande", "winogrande_debiased"),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "winogrande"

    def _clean_winogrande_cache(self, cache_base_dir=None):
        """
        Clean corrupted or conflicting winogrande cache files and directories.
        Recursively searches all subdirectories and removes winogrande-related cache.
        
        Args:
            cache_base_dir: Base directory for HuggingFace datasets cache.
                          If None, uses HF_HOME environment variable or default location.
        """
        if cache_base_dir is None:
            # Get cache directory from environment or use default
            # Check HF_HOME first (set in training.py)
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_base_dir = os.path.join(hf_home, "datasets")
            else:
                # Fallback to default HuggingFace cache location
                cache_base_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        
        if not os.path.exists(cache_base_dir):
            logger.debug(f"Cache directory does not exist: {cache_base_dir}")
            return
        
        logger.info(f"Cleaning winogrande cache from: {cache_base_dir}")
        removed_count = 0
        
        # Pattern 1: Remove entire parquet folders matching winogrande_debiased-* pattern
        parquet_dir = os.path.join(cache_base_dir, "parquet")
        if os.path.exists(parquet_dir):
            winogrande_patterns = [
                os.path.join(parquet_dir, "winogrande_debiased-*"),
                os.path.join(parquet_dir, "winogrande_xl-*"),
                os.path.join(parquet_dir, "winogrande_*-*"),  # Catch all winogrande variants
            ]
            for pattern in winogrande_patterns:
                for cache_dir in glob.glob(pattern):
                    if os.path.isdir(cache_dir):
                        try:
                            shutil.rmtree(cache_dir)
                            logger.info(f"Removed corrupted cache directory: {cache_dir}")
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to remove cache directory {cache_dir}: {e}")
        
        # Pattern 2: Remove lock files with winogrande in name (recursive search)
        # These are in the root datasets directory with long encoded paths
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
                        logger.info(f"Removed corrupted cache file: {lock_file}")
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {lock_file}: {e}")
        
        # Pattern 3: Check and remove old files from downloads/ folder
        downloads_dir = os.path.join(cache_base_dir, "downloads")
        if os.path.exists(downloads_dir):
            # Recursively search for winogrande-related files in downloads
            for root, dirs, files in os.walk(downloads_dir):
                for file in files:
                    if "winogrande" in file.lower():
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            logger.info(f"Removed old download file: {file_path}")
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to remove download file {file_path}: {e}")
        
        logger.info(f"Cache cleanup completed. Removed {removed_count} items.")

    def _read_origin_dataset(self, split):
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            # Clean corrupted winogrande cache before loading dataset
            # This prevents NonMatchingSplitsSizesError from conflicting cache entries
            if split == "train":  # Only clean once, when loading train split first
                self._clean_winogrande_cache()
            
            # Use direct parquet file loading to avoid configuration conflicts
            # This approach bypasses the dataset loading script that causes
            # NonMatchingSplitsSizesError when mixing different winogrande configurations
            # Similar to PAWSReader approach - loads parquet files directly from HuggingFace
            data_files = {
                'train': 'https://huggingface.co/datasets/winogrande/resolve/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande_debiased/train-00000-of-00001.parquet',
                'validation': 'https://huggingface.co/datasets/winogrande/resolve/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande_debiased/validation-00000-of-00001.parquet',
                'test': 'https://huggingface.co/datasets/winogrande/resolve/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande_debiased/test-00000-of-00001.parquet'
            }
            
            # Load dataset directly from parquet files, bypassing the problematic loading script
            huggingFace_data = load_dataset(
                'parquet',
                data_files=data_files,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                # Column 'answer' is a string "1" or "2", need to parse it
                example["lbl"] = int(example["answer"]) - 1
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class CBReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("super_glue", "cb"), template_stash=("super_glue", "cb"),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "cb"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class StoryClozeReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("story_cloze", "2016"),
            template_stash=("story_cloze", "2016"),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "story_cloze"

    def _read_origin_dataset(self, split):

        # We use the test set of StoryCloze for validation and the validation set of StoryCloze
        # for train - following GPT3
        if split == "train":
            load_split = "validation"
        elif split in ["validation", "validation_full", "test"]:
            load_split = "test"

        if split not in self.cached_origData:
            # Do not use default method for loading dataset since the story_cloze dataset must be
            # downloaded manually and then we have to set data_dir to point to it.
            # The HuggingFace datasets library expects files with specific names:
            # - "cloze_test_val__spring2016 - cloze_test_ALL_val.csv" (for validation split)
            # - "cloze_test_test__spring2016 - cloze_test_ALL_test.csv" (for test split)
            # If you have files with different names (e.g., cloze_testval_spring2016.csv),
            # create symbolic links with the expected names pointing to your actual files.
            story_cloze_data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "datasets",
                "story_cloze"
            )
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
                data_dir=story_cloze_data_dir,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer_right_ending"]) - 1
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR1Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("anli",), template_stash=("anli",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "anli-r1"

    def _read_origin_dataset(self, split):

        load_split = "dev" if "validation" in split.lower() else split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{load_split}_r1",
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]
            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR2Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("anli",), template_stash=("anli",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "anli-r2"

    def _read_origin_dataset(self, split):

        load_split = "dev" if "validation" in split.lower() else split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{load_split}_r2",
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR3Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("anli",), template_stash=("anli",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "anli-r3"

    def _read_origin_dataset(self, split):

        load_split = "dev" if "validation" in split.lower() else split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{load_split}_r3",
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class WSCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        # Używamy "wsc" (bez .fixed) dla dataset_stash, bo HuggingFace datasets
        # rozpoznaje tylko "wsc" jako konfigurację super_glue
        # template_stash pozostaje "wsc.fixed" bo promptsource używa tej wersji dla templates
        super().__init__(
            dataset_stash=("super_glue", "wsc"),
            template_stash=("super_glue", "wsc.fixed"),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "wsc"

    def _read_origin_dataset(self, split):
        """
        Ładuje dataset WSC bezpośrednio z plików parquet, omijając problemy
        z cache HuggingFace datasets. Podobne podejście jak w WinograndeReader i PAWSReader.
        """
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            # Używamy bezpośrednich URL do plików parquet dla wsc
            # aby uniknąć problemu z cache HuggingFace datasets który powoduje
            # TypeError: expected str, bytes or os.PathLike object, not NoneType
            data_files = {
                'train': 'https://huggingface.co/datasets/super_glue/resolve/main/wsc/train-00000-of-00001.parquet',
                'validation': 'https://huggingface.co/datasets/super_glue/resolve/main/wsc/validation-00000-of-00001.parquet',
                'test': 'https://huggingface.co/datasets/super_glue/resolve/main/wsc/test-00000-of-00001.parquet'
            }
            
            huggingFace_data = load_dataset(
                'parquet',
                data_files=data_files,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                if len(orig_data) > self.num_val_samples:
                    orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                        orig_data, self.num_val_samples
                    )
                else:
                    print(f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}")
                    print("splitting equally")
                    orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                        orig_data, len(orig_data)//2
                    )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class CosmosQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("cosmos_qa",), template_stash=("cosmos_qa",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "cosmos_qa"

    def _read_origin_dataset(self, split):

        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class SocialIQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("social_i_qa",), template_stash=("social_i_qa",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "social_i_qa"

    def _read_origin_dataset(self, split):

        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"]) - 1
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates(
            ["Check if a random answer is valid or not"], ["Accuracy"]
        )


class PAWSReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("google-research-datasets/paws", "labeled_final"),
            template_stash=("paws", "labeled_final"),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "paws"

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            # Używamy bezpośrednich URL do plików parquet tylko z labeled_final
            # aby uniknąć problemu z konfiguracją która łączy 3 różne pliki (labeled_final, labeled_swap, unlabeled_final)
            # co powodowało 725450 przykładów zamiast oczekiwanych 49401
            data_files = {
                'train': 'https://huggingface.co/datasets/google-research-datasets/paws/resolve/161ece9501cf0a11f3e48bd356eaa82de46d6a09/labeled_final/train-00000-of-00001.parquet',
                'validation': 'https://huggingface.co/datasets/google-research-datasets/paws/resolve/161ece9501cf0a11f3e48bd356eaa82de46d6a09/labeled_final/validation-00000-of-00001.parquet',
                'test': 'https://huggingface.co/datasets/google-research-datasets/paws/resolve/161ece9501cf0a11f3e48bd356eaa82de46d6a09/labeled_final/test-00000-of-00001.parquet'
            }
            huggingFace_data = load_dataset(
                'parquet',
                data_files=data_files,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["label"]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class QuAILReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("quail",), template_stash=("quail",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "quail"

    def _read_origin_dataset(self, split):

        load_split = "challenge" if split == "test" else split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["correct_answer_id"]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class WikiQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("wiki_qa",), template_stash=("wiki_qa",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "wiki_qa"

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class QuaRTzReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("quartz",), template_stash=("quartz",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "quartz"

        self.string_toLabelIdx = {"A": 0, "B": 1}

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class QASCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("qasc",), template_stash=("qasc",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "qasc"

        self.string_toLabelIdx = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
        }

    def _read_origin_dataset(self, split):

        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class ROPESReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):

        super().__init__(
            dataset_stash=("ropes",), template_stash=("ropes",),
            **(dataset_kwargs or {})
        )

        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                if k not in ['cache_dir', 'num_workers']:  # Already handled by parent
                    setattr(self, k, v)

        self.name = "ropes"

    def _read_origin_dataset(self, split):

        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["answers"]["answer_start"] = [0]
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )

                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Squad"])


DATASET_CLASSES = {
    "rte": RTEReader,
    "h-swag": HSwagReader,
    "copa": COPAReader,
    "wic": WiCReader,
    "winogrande": WinograndeReader,
    "cb": CBReader,
    "story_cloze": StoryClozeReader,
    "anli-r1": ANLIR1Reader,
    "anli-r2": ANLIR2Reader,
    "anli-r3": ANLIR3Reader,
    "wsc": WSCReader,
    "cosmos_qa": CosmosQAReader,
    "social_iqa": SocialIQAReader,
    "paws": PAWSReader,
    "quail": QuAILReader,
    "wiki_qa": WikiQAReader,
    "quartz": QuaRTzReader,
    "qasc": QASCReader,
    "ropes": ROPESReader,
}


def get_datasetReader(dataset_name, dataset_kwargs=None):
    """
    Factory function to create DatasetReader instances with optional cache and multiprocessing support.
    
    Args:
        dataset_name: Name of the dataset (must be in DATASET_CLASSES)
        dataset_kwargs: Dictionary of keyword arguments. Can include:
            - cache_dir: Directory for disk cache (optional, defaults to .cache/templates/)
            - num_workers: Number of workers for multiprocessing (optional, defaults to CPU count - 1)
            - Other dataset-specific kwargs (few_shot_random_seed, num_val_samples, etc.)
            
    Returns:
        DatasetReader instance configured with provided kwargs
        
    Example:
        >>> reader = get_datasetReader("paws", {"cache_dir": "/path/to/cache", "num_workers": 8})
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}
    
    # Pass all kwargs through - DatasetReader.__init__ will extract cache_dir and num_workers
    return DATASET_CLASSES[dataset_name](dataset_kwargs)
