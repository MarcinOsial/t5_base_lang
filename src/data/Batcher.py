import torch
import logging
from torch.utils import data
from pathlib import Path

from src.utils.distributed_utils import is_distributedSetup

from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("root")


class Batcher(object):
    """
    Batcher is responsible for returning batches of data.
    
    Supports pre-tokenization with disk caching to eliminate tokenization bottleneck
    in DataLoader workers.
    """

    def __init__(
        self,
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize,
        eval_batchSize,
        world_size,
        device,
        tokenizer=None,
        max_seq_len=None,
        use_tokenization_cache=True,
    ):
        """
        Initialize Batcher.

        Args:
            dataset_reader: DatasetReader instance
            createPytorchDataset_fn: Function to create PyTorch dataset
            train_batchSize: Batch size for training
            eval_batchSize: Batch size for evaluation
            world_size: World size for distributed training
            device: Device for tensors
            tokenizer: HuggingFace tokenizer (optional, for pre-tokenization cache)
            max_seq_len: Maximum sequence length for tokenization (optional)
            use_tokenization_cache: Whether to use tokenization cache (default: True)
        """
        self.dataset_reader = dataset_reader
        self.createPytorchDataset_fn = createPytorchDataset_fn

        self.train_batchSize = train_batchSize
        self.eval_batchSize = eval_batchSize
        self.world_size = world_size
        self.device = device
        self.current_epoch = 0
        
        # Tokenization cache settings
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len if max_seq_len is not None else 512
        self.use_tokenization_cache = use_tokenization_cache and tokenizer is not None
        
        if self.use_tokenization_cache:
            logger.info(f"Tokenization cache enabled (max_seq_len={self.max_seq_len})")
        else:
            logger.info("Tokenization cache disabled - using on-the-fly tokenization")

    def get_metricsForDataset(self):
        return self.dataset_reader.get_metricsForDataset()

    def create_data_loader(self, pytorch_dataset, batch_size, shuffle, num_workers=None):
        """
        Create PyTorch DataLoader with optimized multiprocessing support.
        
        Uses 'spawn' start method for multiprocessing when CUDA is available to avoid
        CUDA re-initialization errors. For CPU-only training, uses default 'fork' method.
        
        Args:
            pytorch_dataset: PyTorch dataset to load
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes for data loading. 
                        If None, uses 16 workers (aligned with SLURM --cpus-per-task=32)
                        
        Returns:
            Tuple of (sampler, data_loader) for distributed setup, or (None, data_loader) otherwise
            
        Optimizations applied:
            - pin_memory=True when CUDA available (faster CPU->GPU transfer)
            - persistent_workers=True when num_workers > 0 (reduces overhead)
            - prefetch_factor=2 when num_workers > 0 (workers prepare batches ahead)
        """
        import torch
        import multiprocessing as mp
        
        # Set num_workers (default to 16 for SLURM cluster with --cpus-per-task=32)
        # Now using numpy arrays in cache (not tensors), so multiprocessing works without mmap issues
        # Leave some CPUs for main process and system
        if num_workers is None:
            num_workers = 16  # Increased back to 16 - numpy arrays work fine with multiprocessing
            logger.info(f"Using num_workers={num_workers} for DataLoader")
        
        # Set multiprocessing start method to 'spawn' if CUDA is available
        # This allows multiprocessing with CUDA (fork doesn't work with CUDA)
        # Note: set_start_method must be called before creating any processes
        if torch.cuda.is_available() and num_workers > 0:
            try:
                # Try to get current start method
                current_method = mp.get_start_method(allow_none=True)
                if current_method is None:
                    # No method set yet - set to 'spawn' for CUDA compatibility
                    mp.set_start_method('spawn')
                    logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
                elif current_method != 'spawn':
                    # Method already set to something else - warn but continue
                    logger.warning(f"Multiprocessing start method is '{current_method}', not 'spawn'. "
                                f"This may cause CUDA issues. Consider setting it to 'spawn' at program start.")
            except RuntimeError as e:
                # Start method already set by another process - that's fine
                logger.debug(f"Could not set start method: {e}")
        
        if is_distributedSetup(self.world_size):
            sampler = DistributedSampler(
                pytorch_dataset,
                num_replicas=self.world_size,
                rank=self.device,
                shuffle=shuffle,
            )

            data_loader = data.DataLoader(
                pytorch_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                sampler=sampler,
                collate_fn=pytorch_dataset.collate_fn,
                pin_memory=torch.cuda.is_available(),  # Faster CPU->GPU transfer
                persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
            )
            return sampler, data_loader
        else:
            data_loader = data.DataLoader(
                pytorch_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                collate_fn=pytorch_dataset.collate_fn,
                pin_memory=torch.cuda.is_available(),  # Faster CPU->GPU transfer
                persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
            )

            return None, data_loader

    def get_trainBatches(self, split, template_idx):
        dataset = self.dataset_reader.get_dataset(
            split, template_idx, is_evaluation=False
        )
        logger.info(f"\tTotal Train Examples along with Templates: {len(dataset)}")
        
        # Pre-tokenize dataset if cache is enabled
        if self.use_tokenization_cache:
            from src.data.tokenization_cache import pre_tokenize_dataset
            dataset_name = self.dataset_reader.name
            tokenized_dataset = pre_tokenize_dataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                dataset_name=dataset_name,
                split=split,
                template_idx=template_idx,
                max_seq_len=self.max_seq_len,
                use_cache=self.use_tokenization_cache,
            )
            dataset = tokenized_dataset
        
        pytorch_dataset = self.createPytorchDataset_fn(dataset)
        sampler, data_loader = self.create_data_loader(
            pytorch_dataset, self.train_batchSize, True
        )

        while True:
            if is_distributedSetup(self.world_size):
                sampler.set_epoch(self.current_epoch)

            for x in data_loader:
                yield x

            self.current_epoch += 1

    def get_splitOfBatches(self, split, template_idx, is_evaluation):
        assert split.lower() in [
            "validation",
            "validation_full",
            "train",
            "test",
        ], f"Evaluation Split {split} not defined"

        dataset = self.dataset_reader.get_dataset(
            split, template_idx, is_evaluation
        )
        
        # Pre-tokenize dataset if cache is enabled
        if self.use_tokenization_cache:
            from src.data.tokenization_cache import pre_tokenize_dataset
            dataset_name = self.dataset_reader.name
            tokenized_dataset = pre_tokenize_dataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                dataset_name=dataset_name,
                split=split,
                template_idx=template_idx,
                max_seq_len=self.max_seq_len,
                use_cache=self.use_tokenization_cache,
            )
            dataset = tokenized_dataset
        
        pytorch_dataset = self.createPytorchDataset_fn(dataset)
        _, data_loader = self.create_data_loader(
            pytorch_dataset, self.eval_batchSize, False
        )

        for x in data_loader:
            yield x

    def get_evalBatches(self, split, template_idx):
        assert split.lower() in [
            "validation",
            "validation_full",
            "train",
            "test",
        ], f"Evaluation Split {split} not defined"

        dataset = self.dataset_reader.get_dataset(
            split, template_idx, is_evaluation=True
        )
        
        # Pre-tokenize dataset if cache is enabled
        if self.use_tokenization_cache:
            from src.data.tokenization_cache import pre_tokenize_dataset
            dataset_name = self.dataset_reader.name
            tokenized_dataset = pre_tokenize_dataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                dataset_name=dataset_name,
                split=split,
                template_idx=template_idx,
                max_seq_len=self.max_seq_len,
                use_cache=self.use_tokenization_cache,
            )
            dataset = tokenized_dataset
        
        pytorch_dataset = self.createPytorchDataset_fn(dataset)
        _, data_loader = self.create_data_loader(
            pytorch_dataset, self.eval_batchSize, False
        )

        for x in data_loader:
            yield x
