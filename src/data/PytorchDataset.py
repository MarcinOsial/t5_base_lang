import torch
import copy
import numpy as np
from torch.utils import data


class PytorchDataset(data.Dataset):
    """
    Pytorch Dataset that returns a dictionary of tensors for each datapoint.
    
    If dataset is pre-tokenized (contains 'input_ids' field), returns it directly.
    Otherwise, tokenizes on-the-fly (legacy mode for backward compatibility).
    """

    def __init__(self, dataset, tokenizer, device, is_pre_tokenized=None):
        """
        Initialize PytorchDataset.
        
        Args:
            dataset: List of datapoints. If pre-tokenized, should contain 'input_ids', 'input_mask', etc.
                    If not pre-tokenized, should contain 'input', 'target'/'answer_choices' (strings)
            tokenizer: HuggingFace tokenizer (used only if not pre-tokenized)
            device: Device for tensors (not used directly, kept for compatibility)
            is_pre_tokenized: Whether dataset is pre-tokenized. If None, auto-detects from first example
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        
        # Auto-detect if dataset is pre-tokenized
        # Pre-tokenized datasets have 'input_ids' field (as numpy array or tensor)
        if is_pre_tokenized is None:
            self.is_pre_tokenized = (
                len(dataset) > 0 
                and "input_ids" in dataset[0]
                and (isinstance(dataset[0]["input_ids"], (np.ndarray, torch.Tensor)))
            )
        else:
            self.is_pre_tokenized = is_pre_tokenized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, get_idx):
        """
        Returns a dictionary tokenized inputs along with masks for each datapoint.
        
        If dataset is pre-tokenized, converts numpy arrays back to PyTorch tensors.
        Otherwise, tokenizes on-the-fly (legacy mode).
        """
        datapoint = self.dataset[get_idx]
        
        # If pre-tokenized, convert numpy arrays back to PyTorch tensors
        if self.is_pre_tokenized:
            new_datapoint = copy.deepcopy(datapoint)
            
            # Migrate old cache format: all_choices_masks -> all_choices_mask
            # This handles cases where old cache wasn't migrated during loading
            if "all_choices_masks" in new_datapoint and "all_choices_mask" not in new_datapoint:
                new_datapoint["all_choices_mask"] = new_datapoint.pop("all_choices_masks")
            
            # Convert numpy arrays to PyTorch tensors
            for key, value in new_datapoint.items():
                if isinstance(value, np.ndarray):
                    new_datapoint[key] = torch.from_numpy(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    # Handle lists of numpy arrays (e.g., all_choices_ids, all_choices_masks)
                    new_datapoint[key] = [torch.from_numpy(arr) for arr in value]
            return new_datapoint
        
        # Legacy mode: tokenize on-the-fly (for backward compatibility)
        input_dict = self.tokenizer(
            datapoint["input"], return_tensors="pt", truncation=True
        )
        input_ids = input_dict["input_ids"][0]
        input_mask = input_dict["attention_mask"][0]

        allChoices_ids = []
        allChoices_masks = []

        new_datapoint = copy.deepcopy(datapoint)

        new_datapoint.update(
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
            }
        )

        if "answer_choices" in datapoint:
            for choice in datapoint["answer_choices"]:
                # This assumes tokenizer does not add BOS token, which is true for T5
                choice_dict = self.tokenizer(
                    choice, return_tensors="pt", truncation=True
                )
                allChoices_ids.append(choice_dict["input_ids"][0])
                allChoices_masks.append(choice_dict["attention_mask"][0])

            new_datapoint.update(
                {
                    "all_choices_ids": allChoices_ids,
                    "all_choices_mask": allChoices_masks,
                }
            )
        else:
            assert "target" in datapoint
            target_dict = self.tokenizer(
                datapoint["target"], return_tensors="pt", truncation=True
            )
            target_ids = target_dict["input_ids"][0]
            target_mask = target_dict["attention_mask"][0]

            new_datapoint.update(
                {
                    "target_ids": target_ids,
                    "target_mask": target_mask,
                }
            )

        return new_datapoint

    def collate_fn(self, batch_ofDatapoints):
        """
        Convert a batch of datapoints into a datapoint that is batched.  This is meant to
        override the default collate function in pytorch.

        Args:
            batch_ofDatapoints:

        Returns:

        """
        datapoint_batched = {}

        for datapoint in batch_ofDatapoints:
            for (k, v) in datapoint.items():
                if k in datapoint_batched:
                    # Each value in all_choices is already a list, so we extend and not append.
                    if "all_choices" in k:
                        datapoint_batched[k].extend(v)
                    else:
                        datapoint_batched[k].append(v)
                else:
                    # Each value in all_choices is already a list, so we do not need to
                    # initialize a list with v in it, and can just use v.
                    if "all_choices" in k:
                        datapoint_batched[k] = v
                    else:
                        datapoint_batched[k] = [v]

        # Pad ids and mask to maximum length in batch
        for (k, batch_ofValues) in datapoint_batched.items():
            # If id or mask is in key, this means we need to pad to the longest sequence length
            if ("ids" in k) or ("mask" in k):
                if "ids" in k:
                    padToken_id = self.tokenizer.pad_token_id
                    if padToken_id is None:
                        padToken_id = self.tokenizer.eos_token_id
                elif "mask" in k:
                    padToken_id = 0
                else:
                    raise ValueError(
                        f"The key {k} has ids or masks but is not recognized"
                    )
                datapoint_batched[k] = torch.nn.utils.rnn.pad_sequence(
                    batch_ofValues, batch_first=True, padding_value=padToken_id
                )

                # Don't move to device here - do it in main process to avoid CUDA issues
                # Device will be moved in training loop after collate_fn returns

            elif k == "lbl":
                datapoint_batched[k] = torch.tensor(batch_ofValues)

                # Don't move to device here - do it in main process to avoid CUDA issues

        return datapoint_batched
