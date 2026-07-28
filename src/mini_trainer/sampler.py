"""
This script implements a custom data loading and batching pipeline specifically
designed for efficient distributed training of sequence models, particularly
large language models, on multiple GPUs.

Key Features:
- Epoch-based Sampler: Provides shuffled data indices for each epoch,
  suitable for both finite and infinite training modes.
- Initial Batching: Groups samples into initial batches based on a fixed number
  of samples per batch.
- Dynamic Minibatching for Distributed Training: Takes the initial batches and
  further divides them into 'minibatches'. Each minibatch is a list distributed
  across available ranks (GPUs). The allocation process aims to pack sequences
  efficiently such that the total number of tokens processed by any single rank
  within a minibatch step stays below a predefined maximum (`max_tokens_per_gpu`).
  The number of minibatches generated from an initial batch can vary dynamically
  depending on the lengths of the sequences in that batch.
- Token-Based Load Balancing: Ensures that each GPU receives a comparable
  computational load (measured in tokens) per step, optimizing hardware
  utilization and preventing out-of-memory errors when dealing with variable
  sequence lengths.
- Padding/Dummy Samples: Handles cases where ranks might not have enough data
  to fill a minibatch by using dummy samples, ensuring all ranks process the
  same number of minibatches.
"""

import os

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset as HFDataset, load_dataset
from deprecated import deprecated
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

from mini_trainer.batch_packer import batch_lengths_to_minibatches_lpt
from mini_trainer.training_types import PretrainingConfig
from mini_trainer.utils import log_rank_0


def reset_minibatches(num_ranks: int):
    return [[] for _ in range(num_ranks)], np.zeros(num_ranks)


@deprecated("Use batch_lengths_to_minibatches_lpt instead for better load balancing performance")
def batch_lengths_to_minibatches(batch_lengths: list[int], max_tokens_per_rank: int, num_ranks: int, rank: int):
    """Distributes indices from a batch into minibatches across ranks.

    Takes a list of sequence lengths corresponding to samples in an initial batch
    and distributes their indices into multiple 'minibatches'. Each minibatch
    represents a step where data is processed concurrently across `num_ranks` GPUs.

    The distribution aims to assign sequences (represented by their indices `sid`
    in the original `batch_lengths` list) to ranks such that the sum of sequence
    lengths (tokens) assigned to any single rank does not exceed
    `max_tokens_per_rank`. It prioritizes assigning the next sequence to the rank
    currently having the minimum total tokens assigned in the current minibatch.

    If adding the next sequence to the least-loaded rank would exceed the limit,
    the current minibatch is considered complete, and a new minibatch is started.

    If the last minibatch is incomplete, ranks with no assigned sequences are
    given a placeholder index of -1.

    Args:
        batch_lengths: A list where each element is the length (number of tokens)
                       of a sequence in the initial batch.
        max_tokens_per_rank: The maximum number of tokens allowed per rank in a
                             single minibatch.
        num_ranks: The total number of distributed training ranks (GPUs).
        rank: The specific rank for which to retrieve the assigned indices.

    Returns:
        A list of lists. Each inner list contains the indices (from the original
        batch) assigned to the specified `rank` for one minibatch. Placeholder -1
        indicates padding.
    """
    minibatches_indices = []
    current_minibatches_ids, current_minibatches_loads = reset_minibatches(num_ranks)
    for sid, sample_len in enumerate(batch_lengths):
        least_full_batch_id = np.argmin(current_minibatches_loads)

        if current_minibatches_loads[least_full_batch_id] + sample_len > max_tokens_per_rank:
            """when the least full minibatch is full, we need to start a new minibatch"""
            minibatches_indices.append(current_minibatches_ids)
            current_minibatches_ids, current_minibatches_loads = reset_minibatches(num_ranks)
            least_full_batch_id = 0

        """add sample to the least full minibatch"""
        current_minibatches_ids[least_full_batch_id].append(sid)
        current_minibatches_loads[least_full_batch_id] += sample_len

    if any(current_minibatches_loads):
        for i in range(num_ranks):
            if current_minibatches_loads[i] == 0:
                current_minibatches_ids[i].append(-1)
        minibatches_indices.append(current_minibatches_ids)

    return [m[rank] for m in minibatches_indices]


class JsonlDataset(Dataset):
    def __init__(
        self,
        path: str | None = None,
        max_seq_len: int | None = None,
        hf_dataset: HFDataset | None = None,
    ):
        """
        Initializes a JsonlDataset object which we use to load and process the dataset.
        Accepts either a path to a JSONL file or a pre-loaded HuggingFace dataset.

        Args:
            path: Path to the JSONL file or HuggingFace dataset name
            max_seq_len: Maximum sequence length to keep (filters out longer sequences)
            hf_dataset: Pre-loaded HuggingFace dataset
        """
        # dataset can be any of these
        if hf_dataset is not None:
            dataset = hf_dataset
        elif path is not None:
            dataset = load_dataset("json", data_files=path, split="train")
        else:
            raise ValueError("Either 'path' or 'hf_dataset' must be provided")

        # The two required fields on a dataset are `input_ids` and `labels`,
        # everything else is computable. Here we handle the case when we
        # must actually provide them
        dataset = self.add_necessary_fields(dataset)
        if max_seq_len is not None:
            dataset = self.filter_by_max_seq_len(dataset, max_seq_len)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[int(index)]

        # Determine the number of loss-counted tokens if the field is missing.
        if (loss_counted_tokens := sample.get("num_loss_counted_tokens", None)) is None:
            # causal LMs shift labels to the left when calculating cross-entropy loss
            # so we must account for this shift when we calculate cross-entropy
            loss_counted_tokens = sum(1 if label != -100 else 0 for label in sample["labels"][1:])

        item = {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "len": sample["len"],
            "num_loss_counted_tokens": loss_counted_tokens,
        }

        # ensure that an attention mask exists if the sample came with one
        if "attention_mask" in sample:
            item["attention_mask"] = torch.tensor(sample["attention_mask"], dtype=torch.long)
        return item

    @classmethod
    def add_necessary_fields(cls, dataset: HFDataset) -> HFDataset:
        required_fields = ["input_ids", "labels"]
        for field in required_fields:
            if field not in dataset.features:
                raise ValueError(f"Dataset must contain '{field}' field")
        if "len" not in dataset.features:
            dataset = dataset.map(lambda s: {"len": len(s["input_ids"])})
        if "num_loss_counted_tokens" not in dataset.features:
            dataset = dataset.map(
                lambda s: {
                    # causal LMs shift labels to the left when calculating cross-entropy loss
                    "num_loss_counted_tokens": sum(1 for tok in s["labels"][1:] if tok != -100)
                }
            )

        return dataset

    @classmethod
    def filter_by_max_seq_len(cls, dataset: HFDataset, max_seq_len: int) -> HFDataset:
        dataset = dataset.filter(lambda x: x["len"] <= max_seq_len)
        return dataset

    @classmethod
    def load_and_split(
        cls,
        data_path: str,
        validation_split: float = 0.0,
        max_seq_len: int | None = None,
        seed: int = 42,
    ) -> tuple["JsonlDataset", "JsonlDataset | None"]:
        """Load dataset and optionally split into train/validation sets.

        Args:
            data_path: Path to JSONL file or HuggingFace dataset name
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
            max_seq_len: Maximum sequence length (filters out longer sequences)
            seed: Random seed for reproducible splits

        Returns:
            tuple: (train_dataset, val_dataset) where val_dataset is None if validation_split <= 0
        """
        # handle either local or HF dataset
        if os.path.exists(data_path):
            hf_dataset = load_dataset("json", data_files=data_path, split="train")
        else:
            hf_dataset = load_dataset(data_path, split="train")

        # add necessary fields & filter by max_seq_len if specified
        hf_dataset = cls.add_necessary_fields(hf_dataset)
        if max_seq_len is not None:
            original_size = len(hf_dataset)
            hf_dataset = cls.filter_by_max_seq_len(hf_dataset, max_seq_len)
            filtered_size = len(hf_dataset)
            if original_size > filtered_size:
                log_rank_0(
                    f"\033[33mFiltered out {original_size - filtered_size} samples "
                    f"(out of {original_size}) that exceed max_seq_len={max_seq_len}\033[0m"
                )

        val_dataset = None
        if validation_split <= 0.0:
            # default case
            train_dataset = cls(hf_dataset=hf_dataset)
            return train_dataset, val_dataset

        # validation split case
        split_dataset = hf_dataset.train_test_split(test_size=validation_split, seed=seed, shuffle=True)
        train_dataset = cls(hf_dataset=split_dataset["train"])
        val_dataset = cls(hf_dataset=split_dataset["test"])
        return train_dataset, val_dataset


class PretrainingBlockDataset(Dataset):
    """
    Dataset for pretraining that concatenates documents and views them as fixed-size blocks.

    Loads JSONL with {"input_ids": [...]} (no labels), concatenates all documents,
    then provides access to fixed-size blocks.
    """

    def __init__(self, dataset: HFDataset, block_size: int, pad_token_id: int):
        """
        Args:
            dataset: HuggingFace dataset with tokenized documents (must have 'input_ids' column)
            block_size: Size of each block in tokens
            pad_token_id: Token ID to use for padding the last block
        """
        self.block_size = block_size
        self.pad_token_id = pad_token_id

        # Validate required fields
        if "input_ids" not in dataset.column_names:
            raise ValueError("Pretraining data must have 'input_ids' field")

        if pad_token_id < 0:
            raise ValueError("pad_token_id must be a positive integer")

        log_rank_0(f"Concatenating {len(dataset):,} documents for pretraining...")

        # Concatenate all input_ids into one giant list
        all_input_ids = []
        for sample in dataset:
            all_input_ids.extend(sample["input_ids"])

        # calculates the offset of the final block (which may be smaller than block_size)
        # so we can sample from the full list of IDs and avoid wasting data
        total_tokens = len(all_input_ids)
        num_full_blocks, remainder = divmod(total_tokens, block_size)

        # include partial block if there is one
        last_block_len = block_size
        if remainder > 0:
            last_block_len = remainder

        # store information needed to sample from the dataset in `block` strides
        self.num_blocks = num_full_blocks + (1 if remainder else 0)
        self.last_block_len = last_block_len
        self.all_input_ids = all_input_ids  # keep all tokens

        log_rank_0(f"Total tokens: {total_tokens:,}")
        log_rank_0(f"Block size: {block_size}")
        log_rank_0(f"Total blocks: {self.num_blocks:,} ({num_full_blocks} complete, {1 if remainder else 0} partial)")
        if remainder:
            log_rank_0(f"Partial block size: {remainder} tokens")

    def __len__(self):
        return self.num_blocks

    @classmethod
    def from_jsonl_file(cls, data_path: str, block_size: int, pad_token_id: int) -> "PretrainingBlockDataset":
        """Load a dataset from a JSONL file and create a PretrainingBlockDataset."""
        dataset = load_dataset("json", data_files=data_path, split="train")
        return cls(dataset, block_size, pad_token_id)

    @classmethod
    def load_and_split(
        cls,
        data_path: str,
        block_size: int,
        pad_token_id: int,
        validation_split: float = 0.0,
        seed: int = 42,
    ) -> tuple["PretrainingBlockDataset", "PretrainingBlockDataset | None"]:
        """
        Load a dataset from a JSONL file and create a PretrainingBlockDataset,
        optionally splitting it into train and validation sets.

        Args:
            data_path: Path to JSONL file with tokenized documents
            block_size: Size of each block in tokens
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
            seed: Random seed for splitting

        Returns:
            Tuple of (train_dataset, val_dataset). val_dataset is None if validation_split is 0.
        """
        if not (0.0 <= validation_split < 1.0):
            raise ValueError(f"validation_split must be in [0.0, 1.0), got {validation_split}")

        dataset = load_dataset("json", data_files=data_path, split="train")

        if validation_split == 0.0:
            train_dataset = cls(dataset, block_size, pad_token_id)
            return train_dataset, None

        # Split the dataset
        split_dataset = dataset.train_test_split(test_size=validation_split, seed=seed)

        # create and return the datasets
        train_dataset = cls(split_dataset["train"], block_size, pad_token_id)
        val_dataset = cls(split_dataset["test"], block_size, pad_token_id)
        return train_dataset, val_dataset

    def __getitem__(self, idx: int):
        """
        Return a block of tokens starting at idx * block_size.

        For pretraining: labels == input_ids (no masking)
        """
        if idx >= self.num_blocks:
            raise IndexError(f"Index {idx} out of range for {self.num_blocks} blocks")

        start = idx * self.block_size
        end = start + self.block_size

        is_last_block = idx == self.num_blocks - 1
        is_partial = is_last_block and (len(self.all_input_ids) % self.block_size != 0)
        if is_partial:
            # partial block: get actual tokens and pad the rest
            actual_tokens = self.all_input_ids[start:]
            actual_len = len(actual_tokens)
            pad_len = self.block_size - actual_len

            input_ids = actual_tokens + [self.pad_token_id] * pad_len
            # mask padding in labels so it doesn't contribute to loss
            labels = actual_tokens + [-100] * pad_len
            num_loss_counted = actual_len - 1  # causal shift
        else:
            input_ids = self.all_input_ids[start:end]
            labels = list(input_ids)  # Create explicit copy to avoid reference issues
            num_loss_counted = self.block_size - 1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "len": self.block_size,
            "num_loss_counted_tokens": num_loss_counted,
        }


class EpochSampler(Sampler):
    """
    Here we redefine RandomSampler so we can have a consistent signature with InfiniteSampler
    """

    def __init__(self, len_data: int, seed: int = 67, epoch: int = 0):
        self.len_data = len_data
        self.seed = seed
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def generate_samples(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        samples = torch.randperm(self.len_data, generator=g).tolist()
        return samples

    def __iter__(self):
        samples = self.generate_samples()
        yield from samples

    def __len__(self):
        return self.len_data


def mb_collate_fn(minibatch, batch_num_loss_counted_tokens):
    """Collates a list of samples into a single packed batch for Flash Attention.

    This function takes a 'minibatch' (list of pre-fetched dataset samples)
    and concatenates their 'input_ids', 'labels', and generates corresponding
    'position_ids'. It does *not* add padding.

    The resulting batch format is 'packed' or 'unpadded', where multiple sequences
    are concatenated into single tensors. Sequence boundaries are implicitly defined
    by the 'position_ids', which restart from 0 for each concatenated sequence.

    **IMPORTANT**: This format requires the downstream model's attention mechanism
    (e.g., Flash Attention) to correctly handle packed sequences. Standard attention
    implementations may not work correctly as they expect padded inputs and explicit
    attention masks. Flash Attention typically uses mechanisms like `cu_seqlens`
    (cumulative sequence lengths), derived from position IDs or sequence lengths,
    to compute the correct block-diagonal attention implicitly.

    Args:
        minibatch: A list of dictionaries, where each dictionary represents a
                   sample and contains at least 'input_ids' and 'labels'.

    Returns:
        A dictionary containing the collated batch:
        - 'input_ids': Single tensor of concatenated input IDs.
        - 'labels': Single tensor of concatenated labels.
        - 'position_ids': Single tensor of position IDs, reset for each sequence.
        - 'num_loss_counted_tokens': Total number of non-ignored label tokens (-100).
        - 'num_samples': The number of sequences packed into this batch.
    """
    input_ids = []
    labels = []
    position_ids = []
    total_len = 0
    num_loss_counted_tokens = 0
    # from ipdb import set_trace; set_trace()
    # try:
    num_samples = 0
    for item in minibatch:
        item_len = len(item["input_ids"])

        input_ids.extend(item["input_ids"])
        labels.extend(item["labels"])
        position_ids.extend(range(item_len))

        total_len += item_len
        # sample_loss_counted_tokens = (item["labels"] != -100).sum().item()
        num_loss_counted_tokens += item["num_loss_counted_tokens"]

        """dummy samples don't have labels != -100 and should not count"""
        num_samples += 1 if item["num_loss_counted_tokens"] > 0 else 0

    # print(
    #     f"\033[96m total length: {total_len} "
    #     f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
    # )

    pad_len = (8 - total_len % 8) % 8
    if pad_len > 0:
        input_ids.extend([0] * pad_len)
        labels.extend([-100] * pad_len)
        position_ids.extend(range(pad_len))

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "position_ids": torch.tensor([position_ids], dtype=torch.long),
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "num_samples": num_samples,
        "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
    }


def padded_mb_collate_fn(minibatch: list[dict], batch_num_loss_counted_tokens: int, pad_token_id: int) -> dict:
    """Collates a list of samples into a padded batch for standard attention.

    This function takes a minibatch (list of dataset samples) and creates padded
    tensors suitable for standard attention mechanisms. Unlike the flash attention
    version, this pads all sequences to the same length and creates attention masks.

    Args:
        minibatch: A list of dictionaries, where each dictionary represents a
                   sample and contains 'input_ids' and 'labels'.
        batch_num_loss_counted_tokens: Total number of loss-counted tokens in the batch.
        pad_token_id: Token id to use for padding in padded batches.

    Returns:
        A dictionary containing the collated batch:
        - 'input_ids': 2D tensor of padded input IDs [batch_size, max_len]
        - 'labels': 2D tensor of padded labels [batch_size, max_len]
        - 'attention_mask': 2D tensor indicating real vs padding tokens
        - 'position_ids': None (not used for standard attention)
        - 'num_loss_counted_tokens': Total number of non-ignored label tokens
        - 'num_samples': The number of real sequences in this batch
        - 'batch_num_loss_counted_tokens': Total number of loss-counted tokens in the batch
    """
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer")

    if not minibatch:
        return {
            "input_ids": torch.tensor([[]], dtype=torch.long),
            "labels": torch.tensor([[]], dtype=torch.long),
            "attention_mask": torch.tensor([[]], dtype=torch.long),
            "position_ids": None,
            "num_loss_counted_tokens": 0,
            "num_samples": 0,
            "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
        }

    max_len = max(len(item["input_ids"]) for item in minibatch)
    max_len = (max_len + 7) & ~7  # round up to multiple of 8 for torch.compile

    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    num_loss_counted_tokens = 0
    num_samples = 0

    for item in minibatch:
        item_len = len(item["input_ids"])
        pad_len = max_len - item_len

        input_ids = item["input_ids"]
        labels = item["labels"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)

        # add attention masks for datasets which don't already have them
        attn = item.get("attention_mask")
        if isinstance(attn, torch.Tensor):
            attn = attn.tolist()
        if attn is None:
            attn = [1] * item_len
        attention_masks.append(attn + [0] * pad_len)

        # calculate aggregate statistics
        num_loss_counted_tokens += item["num_loss_counted_tokens"]
        num_samples += 1 if item["num_loss_counted_tokens"] > 0 else 0

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "position_ids": None,
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "num_samples": num_samples,
        "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
    }


class MaxTokensPerRankCollator:
    """A collate function for PyTorch DataLoader for distributed training.

    This collator takes a batch of samples (obtained using indices from a sampler
    like InfiniteSampler) and performs two main tasks:
    1. Filters out samples longer than `max_tokens_per_rank`.
    2. Uses `batch_lengths_to_minibatches_lpt` to determine how to distribute the
       remaining samples across ranks into one or more 'minibatches', ensuring
       no rank exceeds `max_tokens_per_rank` per minibatch.
    3. For the current rank, it fetches the assigned samples (or dummy samples
       for padding) for each determined minibatch.
    4. Uses `mb_collate_fn` to collate the samples for each minibatch into the
       packed format required by Flash Attention.

    Args:
        max_tokens_per_rank (int): Maximum number of tokens allowed per rank
            in a single processed minibatch.
        rank (int, optional): The rank of the current process. If None, attempts
            to get it from `torch.distributed`.
        world_size (int, optional): Total number of ranks. If None, attempts
            to get it from `torch.distributed`.
        dummy_sample (dict, optional): A sample used for padding when a rank
            has no real samples assigned in a minibatch.
    """

    def __init__(
        self,
        max_tokens_per_rank: int,
        rank: int = None,
        world_size: int = None,
        dummy_sample=None,
    ):
        self.max_tokens_per_rank = max_tokens_per_rank

        if rank is None:
            self.global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        else:
            self.global_rank = rank
        if world_size is None:
            self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        else:
            self.world_size = world_size
        if dummy_sample is None:
            dummy_sample = {
                "input_ids": torch.tensor([15, 14, 13, 12, 11], dtype=torch.long),
                "labels": torch.tensor([-100, -100, -100, -100, -100], dtype=torch.long),
                "len": 5,
                "num_loss_counted_tokens": 0,
            }
        self.dummy_sample = dummy_sample

    def __call__(self, batch: list[dict]):
        """Processes a batch of samples into a list of packed minibatches for the current rank.

        Args:
            batch: A list of sample dictionaries from the Dataset.

        Returns:
            A list where each element is a dictionary representing a collated minibatch
            (output of `mb_collate_fn`) ready for processing by the current rank.
        """
        batch_ = [b for b in batch if b["len"] <= self.max_tokens_per_rank]
        if len(batch_) < len(batch):
            log_rank_0(
                f"\033[38;5;196mremoved {len(batch) - len(batch_)} samples from batch because they are longer than the max tokens per gpu\033[0m"
            )
        # Use filtered batch for lengths and loss counts
        batch_lengths = [sample["len"] for sample in batch_]
        batch_num_loss_counted_tokens = sum([sample["num_loss_counted_tokens"] for sample in batch_])
        all_minibatches_indices = batch_lengths_to_minibatches_lpt(
            batch_lengths, self.max_tokens_per_rank, self.world_size, self.global_rank
        )
        if not all_minibatches_indices:
            # Ensure every rank returns at least one (dummy) microbatch so downstream
            # metric aggregation always has expected keys.
            all_minibatches_indices = [[-1]]

        all_minibatches = []
        for mb_indices in all_minibatches_indices:
            mb = [batch_[i] if i != -1 else self.dummy_sample for i in mb_indices]
            all_minibatches.append(mb_collate_fn(mb, batch_num_loss_counted_tokens))

        return all_minibatches


def get_data_loader(
    data_path: str,
    batch_size: int,
    max_tokens_per_gpu: int,
    seed: int,
    rank: int | None = None,
    world_size: int | None = None,
    dummy_sample: dict | None = None,
    num_workers: int = 0,
    validation_split: float = 0.0,
    validation_data_path: str | None = None,
    max_seq_len: int | None = None,
    pretraining_config: PretrainingConfig | None = None,
    pad_token_id: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
    """Create data loader(s) with optional train/validation split.

    Efficiently loads the dataset once and splits it if needed, avoiding
    multiple reads of the same data.

    Args:
        data_path: Path to the JSONL data file or HuggingFace dataset
        batch_size: Number of samples per batch
        max_tokens_per_gpu: Maximum tokens per GPU per step
        seed: Random seed for reproducibility
        rank: Rank of the current process (for distributed training)
        world_size: Total number of processes (for distributed training)
        dummy_sample: Sample to use for padding when ranks have uneven data
        num_workers: Number of worker processes for data loading
        validation_split: Fraction of data to use for validation (0.0 to 1.0)
        validation_data_path: Path to a separate validation dataset (JSONL).
            Mutually exclusive with validation_split.
        max_seq_len: Maximum sequence length to keep (filters out longer sequences)
        pretraining_config: Configuration for pretraining mode. If provided, enables
            pretraining with block-based sampling.
        pad_token_id: Token id to use for padding in padded batches.
    Returns:
        tuple: (train_loader, val_loader) where val_loader is None if validation_split <= 0
            and validation_data_path is None
    """
    # Validate parameters
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError(f"validation_split must be between 0 and 1 (exclusive of 1), got {validation_split}")

    if validation_data_path is not None and validation_split > 0.0:
        raise ValueError("validation_data_path and validation_split are mutually exclusive")

    # Create dataset based on mode
    if pretraining_config is not None:
        if pad_token_id is None or pad_token_id < 0:
            raise ValueError(f"pretraining mode requires a valid non-negative pad_token_id, got: {pad_token_id=}")
        # Pretraining mode: use PretrainingBlockDataset
        train_dataset, val_dataset = PretrainingBlockDataset.load_and_split(
            data_path=data_path,
            block_size=pretraining_config.block_size,
            pad_token_id=pad_token_id,
            validation_split=validation_split,
            seed=seed,
        )
        log_rank_0(f"Pretraining dataset: {len(train_dataset)} blocks of size {pretraining_config.block_size}")
        if val_dataset is not None:
            log_rank_0(f"Validation dataset: {len(val_dataset)} blocks of size {pretraining_config.block_size}")
        else:
            log_rank_0("No validation dataset")
    else:
        # Instruction tuning mode: use JsonlDataset with optional validation split
        train_dataset, val_dataset = JsonlDataset.load_and_split(
            data_path=data_path,
            validation_split=validation_split,
            max_seq_len=max_seq_len,
            seed=seed,
        )
        if val_dataset is not None:
            log_rank_0(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        else:
            log_rank_0(f"Dataset split: {len(train_dataset)} train")

    # Load a separate validation dataset if provided (overrides split-based val_dataset)
    if validation_data_path is not None:
        if pretraining_config is not None:
            separate_val, _ = PretrainingBlockDataset.load_and_split(
                data_path=validation_data_path,
                block_size=pretraining_config.block_size,
                pad_token_id=pad_token_id,
                validation_split=0.0,
                seed=seed,
            )
        else:
            separate_val, _ = JsonlDataset.load_and_split(
                data_path=validation_data_path,
                validation_split=0.0,
                max_seq_len=max_seq_len,
                seed=seed,
            )
        val_dataset = separate_val
        log_rank_0(f"Loaded separate validation dataset: {len(val_dataset)} samples from {validation_data_path}")

    # Create collate function
    collate_fn = MaxTokensPerRankCollator(
        max_tokens_per_gpu,
        rank=rank,
        world_size=world_size,
        dummy_sample=dummy_sample,
    )

    # Create train data loader
    train_sampler = EpochSampler(len(train_dataset), seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )

    # Create validation data loader if needed
    val_loader = None
    if val_dataset is not None:
        val_sampler = SequentialSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=False,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    data_loader, _ = get_data_loader(
        data_path="test.jsonl",
        batch_size=40,
        max_tokens_per_gpu=5000,
        seed=37,
        rank=0,
        world_size=2,
    )
    data_loader2, _ = get_data_loader(
        data_path="test.jsonl",
        batch_size=26,
        max_tokens_per_gpu=5000,
        seed=37,
        rank=1,
        world_size=2,
    )
    data_loader = iter(data_loader)
    data_loader2 = iter(data_loader2)
    batch = next(data_loader)
    batch2 = next(data_loader2)
