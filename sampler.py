"""
This script implements a custom data loading and batching pipeline specifically
designed for efficient distributed training of sequence models, particularly
large language models, on multiple GPUs.

Key Features:
- Infinite Sampler: Provides an endless stream of shuffled data indices,
  suitable for training for a fixed number of steps rather than epochs.
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
from itertools import chain
from deprecated import deprecated

import torch
from torch.utils.data import Sampler, Dataset, DataLoader
import torch.distributed as dist
import numpy as np
from datasets import load_dataset
from batch_packer import batch_lengths_to_minibatches_lpt

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
            '''when the least full minibatch is full, we need to start a new minibatch'''
            minibatches_indices.append(current_minibatches_ids)
            current_minibatches_ids, current_minibatches_loads = reset_minibatches(num_ranks)
            least_full_batch_id = 0
        
        '''add sample to the least full minibatch'''
        current_minibatches_ids[least_full_batch_id].append(sid)
        current_minibatches_loads[least_full_batch_id] += sample_len
    
    if any(current_minibatches_loads):
        for i in range(num_ranks):
            if current_minibatches_loads[i] == 0:
                current_minibatches_ids[i].append(-1)
        minibatches_indices.append(current_minibatches_ids)
        
    return [m[rank] for m in minibatches_indices]

class JsonlDataset(Dataset):
    def __init__(self, path: str = "/new_data/aldo/v1_reasoning/math_simplerl_qwen_data_token_ids.jsonl"):
        dataset = load_dataset("json", data_files=path, split="train")
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[int(index)]
        # Ignore the index and return a fresh copy of the sequence tensor.

        # HACK(osilkin): this is just a hack to use the instructlab data processing script
        # determine the number of loss counted tokens here

        if (loss_counted_tokens := sample.get("num_loss_counted_tokens", None)) is None:
            loss_counted_tokens = sum(
                1 if label != -100 else 0 for label in sample["labels"]
            )

        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'labels': torch.tensor(sample['labels'], dtype=torch.long),
            'len': sample['len'],
            'num_loss_counted_tokens': loss_counted_tokens,
        }
    
        
class InfiniteSampler(Sampler):
    """Infinitely yields shuffled dataset indices. Crucially, in distributed
    training, it provides the *same* index sequence to all ranks.

    Reshuffles indices using a seed incremented per cycle. The actual distribution
    of samples/indices to specific ranks must be handled later (e.g., by a collate_fn).

    Args:
        len_data: The size of the dataset.
        seed: Initial random seed for shuffling (incremented each cycle).
    """
    def __init__(self, len_data, seed: int = 37):
        self.len_data = len_data
        self.seed = seed

    def __iter__(self):
        """Yields an infinite stream of shuffled dataset indices."""
        epoch = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            indices = torch.randperm(self.len_data, generator=g).tolist()
            yield from indices
            epoch += 1
    
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
        
        '''dummy samples don't have labels != -100 and should not count'''
        num_samples += 1 if item["num_loss_counted_tokens"] > 0 else 0 

    # print(
    #     f"\033[96m total length: {total_len} "
    #     f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
    # )

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "position_ids": torch.tensor([position_ids], dtype=torch.long),
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "num_samples": num_samples,
        "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
    }
    
class MaxTokensPerRankCollator:
    """A collate function for PyTorch DataLoader for distributed training.

    This collator takes a batch of samples (obtained using indices from a sampler
    like InfiniteSampler) and performs two main tasks:
    1. Filters out samples longer than `max_tokens_per_rank`.
    2. Uses `batch_lengths_to_minibatches` to determine how to distribute the
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
    def __init__(self, max_tokens_per_rank: int, rank: int=None, world_size: int=None, dummy_sample=None):
        self.max_tokens_per_rank = max_tokens_per_rank
        self.rank = rank if rank is not None else dist.get_rank()
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        if dummy_sample is None:
            dummy_sample = {'input_ids': torch.tensor([15, 14, 13, 12, 11], dtype=torch.long),
                            'labels': torch.tensor([-100, -100, -100, -100, -100], dtype=torch.long),
                            'len': 5,
                            'num_loss_counted_tokens': 0}
        self.dummy_sample = dummy_sample

    def __call__(self, batch: list[dict]):
        """Processes a batch of samples into a list of packed minibatches for the current rank.

        Args:
            batch: A list of sample dictionaries from the Dataset.

        Returns:
            A list where each element is a dictionary representing a collated minibatch
            (output of `mb_collate_fn`) ready for processing by the current rank.
        """
        batch_ = [b for b in batch if b['len'] <= self.max_tokens_per_rank]
        if len(batch_) < len(batch):
            print(f"\033[38;5;196mremoved {len(batch) - len(batch_)} samples from batch because they are longer than the max tokens per gpu\033[0m")
        batch_lengths = [sample['len'] for sample in batch]
        batch_num_loss_counted_tokens = sum([sample['num_loss_counted_tokens'] for sample in batch])
        all_minibatches_indices = batch_lengths_to_minibatches_lpt(batch_lengths, self.max_tokens_per_rank, self.world_size, self.rank)
        
        all_minibatches = []
        for mb_indices in all_minibatches_indices:
            mb = [batch[i] if i != -1 else self.dummy_sample for i in mb_indices]
            all_minibatches.append(mb_collate_fn(mb, batch_num_loss_counted_tokens))

        return all_minibatches
    
def get_data_loader(**kwargs):
    # from ipdb import set_trace; set_trace()
    dataset = JsonlDataset(kwargs['data_path'])
    batch_size = kwargs['batch_size']
    max_tokens_per_rank = kwargs['max_tokens_per_gpu']
    seed = kwargs['seed']
    rank = kwargs.get('rank', None)
    world_size = kwargs.get('world_size', None)
    dummy_sample = kwargs.get('dummy_sample', None)
    return DataLoader(dataset, 
                      batch_size, 
                      sampler=InfiniteSampler(len(dataset), seed=seed),
                      collate_fn=MaxTokensPerRankCollator(max_tokens_per_rank, 
                                                          rank=rank, 
                                                          world_size=world_size, 
                                                          dummy_sample=dummy_sample),
                      num_workers=4)

if __name__ == "__main__":
    data_loader = get_data_loader(data_path="test.jsonl",
                                  batch_size=40,
                                  max_tokens_per_gpu=5000,
                                  seed=37,
                                  rank=0,
                                  world_size=2)
    data_loader2 = get_data_loader(data_path="test.jsonl",
                                  batch_size=26,
                                  max_tokens_per_gpu=5000,
                                  seed=37,
                                  rank=1,
                                  world_size=2)
    data_loader = iter(data_loader)
    data_loader2 = iter(data_loader2)
    batch = next(data_loader)
    batch2 = next(data_loader2)
    from IPython import embed
    embed()

                
