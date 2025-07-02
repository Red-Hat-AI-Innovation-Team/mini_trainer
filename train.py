import time
import os
from pathlib import Path
from enum import Enum
import json

from typer import Typer, Option

from async_structured_logger import AsyncStructuredLogger
import torch

from batch_metrics import BatchMetrics
from sampler import get_data_loader
from setup_model_for_training import setup_model, setup_training_components
from utils import init_distributed_environment, log_rank_0, setup_logger
from svd_utils import reconstruct_weight_matrix # Used to reconstruct full weight matrix from SVD components during model save (for Orthogonal Subspace Learning models)

app = Typer(
    pretty_exceptions_show_locals=False,  # Hide local variables in tracebacks
    pretty_exceptions_short=True   
)

def take_gradient_step(model, optimizer, lr_scheduler):
    """Scales gradients, applies clipping, and takes an optimization step."""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if hasattr(model, "project_gradients"):
        model.project_gradients()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    return grad_norm

def save_model(fsdp_model, samples_seen, output_dir, model_name_or_path):
    from huggingface_hub import split_torch_state_dict_into_shards
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    # Only on rank 0
    log_rank_0(f"Saving model at {samples_seen} samples")
    start = time.time()
    rank = torch.distributed.get_rank()
    save_directory = Path(output_dir) / "hf_format" / f"samples_{samples_seen}"
    os.makedirs(save_directory, exist_ok=True)
    # Get full state dict
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    state_dict = get_model_state_dict(fsdp_model, options=StateDictOptions(full_state_dict=True))
    inner = getattr(fsdp_model, "module", fsdp_model)
    # If model uses orthogonal subspace learning, reconstruct full weights from SVD components before saving
    if hasattr(inner, "name_mapping"):
        for orig, safe in inner.name_mapping.items():
            U_high = state_dict.pop(f"{safe}_U_high")
            S_high = state_dict.pop(f"{safe}_S_high")
            V_high = state_dict.pop(f"{safe}_V_high")
            U_low = state_dict.pop(f"svd_params.{safe}.U_low")
            S_low = state_dict.pop(f"svd_params.{safe}.S_low")
            V_low = state_dict.pop(f"svd_params.{safe}.V_low")
            W = reconstruct_weight_matrix(
                {
                    "U_high": U_high,
                    "S_high": S_high,
                    "V_high": V_high,
                    "U_low": U_low,
                    "S_low": S_low,
                    "V_low": V_low,
                }
            ).to(torch.bfloat16)
            state_dict[orig] = W
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    
    if rank == 0:
        pattern = "model{suffix}.safetensors"
        index_name = "model.safetensors.index.json"
        
        # Shard splitting
        split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=pattern, max_shard_size="5GB",
        )
        # Save shards
        for filename, tensors in split.filename_to_tensors.items():
            shard = {k: state_dict[k] for k in tensors}
            path = os.path.join(save_directory, filename)
            save_file(shard, path)
            
        # Save index if sharded
        if split.is_sharded:
            index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
            with open(os.path.join(save_directory, index_name), "w") as f:
                json.dump(index, f, indent=2, sort_keys=True)
        # Save config and tokenizer (unwrap inner module)
        inner = getattr(fsdp_model, "module", fsdp_model)
        inner.config.to_json_file(os.path.join(save_directory, "config.json"))
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(save_directory)
        log_rank_0(f"\033[1;38;2;0;255;255mSaved model at\033[0m {samples_seen} samples in {time.time() - start:.2f} seconds")
    torch.distributed.barrier()

def train(model, optimizer, lr_scheduler, data_loader, output_dir, min_samples_per_checkpoint, model_name_or_path):
    model.train()
    metric_logger = AsyncStructuredLogger(output_dir + f"/training_metrics_{os.environ.get('RANK')}.jsonl")
    world_size = int(os.environ["WORLD_SIZE"])
    is_main_process = os.environ.get("RANK") == "0"

    batch_totals = BatchMetrics()
    step = 0
    total_samples_accumulated = 0
    last_saved_samples = 0
    device = next(model.parameters()).device
    
    data_loader = iter(data_loader)
    for batch in data_loader:
        batch_start_time = time.time()
        batch_totals.reset_batch()
        torch.cuda.reset_peak_memory_stats()
        for grad_accum, mb in enumerate(batch):

            mb_start_time = time.time()
            mb_num_loss_counted_tokens = mb.pop('num_loss_counted_tokens')
            mb_num_samples = mb.pop('num_samples')
            batch_num_loss_counted_tokens = mb.pop('batch_num_loss_counted_tokens')
            mb = {k: v.to(device) for k, v in mb.items()}
            # torch.distributed.breakpoint()
            output = model(**mb)
            loss = output.loss.float().sum() 
            loss_metrics = loss.detach().item()
            '''multiply by world_size to account for the fact that fsdp takes the mean of the gradients across the world_size'''
            '''the loss is a sum of all cross entropy losses for all tokens in the batch, we divide by batch_num_loss_counted_tokens to get the average loss per token'''
            loss = loss * world_size / batch_num_loss_counted_tokens

            loss.backward()
            torch.cuda.empty_cache()

            batch_totals.accumulate_minibatch_metrics(
                num_loss_counted_tokens=mb_num_loss_counted_tokens,
                num_total_tokens=mb['input_ids'].shape[1],
                num_samples=mb_num_samples,
                loss=loss_metrics,
                loss_backward=loss.detach().item()/world_size,
                time_per_minibatch=time.time() - mb_start_time,
            )
        step += 1
        #sum the metrics from all processes
        batch_totals.reduce_batch_metrics(device)
        
        #use accumulated metrics to take a gradient step and logging
        bm = batch_totals.totals
        total_samples_accumulated += bm['num_samples']
        grad_norm = take_gradient_step(model, optimizer, lr_scheduler)

        if is_main_process:
            batch_time = time.time() - batch_start_time
            batch_metrics = {
                    "step": step,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm.item(),
                    "loss": bm['loss']/batch_num_loss_counted_tokens,
                    "avg_loss_backward": bm['loss_backward']/(grad_accum+1),
                    "num_samples": bm['num_samples'],
                    "num_loss_counted_tokens": bm['num_loss_counted_tokens'],
                    "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
                    "num_total_tokens": bm['num_total_tokens'],
                    "grad_accum": grad_accum+1,
                    "avg_time_per_minibatch": bm['time_per_minibatch']/(grad_accum+1)/world_size,
                    "time_per_batch": batch_time,
                    "tokens_per_second": bm['num_total_tokens']/batch_time,
                    "total_samples_accumulated": total_samples_accumulated, 
                    "samples_per_second": bm['num_samples']/batch_time,
                    "peak_memory_usage_GB": float(torch.cuda.max_memory_allocated() / 1e9),
                }
            metric_logger.log_sync(
                batch_metrics
            )
        
        torch.distributed.barrier()
        if total_samples_accumulated - last_saved_samples >= min_samples_per_checkpoint:
            save_model(model, total_samples_accumulated, output_dir, model_name_or_path)
            last_saved_samples = total_samples_accumulated

class LogLevelEnum(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@app.command()
def main(
    model_name_or_path: str = Option("Qwen/Qwen2.5-1.5B-Instruct", help="Model name or path"),
    data_path: str = Option("test.jsonl", help="Path to the training data JSONL file"),
    batch_size: int = Option(1024, help="Initial batch size before dynamic splitting"),
    max_tokens_per_gpu: int = Option(10000, help="Maximum tokens per GPU per minibatch"),
    learning_rate: float = Option(5e-6, help="Peak learning rate"),
    num_warmup_steps: int = Option(10, help="Number of warmup steps for the LR scheduler"),
    lr_scheduler: str = Option("constant_with_warmup", help="Learning rate scheduler type"),
    seed: int = Option(42, help="Random seed for reproducibility"),
    use_liger_kernels: bool = Option(False, help="Whether to use Liger kernels"),
    orthogonal_subspace_learning: bool = Option(False, help="Enable SVD based orthogonal subspace training"),
    output_dir: str = Option(..., help="Directory to save checkpoints and logs (required)"),
    logging_level: LogLevelEnum = Option(
        LogLevelEnum.INFO, 
        help="Logging level", 
        case_sensitive=False
    ),
    min_samples_per_checkpoint: int = Option(..., help="Minimum number of samples processed before saving a checkpoint (required)"),
):
    init_distributed_environment()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Log parameters only on rank 0
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        params = {
            "model_name_or_path": model_name_or_path,
            "data_path": data_path,
            "batch_size": batch_size,
            "max_tokens_per_gpu": max_tokens_per_gpu,
            "learning_rate": learning_rate,
            "num_warmup_steps": num_warmup_steps,
            "lr_scheduler": lr_scheduler,
            "seed": seed,
            "use_liger_kernels": use_liger_kernels,
            "orthogonal_subspace_learning": orthogonal_subspace_learning,
            "output_dir": output_dir,
            "logging_level": logging_level.value,
            "min_samples_per_checkpoint": min_samples_per_checkpoint,
            "RANK": rank, # Include rank itself, though it will be 0 here
            "WORLD_SIZE": int(os.environ.get("WORLD_SIZE", 1))
        }
        params_path = output_path / f"training_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        # Pretty print parameters in a single line using JSON
        print(f"Training with parameters: {json.dumps(params, separators=(',', ':'), indent=4)}")
        print(f"Training parameters saved to {params_path}")

    setup_logger(level=logging_level.value)
    # If Orthogonal Subspace Learning is enabled, loads a model with decomposed trainable low-rank + fixed high-rank subspace weights (see svd_utils)
    model = setup_model(
        model_name_or_path=model_name_or_path,
        use_liger_kernels=use_liger_kernels,
        orthogonal_subspace_learning=orthogonal_subspace_learning,
    )
    model, optimizer, lr_scheduler = setup_training_components(model,
                                                               learning_rate=learning_rate,
                                                               num_warmup_steps=num_warmup_steps,
                                                               lr_scheduler=lr_scheduler)
    data_loader = get_data_loader(data_path=data_path,
                                  batch_size=batch_size,
                                  max_tokens_per_gpu=max_tokens_per_gpu,
                                  seed=seed)
    
    train(model, 
          optimizer, 
          lr_scheduler, 
          data_loader, 
          output_dir, 
          min_samples_per_checkpoint,
          model_name_or_path)
    
if __name__ == "__main__":
    app()


'''
rclone copy --copy-links /new_data/experiments_rh/phi-4_limo_trainer_pipe_cleaner/hf_format/samples_8192.0 /dev/shm/phi-4_limo_trainer_pipe_cleaner_cont
        --data-path /dev/shm/knowledge_processed.jsonl \
        --data-path ./some_product_puzzle_tokenized_qwen1.5b.jsonl \
        --data-path ./mihir_prob.jsonl \
        --output-dir /new_data/experiments_rh/mihir_prob_qwen1.5b_v2     \
torchrun --nnodes=1 --nproc-per-node=8 train.py   \
        --output-dir /new_data/experiments_rh/siddantv2/     \
        --data-path ./siddhant.jsonl \
        --model-name-or-path Qwen/Qwen2.5-1.5B-instruct \
        --min-samples-per-checkpoint 10000      \
        --num-warmup-steps 20 \
        --max-tokens-per-gpu 60000              \
        --batch-size 128                       \
        --use-liger-kernels                    \
        --seed 893                               \
        --fsdp-sharding-strategy FULL_SHARD \
        --learning-rate 6e-6
'''