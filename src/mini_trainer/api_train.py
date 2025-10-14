"""API wrapper for mini_trainer that provides programmatic training interface."""

import subprocess
import os
import json
import logging
from pathlib import Path

from mini_trainer.training_types import TorchrunArgs, TrainingArgs


logger = logging.getLogger(__name__)


class StreamablePopen:
    """A wrapper for subprocess.Popen that streams output in real-time."""
    
    def __init__(self, log_file: str, command: list):
        self.log_file = log_file
        self.command = command
        self.process = None
        
    def listen(self):
        """Start the process and stream output."""
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)
        
        with open(self.log_file, "w") as f:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    print(line, end='')
                    f.write(line)
                    f.flush()
            
            self.process.wait()
    
    def poll(self):
        """Check if process has finished."""
        if self.process:
            return self.process.poll()
        return None
    
    def terminate(self):
        """Terminate the process."""
        if self.process:
            self.process.terminate()
    
    def wait(self, timeout=None):
        """Wait for process to finish."""
        if self.process:
            return self.process.wait(timeout=timeout)
    
    def kill(self):
        """Kill the process."""
        if self.process:
            self.process.kill()


def run_training(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    """
    Wrapper around the main training job that calls torchrun.
    
    Args:
        torch_args: Torchrun configuration for distributed training
        train_args: Training configuration parameters
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting training setup...")
    
    # Ensure output directory exists
    output_path = Path(train_args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build torchrun command
    train_script = Path(__file__).parent / "train.py"

    command = [
        "torchrun",
        f"--nnodes={torch_args.nnodes}",
        f"--node-rank={torch_args.node_rank}",
        f"--nproc-per-node={torch_args.nproc_per_node}",
        f"--rdzv-id={torch_args.rdzv_id}",
    ]

    if torch_args.master_addr and torch_args.rdzv_endpoint:
        raise ValueError("Provide either `rdzv_endpoint` OR `master_addr`, not both.")

    if torch_args.master_addr:
        # master-addr + master-port are only compatible with the static backend
        # so here we pass it explicitly
        command += [
            f"--master-addr={torch_args.master_addr}",
            "--rdzv-backend=static"
        ]
        if torch_args.master_port:
            command += [f"--master-port={torch_args.master_port}"]

    elif torch_args.rdzv_endpoint:
        command += [f"--rdzv-endpoint={torch_args.rdzv_endpoint}"]
    else:
        command += ["--standalone"]

    command.extend([
        str(train_script),
        f"--model-name-or-path={train_args.model_name_or_path}",
        f"--data-path={train_args.data_path}",
        f"--batch-size={train_args.batch_size}",
        f"--max-tokens-per-gpu={train_args.max_tokens_per_gpu}",
        f"--learning-rate={train_args.learning_rate}",
        f"--num-warmup-steps={train_args.num_warmup_steps}",
        f"--lr-scheduler={train_args.lr_scheduler}",
        f"--lr-scheduler-kwargs={json.dumps(train_args.lr_scheduler_kwargs) if train_args.lr_scheduler_kwargs else '{}'}",
        f"--seed={train_args.seed}",
        f"--output-dir={train_args.output_dir}",
        f"--training-mode={train_args.training_mode.value}",
        f"--max-epochs={train_args.max_epochs}",
        f"--max-steps={train_args.max_steps}",
        f"--max-tokens={train_args.max_tokens}",
        f"--train-dtype={train_args.train_dtype}",
    ])

    
    # wandb-related arguments
    if train_args.wandb_project:
        command.append(f"--wandb-project={train_args.wandb_project}")
        if train_args.wandb_run_name:
            command.append(f"--wandb-run-name={train_args.wandb_run_name}")
        if train_args.wandb_entity:
            command.append(f"--wandb-entity={train_args.wandb_entity}")

    # validation-related arguments
    if train_args.validation_split > 0.0:
        command.append(f"--validation-split={train_args.validation_split}")
        if train_args.validation_frequency is not None:
            command.append(f"--validation-frequency={train_args.validation_frequency}")

        if train_args.save_best_val_loss:
            command.append(f"--save-best-val-loss")
            command.append(f"--val-loss-improvement-threshold={train_args.val_loss_improvement_threshold}")

    # Add optional min_samples_per_checkpoint if specified
    if train_args.min_samples_per_checkpoint is not None:
        command.append(f"--min-samples-per-checkpoint={train_args.min_samples_per_checkpoint}")
    
    # Add optional boolean flags
    if train_args.use_liger_kernels:
        command.append("--use-liger-kernels")
    
    if train_args.osft:
        if train_args.osft_unfreeze_rank_ratio is None:
            raise ValueError("osft_unfreeze_rank_ratio is required when osft is True")
        command.append("--osft")
        command.append(f"--osft-unfreeze-rank-ratio={train_args.osft_unfreeze_rank_ratio}")
        if train_args.osft_target_patterns:
            command.append(f"--osft-target-patterns={','.join(train_args.osft_target_patterns)}")
        if train_args.osft_upcast_dtype:
            command.append(f"--osft-upcast-dtype={train_args.osft_upcast_dtype}")
        if train_args.osft_output_dtype:
            command.append(f"--osft-output-dtype={train_args.osft_output_dtype}")
        if train_args.osft_memory_efficient_init:
            command.append("--osft-memory-efficient-init")
    
    if train_args.checkpoint_at_epoch:
        command.append("--checkpoint-at-epoch")
    
    if train_args.save_final_checkpoint:
        command.append("--save-final-checkpoint")
 
    if train_args.save_dtype:
        command.append(f"--save-dtype={train_args.save_dtype}")
    
    logger.info("Running training command as subprocess: %s", " ".join(command))
    
    # Run the training process
    log_file = output_path / f"training_log_node{torch_args.node_rank}.log"
    process = None
    interrupt = None
    failure = False
    
    try:
        process = StreamablePopen(str(log_file), command)
        print(f"Command: {' '.join(command)}")
        print(f"Logs will be saved to: {log_file}")
        process.listen()
    except KeyboardInterrupt as e:
        logger.info("Training subprocess interrupted by user.")
        interrupt = e
    except Exception as e:
        logger.error("Unexpected exception during training", exc_info=e)
        interrupt = e
    finally:
        if process is None:
            return
        
        failure = process.poll() != 0
        if not failure:
            logger.info("Training completed successfully! 🎉")
        else:
            logger.error("Training subprocess failed. Check logs for details.")
        
        process.terminate()
        try:
            logger.info("Waiting for process to exit (60s timeout)...")
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.error("Training subprocess did not terminate, sending SIGKILL.")
            process.kill()
        
        if interrupt:
            raise interrupt
        if failure:
            raise RuntimeError(
                f"Training failed. Please check the logs at {log_file} for details."
            )

