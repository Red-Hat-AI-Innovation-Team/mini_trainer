# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
import asyncio
import json
import sys
import threading
import torch.distributed as dist

# Third Party
import aiofiles
from rich.console import Console
from tqdm import tqdm

# Local imports
from mini_trainer import wandb_wrapper, mlflow_wrapper
from mini_trainer.wandb_wrapper import check_wandb_available
from mini_trainer.mlflow_wrapper import check_mlflow_available


class AsyncStructuredLogger:
    def __init__(
        self, file_name="training_log.jsonl", use_wandb=False, use_mlflow=False
    ):
        self.file_name = file_name

        # wandb init is a special case -- if it is requested but unavailable,
        # we should error out early
        if use_wandb:
            check_wandb_available("initialize wandb")
        self.use_wandb = use_wandb

        # mlflow init - same pattern as wandb
        if use_mlflow:
            check_mlflow_available("initialize mlflow")
        self.use_mlflow = use_mlflow

        # Rich console for prettier output (force_terminal=True works with subprocess streaming)
        self.console = Console(force_terminal=True, force_interactive=False)

        # tqdm for state tracking (lazy init to avoid early printing)
        self.train_pbar = None
        self.train_bar_format = None

        self.logs = []
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(
            target=self._run_event_loop, args=(self.loop,), daemon=True
        )
        t.start()
        asyncio.run_coroutine_threadsafe(self._initialize_log_file(), self.loop)

    def _run_event_loop(self, loop):
        asyncio.set_event_loop(loop)  #
        loop.run_forever()

    async def _initialize_log_file(self):
        self.logs = []
        try:
            async with aiofiles.open(self.file_name, "r") as f:
                async for line in f:
                    if line.strip():  # Avoid empty lines
                        self.logs.append(json.loads(line.strip()))
        except FileNotFoundError:
            # File does not exist but the first log will create it.
            pass

    async def log(self, data):
        """logs a dictionary as a new line in a jsonl file with a timestamp"""
        try:
            if not isinstance(data, dict):
                raise ValueError("Logged data must be a dictionary")
            data["timestamp"] = datetime.now().isoformat()
            self.logs.append(data)
            await self._write_logs_to_file(data)

            # log to wandb/mlflow if enabled, but only log this on the MAIN rank
            # Guard rank checks when the process group isn't initialized (single-process runs)
            is_rank0 = not dist.is_initialized() or dist.get_rank() == 0

            # wandb already handles timestamps so no need to include
            if self.use_wandb and is_rank0:
                wandb_data = {k: v for k, v in data.items() if k != "timestamp"}
                wandb_wrapper.log(wandb_data)

            # log to mlflow if enabled, only on MAIN rank
            # Filter out step from data since it's passed as a separate argument
            if self.use_mlflow and is_rank0:
                step = data.get("step")
                mlflow_data = {
                    k: v for k, v in data.items() if k not in ("timestamp", "step")
                }
                mlflow_wrapper.log(mlflow_data, step=step)
        except Exception as e:
            print(f"\033[1;38;2;0;255;255mError logging data: {e}\033[0m")

    async def _write_logs_to_file(self, data):
        """appends to the log instead of writing the whole log each time"""
        async with aiofiles.open(self.file_name, "a") as f:
            await f.write(json.dumps(data, indent=None) + "\n")

    def log_sync(self, data: dict):
        """Runs the log coroutine non-blocking and prints metrics with tqdm-styled progress bar.

        Args:
            data: Dictionary of metrics to log. Will automatically print a tqdm-formatted
                  progress bar with ANSI colors if step and steps_per_epoch are present.
        """
        if not isinstance(data, dict):
            raise ValueError("Logged data must be a dictionary")

        # Print to console synchronously, but only on rank 0
        # to avoid duplicate outputs in distributed training
        should_print = not dist.is_initialized() or dist.get_rank() == 0
        if should_print:
            data_with_timestamp = {**data, "timestamp": datetime.now().isoformat()}

            # Print the JSON using Rich for syntax highlighting
            self.console.print_json(json.dumps(data_with_timestamp))

            # Print tqdm-styled progress bar after JSON (prints as new line each time)
            # This works correctly with subprocess streaming
            if "step" in data and "steps_per_epoch" in data and "epoch" in data:
                # Initialize tqdm on first call (lazy init to avoid early printing)
                if self.train_pbar is None:
                    # Simple bar format with ANSI colors - we'll add epoch and metrics manually
                    self.train_bar_format = (
                        "{bar} "
                        "\033[33m{percentage:3.0f}%\033[0m │ "
                        "\033[37m{n}/{total}\033[0m"
                    )
                    self.train_pbar = tqdm(
                        total=data["steps_per_epoch"],
                        bar_format=self.train_bar_format,
                        ncols=None,
                        leave=False,
                        position=0,
                        file=sys.stdout,
                        ascii="━╺─",  # custom characters matching Rich style
                        disable=True,  # disable auto-display, we'll manually call display()
                    )

                # Reset tqdm if we're in a new epoch
                current_step_in_epoch = (data["step"] - 1) % data["steps_per_epoch"] + 1
                if current_step_in_epoch == 1:
                    self.train_pbar.reset(total=data["steps_per_epoch"])

                # Update tqdm position
                self.train_pbar.n = current_step_in_epoch

                # Manually format the complete progress line with metrics using format_meter
                bar_str = self.train_pbar.format_meter(
                    n=current_step_in_epoch,
                    total=data["steps_per_epoch"],
                    elapsed=0,  # we don't track elapsed time
                    ncols=None,
                    bar_format=self.train_bar_format,
                    ascii="━╺─",
                )

                # Prepend the epoch number (1-indexed)
                epoch_prefix = f"\033[1;34mEpoch {data['epoch'] + 1}:\033[0m "
                bar_str = epoch_prefix + bar_str

                # Add the metrics to the bar string
                metrics_str = (
                    f" │ \033[32mloss:\033[0m \033[37m{data['loss']:.4f}\033[0m"
                    f" │ \033[32mlr:\033[0m \033[37m{data['lr']:.2e}\033[0m"
                    f" │ \033[35m{data['tokens_per_second']:.0f}\033[0m \033[2mtok/s\033[0m"
                )

                # Print the complete line
                print(bar_str + metrics_str, file=sys.stdout, flush=True)

        # Run async logging for file and wandb
        asyncio.run_coroutine_threadsafe(self.log(data), self.loop)

    def __repr__(self):
        return f"<AsyncStructuredLogger(file_name={self.file_name})>"
