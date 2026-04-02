"""Tests for full-state on-demand checkpointing."""

import os
import random
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist

from mini_trainer.training_types import TrainingArgs


class TestTrainingArgsCheckpointFields:
    def test_on_demand_checkpointing_defaults_false(self):
        args = TrainingArgs(
            model_name_or_path="test",
            data_path="test.jsonl",
            batch_size=1,
            max_tokens_per_gpu=512,
            learning_rate=1e-4,
            output_dir="/tmp/test",
        )
        assert args.on_demand_checkpointing is False

    def test_resume_from_full_state_checkpoint_defaults_none(self):
        args = TrainingArgs(
            model_name_or_path="test",
            data_path="test.jsonl",
            batch_size=1,
            max_tokens_per_gpu=512,
            learning_rate=1e-4,
            output_dir="/tmp/test",
        )
        assert args.resume_from_full_state_checkpoint is None

    def test_on_demand_checkpointing_can_be_set(self):
        args = TrainingArgs(
            model_name_or_path="test",
            data_path="test.jsonl",
            batch_size=1,
            max_tokens_per_gpu=512,
            learning_rate=1e-4,
            output_dir="/tmp/test",
            on_demand_checkpointing=True,
        )
        assert args.on_demand_checkpointing is True

    def test_resume_path_can_be_set(self):
        args = TrainingArgs(
            model_name_or_path="test",
            data_path="test.jsonl",
            batch_size=1,
            max_tokens_per_gpu=512,
            learning_rate=1e-4,
            output_dir="/tmp/test",
            resume_from_full_state_checkpoint="/tmp/checkpoint",
        )
        assert args.resume_from_full_state_checkpoint == "/tmp/checkpoint"
