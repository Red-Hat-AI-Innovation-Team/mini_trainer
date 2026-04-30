"""Tests for full-state on-demand checkpointing."""

import os
import random
import signal
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mini_trainer.full_state_checkpoint import find_latest_full_state_checkpoint
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


from mini_trainer.full_state_checkpoint import FullStateCheckpointer


class TestFullStateCheckpointerSignals:
    def test_trigger_file_created_on_signal(self, tmp_path):
        trigger_path = tmp_path / "trigger"
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path / "ckpts"),
            rank=0,
            world_size=1,
            trigger_path=str(trigger_path),
        )
        ckpt.install_signal_handlers()
        os.kill(os.getpid(), signal.SIGUSR1)
        assert trigger_path.exists()
        ckpt.cleanup()

    def test_cleanup_removes_trigger_file(self, tmp_path):
        trigger_path = tmp_path / "trigger"
        trigger_path.touch()
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path / "ckpts"),
            rank=0,
            world_size=1,
            trigger_path=str(trigger_path),
        )
        ckpt.cleanup()
        assert not trigger_path.exists()

    def test_cleanup_noop_if_no_trigger_file(self, tmp_path):
        trigger_path = tmp_path / "trigger"
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path / "ckpts"),
            rank=0,
            world_size=1,
            trigger_path=str(trigger_path),
        )
        ckpt.cleanup()

    def test_all_signals_handled(self, tmp_path):
        trigger_path = tmp_path / "trigger"
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path / "ckpts"),
            rank=0,
            world_size=1,
            trigger_path=str(trigger_path),
        )
        ckpt.install_signal_handlers()
        os.kill(os.getpid(), signal.SIGUSR2)
        assert trigger_path.exists()
        ckpt.cleanup()


class TestFullStateCheckpointerTriggerDetection:
    def test_should_save_false_when_no_trigger(self, tmp_path):
        trigger_path = tmp_path / "trigger"
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path / "ckpts"),
            rank=0,
            world_size=1,
            trigger_path=str(trigger_path),
        )
        with patch("mini_trainer.full_state_checkpoint.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            assert ckpt.should_save(torch.device("cpu")) is False

    def test_should_save_true_when_trigger_exists(self, tmp_path):
        trigger_path = tmp_path / "trigger"
        trigger_path.touch()
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path / "ckpts"),
            rank=0,
            world_size=1,
            trigger_path=str(trigger_path),
        )
        with patch("mini_trainer.full_state_checkpoint.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            assert ckpt.should_save(torch.device("cpu")) is True


class TestFullStateCheckpointerMetadata:
    def test_save_training_metadata_creates_file(self, tmp_path):
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path),
            rank=0,
            world_size=1,
            trigger_path=str(tmp_path / "trigger"),
        )
        training_state = {
            "step": 10,
            "epoch": 1,
            "total_samples_accumulated": 100,
            "total_tokens_processed": 5000,
            "last_validation_loss": 2.5,
        }
        checkpointer_state = {
            "last_saved_samples": 80,
            "last_frequency_saved_samples": 80,
            "best_val_loss": 2.7,
        }
        lr_scheduler_state = {"last_epoch": 10, "_step_count": 11}

        save_dir = tmp_path / "step_10"
        save_dir.mkdir()

        ckpt._save_metadata(
            save_dir=save_dir,
            training_state=training_state,
            checkpointer_state=checkpointer_state,
            lr_scheduler_state=lr_scheduler_state,
            sampler_epoch=1,
        )

        meta_path = save_dir / "training_state.pt"
        assert meta_path.exists()
        meta = torch.load(meta_path, weights_only=False)
        assert meta["step"] == 10
        assert meta["epoch"] == 1
        assert meta["total_samples_accumulated"] == 100
        assert meta["total_tokens_processed"] == 5000
        assert meta["last_validation_loss"] == 2.5
        assert meta["checkpointer_state"]["best_val_loss"] == 2.7
        assert meta["lr_scheduler_state"]["last_epoch"] == 10
        assert meta["sampler_epoch"] == 1

    def test_save_rng_state_creates_per_rank_file(self, tmp_path):
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path),
            rank=3,
            world_size=8,
            trigger_path=str(tmp_path / "trigger"),
        )
        save_dir = tmp_path / "step_5"
        save_dir.mkdir()

        ckpt._save_rng_states(save_dir)

        rng_path = save_dir / "rng_state_rank_3.pt"
        assert rng_path.exists()
        rng = torch.load(rng_path, weights_only=False)
        assert "torch_rng" in rng
        assert "python_rng" in rng
        assert "numpy_rng" in rng

    def test_metadata_roundtrip(self, tmp_path):
        ckpt = FullStateCheckpointer(
            checkpoint_dir=str(tmp_path),
            rank=0,
            world_size=1,
            trigger_path=str(tmp_path / "trigger"),
        )
        save_dir = tmp_path / "step_7"
        save_dir.mkdir()

        original_state = {
            "step": 7,
            "epoch": 0,
            "total_samples_accumulated": 42,
            "total_tokens_processed": 2100,
            "last_validation_loss": None,
        }
        ckpt._save_metadata(
            save_dir=save_dir,
            training_state=original_state,
            checkpointer_state={"last_saved_samples": 0, "last_frequency_saved_samples": 0, "best_val_loss": None},
            lr_scheduler_state={"_step_count": 8},
            sampler_epoch=0,
        )
        ckpt._save_rng_states(save_dir)

        loaded_meta = torch.load(save_dir / "training_state.pt", weights_only=False)
        loaded_rng = torch.load(save_dir / "rng_state_rank_0.pt", weights_only=False)

        assert loaded_meta["step"] == 7
        assert loaded_meta["last_validation_loss"] is None
        assert loaded_rng["torch_rng"] is not None


class TestFullStateCheckpointerLoad:
    def test_load_metadata(self, tmp_path):
        save_dir = tmp_path / "step_5"
        save_dir.mkdir()
        metadata = {
            "step": 5,
            "epoch": 0,
            "total_samples_accumulated": 50,
            "total_tokens_processed": 2500,
            "last_validation_loss": 3.1,
            "checkpointer_state": {
                "last_saved_samples": 40,
                "last_frequency_saved_samples": 40,
                "best_val_loss": None,
            },
            "lr_scheduler_state": {"_step_count": 6},
            "sampler_epoch": 0,
        }
        torch.save(metadata, save_dir / "training_state.pt")

        loaded = FullStateCheckpointer.load_metadata(save_dir)
        assert loaded["step"] == 5
        assert loaded["total_samples_accumulated"] == 50
        assert loaded["checkpointer_state"]["last_saved_samples"] == 40

    def test_load_rng_states(self, tmp_path):
        save_dir = tmp_path / "step_5"
        save_dir.mkdir()

        original_torch_rng = torch.get_rng_state()
        original_python_rng = random.getstate()
        original_numpy_rng = np.random.get_state()

        rng_state = {
            "torch_rng": original_torch_rng,
            "python_rng": original_python_rng,
            "numpy_rng": original_numpy_rng,
        }
        torch.save(rng_state, save_dir / "rng_state_rank_0.pt")

        # Advance RNG states
        torch.randn(100)
        random.random()
        np.random.rand(100)

        # Restore
        FullStateCheckpointer.load_rng_states(save_dir, rank=0)

        assert torch.equal(torch.get_rng_state(), original_torch_rng)

    def test_load_metadata_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FullStateCheckpointer.load_metadata(tmp_path / "nonexistent")

    def test_load_rng_states_missing_file(self, tmp_path):
        save_dir = tmp_path / "step_5"
        save_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            FullStateCheckpointer.load_rng_states(save_dir, rank=99)


class TestFindLatestFullStateCheckpoint:
    def _make_checkpoint(self, base_dir, step):
        step_dir = base_dir / "full_state_checkpoints" / f"step_{step}"
        step_dir.mkdir(parents=True)
        torch.save({"step": step}, step_dir / "training_state.pt")
        return step_dir

    def test_returns_none_when_no_checkpoint_dir(self, tmp_path):
        assert find_latest_full_state_checkpoint(tmp_path) is None

    def test_returns_none_when_empty_checkpoint_dir(self, tmp_path):
        (tmp_path / "full_state_checkpoints").mkdir()
        assert find_latest_full_state_checkpoint(tmp_path) is None

    def test_finds_single_checkpoint(self, tmp_path):
        step_dir = self._make_checkpoint(tmp_path, 10)
        assert find_latest_full_state_checkpoint(tmp_path) == str(step_dir)

    def test_finds_latest_among_multiple(self, tmp_path):
        self._make_checkpoint(tmp_path, 5)
        self._make_checkpoint(tmp_path, 20)
        self._make_checkpoint(tmp_path, 10)
        result = find_latest_full_state_checkpoint(tmp_path)
        assert result.endswith("step_20")

    def test_skips_incomplete_checkpoints(self, tmp_path):
        valid = self._make_checkpoint(tmp_path, 5)
        incomplete = tmp_path / "full_state_checkpoints" / "step_100"
        incomplete.mkdir(parents=True)
        assert find_latest_full_state_checkpoint(tmp_path) == str(valid)

    def test_skips_non_step_directories(self, tmp_path):
        valid = self._make_checkpoint(tmp_path, 10)
        other = tmp_path / "full_state_checkpoints" / "not_a_step"
        other.mkdir(parents=True)
        torch.save({}, other / "training_state.pt")
        assert find_latest_full_state_checkpoint(tmp_path) == str(valid)

    def test_skips_non_numeric_step_suffix(self, tmp_path):
        valid = self._make_checkpoint(tmp_path, 7)
        malformed = tmp_path / "full_state_checkpoints" / "step_latest"
        malformed.mkdir(parents=True)
        torch.save({}, malformed / "training_state.pt")
        assert find_latest_full_state_checkpoint(tmp_path) == str(valid)
