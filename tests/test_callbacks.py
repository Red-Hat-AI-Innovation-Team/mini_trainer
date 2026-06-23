"""
Test suite for the callback system.

Tests async fire-and-forget dispatch, per-rank context exposure, mutable
context with snapshot isolation, class-based interface, and serialization
for crossing the torchrun subprocess boundary.
"""

import asyncio
import logging
import threading
import time

import pytest

from mini_trainer.callbacks import (
    CallbackManager,
    TrainerCallback,
    TrainingContext,
    deserialize_callback,
    deserialize_callbacks_from_cli,
    serialize_callback,
    serialize_callbacks_for_cli,
)


class TestTrainingContext:
    """Test suite for TrainingContext dataclass."""

    def test_default_values(self):
        """Test TrainingContext default field values."""
        ctx = TrainingContext()
        assert ctx.step == 0
        assert ctx.epoch == 0
        assert ctx.loss is None
        assert ctx.batch_metrics == {}
        assert ctx.world_size == 1
        assert ctx.is_local_process_zero is True
        assert ctx.is_world_process_zero is True

    def test_mutable_fields(self):
        """Test that TrainingContext fields can be updated in place."""
        ctx = TrainingContext()
        ctx.step = 5
        ctx.epoch = 2
        ctx.loss = 0.42
        assert ctx.step == 5
        assert ctx.epoch == 2
        assert ctx.loss == 0.42

    def test_all_fields_populated(self):
        """Test TrainingContext with all fields explicitly set."""
        ctx = TrainingContext(
            hook_name="on_step_end",
            step=42,
            epoch=2,
            total_samples=1000,
            total_tokens=50000,
            loss=0.5,
            learning_rate=1e-4,
            grad_norm=1.2,
            batch_metrics={"loss": 0.5, "lr": 1e-4},
            val_metrics={"val_loss": 0.6},
            checkpoint_path="/tmp/ckpt",
            output_dir="/tmp/output",
            model_name_or_path="meta-llama/Llama-3.1-8B",
            training_mode="epoch",
            max_epochs=3,
            max_steps=0,
            max_tokens=0,
            world_size=8,
        )
        assert ctx.step == 42
        assert ctx.total_tokens == 50000
        assert ctx.checkpoint_path == "/tmp/ckpt"


class _StepRecorderCallback(TrainerCallback):
    def __init__(self):
        self.event = threading.Event()
        self.captured = {}

    def on_train_begin(self, context):
        self.captured["step"] = context.step
        self.captured["hook"] = context.hook_name
        self.event.set()


class _SignalCallback(TrainerCallback):
    def __init__(self):
        self.event = threading.Event()

    def on_step_end(self, context):
        self.event.set()


class _ExplodingCallback(TrainerCallback):
    def __init__(self):
        self.event = threading.Event()

    def on_step_end(self, context):
        self.event.set()
        raise RuntimeError("boom")

    def on_log(self, context):
        self.event.set()
        raise ValueError("test error 12345")


class _AsyncCallback(TrainerCallback):
    def __init__(self):
        self.event = threading.Event()
        self.captured = {}

    async def on_epoch_end(self, context):
        self.captured["step"] = context.step
        self.event.set()


class _SlowTrainEndCallback(TrainerCallback):
    def __init__(self):
        self.result = {}

    def on_train_end(self, context):
        time.sleep(0.5)
        self.result["done"] = True


class _SaveRecorderCallback(TrainerCallback):
    def __init__(self):
        self.event = threading.Event()
        self.captured = {}

    def on_save(self, context):
        self.captured["checkpoint_path"] = context.checkpoint_path
        self.captured["step"] = context.step
        self.event.set()


class _StepEndRecorderCallback(TrainerCallback):
    def __init__(self):
        self.event = threading.Event()
        self.captured = {}

    def on_step_end(self, context):
        self.captured["step"] = context.step
        self.event.set()


class _MultiHookCallback(TrainerCallback):
    """Callback that implements multiple hooks."""

    def __init__(self):
        self.events = []
        self.done = threading.Event()

    def on_train_begin(self, context):
        self.events.append("on_train_begin")

    def on_log(self, context):
        self.events.append("on_log")

    def on_train_end(self, context):
        self.events.append("on_train_end")
        self.done.set()


class TestCallbackManager:
    """Test suite for CallbackManager dispatch and lifecycle."""

    def test_add_callback(self):
        """Test adding a TrainerCallback instance."""
        mgr = CallbackManager()
        cb = _SignalCallback()
        mgr.add_callback(cb)
        assert mgr.has_callbacks("on_step_end")

    def test_add_callback_rejects_non_instance(self):
        """Test that passing a class instead of an instance raises TypeError."""
        mgr = CallbackManager()
        with pytest.raises(TypeError, match="Expected a TrainerCallback instance"):
            mgr.add_callback(_SignalCallback)

    def test_remove_callback_by_instance(self):
        """Test removing a specific callback instance."""
        mgr = CallbackManager()
        cb = _SignalCallback()
        mgr.add_callback(cb)
        assert mgr.has_callbacks("on_step_end")
        mgr.remove_callback(cb)
        assert not mgr.has_callbacks("on_step_end")

    def test_remove_callback_by_type(self):
        """Test removing all callbacks of a given type."""
        mgr = CallbackManager()
        mgr.add_callback(_SignalCallback())
        mgr.add_callback(_SignalCallback())
        assert mgr.has_callbacks("on_step_end")
        mgr.remove_callback(_SignalCallback)
        assert not mgr.has_callbacks("on_step_end")

    def test_fire_invokes_callback(self):
        """Test that fire() dispatches the callback with correct context."""
        cb = _StepRecorderCallback()
        mgr = CallbackManager()
        mgr.context.step = 7
        mgr.add_callback(cb)
        mgr.fire("on_train_begin")
        assert cb.event.wait(timeout=5), "Callback was not invoked within timeout"
        assert cb.captured["step"] == 7
        assert cb.captured["hook"] == "on_train_begin"

    def test_fire_multiple_callbacks(self):
        """Test that fire() invokes all registered callbacks for a hook."""
        cb1 = _SignalCallback()
        cb2 = _SignalCallback()
        mgr = CallbackManager()
        mgr.add_callback(cb1)
        mgr.add_callback(cb2)
        mgr.fire("on_step_end")
        assert cb1.event.wait(timeout=5)
        assert cb2.event.wait(timeout=5)

    def test_fire_on_all_ranks(self):
        """Test that callbacks fire regardless of rank; rank info is on context."""
        cb = _StepRecorderCallback()
        mgr = CallbackManager()
        mgr.context.is_world_process_zero = False
        mgr.context.is_local_process_zero = False
        mgr.add_callback(cb)
        mgr.fire("on_train_begin")
        assert cb.event.wait(timeout=5), "Callback should fire on non-rank-0 processes"

    def test_rank_info_on_context(self):
        """Test that is_local_process_zero and is_world_process_zero are on context."""
        mgr = CallbackManager()
        assert mgr.context.is_local_process_zero is True
        assert mgr.context.is_world_process_zero is True
        mgr.context.is_local_process_zero = False
        mgr.context.is_world_process_zero = False
        assert mgr.context.is_local_process_zero is False
        assert mgr.context.is_world_process_zero is False

    def test_fire_exception_swallowed(self):
        """Test that callback exceptions do not propagate to the caller."""
        cb = _ExplodingCallback()
        mgr = CallbackManager()
        mgr.add_callback(cb)
        mgr.fire("on_step_end")
        assert cb.event.wait(timeout=5), "Callback should have been invoked"
        time.sleep(0.1)

    def test_fire_exception_logged(self):
        """Test that callback exceptions are logged with the error message."""
        cb = _ExplodingCallback()
        logged_messages = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                logged_messages.append(self.format(record))

        cb_logger = logging.getLogger("mini_trainer.callbacks")
        handler = CaptureHandler()
        cb_logger.addHandler(handler)
        cb_logger.setLevel(logging.ERROR)
        try:
            mgr = CallbackManager()
            mgr.add_callback(cb)
            mgr.fire("on_log")
            cb.event.wait(timeout=5)
            time.sleep(0.3)
            assert any("test error 12345" in msg for msg in logged_messages)
        finally:
            cb_logger.removeHandler(handler)

    def test_has_callbacks_detects_overridden(self):
        """Test has_callbacks() returns True only for overridden hooks."""
        mgr = CallbackManager()
        mgr.add_callback(_SignalCallback())
        assert mgr.has_callbacks("on_step_end")
        assert not mgr.has_callbacks("on_train_begin")
        assert not mgr.has_callbacks("on_log")

    def test_has_callbacks_no_callbacks(self):
        """Test has_callbacks() returns False when no callbacks registered."""
        mgr = CallbackManager()
        assert not mgr.has_callbacks("on_train_begin")

    def test_fire_no_callbacks_is_noop(self):
        """Test that fire() with no registered callbacks is a safe no-op."""
        mgr = CallbackManager()
        mgr.fire("on_train_begin")

    def test_async_callback_support(self):
        """Test that async (coroutine) callbacks are awaited correctly."""
        cb = _AsyncCallback()
        mgr = CallbackManager()
        mgr.context.step = 99
        mgr.add_callback(cb)
        mgr.fire("on_epoch_end")
        assert cb.event.wait(timeout=5)
        assert cb.captured["step"] == 99

    def test_on_train_end_waits_for_completion(self):
        """Test that on_train_end blocks until callbacks finish."""
        cb = _SlowTrainEndCallback()
        mgr = CallbackManager()
        mgr.add_callback(cb)
        mgr.fire("on_train_end")
        assert cb.result.get("done"), "on_train_end should wait for callback to complete"

    def test_fire_kwargs_override_context(self):
        """Test that kwargs passed to fire() override context fields in the snapshot."""
        cb = _SaveRecorderCallback()
        mgr = CallbackManager()
        mgr.context.step = 10
        mgr.context.checkpoint_path = None
        mgr.add_callback(cb)
        mgr.fire("on_save", checkpoint_path="/tmp/ckpt-10")
        assert cb.event.wait(timeout=5)
        assert cb.captured["checkpoint_path"] == "/tmp/ckpt-10"
        assert cb.captured["step"] == 10
        assert mgr.context.checkpoint_path is None, "kwargs should not mutate the shared context"

    def test_snapshot_isolation(self):
        """Test that callbacks receive a snapshot, not a reference to the shared context."""
        cb = _StepEndRecorderCallback()
        mgr = CallbackManager()
        mgr.context.step = 42
        mgr.add_callback(cb)
        mgr.fire("on_step_end")
        mgr.context.step = 999
        assert cb.event.wait(timeout=5)
        assert cb.captured["step"] == 42, "Callback should see the step value at fire() time"

    def test_multi_hook_callback(self):
        """Test that a single callback can implement multiple hooks."""
        cb = _MultiHookCallback()
        mgr = CallbackManager()
        mgr.add_callback(cb)
        mgr.fire("on_train_begin")
        mgr.fire("on_log")
        mgr.fire("on_train_end")
        assert cb.done.wait(timeout=5)
        assert "on_train_begin" in cb.events
        assert "on_log" in cb.events
        assert "on_train_end" in cb.events


class _StandaloneCallback(TrainerCallback):
    def on_train_begin(self, context):
        import os

        _ = os.getpid()


class _InlineImportCallback(TrainerCallback):
    def on_log(self, context):
        import json

        _ = json.dumps({"step": context.step})


class TestSerialization:
    """Test suite for callback serialization across the torchrun subprocess boundary."""

    def test_serialize_deserialize_roundtrip(self):
        """Test single callback serialize/deserialize round-trip."""
        original = _StandaloneCallback()
        encoded = serialize_callback(original)
        restored = deserialize_callback(encoded)
        assert isinstance(restored, TrainerCallback)
        ctx = TrainingContext(hook_name="test")
        restored.on_train_begin(ctx)

    def test_serialize_self_contained_callback(self):
        """Test serialization of a callback with inline imports."""
        original = _InlineImportCallback()
        encoded = serialize_callback(original)
        restored = deserialize_callback(encoded)
        ctx = TrainingContext(hook_name="test", step=42)
        restored.on_log(ctx)

    def test_serialize_callbacks_for_cli_roundtrip(self):
        """Test full callbacks list serialize/deserialize for CLI transport."""
        cb1 = _StandaloneCallback()
        cb2 = _InlineImportCallback()
        encoded = serialize_callbacks_for_cli([cb1, cb2])
        decoded = deserialize_callbacks_from_cli(encoded)
        assert len(decoded) == 2
        assert all(isinstance(cb, TrainerCallback) for cb in decoded)
        ctx = TrainingContext(hook_name="test")
        decoded[0].on_train_begin(ctx)
        decoded[1].on_log(ctx)

    def test_deserialize_invalid_base64(self):
        """Test that invalid base64 input raises an exception."""
        with pytest.raises(Exception):
            deserialize_callback("not-valid-base64!!!")

    def test_empty_callbacks_list(self):
        """Test serialization of an empty callbacks list."""
        encoded = serialize_callbacks_for_cli([])
        decoded = deserialize_callbacks_from_cli(encoded)
        assert decoded == []


class TestApiTrainSerialization:
    """Test suite for callback integration with TrainingArgs and api_train."""

    def test_training_args_with_callbacks(self):
        """Test that TrainingArgs accepts a list of TrainerCallback instances."""
        from mini_trainer.training_types import TrainingArgs

        class MyCallback(TrainerCallback):
            def on_train_begin(self, context):
                pass

        args = TrainingArgs(
            model_name_or_path="test-model",
            data_path="test.jsonl",
            batch_size=4,
            max_tokens_per_gpu=1024,
            learning_rate=1e-4,
            output_dir="/tmp/test_output",
            callbacks=[MyCallback()],
        )
        assert args.callbacks is not None
        assert len(args.callbacks) == 1
        assert isinstance(args.callbacks[0], TrainerCallback)

    def test_training_args_no_callbacks_default(self):
        """Test that callbacks defaults to None when not provided."""
        from mini_trainer.training_types import TrainingArgs

        args = TrainingArgs(
            model_name_or_path="test-model",
            data_path="test.jsonl",
            batch_size=4,
            max_tokens_per_gpu=1024,
            learning_rate=1e-4,
            output_dir="/tmp/test_output",
        )
        assert args.callbacks is None
