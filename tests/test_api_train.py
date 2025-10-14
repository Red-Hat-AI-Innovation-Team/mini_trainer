"""Unit tests for the mini_trainer API wrapper."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

from mini_trainer.api_train import (
    run_training,
    StreamablePopen
)
from mini_trainer.training_types import (
    TorchrunArgs,
    TrainingArgs,
    TrainingMode
)


class TestDataclasses:
    """Test the dataclass configurations."""
    
    def test_torchrun_args_defaults(self):
        """Test TorchrunArgs default values."""
        args = TorchrunArgs()  # Use all defaults
        assert args.nnodes == 1
        assert args.nproc_per_node == 1
        assert args.node_rank == 0
        assert args.rdzv_id == 123
        assert args.rdzv_endpoint == None
        
        # Test with custom nproc_per_node only
        args = TorchrunArgs(nproc_per_node=8)
        assert args.nnodes == 1  # Should still use default
        assert args.nproc_per_node == 8
    
    def test_torchrun_args_custom(self):
        """Test TorchrunArgs with custom values."""
        args = TorchrunArgs(
            nnodes=2,
            nproc_per_node=4,
            node_rank=1,
            rdzv_id=123,
            rdzv_endpoint="localhost:9999"
        )
        assert args.nnodes == 2
        assert args.nproc_per_node == 4
        assert args.node_rank == 1
        assert args.rdzv_id == 123
        assert args.rdzv_endpoint == "localhost:9999"
    
    def test_training_args_defaults(self):
        """Test TrainingArgs default values."""
        args = TrainingArgs(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            data_path="test.jsonl",
            batch_size=1024,
            max_tokens_per_gpu=10000,
            learning_rate=5e-6,
            output_dir="./output"
        )
        assert args.model_name_or_path == "Qwen/Qwen2.5-1.5B-Instruct"
        assert args.data_path == "test.jsonl"
        assert args.batch_size == 1024
        assert args.max_tokens_per_gpu == 10000
        assert args.learning_rate == 5e-6
        assert args.num_warmup_steps == 0
        assert args.lr_scheduler == "cosine"
        assert args.seed == 42
        assert args.use_liger_kernels is False
        assert args.osft is False
        assert args.output_dir == "./output"
        assert args.min_samples_per_checkpoint is None
        assert args.training_mode == TrainingMode.EPOCH
        assert args.max_epochs == 1
        assert args.max_steps == 0
        assert args.max_tokens == 0
        assert args.checkpoint_at_epoch is False
        assert args.save_final_checkpoint is True
    
    def test_training_args_custom(self):
        """Test TrainingArgs with custom values."""
        args = TrainingArgs(
            model_name_or_path="gpt2",
            data_path="/path/to/data.jsonl",
            batch_size=512,
            max_tokens_per_gpu=5000,
            learning_rate=1e-4,
            output_dir="/custom/output",
            use_liger_kernels=True
        )
        assert args.model_name_or_path == "gpt2"
        assert args.data_path == "/path/to/data.jsonl"
        assert args.batch_size == 512
        assert args.max_tokens_per_gpu == 5000
        assert args.learning_rate == 1e-4
        assert args.output_dir == "/custom/output"
        assert args.use_liger_kernels is True


class TestStreamablePopen:
    """Test the StreamablePopen wrapper."""
    buffer_time = 1.0 # seconds
    
    def test_streamable_popen_success(self):
        """Test StreamablePopen with successful command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            command = ["echo", "test output"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check log file was created and contains output
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "test output" in content
            
            # Check process finished successfully
            assert popen.poll() == 0
    
    def test_streamable_popen_failure(self):
        """Test StreamablePopen with failing command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            command = ["ls", "/nonexistent/path/that/should/not/exist"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check process failed
            assert popen.poll() != 0


    def test_streamable_popen_stdout_capture(self):
        """Test StreamablePopen captures stdout correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stdout_test.log"
            command = ["python", "-c", "print('Hello stdout'); print('Line 2 stdout')"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check log file contains stdout
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "Hello stdout" in content
                assert "Line 2 stdout" in content
            
            assert popen.poll() == 0

    def test_streamable_popen_stderr_capture(self):
        """Test StreamablePopen captures stderr correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stderr_test.log"
            command = ["python", "-c", "import sys; print('Error message', file=sys.stderr); print('Another error', file=sys.stderr)"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check log file contains stderr
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "Error message" in content
                assert "Another error" in content
            
            assert popen.poll() == 0

    def test_streamable_popen_mixed_output(self):
        """Test StreamablePopen captures both stdout and stderr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "mixed_test.log"
            command = ["python", "-c", 
                      "import sys; "
                      "print('stdout line 1'); "
                      "print('stderr line 1', file=sys.stderr); "
                      "print('stdout line 2'); "
                      "print('stderr line 2', file=sys.stderr)"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check log file contains both outputs
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "stdout line 1" in content
                assert "stdout line 2" in content
                assert "stderr line 1" in content
                assert "stderr line 2" in content
            
            assert popen.poll() == 0

    def test_streamable_popen_stdout_realtime(self):
        """Test StreamablePopen captures stdout in real-time."""
        import time
        import threading
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stdout_realtime_test.log"
            # Script that outputs "1", "2", "3" with delays
            command = ["python", "-c", 
                      "import time, sys; "
                      "print('1', flush=True); "
                      f"time.sleep({self.buffer_time}); "
                      "print('2', flush=True); "
                      f"time.sleep({self.buffer_time}); "
                      "print('3', flush=True)"]
            
            popen = StreamablePopen(str(log_file), command)
            
            # Start listening in a separate thread
            listen_thread = threading.Thread(target=popen.listen)
            listen_thread.start()
            
            # Check that "1" appears first
            time.sleep(self.buffer_time)  # Increased delay for CI environments to start subprocess
            with open(log_file) as f:
                content = f.read()
                assert "1" in content, f"Expected '1' in content but got: {repr(content)}"
                assert "2" not in content
                assert "3" not in content
            
            # Check that "2" appears next
            time.sleep(self.buffer_time)  # Wait for "2" to be printed
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" in content, f"Expected '2' in content but got: {repr(content)}"
                assert "3" not in content
            
            # Check that "3" appears last
            time.sleep(self.buffer_time)  # Wait for "3" to be printed
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" in content
                assert "3" in content, f"Expected '3' in content but got: {repr(content)}"
            
            listen_thread.join()
            assert popen.poll() == 0

    def test_streamable_popen_stderr_realtime(self):
        """Test StreamablePopen captures stderr in real-time."""
        import time
        import threading
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stderr_realtime_test.log"
            # Script that outputs "1", "2", "3" to stderr with delays
            command = ["python", "-c", 
                      "import time, sys; "
                      "print('1', file=sys.stderr, flush=True); "
                      f"time.sleep({self.buffer_time}); "
                      "print('2', file=sys.stderr, flush=True); "
                      f"time.sleep({self.buffer_time}); "
                      "print('3', file=sys.stderr, flush=True)"]
            
            popen = StreamablePopen(str(log_file), command)
            
            # Start listening in a separate thread
            listen_thread = threading.Thread(target=popen.listen)
            listen_thread.start()
            
            # Check that "1" appears first
            time.sleep(self.buffer_time)  # Increased delay for CI environments to start subprocess
            with open(log_file) as f:
                content = f.read()
                assert "1" in content, f"Expected '1' in content but got: {repr(content)}"
                assert "2" not in content
                assert "3" not in content
            
            # Check that "2" appears next
            time.sleep(self.buffer_time)  # Wait for "2" to be printed
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" in content, f"Expected '2' in content but got: {repr(content)}"
                assert "3" not in content
            
            # Check that "3" appears last
            time.sleep(self.buffer_time)  # Wait for "3" to be printed
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" in content
                assert "3" in content, f"Expected '3' in content but got: {repr(content)}"
            
            listen_thread.join()
            assert popen.poll() == 0

    def test_streamable_popen_mixed_realtime(self):
        """Test StreamablePopen captures mixed stdout/stderr in real-time."""
        import time
        import threading
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "mixed_realtime_test.log"
            # Script that alternates between stdout and stderr with delays
            command = ["python", "-c", 
                      "import time, sys; "
                      "print('stdout-1', flush=True); "
                      f"time.sleep({self.buffer_time}); "
                      "print('stderr-1', file=sys.stderr, flush=True); "
                      f"time.sleep({self.buffer_time}); "
                      "print('stdout-2', flush=True); "
                      f"time.sleep({self.buffer_time}); "
                      "print('stderr-2', file=sys.stderr, flush=True)"]
            
            popen = StreamablePopen(str(log_file), command)
            
            # Start listening in a separate thread
            listen_thread = threading.Thread(target=popen.listen)
            listen_thread.start()
            
            # Check first stdout appears
            time.sleep(self.buffer_time)  # Increased initial delay for CI environments
            with open(log_file) as f:
                content = f.read()
                assert "stdout-1" in content, f"Expected 'stdout-1' in content but got: {repr(content)}"
                assert "stderr-1" not in content
                assert "stdout-2" not in content
                assert "stderr-2" not in content
            
            # Check first stderr appears
            time.sleep(self.buffer_time)
            with open(log_file) as f:
                content = f.read()
                assert "stdout-1" in content
                assert "stderr-1" in content, f"Expected 'stderr-1' in content but got: {repr(content)}"
                assert "stdout-2" not in content
                assert "stderr-2" not in content
            
            # Check second stdout appears
            time.sleep(self.buffer_time)
            with open(log_file) as f:
                content = f.read()
                assert "stdout-1" in content
                assert "stderr-1" in content
                assert "stdout-2" in content, f"Expected 'stdout-2' in content but got: {repr(content)}"
                assert "stderr-2" not in content
            
            # Check second stderr appears
            time.sleep(self.buffer_time)  # Increased delay to ensure last output is written
            with open(log_file) as f:
                content = f.read()
                assert "stdout-1" in content
                assert "stderr-1" in content
                assert "stdout-2" in content
                assert "stderr-2" in content, f"Expected 'stderr-2' in content but got: {repr(content)}"
            
            listen_thread.join()
            assert popen.poll() == 0


class TestRunTraining:
    """Test the run_training function."""
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_run_training_command_construction(self, mock_popen_class):
        """Test that run_training constructs the correct command."""
        mock_popen = MagicMock()
        mock_popen.poll.return_value = 0  # Success
        mock_popen_class.return_value = mock_popen
 
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(
                nnodes=2,
                nproc_per_node=4,
                node_rank=1,
                rdzv_id=999,
                rdzv_endpoint="master:1234"
            )
            train_args = TrainingArgs(
                model_name_or_path="my-model",
                data_path="/data/train.jsonl",
                batch_size=256,
                max_tokens_per_gpu=5000,
                learning_rate=2e-5,
                num_warmup_steps=100,
                lr_scheduler="cosine",
                seed=123,
                output_dir=tmpdir,
                use_liger_kernels=True,
                osft=True,
                osft_unfreeze_rank_ratio=0.5,
                min_samples_per_checkpoint=5000
            )
 
            run_training(torch_args, train_args)
 
            # Check that StreamablePopen was called with correct arguments
            assert mock_popen_class.called
            call_args = mock_popen_class.call_args
            log_file, command = call_args[0]
            
            # Verify log file path
            assert tmpdir in log_file
            assert "training_log_node1.log" in log_file
            
            # Verify command structure
            assert command[0] == "torchrun"
            assert "--nnodes=2" in command
            assert "--node-rank=1" in command
            assert "--nproc-per-node=4" in command
            assert "--rdzv-id=999" in command
            assert "--rdzv-endpoint=master:1234" in command
            
            # Verify training arguments
            assert "--model-name-or-path=my-model" in command
            assert "--data-path=/data/train.jsonl" in command
            assert "--batch-size=256" in command
            assert "--max-tokens-per-gpu=5000" in command
            assert "--learning-rate=2e-05" in command
            assert "--num-warmup-steps=100" in command
            assert "--lr-scheduler=cosine" in command
            assert "--seed=123" in command
            assert f"--output-dir={tmpdir}" in command
            # min_samples_per_checkpoint=5000 should be in the command
            assert "--min-samples-per-checkpoint=5000" in command
            assert "--use-liger-kernels" in command
            assert "--osft" in command
            assert "--osft-unfreeze-rank-ratio=0.5" in command
            
            # Verify listen was called
            mock_popen.listen.assert_called_once()

    @patch('mini_trainer.api_train.StreamablePopen')
    def test_run_training_osft_scenarios(self, mock_popen_class):
        """Test OSFT parameter validation scenarios."""
        mock_popen = MagicMock()
        mock_popen.poll.return_value = 0  # Success
        mock_popen_class.return_value = mock_popen

        with tempfile.TemporaryDirectory() as tmpdir:
            base_torch_args = TorchrunArgs(nproc_per_node=8)
            
            # Scenario 1: osft is not provided, this should succeed
            train_args_no_osft = TrainingArgs(
                model_name_or_path="my-model",
                data_path="/data/train.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=False  # Default case
            )
            
            run_training(base_torch_args, train_args_no_osft)
            
            # Verify command was constructed without osft flags
            call_args = mock_popen_class.call_args
            _, command = call_args[0]
            assert "--osft" not in command
            assert "--osft-unfreeze-rank-ratio" not in " ".join(command)
            
            # Reset mock for next test
            mock_popen_class.reset_mock()
            
            # Scenario 2: osft is passed but not unfreeze rank ratio, this should fail
            train_args_osft_no_ratio = TrainingArgs(
                model_name_or_path="my-model",
                data_path="/data/train.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=None  # Missing required parameter
            )
            
            with pytest.raises(ValueError, match="osft_unfreeze_rank_ratio is required when osft is True"):
                run_training(base_torch_args, train_args_osft_no_ratio)
            
            # Verify StreamablePopen was not called due to validation failure
            assert not mock_popen_class.called
            
            # Scenario 3: osft is passed with unfreeze rank ratio, this should succeed
            train_args_osft_with_ratio = TrainingArgs(
                model_name_or_path="my-model",
                data_path="/data/train.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.3
            )
            
            run_training(base_torch_args, train_args_osft_with_ratio)
            
            # Verify command includes both osft flags
            call_args = mock_popen_class.call_args
            _, command = call_args[0]
            assert "--osft" in command
            assert "--osft-unfreeze-rank-ratio=0.3" in command
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_run_training_keyboard_interrupt(self, mock_popen_class):
        """Test that run_training handles keyboard interrupt properly."""
        mock_popen = MagicMock()
        mock_popen.poll.return_value = 1  # Failure
        mock_popen.listen.side_effect = KeyboardInterrupt()
        mock_popen_class.return_value = mock_popen
        
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir
            )
            
            with pytest.raises(KeyboardInterrupt):
                run_training(torch_args, train_args)
            
            # Verify cleanup was attempted
            mock_popen.terminate.assert_called_once()
            mock_popen.wait.assert_called_once()
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_run_training_process_failure(self, mock_popen_class):
        """Test that run_training raises error on process failure."""
        mock_popen = MagicMock()
        mock_popen.poll.return_value = 1  # Failure
        mock_popen_class.return_value = mock_popen
        
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir
            )
            
            with pytest.raises(RuntimeError) as exc_info:
                run_training(torch_args, train_args)
            
            assert "Training failed" in str(exc_info.value)
            mock_popen.terminate.assert_called_once()


class TestEnums:
    """Test the enum types."""
    
    def test_training_mode_values(self):
        """Test that TrainingMode enum has correct values."""
        assert TrainingMode.EPOCH.value == "epoch"
        assert TrainingMode.STEP.value == "step"
        assert TrainingMode.TOKEN.value == "token"
        assert TrainingMode.INFINITE.value == "infinite"
    
    def test_training_mode_string_comparison(self):
        """Test that TrainingMode can be compared with strings."""
        assert TrainingMode.EPOCH == "epoch"
        assert TrainingMode.INFINITE == "infinite"


class TestParameterPassing:
    """Test that parameters are correctly passed through to the training script."""
    
    def test_lr_scheduler_kwargs_empty(self):
        """Test that empty lr_scheduler_kwargs is passed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock training script that just prints the received arguments
            mock_script = Path(tmpdir) / "mock_train.py"
            mock_script.write_text("""
import sys
import json

# Parse arguments and print lr_scheduler_kwargs
for i, arg in enumerate(sys.argv):
    if arg.startswith("--lr-scheduler-kwargs="):
        kwargs_str = arg.split("=", 1)[1]
        kwargs = json.loads(kwargs_str)
        print(f"LR_SCHEDULER_KWARGS:{json.dumps(kwargs)}")
        sys.exit(0)
""")
            
            with patch('mini_trainer.api_train.Path') as mock_path:
                # Make the train script path point to our mock script
                mock_path.return_value.__truediv__.return_value = mock_script
                
                torch_args = TorchrunArgs()  # Both are defaults now
                train_args = TrainingArgs(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=32,
                    max_tokens_per_gpu=1000,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    lr_scheduler_kwargs=None  # Should default to empty dict
                )
                
                # Capture the subprocess output
                with patch('subprocess.Popen') as mock_popen:
                    mock_process = MagicMock()
                    mock_process.stdout.readline.side_effect = [
                        "LR_SCHEDULER_KWARGS:{}\n",
                        ""  # End of output
                    ]
                    mock_process.poll.return_value = 0
                    mock_process.wait.return_value = 0
                    mock_popen.return_value = mock_process
                    
                    process = StreamablePopen(str(Path(tmpdir) / "test.log"), ["dummy"])
                    process.process = mock_process
                    
                    # Check that empty dict is passed
                    output = []
                    for line in iter(mock_process.stdout.readline, ''):
                        if line:
                            output.append(line.strip())
                    
                    assert "LR_SCHEDULER_KWARGS:{}" in output[0]
    
    def test_lr_scheduler_kwargs_complex(self):
        """Test that complex lr_scheduler_kwargs dict is passed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock training script
            mock_script = Path(tmpdir) / "mock_train.py"
            mock_script.write_text("""
import sys
import json

# Parse arguments and validate lr_scheduler_kwargs
for i, arg in enumerate(sys.argv):
    if arg.startswith("--lr-scheduler-kwargs="):
        kwargs_str = arg.split("=", 1)[1]
        kwargs = json.loads(kwargs_str)
        # Validate the complex kwargs
        assert "min_lr" in kwargs
        assert kwargs["min_lr"] == 1e-6
        assert "T_max" in kwargs
        assert kwargs["T_max"] == 1000
        assert "eta_min" in kwargs
        assert kwargs["eta_min"] == 0.0
        assert "nested" in kwargs
        assert kwargs["nested"]["key1"] == "value1"
        assert kwargs["nested"]["key2"] == 42
        print("KWARGS_VALIDATED:SUCCESS")
        sys.exit(0)

print("KWARGS_VALIDATED:FAILED")
sys.exit(1)
""")
            
            complex_kwargs = {
                "min_lr": 1e-6,
                "T_max": 1000,
                "eta_min": 0.0,
                "nested": {
                    "key1": "value1",
                    "key2": 42
                }
            }
            
            # Actually run the subprocess to test end-to-end
            command = [
                "python", str(mock_script),
                f"--lr-scheduler-kwargs={json.dumps(complex_kwargs)}"
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "KWARGS_VALIDATED:SUCCESS" in result.stdout
    
    def test_command_construction_with_special_chars(self):
        """Test that special characters in kwargs are properly escaped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                lr_scheduler_kwargs={
                    "description": "A string with spaces and special chars: @#$%",
                    "path": "/path/with/slashes",
                    "float_val": 3.14159,
                    "bool_val": True,
                    "null_val": None
                }
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                # Get the constructed command
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Find the lr-scheduler-kwargs argument
                kwargs_arg = None
                for arg in command:
                    if arg.startswith("--lr-scheduler-kwargs="):
                        kwargs_arg = arg
                        break
                
                assert kwargs_arg is not None
                
                # Extract and parse the JSON
                json_str = kwargs_arg.split("=", 1)[1]
                parsed_kwargs = json.loads(json_str)
                
                # Verify all values are preserved correctly
                assert parsed_kwargs["description"] == "A string with spaces and special chars: @#$%"
                assert parsed_kwargs["path"] == "/path/with/slashes"
                assert parsed_kwargs["float_val"] == 3.14159
                assert parsed_kwargs["bool_val"] is True
                assert parsed_kwargs["null_val"] is None
    
    def test_all_boolean_flags_passed(self):
        """Test that all boolean flags are correctly passed when True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                use_liger_kernels=True,
                checkpoint_at_epoch=True,
                save_final_checkpoint=True,
                # OSFT requires rank ratio
                osft=True,
                osft_unfreeze_rank_ratio=0.5
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Verify all boolean flags are present
                assert "--use-liger-kernels" in command
                assert "--osft" in command
                assert "--checkpoint-at-epoch" in command
                assert "--save-final-checkpoint" in command
    
    def test_boolean_flags_not_passed_when_false(self):
        """Test that boolean flags are not passed when False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                use_liger_kernels=False,
                osft=False,
                checkpoint_at_epoch=False,
                save_final_checkpoint=False
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Verify boolean flags are NOT present when False
                assert "--use-liger-kernels" not in command
                assert "--osft" not in command
                assert "--checkpoint-at-epoch" not in command
                assert "--save-final-checkpoint" not in command
    
    def test_training_mode_enum_passed_correctly(self):
        """Test that TrainingMode enum values are correctly converted to strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in [TrainingMode.EPOCH, TrainingMode.STEP, TrainingMode.TOKEN, TrainingMode.INFINITE]:
                torch_args = TorchrunArgs(nproc_per_node=8)
                train_args = TrainingArgs(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=32,
                    max_tokens_per_gpu=1000,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    training_mode=mode
                )
                
                with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                    mock_popen = MagicMock()
                    mock_popen.poll.return_value = 0
                    mock_popen_class.return_value = mock_popen
                    
                    run_training(torch_args, train_args)
                    
                    call_args = mock_popen_class.call_args
                    _, command = call_args[0]
                    
                    # Find the training-mode argument
                    mode_arg = None
                    for arg in command:
                        if arg.startswith("--training-mode="):
                            mode_arg = arg
                            break
                    
                    assert mode_arg == f"--training-mode={mode.value}"
    
