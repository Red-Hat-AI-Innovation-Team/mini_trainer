# Changelog

All notable changes to mini_trainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.0] - 2025-09-16

### Added
- **GPT-OSS Model Support**
  - Full support for OpenAI's new open-weight GPT-OSS models (20B and 120B variants)
  - Native MXFP4 quantization implementation
  - New `gpt_oss_utils.py` module (430+ lines)
- **Memory-Efficient OSFT Initialization**
  - New `osft_memory_efficient_init` flag for optimized initialization of large models
  - Significant memory savings during model loading
- **Training Dtype Control**
  - New `train_dtype` parameter for switching models to bf16/fp16 training
  - Reduces memory usage (use sparingly as lower precision may impact results)
- **Pretraining Data Conversion**
  - New `convert_to_pretrain.py` script for converting conversation datasets
- **OSFT Dtype Controls**
  - `osft_upcast_dtype` for computation precision (default: float32)
  - `osft_output_dtype` for output precision control
- **Enhanced Data Processing**
  - Improved `process_data.py` with additional functionality
- **Weights & Biases (wandb) integration** for experiment tracking
  - New `wandb_wrapper.py` module
  - Automatic logging of training/validation metrics, gradients, and system stats
  - Opt-in via `--wandb` CLI flag or corresponding config entry
- **Train/Validation Split Support**
  - Deterministic split into train and validation shards in sampler
  - New `--validation-split` argument (default 0.05) controls hold-out fraction
  - Validation loop runs every `validation_frequency` steps
- **Validation Loss Tracking**
  - Validation loss computation and reporting
  - Integration with console logs and wandb dashboards

### Changed
- **Dependencies**
  - Updated transformers to `>=4.55.0`
  - Added liger-kernel for optimized operations
  - Added kernels package for flash-attention-3 support
- Simplified implementation of memory efficient + GPT-OSS loading
- Enhanced test coverage for validation and sampler behavior
- Updated dependencies in `pyproject.toml`

### Fixed
- Various test case failures
- Code optimization and cleanup based on PR feedback
- GPT-OSS checkpoint saving during SFT
- Distributed torch tests stability by mocking `torch.distributed` checks
- Dtype conversion edge-cases
- Default `validation_frequency` is now `None` instead of `0`

## [v0.1.1] - Previous Release

[Previous release details would go here]

---

## Usage Examples

### GPT-OSS-20B Training
```python
from mini_trainer.api_train import run_training
from mini_trainer.training_types import TrainingArgs, TorchrunArgs

train_args = TrainingArgs(
    model_name="openai/gpt-oss-20b",
    osft_memory_efficient_init=True,
    train_dtype="bfloat16",
    wandb=True,  # Enable wandb logging
    validation_split=0.05,  # 5% validation split
    validation_frequency=100,  # Validate every 100 steps
    ...  # other training arguments
)

run_training(torch_args, train_args)
```

### Upgrade Notes
- v0.2.0: No breaking API changes. Primary focus on GPT-OSS 20B model support (120B variant potentially supported but not extensively tested). WandB logging requires `wandb>=0.16`.

### Contributors
- @NikhilNayak-debug 
- @Maxusmusti 
- @RobotSail

### Links
- [Full Changelog v0.1.1...v0.2.0](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/compare/v0.1.1...v0.2.0)
