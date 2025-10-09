# 🚀 Mini Trainer

[![PR Tests](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/actions/workflows/pr-tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A lightweight, high-performance training library for efficient fine-tuning of large language models up to 70B parameters. Built for speed, simplicity, and scalability.

---

## ✨ Features

- 🔥 **[Liger Kernels](https://github.com/linkedin/Liger-Kernel)** - Minimized memory footprint through chunked loss computation
- ⚡ **Smart Batch Packing** - Automatic minibatching with numba-optimized LPT algorithm for optimal GPU load balancing
- 🎯 **FSDP2 Support** - Native PyTorch distributed training with FullyShardedDataParallel
- 🚫 **Padding-Free** - Leverages Flash Attention for efficient computation without padding overhead
- ♾️ **Infinite Sampling** - Continuous data streaming without manual epoch configuration
- 🔬 **OSFT (Orthogonal Subspace Fine-Tuning)** - Advanced continual learning technique for parameter-efficient training
- 📊 **Flexible Logging** - JSONL metrics logging with optional Weights & Biases integration

---

## 📦 Installation

### From PyPI

```bash
# Install base package
pip install rhai-innovation-mini-trainer

# Install CUDA dependencies (required for GPU training)
pip install rhai-innovation-mini-trainer[cuda] --no-build-isolation
```

### From Source (Editable)

```bash
# Clone the repository
git clone https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer.git
cd mini_trainer

# Install in editable mode
pip install -e .

# Install CUDA dependencies
pip install -e .[cuda] --no-build-isolation
```

---

## 🎯 Usage

Training is orchestrated through the `api_train.py` module, which provides a programmatic interface for launching training jobs. You can run training using `torchrun` for distributed setups:

```bash
torchrun --nnodes=1 --nproc-per-node=8 -m mini_trainer.train \
    --output-dir ./checkpoints \
    --data-path ./data.jsonl \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 128 \
    --max-tokens-per-gpu 128000 \
    --learning-rate 5e-6 \
    --use-liger-kernels
```

### Key Parameters

- `--model-name-or-path` - HuggingFace model identifier or local path
- `--data-path` - Path to tokenized training data (JSONL format)
- `--batch-size` - Target batch size for training
- `--max-tokens-per-gpu` - Maximum tokens per GPU (auto-balances minibatches)
- `--output-dir` - Directory for checkpoints and logs
- `--use-liger-kernels` - Enable memory-efficient Liger kernels
- `--osft` - Enable Orthogonal Subspace Fine-Tuning mode
- `--osft-unfreeze-rank-ratio` - Ratio of model parameters to train with OSFT (0.0-1.0)

For the complete list of arguments and advanced configuration options, see [`src/mini_trainer/api_train.py`](src/mini_trainer/api_train.py).

---

## 📊 Data Format

Mini Trainer expects pre-tokenized data in **JSONL format** with the following structure:

```json
{"input_ids": [1, 2, 3, ...], "labels": [1, 2, 3, ...], "len": 128}
{"input_ids": [4, 5, 6, ...], "labels": [-100, -100, 6, ...], "len": 256}
```

Each line should contain:
- `input_ids` - Tokenized input sequence
- `labels` - Target labels (use `-100` for tokens to ignore in loss computation)
- `len` - Sequence length (optional, computed automatically if missing)

### 🔄 Data Processing

**Mini Trainer does not include data processing utilities.** For tokenization and data preparation, please use the **[instructlab-training](https://github.com/instructlab/training)** APIs, which provide robust data processing pipelines compatible with Mini Trainer's input format.

---

## 🐛 Bug Reports & Issues

Found a bug or have a feature request? We'd love to hear from you! Please [open an issue](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/issues) on GitHub with:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, GPU type, etc.)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with ❤️ by the Red Hat AI Innovation Team. Special thanks to the open-source community for contributions and feedback!
