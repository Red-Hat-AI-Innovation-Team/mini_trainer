<div align="center">


# Mini Trainer


[![PR Tests](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/actions/workflows/pr-tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/mini_trainer/graph/badge.svg?token=FHCFYB1HJZ)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/mini_trainer)

### A lightweight, high-performance training library for efficient fine-tuning of large language models up to 70B parameters.

<img src="https://ai-innovation.team/images/toolkit%20logos/Black%20and%20White%20Labs/lab-mini-trainer.png" alt="Mini Trainer Logo" height="150"/>

**Built for speed, simplicity, and scalability** 🚀

</div>

---

## ✨ Features

- 🔥 **[Liger Kernels](https://github.com/linkedin/Liger-Kernel)** - Minimized memory footprint through chunked loss computation
- ⚡ **Smart Batch Packing** - Automatic minibatching with numba-optimized LPT algorithm for optimal GPU load balancing
- 🎯 **FSDP2 Support** - Native PyTorch distributed training with FullyShardedDataParallel
- 🚫 **Padding-Free** - Leverages Flash Attention for efficient computation without padding overhead
- ♾️ **Infinite Sampling** - Continuous data streaming without manual epoch configuration
- 🔬 **Orthogonal Subspace Fine-Tuning (OSFT)** - Advanced continual learning technique for parameter-efficient training
- 📊 **Flexible Logging** - JSONL metrics logging with optional Weights & Biases integration

---

## 🔬 Orthogonal Subspace Fine-Tuning (OSFT)

[![arXiv](https://img.shields.io/badge/arXiv-2504.07097-b31b1b.svg)](https://arxiv.org/abs/2504.07097)

Mini Trainer implements **Orthogonal Subspace Fine-Tuning (OSFT)**, a breakthrough continual learning technique that enables models to learn new tasks **without catastrophic forgetting**. OSFT uses adaptive SVD-based decomposition to intelligently update models in unused parameter subspaces while preserving crucial prior knowledge.

### 🎥 Learn More

<div align="center">

[![Orthogonal Subspace Learning](https://img.youtube.com/vi/iVp8aWkF_5M/0.jpg)](https://www.youtube.com/watch?v=iVp8aWkF_5M)

**Watch our technical deep-dive on Orthogonal Subspace Learning**

</div>

### 📚 Resources

- 📝 **Blog Post**: [Sculpting Subspaces: How We Solved Continual Learning in LLMs](https://ai-innovation.team/blog/orthogonal-subspace-learning)
- 📄 **Research Paper**: [arXiv:2504.07097](https://arxiv.org/abs/2504.07097)

### 🚀 Using OSFT

Enable OSFT in your training runs with the `--osft` flag:

```bash
torchrun --nnodes=1 --nproc-per-node=8 -m mini_trainer.train \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --data-path ./data.jsonl \
    --output-dir ./checkpoints \
    --osft \
    --osft-unfreeze-rank-ratio 0.25  # train the 25% least important parameters
```

The `--osft-unfreeze-rank-ratio` parameter controls how much of the model to update (0.0 = everything frozen, 1.0 = full training).

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.

---

## 🙏 Acknowledgments

Built with ❤️ by the [Red Hat AI Innovation Team](https://ai-innovation.team/). 

Mini Trainer is part of a broader ecosystem of LLM tools developed by the AI Innovation Team. Check out our other projects:
- [training_hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) - Post-training algorithms for LLMs
- [its_hub](https://github.com/Red-Hat-AI-Innovation-Team/its_hub) - Inference-time scaling for LLMs
- [sdg_hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub) - Synthetic data generation pipelines
- [reward_hub](https://github.com/Red-Hat-AI-Innovation-Team/reward_hub) - State-of-the-art reward models

Visit [ai-innovation.team](https://ai-innovation.team/) to explore all our open-source tools and research.

Special thanks to the open-source community for contributions and feedback!
