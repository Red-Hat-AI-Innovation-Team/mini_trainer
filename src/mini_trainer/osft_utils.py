import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import typing as t
from typing import Protocol
from weakref import ref as weakref
from dataclasses import dataclass
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
import gc
import types

from tqdm import tqdm

from mini_trainer.utils import log_rank_0, get_control_process_group
from mini_trainer.gpt_oss_utils import is_gpt_oss_model
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from mini_trainer.fsdp2_lazy_init import (
    FSDP2_LAZY_INIT_OSFT,
    get_fsdp2_lazy_init_mode,
    set_fsdp2_lazy_init_mode,
)

import os

# Memory optimization constants
OSFT_CACHE_CLEAR_INTERVAL = int(
    os.getenv("OSFT_CACHE_CLEAR_INTERVAL", 5)
)  # Clear GPU cache every N parameters during matrix reconstruction


Role = t.Literal["osft_target", "non_osft"]


@dataclass(frozen=True)
class ParamSpec:
    logical_key: str  # e.g., "transformer.blocks.12.attn.q_proj.weight"
    shape: tuple[int, ...]
    dtype: torch.dtype
    role: Role


@dataclass(frozen=True)
class OSFTFactorSpec:
    parent_key: str  # e.g., "transformer.blocks.12.attn.q_proj"
    # Derived runtime keys that will exist after OSFT install:
    U_high: str
    S_high: str
    V_high: str
    U_low: str
    S_low: str
    V_low: str
    rank_high: str


class SVDDictBase(t.TypedDict):
    U_high: torch.Tensor
    S_high: torch.Tensor
    V_high: torch.Tensor
    U_low: nn.Parameter
    S_low: nn.Parameter
    V_low: nn.Parameter


class SVDDecompositionDict(SVDDictBase, total=False):
    rank_high: int


class OSFTModelProtocol(Protocol):
    """
    Protocol defining the interface for models with OSFT capabilities.

    This allows type hints throughout the codebase without depending on the dynamically
    created class from create_osft_model_class().
    """

    osft_config: dict[str, int]
    name_mapping: dict[str, str]
    osft_params: nn.ModuleDict
    upcast_dtype: torch.dtype
    output_dtype: torch.dtype

    def reinitialize_osft(
        self,
        decompose_existing_weights: bool,
        assigned_params: list[tuple[str, torch.Tensor]] | None = None,
    ) -> None: ...

    def reinitialize_osft_distributed(self) -> None: ...

    def project_gradients(self) -> None: ...

    def _reconstruct_weight_by_safe_name(
        self,
        safe_name: str,
        upcast_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor: ...

    def _reconstruct_weight(
        self,
        original_name: str,
        upcast_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor: ...


# Type alias for any model that implements OSFT
OSFTModel = OSFTModelProtocol


# OSFT parameter classification constants
OSFT_ALL_PARAMS = {
    "osft_config",
    "initialize_osft",
    "rank_ratio",
    "target_patterns",
    "upcast_dtype",
    "output_dtype",
    "model_name_or_class",
}

# Parameters that GPT-OSS constructors don't accept and must be filtered out
OSFT_GPT_OSS_FILTERED_PARAMS = OSFT_ALL_PARAMS

# Parameters that base model loading doesn't accept (subset of GPT-OSS filtered)
OSFT_BASE_MODEL_FILTERED_PARAMS = {
    "osft_config",
    "initialize_osft",
    "rank_ratio",
    "target_patterns",
}

# Parameters that OSFT class constructors can handle
OSFT_CLASS_PARAMS = {
    "upcast_dtype",
    "output_dtype",
    "model_name_or_class",
    "lazy_init_tokenizer_align_fn",
}


# Pre-defined model configurations for common architectures
MODEL_CONFIGS = {
    "llama": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "mlp.up_proj",
        ]
    },
    "gpt2": {
        "patterns": [
            "attn.c_proj",
            "attn.c_attn",
            "mlp.c_fc",
            "mlp.c_proj",
        ]
    },
    "mistral": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    },
    "gpt-j": {
        "patterns": [
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.out_proj",
            "mlp.fc_in",
            "mlp.fc_out",
        ]
    },
    "gpt-neo": {
        "patterns": [
            "attn.attention.q_proj",
            "attn.attention.k_proj",
            "attn.attention.v_proj",
            "attn.attention.out_proj",
            "mlp.c_fc",
            "mlp.c_proj",
        ]
    },
    "opt": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ]
    },
    "qwen": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "mlp.up_proj",
        ]
    },
    "gemma": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    },
    "phi3": {
        "patterns": [
            "self_attn.o_proj",
            "self_attn.qkv_proj",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        ]
    },
    # granite-4 architecture may change so this will likely
    # need to be updated then
    "granite": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "mlp.up_proj",
        ]
    },
    "gpt-oss": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            # Removed expert layer patterns to avoid MoE complexity
            # "experts.gate_up_proj",
            # "experts.down_proj",
        ]
    },
    "default": {
        "patterns": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "mlp.up_proj",
        ]
    },
}

# Define model name mappings at module level
MODEL_NAME_MAPPINGS = {
    "llama": "llama",
    "gpt-j": "gpt-j",
    "gptj": "gpt-j",  # Handle both "gpt-j" and "gptj" variants
    "gpt-neo": "gpt-neo",
    "gptneo": "gpt-neo",  # Handle both "gpt-neo" and "gptneo" variants
    "gpt-oss": "gpt-oss",
    "opt": "opt",
    "qwen": "qwen",
    "gemma": "gemma",
    "phi4": "phi3",
    "phi-4": "phi3",  # this should handle phi-4, phi-4-mini, and phi-4-mini-instruct
    "phi3": "phi3",
    "phi-3": "phi3",
    "mistral": "mistral",
    "granite": "granite",
    "gpt2": "gpt2",
    # Easy to add more mappings
    # "phi": "phi",
}


def is_osft_param(name: str, param: torch.Tensor, osft_config: dict) -> bool:
    """
    Utility function to make it easier to classify OSFT parameters.
    """
    return len(param.shape) == 2 and name in osft_config and osft_config[name] > 0


def is_osft_model(model: torch.nn.Module) -> bool:
    """
    Check if a model implements the OSFT interface.

    Args:
        model: The model to check

    Returns:
        True if the model has OSFT capabilities, False otherwise
    """
    required_attrs = [
        "osft_config",
        "osft_params",
        "project_gradients",
        "reinitialize_osft",
    ]
    return all(hasattr(model, attr) for attr in required_attrs)


def cast_to_osft_model(model: torch.nn.Module) -> OSFTModel:
    """
    Cast a model to OSFTModel type for type checkers.

    Args:
        model: The model to cast (should implement OSFTModelProtocol)

    Returns:
        The same model, but typed as OSFTModel

    Raises:
        TypeError: If the model doesn't implement the OSFT interface
    """
    if not is_osft_model(model):
        raise TypeError(f"Model {type(model)} does not implement OSFT interface")
    return model  # type: ignore


def create_svd_dict(
    weight: torch.Tensor,
    top_k: int,
    decompose_existing: bool = True,
    upcast_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
    use_meta: bool = False,
) -> SVDDecompositionDict:
    """
    Decomposes a 2D weight matrix into two components using Singular Value Decomposition (SVD):
    - The top `top_k` singular components (U_high, S_high, V_high) are treated as frozen and encode
      critical directions that should not be updated in new tasks.
    - The remaining components (U_low, S_low, V_low) are made trainable and are used to learn new tasks.

    This decomposition separates the weight space into high-rank subspaces for knowledge retention
    and low-rank subspaces for task-specific adaptation, helping to mitigate catastrophic forgetting
    in continual learning scenarios.
    """
    device_local = weight.device
    use_meta = use_meta or device_local.type == "meta"

    # handle casting data-types
    if not output_dtype:
        output_dtype = upcast_dtype

    if weight.ndim != 2:
        raise ValueError(
            "creating SVD dict from a non-2D tensor is currently unsupported!"
        )

    if decompose_existing and use_meta:
        raise ValueError("cannot decompose meta weights into SVD!")

    # N: output dim, M: input dim
    N, M = weight.shape

    if decompose_existing:
        # To minimize numerical error, we perform the SVD decomposition
        # in high precision, before casting back to the original data-type
        # since FSDP requires homogenous data-types.
        W = weight.to(upcast_dtype)  # Ensure numerical stability for SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        if upcast_dtype != output_dtype:
            U = U.to(output_dtype)
            S = S.to(output_dtype)
            Vt = Vt.to(output_dtype)
    else:
        # Note(osilkin):
        # Here we create dummy versions of the weights initialized to 0
        # So that we can later populate them with the SVD from another process
        # this is how pytorch reshapes the SVD matrices with `full_matrices=False`
        R = min(N, M)

        # recreate how the matrices would be shaped inside of pytorch
        if use_meta:
            meta_device = torch.device("meta")
            U = torch.zeros((N, R), dtype=output_dtype, device=meta_device)
            S = torch.zeros((R,), dtype=output_dtype, device=meta_device)
            Vt = torch.zeros((R, M), dtype=output_dtype, device=meta_device)
        else:
            U = torch.zeros((N, R), dtype=output_dtype)
            S = torch.zeros((R,), dtype=output_dtype)
            Vt = torch.zeros((R, M), dtype=output_dtype)

    k = min(top_k, S.shape[0])  # Cap to matrix rank

    # Split high-rank (frozen) and low-rank (trainable) subspaces
    svd = {
        "U_high": U[:, :k].contiguous().detach().to(device=device_local),
        "S_high": S[:k].contiguous().detach().to(device=device_local),
        "V_high": Vt[:k, :].contiguous().detach().to(device=device_local),
        "U_low": nn.Parameter(U[:, k:].contiguous().detach().to(device=device_local)),
        "S_low": nn.Parameter(S[k:].contiguous().detach().to(device=device_local)),
        "V_low": nn.Parameter(Vt[k:, :].contiguous().detach().to(device=device_local)),
        "rank_high": k,  # Store for later use in orthogonal projection
    }
    return svd


def reconstruct_weight_matrix(
    svd_dict: SVDDecompositionDict,
    upcast_dtype: torch.dtype,
    output_dtype: torch.dtype | None = None,
):
    """
    Reconstructs the original weight matrix from its SVD components.

    Used for replacing linear layers during inference or forward pass to preserve the weight structure.
    The final matrix is the sum of contributions from both the high-rank (frozen) and low-rank (trainable) components.
    """
    U_high = svd_dict["U_high"].to(upcast_dtype)
    S_high = svd_dict["S_high"].to(upcast_dtype)
    V_high = svd_dict["V_high"].to(upcast_dtype)
    U_low = svd_dict["U_low"].to(upcast_dtype)
    S_low = svd_dict["S_low"].to(upcast_dtype)
    V_low = svd_dict["V_low"].to(upcast_dtype)

    # Reconstruct high-rank component (frozen during continual learning)
    if U_high.numel() > 0 and S_high.numel() > 0:
        high_part = torch.mm(U_high * S_high.unsqueeze(0), V_high)
    else:
        high_part = torch.zeros(
            U_low.size(0), V_low.size(1), device=U_high.device, dtype=upcast_dtype
        )

    # Reconstruct low-rank component (receives task-specific updates)
    if U_low.numel() > 0 and S_low.numel() > 0:
        low_part = torch.mm(U_low * S_low.unsqueeze(0), V_low)
    else:
        low_part = torch.zeros(
            U_high.size(0),
            V_high.size(1),
            device=U_low.device,
            dtype=upcast_dtype,
        )

    # Combine the low-rank & high-rank components
    reconstructed = high_part + low_part
    if output_dtype:
        reconstructed = reconstructed.to(output_dtype)
    return reconstructed


def project_gradient_to_orthogonal_space(svd_dict: SVDDecompositionDict):
    """
    Projects the gradient of the low-rank parameters (U_low, V_low) to be orthogonal to the frozen high-rank subspace.

    This step ensures that learning new tasks does not interfere with previously learned representations by enforcing an orthogonality constraint.

    TODO(osilkin): Add mixed-precision gradients here
    """
    # Skip if no gradients present (sanity check)
    if (
        svd_dict["U_low"].grad is None
        and svd_dict["S_low"].grad is None
        and svd_dict["V_low"].grad is None
    ):
        return

    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]

    # Project U_low gradients to space orthogonal to U_high
    if svd_dict["U_low"].grad is not None:
        dU = svd_dict["U_low"].grad
        # Support distributed tensors by operating on the local shard
        local_U_high = getattr(U_high, "to_local", lambda: U_high)()
        local_dU = getattr(dU, "to_local", lambda: dU)()

        # Perform projection computation using memory-efficient operations
        # Memory-optimized projection: dU = dU - U_high @ (U_high.T @ dU)
        # Use addmm_ for efficient in-place operation
        # Compute local contribution to (U_high^T @ dU); all-reduce to get global projection
        proj_coeff = torch.mm(local_U_high.transpose(0, 1), local_dU)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(proj_coeff, op=dist.ReduceOp.SUM)
        # Apply projection using only local rows of U_high
        local_dU.addmm_(local_U_high, proj_coeff, alpha=-1.0)

        if hasattr(dU, "_local_tensor"):
            dU._local_tensor.copy_(local_dU)
        else:
            dU.copy_(local_dU)

    # Repeat projection for V_low using V_high
    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        local_V_high = getattr(V_high, "to_local", lambda: V_high)()
        local_dV = getattr(dV, "to_local", lambda: dV)()

        # Compute Gram matrix G = V_high^T @ V_high for global projection across row-sharded V_high
        # Assumes column dimension is consistent across ranks (row sharding over singular vectors)
        G_local = torch.mm(local_V_high.transpose(0, 1), local_V_high)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(G_local, op=dist.ReduceOp.SUM)

        # Apply projection: dV = dV - dV @ G (use local shard of dV)
        update = torch.mm(local_dV, G_local)
        local_dV.add_(update, alpha=-1.0)

        if hasattr(dV, "_local_tensor"):
            dV._local_tensor.copy_(local_dV)
        else:
            dV.copy_(local_dV)


def get_osft_target_parameters(model, osft_config):
    """
    Determines which parameters will be OSFT decomposed based on the OSFT configuration.

    Returns a list of (name, param) tuples for parameters that will be decomposed.
    """
    target_params = []
    for name, param in model.named_parameters():
        # TODO(osilkin): Right now we are only training 2D parameters, but some 1D parameters (like bias vectors)
        # are vectors stored as a list, but may be interpreted as a (1, N) or (N, 1) matrix.
        # SVD can processs these in general, but maybe they should be targeted normally
        if is_osft_param(name, param, osft_config):
            target_params.append((name, param))
    return target_params


def _get_model_patterns_from_name(name: str) -> list:
    """
    Get model patterns from a model name string.

    Args:
        name: Model name string

    Returns:
        List of patterns for the model
    """
    # Find first matching model type
    for identifier, config_key in MODEL_NAME_MAPPINGS.items():
        if identifier in name.lower():
            return MODEL_CONFIGS[config_key]["patterns"]

    # Default fallback
    return MODEL_CONFIGS["default"]["patterns"]


def get_model_patterns(model_name_or_class):
    """Get patterns for a model from name string or class object."""
    # Handle string model names
    name = model_name_or_class
    if not isinstance(name, str):
        if hasattr(name, "__name__"):
            name = name.__name__
        else:
            raise ValueError(
                f"Invalid model name or class: {model_name_or_class} (expected str or class object)"
            )

    return _get_model_patterns_from_name(name)


def get_model_config(model_name_or_class=None, target_patterns=None):
    """
    Get SVD target patterns for a model.

    Args:
        model_name_or_class: Model name/class to get predefined patterns for, or None
        target_patterns: Custom list of patterns to use instead of predefined ones

    Returns:
        List of patterns to match against parameter names
    """
    if target_patterns is not None:
        return target_patterns

    if model_name_or_class is None:
        return MODEL_CONFIGS["default"]["patterns"]

    return get_model_patterns(model_name_or_class)


def auto_generate_target_osft_config(
    model, model_name_or_class=None, target_patterns=None, rank_ratio=0.5
) -> dict[str, int]:
    """
    Automatically selects which weight matrices to decompose for OSFT and determines their top-k values.

    Args:
        model: The model to analyze
        model_name_or_class: Model name/class to get predefined patterns for, or None for auto-detection
        target_patterns: Custom list of patterns to use instead of predefined ones
        rank_ratio: Ratio of the smaller dimension to use for top-k rank (default: 0.5)

    Returns:
        Dictionary mapping parameter names to their top-k values
    """
    target_patterns = get_model_config(model_name_or_class, target_patterns)

    config = {}
    for name, param in model.named_parameters():
        if any(pat in name for pat in target_patterns) and len(param.shape) == 2:
            # Use specified ratio of effective rank
            top_k = int(np.floor(min(param.shape) * rank_ratio))
            full_rank = min(param.shape)
            if top_k >= full_rank:
                top_k = full_rank - 1
            config[name] = top_k
    return config


def _filter_osft_parameters(kwargs: dict, filter_set: set[str]) -> dict:
    """
    Filter out OSFT-specific parameters using the specified filter set.

    Args:
        kwargs: Dictionary of keyword arguments
        filter_set: Set of parameter names to filter out

    Returns:
        Filtered dictionary with specified parameters removed
    """
    return {k: v for k, v in kwargs.items() if k not in filter_set}


def _extract_osft_class_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """
    Separate OSFT class-specific kwargs from other kwargs.

    Args:
        kwargs: Full kwargs dictionary

    Returns:
        Tuple of (osft_class_kwargs, filtered_kwargs)
    """
    osft_class_kwargs = {k: v for k, v in kwargs.items() if k in OSFT_CLASS_PARAMS}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in OSFT_CLASS_PARAMS}
    return osft_class_kwargs, filtered_kwargs


def _load_model_memory_efficient(
    actual_osft_cls,
    pretrained_model_name_or_path: str,
    model_args: tuple,
    base_kwargs: dict,
    osft_class_kwargs: dict,
):
    """
    Memory-efficient loading for OSFT models to avoid CUDA/CPU OOM.
    This is only supported in distributed environments.


    This function loads models to CPU first, extracts state dict, then creates
    the OSFT model to minimize peak memory usage during initialization.

    Args:
        actual_osft_cls: The OSFT model class to instantiate
        pretrained_model_name_or_path: Model path or name
        model_args: Positional arguments for model loading
        base_kwargs: Base model kwargs (already filtered)
        init_cfg: OSFT configuration
        osft_class_kwargs: OSFT class-specific parameters

    Returns:
        Loaded OSFT model
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "memory efficient initialization is only supported in distributed"
        )

    # Get the base model class from the OSFT class inheritance chain
    base_model_class = None
    for base in actual_osft_cls.__mro__:
        if (
            hasattr(base, "from_pretrained")
            and base != actual_osft_cls
            and "WithOSFT" not in base.__name__
        ):
            base_model_class = base
            break

    if base_model_class is None:
        raise ValueError(
            f"Could not find base model class in inheritance chain of {actual_osft_cls}"
        )

    log_rank_0(f"🎯 Using base model class: {base_model_class.__name__}")
    log_rank_0("🧠 Using memory-efficient loading to avoid CUDA OOM")

    # Remove additional OSFT parameters before calling base model's from_pretrained
    final_base_kwargs = _filter_osft_parameters(
        base_kwargs, OSFT_BASE_MODEL_FILTERED_PARAMS
    )

    # Force CPU loading via default behavior and match the train_dtype for FSDP2
    # Need to get train_dtype from base_kwargs or default to float32
    load_dtype = base_kwargs.get("torch_dtype", None)
    if load_dtype is None:
        raise ValueError(
            "error: model does not have a `torch_dtype` setting, please report this to the developers"
        )
    final_base_kwargs["torch_dtype"] = load_dtype

    # initialize params to instance the OSFT model
    # global rank 0 process actually loads the model, and all other procs
    # will receive and store the data in these vars
    config = None
    state_dict = None
    param_keys = []
    buffer_dict = {}

    # This section is responsible for loading the original model's state dict
    # which gets reused later for memory-efficient initialization. The extracted config
    # is broadcast to all processes so they can correctly initialize a model on the meta
    # device, allowing them to create a prototype of the same model without loading any data.

    if dist.get_rank() == 0:
        with torch.no_grad():
            log_rank_0(f"📥 Loading base model to CPU in {load_dtype}...")
            base_model = base_model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **final_base_kwargs,
            )

            align_fn = osft_class_kwargs.get("lazy_init_tokenizer_align_fn")
            if align_fn:
                base_model = align_fn(base_model)

            # Extract config and state dict immediately
            config = base_model.config
            state_dict = base_model.state_dict()

            # set the param keys directly from the state dict
            param_keys = list(state_dict.keys())

            # export the buffer dict - FSDP2 doesn't wrap this by default
            buffer_dict = dict(base_model.named_buffers())

            # Delete base model immediately to free memory
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # we need this for the pickle
        if not config:
            raise ValueError(
                "during memory-efficient OSFT loading, rank 0 loaded a model with no config!"
            )

    # other processes wait
    dist.barrier()

    # now we need to distribute metadata from main proc to the other procs
    mailbox = [config, param_keys, buffer_dict]
    dist.broadcast_object_list(mailbox, src=0)

    # non-main procs read the keys
    if dist.get_rank() != 0:
        config, param_keys, buffer_dict = mailbox

    # config is necessary for properly loading the model
    if not config:
        raise ValueError(
            f"during memory-efficient OSFT loading, rank {dist.get_rank()} loaded a model with no config!"
        )

    log_rank_0("instantiating OSFT model on meta device")
    extra_kwargs = {}
    if dist.get_rank() == 0:
        # rank 0 passes the OG state dict to the OSFT model for later access
        extra_kwargs["lazy_init_og_state_dict"] = state_dict

    # this instantiates the model as we'd expect, except that params and buffers
    # only load their metadata (dtype, shape) without loading the raw data or creating new random weights
    with torch.device("meta"):
        model = actual_osft_cls(
            config=config,
            initialize_osft=False,
            # TODO: remove `upcast_dtype` and `output_dtype`
            upcast_dtype=osft_class_kwargs.get("upcast_dtype", torch.float32),
            output_dtype=osft_class_kwargs.get("output_dtype", None),
            fsdp2_lazy_init=True,
            # provide the osft model with the logical set of original parameter keys
            lazy_init_param_keys=param_keys,
            lazy_init_buffer_dict=buffer_dict,
            **extra_kwargs,
        )

    return model


def _build_osft_kwargs(osft_rank_ratio, osft_target_patterns):
    """
    Build OSFT kwargs from parameters, eliminating duplication.

    Args:
        osft_rank_ratio: Rank ratio parameter
        osft_target_patterns: Target patterns parameter

    Returns:
        Dictionary of OSFT kwargs
    """
    osft_kwargs = {}
    if osft_rank_ratio:
        osft_kwargs["rank_ratio"] = osft_rank_ratio
    if osft_target_patterns:
        osft_kwargs["target_patterns"] = osft_target_patterns
    return osft_kwargs


def _set_osft_dtypes(model, osft_upcast_dtype, osft_output_dtype):
    """
    Set OSFT dtype attributes on model for computation precision control.

    Args:
        model: The OSFT model to configure
        osft_upcast_dtype: Upcast dtype for computations
        osft_output_dtype: Output dtype for results
    """
    model.upcast_dtype = osft_upcast_dtype
    if osft_output_dtype:
        model.output_dtype = osft_output_dtype


def create_osft_model_class(base_cls) -> type[OSFTModel]:
    """
    Dynamically creates a subclass of the given `base_cls` that replaces selected linear weights
    with low-rank + high-rank SVD-decomposed versions for OSFT training.

    This class:
    - Initializes frozen high-rank buffers and trainable low-rank parameters.
    - Replaces the forward pass of targeted modules to use reconstructed weights.
    - Projects gradients during training to enforce orthogonality with high-rank subspaces.

    This class enables constrained full fine-tuning using OSFT (Orthogonal Subspace Fine-Tuning).

    Returns:
        A class that implements OSFTModelProtocol and inherits from base_cls.
    """

    class ModelWithOSFT(base_cls):
        osft_config: dict[str, int]

        def __init__(
            self,
            config,
            osft_config: dict[str, int] | None = None,
            initialize_osft=True,
            upcast_dtype: torch.dtype = torch.float32,
            output_dtype: torch.dtype | None = None,
            fsdp2_lazy_init: bool = False,
            lazy_init_og_state_dict: dict | None = None,
            lazy_init_param_keys: list[str] | None = None,
            lazy_init_buffer_dict: dict[str, torch.Tensor] | None = None,
            **kwargs,
        ):
            # validation
            if fsdp2_lazy_init:
                if not dist.is_available() or not dist.is_initialized():
                    raise ValueError(
                        "cannot use fsdp2 lazy init when torch.distributed is unavailable"
                    )

                if initialize_osft:
                    raise ValueError(
                        "cannot initialize in the __init__ method when calling lazy init"
                    )

                if dist.get_rank() == 0 and (
                    not isinstance(lazy_init_og_state_dict, dict)
                    or not lazy_init_og_state_dict
                ):
                    raise ValueError(
                        "expected the original state dict on rank 0 but it wasn't provided!"
                    )

            # Filter out OSFT-specific parameters for GPT-OSS compatibility
            is_gpt_oss = is_gpt_oss_model(config)
            if is_gpt_oss:
                # Remove any OSFT-specific parameters that GPT-OSS constructor won't accept
                kwargs = _filter_osft_parameters(kwargs, OSFT_GPT_OSS_FILTERED_PARAMS)
            super().__init__(config, **kwargs)
            self.osft_config = (
                osft_config if osft_config else {}
            )  # Maps parameter names → top_k

            # for fsdp2 lazy initialization
            self.fsdp2_lazy_init = fsdp2_lazy_init
            self._lazy_init_pending = fsdp2_lazy_init
            self._lazy_init_og_state_dict = lazy_init_og_state_dict
            set_fsdp2_lazy_init_mode(
                self,
                FSDP2_LAZY_INIT_OSFT if fsdp2_lazy_init else None,
            )

            # create a set of logical keys
            self.lazy_init_param_keys = (
                lazy_init_param_keys if lazy_init_param_keys else []
            )
            self.lazy_init_buffer_dict = (
                lazy_init_buffer_dict if lazy_init_buffer_dict else {}
            )
            self._osft_handles: dict[str, tuple[weakref[nn.Module], str]] = {}
            self.logical_osft_keys = []
            self.orig_param_registry: dict[
                str, ParamSpec
            ] = {}  # stores all of the original params
            self.osft_paramspec_registry: dict[str, OSFTFactorSpec] = {}

            # We want to define how we will upcast & what precision we'll store the SVD
            # params in. Higher precision is best, but expensive during training, so
            # we use a higher precision data type for computing to/from SVD components
            # and store in the original data-type by default (usually bf16)
            self.upcast_dtype = upcast_dtype
            # Handle cases where the base model doesn't have a dtype attribute
            default_dtype = getattr(self, "dtype", torch.bfloat16)
            self.output_dtype = (
                output_dtype if output_dtype is not None else default_dtype
            )

            self._reset_osft_metadata()

            if initialize_osft:
                log_rank_0("initializing OSFT model parameters")
                self._initialize_osft_parameters(decompose_existing_weights=True)

        def _reset_osft_metadata(self):
            """
            Reset tracking structures that tie original parameter names to their OSFT
            projections. This keeps the non-distributed path aligned with the new
            distributed initialization flow.
            """
            self.name_mapping = {}
            self.logical_osft_keys = []
            self.orig_param_registry = {}
            self.osft_paramspec_registry = {}
            self._osft_handles = {}
            self.osft_params = {}

        @staticmethod
        def _load_non_distributed(
            actual_osft_cls,
            pretrained_model_name_or_path: str,
            model_args: tuple,
            base_kwargs: dict,
            osft_class_kwargs: dict,
        ):
            """
            Standard non-distributed loading for OSFT models.

            This method uses the parent class's from_pretrained directly,
            which is simpler and doesn't require distributed coordination.

            Args:
                actual_osft_cls: The OSFT model class to instantiate
                pretrained_model_name_or_path: Model path or name
                model_args: Positional arguments for model loading
                base_kwargs: Base model kwargs (already filtered)
                osft_class_kwargs: OSFT class-specific parameters

            Returns:
                Loaded OSFT model
            """
            log_rank_0("using simple (non-distributed) OSFT loading")

            # get the base model class from the OSFT class inheritance chain
            base_model_class = None
            for base in actual_osft_cls.__mro__:
                if (
                    hasattr(base, "from_pretrained")
                    and base != actual_osft_cls
                    and "WithOSFT" not in base.__name__
                ):
                    base_model_class = base
                    break

            if base_model_class is None:
                raise ValueError(
                    f"Could not find base model class in inheritance chain of {actual_osft_cls}"
                )

            log_rank_0(f"🎯 Using base model class: {base_model_class.__name__}")

            # remove additional OSFT parameters before calling base model's from_pretrained
            final_base_kwargs = _filter_osft_parameters(
                base_kwargs, OSFT_BASE_MODEL_FILTERED_PARAMS
            )

            # load the base model directly using parent's from_pretrained
            log_rank_0(f"📥 Loading base model from {pretrained_model_name_or_path}...")
            base_model = base_model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **final_base_kwargs,
            )

            # extract config and state dict
            config = base_model.config
            state_dict = base_model.state_dict()

            # create OSFT model with config
            log_rank_0("🔧 Creating OSFT model wrapper...")
            model = actual_osft_cls(
                config=config,
                osft_config={},  # Will be set later
                initialize_osft=False,
                upcast_dtype=osft_class_kwargs.get("upcast_dtype", torch.float32),
                output_dtype=osft_class_kwargs.get("output_dtype", None),
                fsdp2_lazy_init=False,
            )

            # load the state dict into the OSFT model
            model.load_state_dict(state_dict)

            # clean up base model
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_rank_0("✅ Non-distributed OSFT model loaded successfully")
            return model

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            *model_args,
            target_patterns=None,
            rank_ratio=0.5,
            fsdp2_lazy_init=False,
            **kwargs,
        ) -> type[OSFTModel]:
            """Load pretrained weights and automatically initialize OSFT parameters.

            Args:
                fsdp2_lazy_init:
                    When this setting is enabled, the model is initialized in a memory-efficient way and assumes
                    that there is a future FSDP2 sharding that will happen. This is only compatible with torch.distributed.
            """
            log_rank_0("\033[33m!!!! Calling from_pretrained !!!!\033[0m")

            initialize_osft = kwargs.pop("initialize_osft", False)

            # validation
            if fsdp2_lazy_init:
                if not dist.is_available() or not dist.is_initialized():
                    raise ValueError(
                        "FSDP2 lazy initialization requires torch.distributed to be available and initialized. "
                        "Either initialize distributed training or set fsdp2_lazy_init=False for non-distributed loading."
                    )

            # Check if this is a GPT-OSS model
            is_gpt_oss = is_gpt_oss_model(pretrained_model_name_or_path)

            # Extract OSFT class-specific kwargs
            osft_class_kwargs, filtered_kwargs = _extract_osft_class_kwargs(kwargs)

            # Apply model-specific parameter filtering
            if is_gpt_oss:
                base_kwargs = _filter_osft_parameters(
                    filtered_kwargs, OSFT_GPT_OSS_FILTERED_PARAMS
                )
                # For GPT-OSS, we need to use the specific model class
                actual_osft_cls = create_osft_model_class(GptOssForCausalLM)
            else:
                base_kwargs = filtered_kwargs.copy()
                base_kwargs["initialize_osft"] = False
                actual_osft_cls = cls

            # choose loading path based on fsdp2_lazy_init flag
            if fsdp2_lazy_init:
                # memory-efficient distributed loading
                log_rank_0(
                    "🧠 distributed environment detected, using memory-efficient loading strategy"
                )
                model = _load_model_memory_efficient(
                    actual_osft_cls,
                    pretrained_model_name_or_path,
                    model_args,
                    base_kwargs,
                    osft_class_kwargs,
                )
            else:
                # standard non-distributed loading
                log_rank_0(
                    f"⚡ Using standard model loading (model_type: {'gpt_oss' if is_gpt_oss else 'standard'})"
                )
                model = cls._load_non_distributed(
                    actual_osft_cls,
                    pretrained_model_name_or_path,
                    model_args,
                    base_kwargs,
                    osft_class_kwargs,
                )

            # quickly check this
            if fsdp2_lazy_init and not model._lazy_init_pending:
                raise ValueError(
                    "FSDP2 lazy initialization was requested but model._lazy_init_pending is False"
                )

            # we always generate and OSFT config since we have no usage in the current codebase
            # where we provide the osft config when calling from_pretrained
            osft_config = auto_generate_target_osft_config(
                model,
                model_name_or_class=pretrained_model_name_or_path,
                target_patterns=target_patterns,
                rank_ratio=rank_ratio,
            )

            model.osft_config = osft_config

            # Decompose weights into high/low rank components
            if initialize_osft:
                log_rank_0("initializing OSFT model")
                if fsdp2_lazy_init:
                    log_rank_0(
                        "distributed loading strategy enabled, loading state dict on rank 0 and preparing to distribute to all ranks"
                    )
                    model._pre_fsdp2_wrap_initialize_lazy_osft()
                    model._pre_fsdp2_wrap_synchronize_buffers()
                    log_rank_0("✅ Prepared OSFT model for distributed loading")
                else:
                    model.reinitialize_osft(decompose_existing_weights=True)
                    log_rank_0("✅ OSFT model initialized successfully")
            return model

        @staticmethod
        def compute_distributed_svd(
            dist_model,
            rank0_og_osft_state_dict: dict[str, torch.Tensor],
            offload_to_cpu: bool = False,
        ):
            """Rank 0 distributes work to all nodes, SVD gets computed across all ranks,
            then they communicate the SVD back to the originating node.

            Args:
                params_to_compute:
                    A dict mapping logical keys to the tensors the SVD will be
                    computed on. For the rank 0 node, the tensors will contain live data.
                    For all other ranks, the mapping will be to meta devices.

            """
            log_rank_0(
                "🔄 [compute_distributed_svd] Starting distributed SVD computation"
            )

            # Helfpul vars
            current_rank = dist.get_rank()
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            local_device = torch.device("cuda", local_rank)
            main_proc_rank = 0
            is_main_proc = current_rank == main_proc_rank
            world_size = dist.get_world_size()

            # all params to be computed in distributed SVD are moved from the OG state dict
            # into `params_to_compute` on rank 0, while all other procs initialize a meta
            # tensor of the same shape + dtype in order to properly populate the data afterwards
            params_to_compute = {}
            for lk, spec in dist_model.orig_param_registry.items():
                if spec.role == "osft_target":
                    # only the first rank will send out the real parameter, all other processes
                    # will use the metadata to create a simple one
                    if current_rank == 0:
                        # remove the param from the state dict so we can clear the data properly
                        # afterwards
                        params_to_compute[lk] = rank0_og_osft_state_dict.pop(lk)
                    else:
                        # no other ranks will have the full state dict, so instead they initialize a tensor of equal shape on the meta device
                        params_to_compute[lk] = torch.zeros(
                            size=spec.shape,
                            dtype=spec.dtype,
                            device=torch.device("meta"),
                        )

            # Sanity check:
            # We need to make sure that we clear the data of the old parameters as we move
            # each param from rank 0 to the intended target.
            if is_main_proc:
                if any(k in rank0_og_osft_state_dict for k in params_to_compute.keys()):
                    raise ValueError(
                        "key still in rank0_og_osft_state_dict after removing it"
                    )

            # At this point we should have OSFT params to train. If not, raise an error
            # since this degenerates down to SFT if not managed
            if not any(
                [
                    x.role == "osft_target"
                    for x in dist_model.orig_param_registry.values()
                ]
            ):
                raise ValueError(
                    "No OSFT target parameters found in orig_param_registry"
                )
            if len(dist_model.orig_param_registry) == 0:
                raise ValueError("orig_param_registry is empty")
            if len(params_to_compute) == 0:
                raise ValueError("No parameters to compute SVD for")

            # The following section assigns each rank in the process group a set of
            # parameters to compute the SVD on. This will determine where rank 0 sends the data.

            # Compute the assignments
            params_per_rank = len(params_to_compute) // world_size
            params_to_compute_list = list[tuple[str, torch.Tensor]](
                params_to_compute.items()
            )
            work_assignments = []
            for rank_idx in range(world_size):
                start_idx = rank_idx * params_per_rank
                # give the last rank any remaining parameters
                if rank_idx == world_size - 1:
                    end_idx = len(params_to_compute_list)
                else:
                    end_idx = start_idx + params_per_rank
                work_assignments.append(params_to_compute_list[start_idx:end_idx])

            log_rank_0(
                f"📊 [compute_distributed_svd] Distributing work across {world_size} ranks"
            )
            log_rank_0(f"   • Total parameters: {len(params_to_compute)}")
            log_rank_0(f"   • Parameters per rank (base): {params_per_rank}")
            log_rank_0(f"   • Last rank gets: {len(work_assignments[-1])} parameters")

            # The torch.distributed.*_object_list set of APIs have an active memory leak issue when sending CPU-based objects
            # in a NCCL-based process group. To prevent these APIs from allocating uncreclaimable memory, we create or obtain a CPU-based process-group
            # based on the Gloo backend, which we use for communcating CPU-based objects which aren't yet ready to be moved onto a CUDA device.
            control_pg = get_control_process_group()

            # Next, rank 0 sends the assigned params to its intended recipients
            my_work = []
            for target_rank in range(len(work_assignments)):
                assignment = work_assignments[target_rank]
                if target_rank == main_proc_rank:
                    # main process doesn't need to send
                    if is_main_proc:
                        my_work = assignment

                    # Equally clear this out
                    work_assignments[target_rank] = None
                    continue

                mailbox = [None]

                # transfer logic:
                #   main proc: sends the names + tensors assigned to the proc and deletes the data afterwards
                #   non-main proc: receives the data and prepares to process it in the next step
                if is_main_proc:
                    mailbox = [assignment]
                    dist.send_object_list(
                        mailbox, dst=target_rank, use_batch=True, group=control_pg
                    )

                    # delete params from local list
                    mailbox.pop()
                    del mailbox
                    for _ in range(len(assignment)):
                        _, param = assignment.pop()
                        del param
                        param = None

                    # null the respective assignment entry
                    work_assignments[target_rank] = None

                    # garbage collection
                    torch.cuda.empty_cache()
                    gc.collect()

                elif target_rank == current_rank:
                    # target ranks sends
                    dist.recv_object_list(
                        mailbox, src=main_proc_rank, use_batch=True, group=control_pg
                    )
                    my_work = mailbox.pop()

                # everyone else waits until they're done
                dist.barrier()

            log_rank_0(
                "✅ [compute_distributed_svd] Work distribution complete, starting SVD computation"
            )
            # now each process computes the SVD separately
            processed_svd_dicts = {}
            for logical_key, param in my_work:
                param_gpu = param.to(device=local_device)
                svd_dict = dist_model.process_param_into_svd_dict(
                    param_gpu, logical_key
                )

                # store it and make sure we don't have a lingering reference
                processed_svd_dicts[logical_key] = svd_dict

                # clear the old params
                del param_gpu
                del param
                torch.cuda.empty_cache()
                gc.collect()

            log_rank_0(
                "✅ [compute_distributed_svd] SVD computation complete, gathering results"
            )
            # by now, each rank has a mapping of logical keys --> new SVD params
            # worker procs need to return the processed dicts
            gathered_results = {}
            for sender_rank in range(dist.get_world_size()):
                # again, rank 0 doesn't need to send to itself
                if sender_rank == main_proc_rank:
                    if is_main_proc:
                        gathered_results.update(processed_svd_dicts)
                    continue

                # worker procs send back to main proc
                mailbox = [None]
                if sender_rank == current_rank:
                    mailbox = [processed_svd_dicts]
                    dist.send_object_list(
                        mailbox, dst=main_proc_rank, use_batch=True, group=control_pg
                    )

                    # now we delete the data from memory
                    for k in list(processed_svd_dicts.keys()):
                        w = processed_svd_dicts.pop(k)
                        del w
                        w = None
                    processed_svd_dicts = None

                    # empty cache and gc
                    torch.cuda.empty_cache()
                    gc.collect()

                # main process receives
                elif is_main_proc:
                    dist.recv_object_list(
                        mailbox, src=sender_rank, use_batch=True, group=control_pg
                    )
                    gathered_results.update(mailbox.pop())

            log_rank_0(
                "[compute_distributed_svd] gathering SVD results from world onto main process"
            )
            # this is the final state dict
            finalized_sd = {}
            if is_main_proc:
                for lk, svd_dict in gathered_results.items():
                    # we want to make sure we have this
                    if lk not in dist_model.osft_paramspec_registry:
                        raise RuntimeError(
                            f"key {lk} not in osft param registry, this is what exists:\n{dist_model.osft_paramspec_registry.keys()}"
                        )

                    osft_spec = dist_model.osft_paramspec_registry[lk]

                    # get expected dtype from original parameter spec
                    expected_dtype = (
                        dist_model.orig_param_registry[lk].dtype
                        if lk in dist_model.orig_param_registry
                        else None
                    )

                    # helper to convert dtype if needed
                    def ensure_dtype(tensor, expected_dtype):
                        if expected_dtype and tensor.dtype != expected_dtype:
                            return tensor.to(dtype=expected_dtype)
                        return tensor

                    finalized_sd.update(
                        {
                            osft_spec.U_low: ensure_dtype(
                                svd_dict["U_low"], expected_dtype
                            ),
                            osft_spec.S_low: ensure_dtype(
                                svd_dict["S_low"], expected_dtype
                            ),
                            osft_spec.V_low: ensure_dtype(
                                svd_dict["V_low"], expected_dtype
                            ),
                            osft_spec.U_high: ensure_dtype(
                                svd_dict["U_high"], expected_dtype
                            ),
                            osft_spec.S_high: ensure_dtype(
                                svd_dict["S_high"], expected_dtype
                            ),
                            osft_spec.V_high: ensure_dtype(
                                svd_dict["V_high"], expected_dtype
                            ),
                            osft_spec.rank_high: svd_dict["rank_high"],
                        }
                    )

            # finally, the main process distributes the state dict
            log_rank_0(
                "📤 [compute_distributed_svd] Distributing computed SVD data to sharded models"
            )
            set_model_state_dict(
                model=dist_model,
                model_state_dict=finalized_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                ),
            )

            log_rank_0(
                "✅ [compute_distributed_svd] Distributed SVD computation complete!"
            )

        def _pre_fsdp2_wrap_synchronize_buffers(self):
            """
            This method populates the initial buffers which are on a meta-device with the true
            buffers which were provided during memory-efficient initialization.
            Assumes that the buffers were synchronized during initialization properly, and
            that user is already in a distributed environment.

            **Important**:
                FSDP2 doesn't allow post-wrap mutation, so this method must be called BEFORE
                invoking `fully_shard` on the model.

            """
            # now we can go ahead and make the update
            for bk, data in self.lazy_init_buffer_dict.items():
                mod, attr = self._get_module_by_name(bk)
                if not mod:
                    raise ValueError(
                        f"requested module for buffer '{bk}' came back as None"
                    )

                # checks that the current buffer IS meta
                if (
                    curr_buff := getattr(mod, attr, None)
                ) is None or curr_buff.device.type != "meta":
                    raise RuntimeError(
                        f"expected buffer {attr} of module {mod} to be meta, but got: {curr_buff.device.type}"
                    )

                # check expected dtype from registry
                if bk in self.orig_param_registry:
                    expected_dtype = self.orig_param_registry[bk].dtype
                    if data.dtype != expected_dtype:
                        log_rank_0(
                            f"Converting buffer {bk} from {data.dtype} to {expected_dtype}"
                        )
                        data = data.to(dtype=expected_dtype)

                # this overwrites the buffer currently present (should be meta)
                new_data = data.detach().clone()
                mod.register_buffer(attr, new_data, persistent=True)

                # checks that the current buffer IS meta
                if (
                    curr_buff := getattr(mod, attr, None)
                ) is None or curr_buff.device.type == "meta":
                    raise RuntimeError(
                        f"expected buffer {attr} of module {mod} to be not meta, but got: {curr_buff.device.type}"
                    )

        @staticmethod
        def post_fsdp2_wrap_synchronize_state_dict_across_procs(
            dist_model: "ModelWithOSFT", og_state_dict: dict[str, torch.Tensor]
        ):
            """
            This method broadcasts non-OSFT params across all processes. After these are shared,
            they are cleared from the original state dict in order to free up memory.
            """
            non_osft_sd = {}
            if dist.get_rank() == 0:
                # sanity check
                # Check if all registered parameters are present in the state dict
                missing_keys = [
                    lk
                    for lk in dist_model.orig_param_registry.keys()
                    if lk not in og_state_dict
                ]
                if missing_keys:
                    log_rank_0(
                        f"\033[33m⚠️  WARNING ⚠️  Some registered parameters are missing from state dict: {missing_keys}\033[0m"
                    )

                # now we need to share out all of the non-osft params
                for lk, spec in dist_model.orig_param_registry.items():
                    if spec.role == "osft_target":
                        # all of these are ignored
                        continue

                    # convert dtype to match what was registered during initialization
                    param_value = og_state_dict.pop(lk)
                    expected_dtype = spec.dtype
                    if param_value.dtype != expected_dtype:
                        log_rank_0(
                            f"Converting {lk} from {param_value.dtype} to {expected_dtype}"
                        )
                        param_value = param_value.to(dtype=expected_dtype)

                    non_osft_sd[lk] = param_value

            # now the rank 0 proc shares the state dict
            set_model_state_dict(
                model=dist_model,
                model_state_dict=non_osft_sd,
                options=StateDictOptions(
                    broadcast_from_rank0=True,
                    strict=False,
                    full_state_dict=True,
                ),
            )

            # keys to clear
            keys_to_clear = list[str](non_osft_sd.keys())
            for k in keys_to_clear:
                p = non_osft_sd.pop(k)
                del p

            # empty cache and gc
            torch.cuda.empty_cache()
            gc.collect()

        def _pre_fsdp2_wrap_initialize_lazy_osft(self):
            """
            This method initializes our OSFT parameters as meta devices based on the OSFT config
            and populates the original parameter registry, which we use to recover
            the original model.

            During the procedure for initializing lazy OSFT, we assume all devices on the state dict
            to be meta. Assuming this is the case, we then stage the original weight matrices to become
            OSFT params.
            """
            self._reset_osft_metadata()

            # verify everything is a meta device
            for k, p in self.state_dict().items():
                if p.device.type != "meta":
                    raise RuntimeError(
                        f"excpected '{k}' to be a meta device, but got: {p.device}"
                    )

            # once we've done this, the goal is to iterate through the parameters in the model
            # and place everything in the set of keys

            # populate the parameter registry
            for pk, v in self.state_dict().items():
                # add the param to our registry
                is_osft_param = pk in self.osft_config
                param_role: Role = "osft_target" if is_osft_param else "non_osft"
                self.orig_param_registry[pk] = ParamSpec(
                    role=param_role, dtype=v.dtype, logical_key=pk, shape=v.shape
                )

                # skip if it's not in the registry
                if is_osft_param:
                    self.logical_osft_keys.append(pk)

            # next, we actually register these keys
            # and replace them with the OSFT equivalents
            for key in self.logical_osft_keys:
                # here we build the association to the original key
                mod, attr = self._get_module_by_name(key)
                self._register_osft_target(key, mod, attr)
                self._prepare_osft_param(key)

        def reinitialize_osft(
            self, decompose_existing_weights: bool, assigned_params=None
        ):
            """
            Reinitializes the OSFT decomposition (e.g., when learning a new task in continual learning).

            Arguments:
                decompose_existing_weights (bool):
                    When true, the targeted weights are decomposed to create the OSFT params.
                    Otherwise, we simply create parameters with the expected shapes.
                assigned_params (list, optional):
                    List of (name, param) tuples to process. If None, processes all parameters.
            """
            log_rank_0("🔄 [reinitialize_osft] Starting OSFT reinitialization")
            log_rank_0(f"   • decompose_existing_weights: {decompose_existing_weights}")
            log_rank_0(
                f"   • assigned_params: {len(assigned_params) if assigned_params else 'None (all params)'}"
            )

            self._reset_osft_metadata()

            log_rank_0("🚀 [reinitialize_osft] Calling _initialize_osft_parameters")
            self._initialize_osft_parameters(
                decompose_existing_weights=decompose_existing_weights,
                assigned_params=assigned_params,
            )
            log_rank_0("✅ [reinitialize_osft] Completed _initialize_osft_parameters")

        def reinitialize_osft_distributed(self):
            """
            Convenience wrapper that mirrors the distributed lazy-init pipeline.

            This is primarily used for tests to ensure the method exists, but it also
            provides a single entry point for re-running the distributed initialization
            sequence when torch.distributed is active.
            """
            if not self.fsdp2_lazy_init:
                raise RuntimeError(
                    "reinitialize_osft_distributed is only valid when fsdp2_lazy_init=True"
                )
            self._pre_fsdp2_wrap_initialize_lazy_osft()
            self._pre_fsdp2_wrap_synchronize_buffers()

        @property
        def is_initialized(self):
            return not self._lazy_init_pending

        @property
        def requires_fsdp2_initialization(self):
            """
            Returns true when the OSFT module is loading via the FSDP2
            lazy init method.
            """
            return (
                get_fsdp2_lazy_init_mode(self) == FSDP2_LAZY_INIT_OSFT
                and not self.is_initialized
            )

        def mark_fsdp2_initialized(self):
            """Mark FSDP2 lazy initialization as complete."""
            self._lazy_init_pending = False
            set_fsdp2_lazy_init_mode(self, None)

        def _get_module_by_logical_key(self, logical_key: str):
            """Return (module, attr) using the stable handle; independent of FQNs/wrappers."""
            try:
                wr, attr = self._osft_handles[logical_key]
            except KeyError:
                return None, None
            mod = wr()
            return (mod, attr) if mod is not None else (None, None)

        # call this BEFORE activation checkpointing / fully_shard
        def _register_osft_target(self, logical_key: str, module: nn.Module, attr: str):
            """Record a stable handle to the parent module + attribute name for a target param."""
            # Optional: tag the Parameter for debugging/validation
            p = getattr(module, attr)
            setattr(p, "_osft_key", logical_key)
            self._osft_handles[logical_key] = (weakref(module), attr)

        def _record_osft_factor_spec(
            self, logical_key: str, attr: str
        ) -> OSFTFactorSpec:
            """Create and store the factor spec describing where OSFT tensors live."""
            parent_logical_key = (
                logical_key.rsplit(".", 1)[0] if "." in logical_key else ""
            )

            def _compose(parent: str, suffix: str) -> str:
                return f"{parent}.{suffix}" if parent else suffix

            spec = OSFTFactorSpec(
                parent_key=parent_logical_key,
                U_high=_compose(parent_logical_key, "osft_U_high"),
                S_high=_compose(parent_logical_key, "osft_S_high"),
                V_high=_compose(parent_logical_key, "osft_V_high"),
                U_low=_compose(parent_logical_key, "osft_params.U_low"),
                S_low=_compose(parent_logical_key, "osft_params.S_low"),
                V_low=_compose(parent_logical_key, "osft_params.V_low"),
                rank_high=_compose(parent_logical_key, "osft_params.rank_high"),
            )
            self.osft_paramspec_registry[logical_key] = spec
            return spec

        def eject_og_state_dict(self):
            """
            Removes the original state dict on the rank 0 process and returns it,
            setting its original reference on the model to be None.
            This way, the state dict can remain instanced, but will no longer
            be attached to the lifecyle of this object.
            """

            sd = self._lazy_init_og_state_dict
            self._lazy_init_og_state_dict = None
            return sd

        def _prepare_osft_param(self, logical_key: str):
            """
            Prepares an OSFT parameter by initializing an OSFT module at the given
            key and removes the actual weight.
            """
            mod_ref, attr = self._osft_handles[logical_key]
            mod = mod_ref()
            if mod is None:
                raise ValueError(f"requested module {logical_key} but ref is None")

            # next we register the parameter and remove the attribute
            meta_weight = getattr(mod, attr)
            top_K = self.osft_config[logical_key]
            svd_dict = create_svd_dict(
                meta_weight,
                top_K,
                decompose_existing=False,
                use_meta=True,
                upcast_dtype=torch.float32,
                output_dtype=meta_weight.dtype,
            )

            # next, we create a new module and register the parameters
            # TODO: move these no-grad params onto the SVD module
            mod.register_parameter(
                "osft_U_high", nn.Parameter(svd_dict["U_high"], requires_grad=False)
            )
            mod.register_parameter(
                "osft_S_high", nn.Parameter(svd_dict["S_high"], requires_grad=False)
            )
            mod.register_parameter(
                "osft_V_high", nn.Parameter(svd_dict["V_high"], requires_grad=False)
            )

            # Trainable low-rank components
            module_svd = nn.Module()
            module_svd.U_low = svd_dict["U_low"]
            module_svd.S_low = svd_dict["S_low"]
            module_svd.V_low = svd_dict["V_low"]
            module_svd.rank_high = svd_dict["rank_high"]

            # this we want to improve
            mod.add_module("osft_params", module_svd)

            # Override linear projection to use module-local OSFT params
            # Note: we use the logical key to look up the module dynamically via the handle registry
            # to ensure the reference survives FSDP2 wrapping and activation checkpointing
            def make_forward(lkey):
                def forward(x):
                    owner_mod, _ = self._get_module_by_logical_key(lkey)
                    if owner_mod is None:
                        raise RuntimeError(
                            f"Module for logical key '{lkey}' not found in handle registry"
                        )
                    svd_dict = {
                        "U_high": owner_mod.osft_U_high,
                        "S_high": owner_mod.osft_S_high,
                        "V_high": owner_mod.osft_V_high,
                        "U_low": owner_mod.osft_params.U_low,
                        "S_low": owner_mod.osft_params.S_low,
                        "V_low": owner_mod.osft_params.V_low,
                        "rank_high": owner_mod.osft_params.rank_high,
                    }
                    # retrieve bias dynamically to avoid meta tensor issues
                    bias = getattr(owner_mod, "bias", None)
                    return self._factorized_linear(x, svd_dict, bias)

                return forward

            # update the forward
            mod.forward = make_forward(logical_key)
            meta_weight.requires_grad = False
            self._record_osft_factor_spec(logical_key, attr)

            mod._parameters.pop(attr)

        @torch.no_grad()
        def process_param_into_svd_dict(
            self, param: torch.Tensor, name: str
        ) -> SVDDecompositionDict:
            # Perform SVD on GPU
            top_K = self.osft_config[name]
            svd_dict = create_svd_dict(
                param,
                top_k=top_K,
                decompose_existing=True,
                upcast_dtype=self.upcast_dtype,
                output_dtype=self.output_dtype,
                use_meta=False,
            )
            return svd_dict

        def _get_module_by_name(
            self, name
        ) -> tuple[nn.Module, str] | tuple[None, None]:
            """Helper to traverse and retrieve a module and its attribute by name string (e.g., `model.layers.0.attn.q_proj.weight`)."""
            parts = name.split(".")
            attr = parts[-1]
            mod = self
            for p in parts[:-1]:
                if hasattr(mod, p):
                    mod = getattr(mod, p)
                elif p.isdigit():
                    mod = mod[int(p)]
                else:
                    return None, None
            return mod, attr

        def _initialize_osft_parameters(
            self, decompose_existing_weights: bool, assigned_params=None
        ):
            """
            Applies SVD decomposition to targeted parameters for OSFT and replaces their forward logic.

            This is the key transformation that enables constrained full-parameter updates by:
            - Freezing high-rank components
            - Training only low-rank ones
            - Intercepting the forward pass to use the reconstructed matrix

            Arguments:
                decompose_existing_weights (bool):
                    When true, the targeted weights are decomposed to create the OSFT params.
                    Otherwise, we simply create parameters with the expected shapes.
                assigned_params (list, optional):
                    List of (name, param) tuples to process. If None, processes all parameters.
            """
            log_rank_0(
                "⚙️  [_initialize_osft_parameters] Starting parameter initialization"
            )
            log_rank_0(f"   • decompose_existing_weights: {decompose_existing_weights}")

            self._reset_osft_metadata()

            local_rank = int(os.getenv("LOCAL_RANK", 0))

            all_named_params = list(self.named_parameters())
            if assigned_params is not None:
                named_params = assigned_params
                log_rank_0(
                    f"   • Processing {len(assigned_params)} assigned parameters"
                )
            else:
                named_params = all_named_params
                log_rank_0(f"   • Processing all {len(named_params)} model parameters")

            # Populate registry metadata so later distributed utilities can rely on it
            for param_name, param in all_named_params:
                role: Role = (
                    "osft_target" if param_name in self.osft_config else "non_osft"
                )
                self.orig_param_registry[param_name] = ParamSpec(
                    role=role,
                    dtype=param.dtype,
                    logical_key=param_name,
                    shape=param.shape,
                )
                if role == "osft_target":
                    self.logical_osft_keys.append(param_name)

            # Show progress bar only on the rank doing the work
            if assigned_params is not None and len(assigned_params) > 0:
                if torch.distributed.is_initialized():
                    global_rank = torch.distributed.get_rank()
                    named_params = tqdm(
                        named_params,
                        total=len(named_params),
                        desc=f"[OSFT Init Rank {global_rank}, Local Rank {local_rank}] Decomposing params",
                    )
                else:
                    named_params = tqdm(
                        named_params,
                        total=len(named_params),
                        desc="[OSFT Init] Decomposing params",
                    )

            # Set up target device for memory-efficient operations
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            target_device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            log_rank_0(f"   • target_device: {target_device}")

            osft_params_processed = 0
            for name, param in named_params:
                # Apply SVD only to 2D matrices in the target config (e.g., q_proj, down_proj, etc.)
                if is_osft_param(name, param, self.osft_config):
                    top_k = self.osft_config[name]

                    # Memory monitoring before processing
                    if torch.cuda.is_available():
                        mem_before = torch.cuda.memory_allocated(target_device) / 1e9
                        log_rank_0(
                            f"🔄 Processing {name} with incremental GPU usage (top_k={top_k}) - GPU mem: {mem_before:.2f}GB"
                        )

                    # Memory-efficient processing: move parameter to GPU temporarily for SVD
                    param_gpu = param.data.to(target_device)

                    # Perform SVD on GPU
                    svd_dict = create_svd_dict(
                        param_gpu,
                        top_k=top_k,
                        decompose_existing=decompose_existing_weights,
                        upcast_dtype=self.upcast_dtype,
                        output_dtype=self.output_dtype,
                    )

                    # Move SVD components to target device and clear GPU cache
                    for key in svd_dict:
                        if isinstance(svd_dict[key], torch.Tensor):
                            svd_dict[key] = svd_dict[key].to(target_device)

                    # Clear the temporary GPU + CPU tensor
                    del param_gpu
                    torch.cuda.empty_cache()

                    # Memory monitoring after processing
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated(target_device) / 1e9
                        log_rank_0(
                            f"✅ Completed {name} - GPU mem: {mem_after:.2f}GB (freed: {mem_before - mem_after:.2f}GB)"
                        )
                    safe_name = name.replace(".", "_")
                    self.name_mapping[name] = safe_name

                    # Attach OSFT components to the owning module so only block-local params materialize
                    mod, attr = self._get_module_by_name(name)

                    # Register this target in the handle registry for stable lookups
                    self._register_osft_target(name, mod, attr)

                    # High-rank frozen components
                    mod.register_parameter(
                        "osft_U_high",
                        nn.Parameter(svd_dict["U_high"], requires_grad=False),
                    )
                    mod.register_parameter(
                        "osft_S_high",
                        nn.Parameter(svd_dict["S_high"], requires_grad=False),
                    )
                    mod.register_parameter(
                        "osft_V_high",
                        nn.Parameter(svd_dict["V_high"], requires_grad=False),
                    )
                    # Trainable low-rank components
                    module_svd = nn.Module()
                    module_svd.U_low = svd_dict["U_low"]
                    module_svd.S_low = svd_dict["S_low"]
                    module_svd.V_low = svd_dict["V_low"]
                    module_svd.rank_high = svd_dict["rank_high"]
                    module_svd.safe_name = safe_name
                    mod.add_module("osft_params", module_svd)

                    # Override linear projection to use module-local OSFT params
                    # Note: we use the logical key to look up the module dynamically via the handle registry
                    # to ensure the reference survives FSDP2 wrapping and activation checkpointing
                    def make_forward(lkey):
                        def forward(x):
                            owner_mod, _ = self._get_module_by_logical_key(lkey)
                            if owner_mod is None:
                                raise RuntimeError(
                                    f"Module for logical key '{lkey}' not found in handle registry"
                                )
                            svd_dict = {
                                "U_high": owner_mod.osft_U_high,
                                "S_high": owner_mod.osft_S_high,
                                "V_high": owner_mod.osft_V_high,
                                "U_low": owner_mod.osft_params.U_low,
                                "S_low": owner_mod.osft_params.S_low,
                                "V_low": owner_mod.osft_params.V_low,
                                "rank_high": owner_mod.osft_params.rank_high,
                            }
                            # retrieve bias dynamically to avoid meta tensor issues
                            bias = getattr(owner_mod, "bias", None)
                            return self._factorized_linear(x, svd_dict, bias)

                        return forward

                    mod.forward = make_forward(name)
                    param.requires_grad = False
                    # Remove original parameter so it doesn't get updated
                    mod._parameters.pop(attr, None)
                    self._record_osft_factor_spec(name, attr)
                    torch.cuda.empty_cache()

                    osft_params_processed += 1

            # Barrier for synchronization in distributed setting
            log_rank_0(
                f"✅ [_initialize_osft_parameters] Processed {osft_params_processed} OSFT parameters"
            )
            if dist.is_initialized():
                torch.distributed.barrier()
                log_rank_0("🔄 [_initialize_osft_parameters] All ranks synchronized")

        def _reconstruct_weight_by_safe_name(
            self,
            safe_name,
            upcast_dtype: torch.dtype | None = None,
            output_dtype: torch.dtype | None = None,
        ):
            """
            Reconstructs a decomposed weight matrix from saved buffers + trainable low-rank parameters
            to rebuild the full matrix used in forward.
            """
            upcast_dtype = (
                upcast_dtype if upcast_dtype is not None else self.upcast_dtype
            )
            output_dtype = (
                output_dtype if output_dtype is not None else self.output_dtype
            )
            original = None
            for k, v in self.name_mapping.items():
                if v == safe_name:
                    original = k
                    break
            if original is None:
                raise ValueError(
                    f"Could not find original name for safe name {safe_name}"
                )
            mod, _ = self._get_module_by_name(original)
            svd_dict = self.get_svd_dict_for_module(mod)
            return reconstruct_weight_matrix(
                svd_dict, upcast_dtype=upcast_dtype, output_dtype=output_dtype
            )

        def _reconstruct_weight(
            self,
            original_name,
            upcast_dtype: torch.dtype | None = None,
            output_dtype: torch.dtype | None = None,
        ):
            """Convenience wrapper to reconstruct using the original parameter name."""
            mod, _ = self._get_module_by_name(original_name)
            svd_dict = self.get_svd_dict_for_module(mod)
            return reconstruct_weight_matrix(
                svd_dict, upcast_dtype=upcast_dtype, output_dtype=output_dtype
            )

        def _factorized_linear(self, x, svd_dict, bias=None):
            """
            Efficient factorized linear operation using SVD components.

            Computes: x @ (U_high @ S_high @ V_high + U_low @ S_low @ V_low).T + bias
            As: (x @ V_high.T) @ (S_high * U_high).T + (x @ V_low.T) @ (S_low * U_low).T
            """
            # Extract components
            U_high = svd_dict["U_high"]
            S_high = svd_dict["S_high"]
            V_high = svd_dict["V_high"]
            U_low = svd_dict["U_low"]
            S_low = svd_dict["S_low"]
            V_low = svd_dict["V_low"]

            device = x.device
            dtype = x.dtype

            # Move to correct device (keep native dtype)
            U_high = U_high.to(device=device)
            S_high = S_high.to(device=device)
            V_high = V_high.to(device=device)
            U_low = U_low.to(device=device)
            S_low = S_low.to(device=device)
            V_low = V_low.to(device=device)

            # High-rank path (frozen): x @ V_high.T -> (batch, seq, rank_high)
            x_V_high = x @ V_high.transpose(0, 1)
            result_high = (x_V_high * S_high) @ U_high.transpose(0, 1)

            # Low-rank path (trainable): x @ V_low.T -> (batch, seq, rank_low)
            x_V_low = x @ V_low.transpose(0, 1)
            result_low = (x_V_low * S_low) @ U_low.transpose(0, 1)

            # Combine both paths
            result = result_high + result_low

            # Add bias if present
            if bias is not None:
                result = result + bias.to(device=device, dtype=dtype)

            return result

        def get_svd_dict_for_module(self, module) -> SVDDecompositionDict:
            if not hasattr(module, "osft_params"):
                raise ValueError("Module does not have OSFT parameters attached")
            # Ensure module-local high-rank components exist (skip non-attached holders)
            if not (
                hasattr(module, "osft_U_high")
                and hasattr(module, "osft_S_high")
                and hasattr(module, "osft_V_high")
            ):
                raise ValueError(
                    "Module is missing OSFT high-rank tensors (U/S/V_high)"
                )
            module_svd = module.osft_params
            S_high = module.osft_S_high
            rank_high = S_high.shape[0]
            svd_dict: SVDDecompositionDict = {
                "U_high": module.osft_U_high,
                "S_high": S_high,
                "V_high": module.osft_V_high,
                "U_low": module_svd.U_low,
                "S_low": module_svd.S_low,
                "V_low": module_svd.V_low,
                "rank_high": rank_high,
            }
            return svd_dict

        def project_gradients(self):
            """
            Applies orthogonal projection to gradients of low-rank components to avoid interfering
            with the high-rank subspace encoding prior task knowledge.

            This method should be called after backpropagation and before optimizer step.
            """
            for module in self.modules():
                # Only process real OSFT-attached linear modules, not the top-level container
                if (
                    hasattr(module, "osft_params")
                    and hasattr(module, "osft_U_high")
                    and hasattr(module, "osft_S_high")
                    and hasattr(module, "osft_V_high")
                ):
                    try:
                        svd_dict = self.get_svd_dict_for_module(module)
                    except ValueError:
                        raise ValueError(
                            f"error in projecting gradients for module: {module}"
                        )
                    project_gradient_to_orthogonal_space(svd_dict)

        def prepare_state_dict_for_save(self, state_dict):
            """Reconstruct dense weights into ``state_dict`` for saving with memory optimization."""
            log_rank_0("Reconstructing OSFT weights for checkpoint saving...")

            # Process parameters one at a time to minimize peak memory usage
            main_local_rank = int(os.getenv("LOCAL_RANK", 0))
            for i, (orig, osft_factors) in enumerate(
                tqdm(
                    self.osft_paramspec_registry.items(),
                    desc="Reconstructing OSFT weights, this may take a while...",
                    disable=main_local_rank != 0,
                )
            ):
                # Extract SVD components from CPU state_dict (avoid touching FSDP sharded params)
                U_high = state_dict.pop(osft_factors.U_high)
                S_high = state_dict.pop(osft_factors.S_high)
                V_high = state_dict.pop(osft_factors.V_high)
                U_low = state_dict.pop(osft_factors.U_low)
                S_low = state_dict.pop(osft_factors.S_low)
                V_low = state_dict.pop(osft_factors.V_low)
                W = reconstruct_weight_matrix(
                    {
                        "U_high": U_high,
                        "S_high": S_high,
                        "V_high": V_high,
                        "U_low": U_low,
                        "S_low": S_low,
                        "V_low": V_low,
                    },
                    output_dtype=self.dtype,
                    upcast_dtype=self.upcast_dtype,
                )
                state_dict[orig] = W

                # Clear GPU cache every few parameters to prevent accumulation
                if (i + 1) % OSFT_CACHE_CLEAR_INTERVAL == 0:
                    torch.cuda.empty_cache()
                    log_rank_0(
                        f"Processed {i + 1}/{len(self.osft_paramspec_registry)} OSFT parameters"
                    )

            # Final cleanup
            torch.cuda.empty_cache()
            log_rank_0(
                f"Finished reconstructing {len(self.osft_paramspec_registry)} OSFT parameters"
            )

            return state_dict

    ModelWithOSFT.__name__ = f"{base_cls.__name__}WithOSFT"
    return ModelWithOSFT


def optim_wrapper(optimizer, model):
    """Wrap optimizer.step to project gradients before each update."""
    if not hasattr(model, "project_gradients"):
        return optimizer

    orig_step = optimizer.step

    def step(self, *args, **kwargs):
        model.project_gradients()
        return orig_step(*args, **kwargs)

    optimizer.step = types.MethodType(step, optimizer)
    return optimizer
