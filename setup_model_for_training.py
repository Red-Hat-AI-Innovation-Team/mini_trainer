import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh

from utils import log_rank_0, patch_target_module

# New simple HF-only activation-checkpointing + FSDP2 wrapper
# This mirrors TorchTitan: checkpoint each block, then shard each block and the full model.
def wrap_fsdp2(model: torch.nn.Module) -> torch.nn.Module:
    # Move model to GPU and disable HuggingFace cache
    model = model.to(torch.device("cuda"))
    if hasattr(model, 'config'):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    # 1) Find the HF transformer block container (GPT2: transformer.h, Llama: model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find transformer block container on model")
    # 2) Activation checkpoint each block
    for idx, block in enumerate(layers):
        layers[idx] = ptd_checkpoint_wrapper(block, preserve_rng_state=False)

    # 3) Build a 1D device mesh over all ranks
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", [world_size], mesh_dim_names=["fsdp"])

    # 4) Mixed-precision policy (bf16)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, 
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16)

    # 4) FSDP2 wrap each block
    for idx, block in enumerate(layers):
        reshard = idx < len(layers) - 1
        fully_shard(block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard)

    # 5) FSDP2 wrap full model
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    return model

def align_model_and_tokenizer(model, tokenizer):
    """
    Aligns the model's vocabulary and special tokens with the tokenizer.
    """
    if len(tokenizer) > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    # Fix any discrepancy between model and tokenizer
    special_tokens = {
        'pad': ('pad_token_id', 'Fixing model pad token id'),
        'bos': ('bos_token_id', 'Fixing model bos token id'),
        'eos': ('eos_token_id', 'Fixing model eos token id')
    }

    for token_type, (token_attr, message) in special_tokens.items():
        model_token = getattr(model.config, token_attr)
        tokenizer_token = getattr(tokenizer, token_attr)
        
        if (model_token is not None and tokenizer_token is not None 
            and model_token != tokenizer_token):
            log_rank_0(
                "\033[38;5;226m"
                f"WARNING: There is a mismatch between {token_type} token id of "
                f"model({model_token}) and tokenizer({tokenizer_token}). "
                f"{message} to be same as tokenizer's {token_type} token id"
                "\033[0m"
            )
            setattr(model.config, token_attr, tokenizer_token)

    return model

def setup_model(model=None, orthogonal_subspace_learning: bool = False, **kwargs):
    base_model_args = {
        "pretrained_model_name_or_path": kwargs['model_name_or_path'],
        "torch_dtype": torch.bfloat16,
    }
    base_model_args["attn_implementation"] = "flash_attention_2"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name_or_path'])

    if kwargs['use_liger_kernels']:
        '''need to patch the loss function to not reduce, so we can reduce across all GPUs'''
        from none_reduction_losses import liger_fixed_fused_linear_cross_entropy_none_reduction
        patch_target_module(
            "liger_kernel.transformers.model.loss_utils.fixed_fused_linear_cross_entropy",
            liger_fixed_fused_linear_cross_entropy_none_reduction,
        )
        from liger_kernel.transformers import AutoLigerKernelForCausalLM as ModelClass
    else:
        from none_reduction_losses import hf_fixed_cross_entropy_none_reduction
        patch_target_module(
            "transformers.loss.loss_utils.fixed_cross_entropy",
            hf_fixed_cross_entropy_none_reduction,
        )
        ModelClass = AutoModelForCausalLM
    
    def load_standard_model():
        model = ModelClass.from_pretrained(**base_model_args)
        return align_model_and_tokenizer(model, tokenizer)
    
    # Load a subclassed model that supports orthogonal subspace learning using SVD decomposition
    def load_svd_model():
        # Import utility to decompose weights and inject projected low-rank updates
        from svd_utils import create_svd_model_class, auto_generate_target_svd_config

        tmp = ModelClass.from_pretrained(**base_model_args)
        tmp = align_model_and_tokenizer(tmp, tokenizer)
        # Estimate the per-layer effective rank for decomposition
        svd_cfg = auto_generate_target_svd_config(tmp)
        # Dynamically subclass model to override linear layers with SVD-decomposed versions
        svd_cls = create_svd_model_class(tmp.__class__)
        cfg = tmp.config
        del tmp
        torch.cuda.empty_cache()
        model = svd_cls.from_pretrained(
            **base_model_args,
            config=cfg,
            svd_config=svd_cfg,
            initialize_svd=False,
        )
        model = align_model_and_tokenizer(model, tokenizer)
        return model
    
    # Choose whether to apply orthogonal subspace learning (OSL) based on `orthogonal_subspace_learning` flag
    # OSL enables continual fine-tuning by constraining updates to low-rank directions orthogonal to critical knowledge that is to be preserved
    model = load_svd_model() if orthogonal_subspace_learning else load_standard_model()

    if model.__class__.__name__ not in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM", 
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "MixtralForCausalLM",
        "GraniteForCausalLM",
    ]:
        log_rank_0(
            f"\033[38;2;255;255;0mWarning: Model class name: {model.__class__.__name__} is not in the list of supported models.\033[0m",
            to_print=True,
        )

    # NOTE: Don't enable HuggingFace gradient checkpointing with FSDP2
    # It causes conflicts. TorchTitan applies PyTorch's checkpoint wrapper
    # BEFORE FSDP2 wrapping if needed.
    # model.gradient_checkpointing_enable()
    # torch.compile(model)
    return model

def setup_training_components(model, **kwargs):
    from transformers import get_scheduler
    
    # Using FSDP2 wrapper
    log_rank_0("Using FSDP2 wrapper")
    model = wrap_fsdp2(model)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=kwargs['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    from svd_utils import optim_wrapper
    optimizer = optim_wrapper(optimizer, model)
    lr_scheduler = get_scheduler(
        name=kwargs['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=kwargs['num_warmup_steps'],
    )
    lr_scheduler.split_batches = True
    lr_scheduler.step() #the scheduler starts at 0 and there's no learning.
    return model, optimizer, lr_scheduler

