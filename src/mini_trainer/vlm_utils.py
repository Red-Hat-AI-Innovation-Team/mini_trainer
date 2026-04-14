"""Utilities for detecting and extracting CausalLM text backbones from VLM models.

Vision-Language Models (VLMs) like Mistral3ForConditionalGeneration wrap a
CausalLM text backbone (e.g. Ministral3ForCausalLM).  This module provides
helpers to detect that wrapping and extract the text backbone so mini-trainer
can treat it as a standard CausalLM for SFT / OSFT training.

For VLMs that have NO standalone CausalLM class (e.g. Qwen3-VL-2B), this
module also provides helpers to load the VLM directly for text-only training.
"""

import torch
import torch.nn as nn
from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING

from mini_trainer.utils import log_rank_0


def is_vlm_with_causal_lm(config) -> bool:
    """Check if a model config is a VLM wrapping a CausalLM text backbone.

    Returns True when the model needs VLM extraction to obtain the trainable
    CausalLM sub-model.  This covers two cases:

    1. The top-level config is NOT in the CausalLM mapping but its nested
       ``text_config`` IS (e.g. Ministral-3 / Mistral3ForConditionalGeneration).
    2. The top-level config IS in the CausalLM mapping, but the resolved class
       is a ``ForConditionalGeneration`` VLM (e.g. Gemma 3, which is
       dual-registered so ``AutoModelForCausalLM`` loads the full VLM).

    Args:
        config: An already-loaded HuggingFace model config object.

    Returns:
        True if the model is a VLM wrapping a CausalLM text backbone.
    """
    text_config = getattr(config, "text_config", None)

    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        # Check what class AutoModelForCausalLM would actually load.
        # Some models (e.g. Gemma 3) are dual-registered and resolve to
        # a ForConditionalGeneration VLM, which still needs extraction.
        resolved_cls = MODEL_FOR_CAUSAL_LM_MAPPING[config.__class__]
        if "ForConditionalGeneration" not in resolved_cls.__name__:
            return False
        if text_config is None:
            return False
        return text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING

    return text_config is not None and text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING


def is_vlm_for_direct_loading(config) -> bool:
    """Check if a model config is a VLM that should be loaded directly for text-only training.

    Returns True when the model is NOT in the CausalLM mapping, has no
    extractable CausalLM text backbone (via ``text_config``), but IS
    registered in the ImageTextToText mapping.  This covers models like
    Qwen3-VL-2B that have no standalone CausalLM class at all.

    Args:
        config: An already-loaded HuggingFace model config object.

    Returns:
        True if the model is a VLM that should be loaded directly.
    """
    from transformers.models.auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING

    # Already a CausalLM — load normally
    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        return False

    # Has an extractable CausalLM text backbone — use extraction path
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        return False

    # Is a VLM with no CausalLM mapping at all — load directly
    return config.__class__ in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING


def load_vlm_for_text_training(model_path: str, load_kwargs: dict) -> nn.Module:
    """Load a VLM directly for text-only training.

    Used for VLM models that have no standalone CausalLM class (detected
    by :func:`is_vlm_for_direct_loading`).  The full VLM is loaded via
    ``AutoModelForImageTextToText`` and used as-is for text-only forward
    passes (input_ids + labels).

    Note: The layer structure for these models is typically
    ``model.model.language_model.layers`` rather than ``model.model.layers``.

    Args:
        model_path: HuggingFace model name or local path.
        load_kwargs: Keyword arguments forwarded to ``from_pretrained``.

    Returns:
        The loaded VLM model ready for text-only training.
    """
    from transformers import AutoModelForImageTextToText

    log_rank_0("🔄 VLM detected (no CausalLM class) – loading directly for text-only training")

    # Filter out None quantization_config to avoid interfering with
    # the model's built-in quantization handling.
    # Also filter out pretrained_model_name_or_path since model_path is passed positionally.
    filtered_kwargs = {
        k: v
        for k, v in load_kwargs.items()
        if k != "pretrained_model_name_or_path" and not (k == "quantization_config" and v is None)
    }
    model = AutoModelForImageTextToText.from_pretrained(model_path, **filtered_kwargs)

    log_rank_0(f"   ✅ Loaded {type(model).__name__} directly for text-only training")
    return model


def _find_text_backbone(vlm_model: nn.Module) -> nn.Module:
    """Auto-detect the text backbone inside a VLM model.

    Tries well-known attribute names first (``language_model``,
    ``text_model``, ``llm``), then falls back to searching
    ``named_children`` for class names containing ``ForCausalLM`` or
    ``TextModel``.

    Args:
        vlm_model: The loaded VLM model.

    Returns:
        The text backbone module.

    Raises:
        ValueError: If no text backbone can be found.
    """
    inner = vlm_model.model if hasattr(vlm_model, "model") else vlm_model

    # Well-known attribute names
    for attr_name in ("language_model", "text_model", "llm"):
        if hasattr(inner, attr_name):
            return getattr(inner, attr_name)

    # Fallback: search named children for common class-name patterns
    for name, child in inner.named_children():
        cls_name = child.__class__.__name__
        if "ForCausalLM" in cls_name or "TextModel" in cls_name:
            return child

    available = [name for name, _ in inner.named_children()]
    raise ValueError(
        f"Cannot find text backbone in {type(vlm_model).__name__}. Available sub-modules on inner model: {available}"
    )


def _dequantize_fp8_model(model: nn.Module) -> None:
    """Dequantize FP8 weights in-place for FSDP compatibility.

    Some models (e.g. Ministral) ship with FP8 quantized weights that include
    scalar parameters like ``weight_scale_inv`` and ``activation_scale``.
    FSDP rejects scalar parameters, so we dequantize the weights back to
    bfloat16 and remove all FP8 scalar parameters before distributed wrapping.

    The original FP8 scales and quantization config are preserved on the model
    (as ``_fp8_scales`` and ``_fp8_quantization_config``) so that
    :func:`requantize_fp8_state_dict` can restore them at checkpoint save time.

    The dequantization formula is:
        real_weight = fp8_weight.to(bfloat16) * weight_scale_inv
    """
    # FP8 scalar parameter names to remove after dequantization.
    # weight_scale_inv: inverse scale for weight quantization
    # activation_scale: scale for activation quantization (inference only)
    _FP8_SCALAR_ATTRS = ("weight_scale_inv", "activation_scale")

    # Store original scales keyed by module path for requantization at save time.
    fp8_scales: dict[str, dict[str, torch.Tensor]] = {}

    dequantized_count = 0
    for mod_name, module in model.named_modules():
        has_fp8 = any(hasattr(module, attr) for attr in _FP8_SCALAR_ATTRS)
        if not has_fp8:
            continue

        # Capture original scales before removing them
        saved = {}
        for attr in _FP8_SCALAR_ATTRS:
            if hasattr(module, attr):
                saved[attr] = getattr(module, attr).detach().clone().cpu()
        if saved:
            fp8_scales[mod_name] = saved

        # Dequantize weight if scale is present
        if hasattr(module, "weight_scale_inv") and hasattr(module, "weight"):
            scale_inv = module.weight_scale_inv
            weight = module.weight
            dtype = torch.bfloat16
            dequantized = weight.to(dtype) * scale_inv.to(dtype)
            module.weight = nn.Parameter(dequantized, requires_grad=weight.requires_grad)

        # Remove all FP8 scalar parameters/buffers
        for attr in _FP8_SCALAR_ATTRS:
            if not hasattr(module, attr):
                continue
            if attr in dict(module.named_parameters(recurse=False)):
                delattr(module, attr)
            elif attr in dict(module.named_buffers(recurse=False)):
                setattr(module, attr, None)

        dequantized_count += 1

    if dequantized_count > 0:
        log_rank_0(f"   Dequantized {dequantized_count} FP8 layers to bfloat16 for FSDP compatibility")
        # Preserve scales and quantization config for checkpoint re-quantization.
        # Store on both the model and its config so the metadata survives
        # model wrapping (OSFT, FSDP) and distributed broadcast.
        model._fp8_scales = fp8_scales
        cfg = getattr(model, "config", None)
        if cfg is not None:
            cfg._fp8_scales = fp8_scales
            if hasattr(cfg, "quantization_config"):
                model._fp8_quantization_config = cfg.quantization_config
                cfg._fp8_quantization_config = cfg.quantization_config
                cfg.quantization_config = None
        # Clear quantization metadata so downstream code doesn't treat
        # the model as quantized during training
        if hasattr(model, "hf_quantizer"):
            model.hf_quantizer = None
        if hasattr(model, "is_loaded_in_8bit"):
            model.is_loaded_in_8bit = False


def requantize_fp8_state_dict(
    state_dict: dict[str, torch.Tensor],
    fp8_scales: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Re-quantize a dequantized state dict back to FP8 for checkpoint saving.

    This is the inverse of :func:`_dequantize_fp8_model`.  It converts
    bfloat16 weights back to ``float8_e4m3fn`` and restores the original
    ``weight_scale_inv`` and ``activation_scale`` entries so the saved
    checkpoint matches the original FP8 format.

    Args:
        state_dict: The model state dict with bfloat16 weights.
        fp8_scales: The ``_fp8_scales`` dict stored by
            :func:`_dequantize_fp8_model`, mapping module paths to their
            original scale tensors.

    Returns:
        A new state dict with FP8 weights and restored scale entries.
    """
    out = {}
    for key, tensor in state_dict.items():
        out[key] = tensor

    for mod_path, scales in fp8_scales.items():
        weight_key = f"{mod_path}.weight"
        if weight_key not in out:
            continue

        weight = out[weight_key]

        # Re-quantize: fp8_weight = real_weight / weight_scale_inv
        if "weight_scale_inv" in scales:
            scale_inv = scales["weight_scale_inv"]
            requantized = (weight.to(torch.float32) / scale_inv.to(torch.float32)).to(torch.float8_e4m3fn)
            out[weight_key] = requantized
            out[f"{mod_path}.weight_scale_inv"] = scale_inv

        # Restore activation_scale as-is
        if "activation_scale" in scales:
            out[f"{mod_path}.activation_scale"] = scales["activation_scale"]

    return out


def extract_causal_lm_from_vlm(model_path: str, load_kwargs: dict) -> nn.Module:
    """Load a VLM and extract the CausalLM text backbone.

    Loads the full VLM via ``AutoModelForImageTextToText``, auto-detects
    the text backbone using :func:`_find_text_backbone`, then creates a
    standalone CausalLM model by transferring weights.

    Args:
        model_path: HuggingFace model name or local path.
        load_kwargs: Keyword arguments forwarded to ``from_pretrained``.

    Returns:
        A standalone CausalLM model with the VLM's text weights.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText

    log_rank_0("🔄 VLM detected – loading full VLM to extract CausalLM text backbone")

    # Filter out None quantization_config to avoid interfering with
    # the model's built-in quantization handling (e.g. FP8 auto-dequant).
    # Also filter out pretrained_model_name_or_path since model_path is passed positionally.
    vlm_kwargs = {
        k: v
        for k, v in load_kwargs.items()
        if k != "pretrained_model_name_or_path" and not (k == "quantization_config" and v is None)
    }
    vlm = AutoModelForImageTextToText.from_pretrained(model_path, **vlm_kwargs)

    # Auto-detect text backbone
    backbone = _find_text_backbone(vlm)

    # Resolve text_config and create standalone CausalLM
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = config.text_config

    # Propagate quantization_config from the VLM config to text_config
    # so FP8 dequantization can preserve and restore it at checkpoint time.
    vlm_quant_cfg = getattr(config, "quantization_config", None)
    if vlm_quant_cfg is not None and not hasattr(text_config, "quantization_config"):
        text_config.quantization_config = vlm_quant_cfg

    causal_lm_class = MODEL_FOR_CAUSAL_LM_MAPPING[text_config.__class__]

    log_rank_0(f"   Extracting {causal_lm_class.__name__} from {type(vlm).__name__}")
    text_model = causal_lm_class(text_config)

    # Transfer backbone weights
    text_model.model = backbone

    # Transfer lm_head
    if hasattr(vlm, "lm_head"):
        text_model.lm_head = vlm.lm_head
    else:
        raise ValueError(f"Cannot extract lm_head from {type(vlm).__name__}")

    del vlm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Dequantize FP8 weights if present — FSDP rejects scalar parameters
    # like weight_scale_inv that come from FP8 quantized models.
    _dequantize_fp8_model(text_model)

    log_rank_0(f"   ✅ Extracted {causal_lm_class.__name__} successfully")
    return text_model


def has_mrope(config) -> bool:
    """Check if a model config uses M-RoPE (multimodal rotary position embeddings).

    Inspects both the top-level config and its ``text_config`` (if present)
    for ``rope_scaling`` or ``rope_parameters`` dicts containing the
    ``mrope_section`` key.

    Args:
        config: An already-loaded HuggingFace model config object.

    Returns:
        True if M-RoPE is detected.
    """
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        for attr in ("rope_scaling", "rope_parameters"):
            rope_obj = getattr(cfg, attr, None)
            if rope_obj is None:
                continue
            # Handle both dict and RopeParameters objects
            if isinstance(rope_obj, dict) and "mrope_section" in rope_obj:
                return True
            if not isinstance(rope_obj, dict) and hasattr(rope_obj, "mrope_section"):
                return True
    return False


def needs_sdpa(config) -> bool:
    """Check if a model requires SDPA instead of Flash Attention 2.

    Returns True when the model has characteristics incompatible with
    Flash Attention 2:
    - M-RoPE (multimodal rotary position embeddings) producing 3D position_ids
    - A timm-based vision tower (TimmWrapperModel rejects flash_attention_2)

    Args:
        config: An already-loaded HuggingFace model config object.

    Returns:
        True if the model should use SDPA attention.
    """
    if has_mrope(config):
        return True

    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        model_type = getattr(vision_config, "model_type", "")
        if model_type in ("timm_wrapper", "gemma3n_vision"):
            return True
        try:
            from transformers.models.auto import MODEL_MAPPING

            if vision_config.__class__ in MODEL_MAPPING:
                vision_cls = MODEL_MAPPING[vision_config.__class__]
                if "Timm" in vision_cls.__name__:
                    return True
        except Exception:
            pass

    return False


def has_timm_vision_tower(config) -> bool:
    """Check if a model config has a timm-based vision tower.

    timm vision towers only support ``eager`` attention. The vision config
    must be patched to use eager while the text model can use FA2/SDPA.

    Args:
        config: An already-loaded HuggingFace model config object.

    Returns:
        True if the model has a timm-based vision tower.
    """
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return False
    model_type = getattr(vision_config, "model_type", "")
    if model_type in ("timm_wrapper", "gemma3n_vision"):
        return True
    try:
        from transformers.models.auto import MODEL_MAPPING

        if vision_config.__class__ in MODEL_MAPPING:
            vision_cls = MODEL_MAPPING[vision_config.__class__]
            if "Timm" in vision_cls.__name__:
                return True
    except Exception:
        pass
    return False
