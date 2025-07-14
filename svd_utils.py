import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def decompose_weight_matrix(weight: torch.Tensor, top_k: int):
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
    orig_dtype = weight.dtype
    W = weight.to(torch.float32)    # Ensure numerical stability for SVD
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(top_k, S.shape[0])  # Cap to matrix rank

    # Split high-rank (frozen) and low-rank (trainable) subspaces
    svd = {
        "U_high": U[:, :k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "S_high": S[:k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "V_high": Vt[:k, :].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "U_low": nn.Parameter(U[:, k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "S_low": nn.Parameter(S[k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "V_low": nn.Parameter(Vt[k:, :].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "rank_high": k, # Store for later use in orthogonal projection
    }
    return svd


def reconstruct_weight_matrix(svd_dict):
    """
    Reconstructs the original weight matrix from its SVD components.

    Used for replacing linear layers during inference or forward pass to preserve the weight structure.
    The final matrix is the sum of contributions from both the high-rank (frozen) and low-rank (trainable) components.
    """
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    # Reconstruct high-rank component (frozen during continual learning)
    if U_high.numel() > 0 and S_high.numel() > 0:
        high_part = torch.mm(U_high * S_high.unsqueeze(0), V_high)
    else:
        high_part = torch.zeros(U_low.size(0), V_low.size(1), device=U_high.device)

    # Reconstruct low-rank component (receives task-specific updates)
    if U_low.numel() > 0 and S_low.numel() > 0:
        low_part = torch.mm(U_low * S_low.unsqueeze(0), V_low)
    else:
        low_part = torch.zeros(U_high.size(0), V_high.size(1), device=U_low.device)

    return high_part + low_part


def project_gradient_to_orthogonal_space(svd_dict):
    """
    Projects the gradient of the low-rank parameters (U_low, V_low) to be orthogonal to the frozen high-rank subspace.

    This step ensures that learning new tasks does not interfere with previously learned representations by enforcing an orthogonality constraint.
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
        # Handle sharded tensors in distributed training
        if local_U_high.size(0) != local_dU.size(0):
            rank = torch.distributed.get_rank()
            start = rank * local_dU.size(0)
            end = start + local_dU.size(0)
            local_U_high = local_U_high[start:end]
        proj = local_U_high @ (local_U_high.transpose(0, 1) @ local_dU)
        local_dU.sub_(proj)
        if hasattr(dU, "_local_tensor"):
            dU._local_tensor.copy_(local_dU)
        else:
            dU.copy_(local_dU)

    # Repeat projection for V_low using V_high
    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        local_V_high = getattr(V_high, "to_local", lambda: V_high)()
        local_dV = getattr(dV, "to_local", lambda: dV)()
        if local_V_high.size(1) != local_dV.size(1):
            rank = torch.distributed.get_rank()
            start = rank * local_dV.size(1)
            end = start + local_dV.size(1)
            local_V_high = local_V_high[:, start:end]
        proj = (local_dV @ local_V_high.transpose(0, 1)) @ local_V_high
        local_dV.sub_(proj)
        if hasattr(dV, "_local_tensor"):
            dV._local_tensor.copy_(local_dV)
        else:
            dV.copy_(local_dV)


def auto_generate_target_svd_config(model):
    """
    Automatically selects which weight matrices (attention and MLP blocks) to decompose using SVD and determines their top-k values.

    This heuristic uses 50% of the smaller dimension as the default top-k rank, ensuring preservation of
    high-curvature directions while leaving the rest for adaptation.
    """
    target_patterns = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
    ]
    config = {}
    for name, param in model.named_parameters():
        if any(pat in name for pat in target_patterns) and len(param.shape) == 2:
            # Use x% of effective rank heuristically
            top_k = int(np.floor(min(param.shape) * 0.5))
            full_rank = min(param.shape)
            if top_k >= full_rank:
                top_k = full_rank - 1
            config[name] = top_k
    return config


def create_svd_model_class(base_cls):
    """
    Dynamically creates a subclass of the given `base_cls` that replaces selected linear weights
    with low-rank + high-rank SVD-decomposed versions.

    This class:
    - Initializes frozen high-rank buffers and trainable low-rank parameters.
    - Replaces the forward pass of targeted modules to use reconstructed weights.
    - Projects gradients during training to enforce orthogonality with high-rank subspaces.

    This class enables constrained full fine-tuning using adaptive SVD.
    """

    class ModelWithSVD(base_cls):
        def __init__(self, config, svd_config=None, initialize_svd=True, **kwargs):
            super().__init__(config, **kwargs)
            self.svd_config = svd_config or {}  # Maps parameter names → top_k
            self.name_mapping = {}
            self.svd_params = nn.ModuleDict()   # Stores low-rank trainable SVD components
            if initialize_svd:
                self._initialize_svd_parameters()

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, svd_config=None, **kwargs):
            """Load pretrained weights and automatically initialize SVD parameters."""
            # Do not initialize SVD during the initial construction so we load
            # the original dense weights first
            # First load the base model normally without any SVD kwargs
            init_cfg = svd_config if svd_config is not None else {}
            model = super(ModelWithSVD, cls).from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                svd_config=init_cfg,
                # initialize_svd=False,
                **kwargs,
            )

            # Auto-generate SVD config if not provided
            if svd_config is None:
                svd_config = auto_generate_target_svd_config(model)

            model.svd_config = svd_config

            # Decompose weights into high/low rank components
            model.reinitialize_svd()
            return model

        def reinitialize_svd(self):
            """Reinitializes the decomposition (e.g., when learning a new task in continual learning)."""
            self.name_mapping = {}
            self.svd_params = nn.ModuleDict()
            self._initialize_svd_parameters()
        def _get_module_by_name(self, name):
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

        def _initialize_svd_parameters(self):
            """
            Applies SVD decomposition to targeted parameters and replaces their forward logic.

            This is the key transformation that enables constrained full-parameter updates by:
            - Freezing high-rank components
            - Training only low-rank ones
            - Intercepting the forward pass to use the reconstructed matrix
            """
            for name, param in list(self.named_parameters()):
                # Apply SVD only to 2D matrices in the target config (e.g., q_proj, down_proj, etc.)
                if len(param.shape) == 2 and name in self.svd_config and self.svd_config[name] > 0:
                    top_k = self.svd_config[name]
                    print(f"[SVD Init] Decomposing {name} with top_k={top_k}")
                    svd_dict = decompose_weight_matrix(param.data, top_k=top_k)
                    safe_name = name.replace(".", "_")  # Required for buffer/module naming in PyTorch
                    self.name_mapping[name] = safe_name

                    # Freeze top-k singular directions (U/S/V_high)
                    self.register_buffer(f"{safe_name}_U_high", svd_dict["U_high"])
                    self.register_buffer(f"{safe_name}_S_high", svd_dict["S_high"])
                    self.register_buffer(f"{safe_name}_V_high", svd_dict["V_high"])

                    # Wrapper to hold trainable components
                    module_svd = nn.Module()
                    module_svd.U_low = svd_dict["U_low"]
                    module_svd.S_low = svd_dict["S_low"]
                    module_svd.V_low = svd_dict["V_low"]
                    module_svd.rank_high = svd_dict["rank_high"]
                    module_svd.safe_name = safe_name
                    self.svd_params[safe_name] = module_svd

                    mod, attr = self._get_module_by_name(name)
                    bias = mod.bias if hasattr(mod, "bias") else None

                    # Override linear projection with dynamic reconstruction
                    def make_forward(sn, bias):
                        def forward(x):
                            W = self._reconstruct_weight_by_safe_name(sn)
                            if W.dtype != x.dtype:
                                W = W.to(x.dtype)
                            return F.linear(x, W, bias)
                        return forward

                    mod.forward = make_forward(safe_name, bias)
                    param.requires_grad = False
                    # Remove original parameter so it doesn't get updated
                    mod._parameters.pop(attr, None)

        def _reconstruct_weight_by_safe_name(self, safe_name):
            """
            Reconstructs a decomposed weight matrix from saved buffers + trainable low-rank parameters
            to rebuild the full matrix used in forward.
            """
            U_high = getattr(self, f"{safe_name}_U_high")
            S_high = getattr(self, f"{safe_name}_S_high")
            V_high = getattr(self, f"{safe_name}_V_high")
            module_svd = self.svd_params[safe_name]
            svd_dict = {
                "U_high": U_high,
                "S_high": S_high,
                "V_high": V_high,
                "U_low": module_svd.U_low,
                "S_low": module_svd.S_low,
                "V_low": module_svd.V_low,
            }
            return reconstruct_weight_matrix(svd_dict)

        def _reconstruct_weight(self, original_name):
            """Convenience wrapper to reconstruct using the original parameter name."""
            return self._reconstruct_weight_by_safe_name(self.name_mapping[original_name])

        def project_gradients(self):
            """
            Applies orthogonal projection to gradients of low-rank components to avoid interfering
            with the high-rank subspace encoding prior task knowledge.

            This method should be called after backpropagation and before optimizer step.
            """
            for safe_name, module_svd in self.svd_params.items():
                svd_dict = {
                    "U_high": getattr(self, f"{safe_name}_U_high"),
                    "S_high": getattr(self, f"{safe_name}_S_high"),
                    "V_high": getattr(self, f"{safe_name}_V_high"),
                    "U_low": module_svd.U_low,
                    "S_low": module_svd.S_low,
                    "V_low": module_svd.V_low,
                }
                project_gradient_to_orthogonal_space(svd_dict)

        def prepare_state_dict_for_save(self, state_dict):
            """Reconstruct dense weights into ``state_dict`` for saving."""
            if not hasattr(self, "name_mapping"):
                return state_dict
            for orig, safe in self.name_mapping.items():
                U_high = state_dict.pop(f"{safe}_U_high")
                S_high = state_dict.pop(f"{safe}_S_high")
                V_high = state_dict.pop(f"{safe}_V_high")
                U_low = state_dict.pop(f"svd_params.{safe}.U_low")
                S_low = state_dict.pop(f"svd_params.{safe}.S_low")
                V_low = state_dict.pop(f"svd_params.{safe}.V_low")
                W = reconstruct_weight_matrix(
                    {
                        "U_high": U_high,
                        "S_high": S_high,
                        "V_high": V_high,
                        "U_low": U_low,
                        "S_low": S_low,
                        "V_low": V_low,
                    }
                )
                state_dict[orig] = W
            return state_dict

    ModelWithSVD.__name__ = f"{base_cls.__name__}WithSVD"
    return ModelWithSVD


def optim_wrapper(optimizer, model):
    """Wrap optimizer.step to project gradients before each update."""
    if not hasattr(model, "project_gradients"):
        return optimizer

    import types

    orig_step = optimizer.step

    def step(self, *args, **kwargs):
        model.project_gradients()
        return orig_step(*args, **kwargs)

    optimizer.step = types.MethodType(step, optimizer)
    return optimizer

