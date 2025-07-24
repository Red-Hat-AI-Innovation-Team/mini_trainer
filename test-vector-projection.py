import torch
import torch.distributed as dist
import torch.nn as nn
import os
import time
from setup_model_for_training import setup_model
from utils import init_distributed_environment, log_rank_0
from svd_utils import get_svd_target_parameters, create_svd_model_class, reconstruct_weight_matrix
import typer
import tempfile
import random



from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizerFast


# For individual values
def is_effectively_zero(value, atol=1e-20, rtol=1e-12):
    return torch.isclose(value, torch.tensor(0.0, dtype=value.dtype), 
                        atol=atol, rtol=rtol)

def check_orthogonal_result(matrix, original_scale=1e-8):
    """
    Check if result of orthogonal matrix operations is effectively zero
    """
    # Values smaller than 1e-15 are likely numerical noise
    threshold = 1e-15
    
    # Alternative: relative to original scale
    # threshold = original_scale * 1e-8  # 1e-16 for your case
    
    is_zero_mask = torch.abs(matrix) < threshold
    
    # Statistics
    total_elements = matrix.numel()
    zero_elements = is_zero_mask.sum().item()
    max_non_zero = torch.abs(matrix[~is_zero_mask]).max() if (~is_zero_mask).any() else 0
    
    print(f"Elements effectively zero: {zero_elements}/{total_elements}")
    print(f"Max non-zero magnitude: {max_non_zero:.2e}")
    
    return is_zero_mask

def zero_small_values(tensor, threshold=1e-16):
    """
    Zero out values smaller than threshold.
    
    Args:
        tensor: Input tensor
        threshold: Values with abs() < threshold will be set to 0
    
    Returns:
        New tensor with small values zeroed
    """
    return torch.where(torch.abs(tensor) < threshold, 
                      torch.zeros_like(tensor), 
                      tensor)


def project_onto(B: torch.Tensor, V: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Given a vector basis B and set of vectors V defined in the space of B,
    project V away from the top K set of basis vectors in B.
    
    That is, V is projected into a subspace of B orthogonal to the top K vectors in B.

    Let $B \in \mathbb{R}^{n \times n}, V \in \mathbb{n \times m}$, and top_k < n
    """

    # inner product,
    # (n x n)^T x (n x m) --> (n x m) [each row corresponds to basis vector, each column corresponds to a vector in V]
    inner_p = B.T @ V

    # Now we calculate the projections against min_K
    # let ks = n - top_k
    # (n x ks) x (ks x m) --> (n x m)
    Ps =  B[:,top_k:] @ inner_p[top_k:,:]
    return Ps

def projection_test_template():
    """
    This function provides a **template** of how we can test the model's ability to
    project gradients in the low-rank subspace correctly.
    
    This function provides a mock orthogonalization using our own projection logic.
    The real test should rely only on the gradients and the decomposed SVD components.
    """
    hidden_size = 4
    # here we test creating a SUPER SMALL transformer to just validate that we get the intended behavior
    config = LlamaConfig(
        vocab_size=4,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        hidden_size=hidden_size,
        num_attention_heads=1,
        pad_token_id=3,
        mlp_bias=False,
        intermediate_size=hidden_size,
        max_position_embeddings=10,
        num_key_value_heads=1,
        num_hidden_layers=1
    )
    random.seed(42)
    torch.random.manual_seed(42)

    tlm = LlamaForCausalLM(config)
    with tempfile.TemporaryDirectory() as temp_dir:
        fp = os.path.join(temp_dir, 'planck-llama')
        print(f'saving model to {fp!r}')
        tlm.save_pretrained(fp)
        svd_cls = create_svd_model_class(tlm.__class__)
        print(f'loading pretrained model from {fp!r}')
        svd_lm = svd_cls.from_pretrained(fp, config=config,initialize_svd=True, output_dtype=torch.float64, upcast_dtype=torch.float64)

    # for n, p in svd_lm.named_parameters():
    #     print(f"{n} -> {p.shape}")

    # enable training
    svd_lm.train()

    # run a forward pass -- single input sample
    input_ids = torch.randint(0, 4, size=((1, 10)))
    labels = torch.randint(0, 4, size=((1, 10)))

    output = svd_lm(input_ids=input_ids, labels=labels)
    loss = output.loss.float()
    scaling_factor = 1
    loss *= scaling_factor
    loss.backward()
    print(f"loss: {loss.item()}")

    total_checked = 0
    U_correct_orthogonal_pieces = 0
    U_correct_low_value_pieces = 0
    V_correct_orthogonal_pieces = 0
    V_correct_low_value_pieces = 0

    # now that we have access to the SVD params, we can just extract the SVD dict
    with torch.no_grad():
        for safe_name in svd_lm.svd_params.keys():
            total_checked += 1
            svd_dict = svd_lm.get_svd_dict(safe_name)
            # print(f"{safe_name}")
            # print(svd_dict)

            U_full = torch.cat((svd_dict['U_high'], svd_dict['U_low']), dim=1)
            Vt_full = torch.cat((svd_dict['V_high'], svd_dict['V_low']), dim=0)
            S_full = torch.cat((svd_dict['S_high'], svd_dict['S_low']), dim=0)

            # convert this into a diagonal to make the computations easier
            # since these are just small tests it doesn't matter a whole lot
            S_diag = torch.diag(S_full)

            # okay so for all intents and purposes, this does recover the intended values
            W_reconstructed = U_full @ (S_diag @ Vt_full)
            # print(W_reconstructed)

            W_svd_reconstructed = reconstruct_weight_matrix(svd_dict, upcast_dtype=torch.float64, output_dtype=torch.float64)

            assert W_svd_reconstructed.allclose(W_reconstructed)

            # now we want to look at the gradients
            # in particular, if we compute the inner product of the gradients against the basis (U / Vt)
            # we should see that when they are projected correctly, the dot product between any of the gradients and the
            # top-K basis singular vectors is 0
            assert svd_dict['U_low'].grad is not None
            U_low = svd_dict['U_low']
            U_high = svd_dict['U_high']
            Vt_low = svd_dict['V_low']
            Vt_high = svd_dict['V_high']
            S_low = svd_dict['S_low']
            S_high = svd_dict['S_high']
            top_k = svd_dict['rank_high']



            # ================================================================================
            # ||   CHECKING THE LEFT SINGULAR VECTORS                                       ||
            # ================================================================================
            # here, we have an (n x m)  matrix, where each entry i is the i-th basis vector, and each column j is the j-th column from v
            # in our case, it's one of the gradients from U_low, and each entry is the dot product between them.
            # Since this needs to be orthogonal, we would expect the subspace within top-K to be zero
            before_proj = U_full.T @ U_low.grad.data   # map projections back into the basis vectors
            actual_projected = project_onto(U_full, U_low.grad.data, top_k=top_k)
            after_proj = U_full.T @ actual_projected  # map projections back into the basis vectors

            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE NOT ORTHOGONAL TO HIGH COMPONENTS BEFORE PROJECTION
            # --------------------------------------------------------------------------------
            # first check that the actual gradients are not orthogonal to the high singular vector components
            zeroed_entries = zero_small_values(before_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            assert not zeros.equal(zeroed_entries)

            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE ORTHOGONAL TO HIGH COMPONENTS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # check that projecting the gradients makes them orthogonal to the left high singular vector componetns
            zeroed_entries = zero_small_values(after_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            U_correct_orthogonal_pieces += 1 if zeros.equal(zeroed_entries) else 0


            # --------------------------------------------------------------------------------
            # CHECK THAT THE GRADIENTS ARE NOT ORTHOGONAL TO LOW COMPONENTS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # now check the left-singular vectors corresponding to the low
            # singular values
            zeroed_entries = zero_small_values(after_proj[top_k:,:])
            zeros = torch.zeros_like(zeroed_entries)
            U_correct_low_value_pieces += 1 if not zeros.equal(zeroed_entries) else 0



            # Now check right sinular vectors (Vt, dVt)
            # ================================================================================
            # ||   CHECKING THE RIGHT SINGULAR VECTORS                                      ||
            # ================================================================================

            before_proj = Vt_full @ Vt_low.grad.data.T
            dV = Vt_low.grad.data 
            projected = project_onto(Vt_full.T, dV.T, top_k=top_k)
            after_proj = Vt_full @ projected
            assert before_proj.shape == after_proj.shape 

            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE NOT ORTHOGONAL TO HIGH COMPONENTS OF THE RIGHT
            # SINGULAR VECTORS BEFORE PROJECTION
            # --------------------------------------------------------------------------------
            zeroed_entries = zero_small_values(before_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            assert not zeros.equal(zeroed_entries)



            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE ORTHOGONAL TO HIGH COMPONENTS OF THE RIGHT
            # SINGULAR VECTORS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # check that that the gradients are actually orthogonal to the
            # right singular vectors corresponding to high singular values
            zeroed_entries = zero_small_values(after_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            V_correct_orthogonal_pieces += 1 if zeros.equal(zeroed_entries) else 0

            # --------------------------------------------------------------------------------
            # CHECK THAT THE GRADIENTS ARE NOT ORTHOGONAL TO LOW COMPONENTS OF
            # RIGHT SINGULAR VECTORS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # ensure that the gradients are not orthogonal to the right singular
            # vectors corresponding to the low singular values 
            zeroed_entries = zero_small_values(after_proj[top_k:,:]) # here we check after the top_k values
            zeros = torch.zeros_like(zeroed_entries)
            V_correct_low_value_pieces += 1 if not zeros.equal(zeroed_entries) else 0


    print("BEFORE VECTOR PROJECTION")
    print(f"U correct orthogonal pieces: {U_correct_orthogonal_pieces/total_checked*100:.1f}%")
    print(f"U correct low value pieces: {U_correct_low_value_pieces/total_checked*100:.1f}%")
    print(f"V correct orthogonal pieces: {V_correct_orthogonal_pieces/total_checked*100:.1f}%")
    print(f"V correct low value pieces: {V_correct_low_value_pieces/total_checked*100:.1f}%")
    print(f"total counted: {total_checked}")

    
    svd_lm.project_gradients()
    
    total_checked = 0
    U_correct_orthogonal_pieces = 0
    U_correct_low_value_pieces = 0
    V_correct_orthogonal_pieces = 0
    V_correct_low_value_pieces = 0

    # now that we have access to the SVD params, we can just extract the SVD dict
    with torch.no_grad():
        for safe_name in svd_lm.svd_params.keys():
            total_checked += 1
            svd_dict = svd_lm.get_svd_dict(safe_name)
            # print(f"{safe_name}")
            # print(svd_dict)

            U_full = torch.cat((svd_dict['U_high'], svd_dict['U_low']), dim=1)
            Vt_full = torch.cat((svd_dict['V_high'], svd_dict['V_low']), dim=0)
            S_full = torch.cat((svd_dict['S_high'], svd_dict['S_low']), dim=0)

            # convert this into a diagonal to make the computations easier
            # since these are just small tests it doesn't matter a whole lot
            S_diag = torch.diag(S_full)

            # okay so for all intents and purposes, this does recover the intended values
            W_reconstructed = U_full @ (S_diag @ Vt_full)
            # print(W_reconstructed)

            W_svd_reconstructed = reconstruct_weight_matrix(svd_dict, upcast_dtype=torch.float64, output_dtype=torch.float64)

            assert W_svd_reconstructed.allclose(W_reconstructed)

            # now we want to look at the gradients
            # in particular, if we compute the inner product of the gradients against the basis (U / Vt)
            # we should see that when they are projected correctly, the dot product between any of the gradients and the
            # top-K basis singular vectors is 0
            assert svd_dict['U_low'].grad is not None
            U_low = svd_dict['U_low']
            U_high = svd_dict['U_high']
            Vt_low = svd_dict['V_low']
            Vt_high = svd_dict['V_high']
            S_low = svd_dict['S_low']
            S_high = svd_dict['S_high']
            top_k = svd_dict['rank_high']



            # ================================================================================
            # ||   CHECKING THE LEFT SINGULAR VECTORS                                       ||
            # ================================================================================
            # here, we have an (n x m)  matrix, where each entry i is the i-th basis vector, and each column j is the j-th column from v
            # in our case, it's one of the gradients from U_low, and each entry is the dot product between them.
            # Since this needs to be orthogonal, we would expect the subspace within top-K to be zero
            # before_proj = U_full.T @ U_low.grad.data   # map projections back into the basis vectors
            # actual_projected = project_onto(U_full, U_low.grad.data, top_k=top_k)
            # after_proj = U_full.T @ actual_projected  # map projections back into the basis vectors

            # we already projected, so this is now our projection
            after_proj = U_full.T @ U_low.grad.data

            # we already projected so cant check now
            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE NOT ORTHOGONAL TO HIGH COMPONENTS BEFORE PROJECTION
            # --------------------------------------------------------------------------------
            # first check that the actual gradients are not orthogonal to the high singular vector components
            # zeroed_entries = zero_small_values(before_proj[:top_k,:])
            # zeros = torch.zeros_like(zeroed_entries)
            # assert not zeros.equal(zeroed_entries)

            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE ORTHOGONAL TO HIGH COMPONENTS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # check that projecting the gradients makes them orthogonal to the left high singular vector componetns
            zeroed_entries = zero_small_values(after_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            U_correct_orthogonal_pieces += 1 if zeros.equal(zeroed_entries) else 0


            # --------------------------------------------------------------------------------
            # CHECK THAT THE GRADIENTS ARE NOT ORTHOGONAL TO LOW COMPONENTS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # now check the left-singular vectors corresponding to the low
            # singular values
            zeroed_entries = zero_small_values(after_proj[top_k:,:])
            zeros = torch.zeros_like(zeroed_entries)
            U_correct_low_value_pieces += 1 if not zeros.equal(zeroed_entries) else 0



            # Now check right sinular vectors (Vt, dVt)
            # ================================================================================
            # ||   CHECKING THE RIGHT SINGULAR VECTORS                                      ||
            # ================================================================================

            # before_proj = Vt_full @ Vt_low.grad.data.T
            # dV = Vt_low.grad.data 
            # projected = project_onto(Vt_full.T, dV.T, top_k=top_k)
            after_proj = Vt_full @ Vt_low.grad.data.T
            # assert before_proj.shape == after_proj.shape 

            # we already projected, so we cannot check this now
            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE NOT ORTHOGONAL TO HIGH COMPONENTS OF THE RIGHT
            # SINGULAR VECTORS BEFORE PROJECTION
            # --------------------------------------------------------------------------------
            # zeroed_entries = zero_small_values(before_proj[:top_k,:])
            # zeros = torch.zeros_like(zeroed_entries)
            # assert not zeros.equal(zeroed_entries)



            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE ORTHOGONAL TO HIGH COMPONENTS OF THE RIGHT
            # SINGULAR VECTORS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # check that that the gradients are actually orthogonal to the
            # right singular vectors corresponding to high singular values
            zeroed_entries = zero_small_values(after_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            V_correct_orthogonal_pieces += 1 if zeros.equal(zeroed_entries) else 0

            # --------------------------------------------------------------------------------
            # CHECK THAT THE GRADIENTS ARE NOT ORTHOGONAL TO LOW COMPONENTS OF
            # RIGHT SINGULAR VECTORS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # ensure that the gradients are not orthogonal to the right singular
            # vectors corresponding to the low singular values 
            zeroed_entries = zero_small_values(after_proj[top_k:,:]) # here we check after the top_k values
            zeros = torch.zeros_like(zeroed_entries)
            V_correct_low_value_pieces += 1 if not zeros.equal(zeroed_entries) else 0


    print("AFTER VECTOR PROJECTION")
    print(f"U correct orthogonal pieces: {U_correct_orthogonal_pieces/total_checked*100:.1f}%")
    print(f"U correct low value pieces: {U_correct_low_value_pieces/total_checked*100:.1f}%")
    print(f"V correct orthogonal pieces: {V_correct_orthogonal_pieces/total_checked*100:.1f}%")
    print(f"V correct low value pieces: {V_correct_low_value_pieces/total_checked*100:.1f}%")
    print(f"total counted: {total_checked}")


if __name__ == '__main__':
    # Define variables
    hidden_size = 2048
    layers = 36
    scaling_factor = 1
    top_k = 512
    upcast_dtype=torch.float64
    output_dtype=torch.bfloat16

    # Print all variables in a formatted way
    print("\nConfiguration:")
    print("-" * 40)
    current_vars = list(locals().items())
    for var_name, var_value in current_vars:
        # Skip internal/special variables
        if not var_name.startswith('_'):
            print(f"{var_name:20s}: {var_value}")
    print("-" * 40 + "\n")





    # here we test creating a SUPER SMALL transformer to just validate that we get the intended behavior
    config = LlamaConfig(
        vocab_size=4,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        hidden_size=hidden_size,
        num_attention_heads=8,
        pad_token_id=3,
        mlp_bias=False,
        intermediate_size=hidden_size,
        max_position_embeddings=10,
        num_key_value_heads=1,
        num_hidden_layers=layers
    )
    random.seed(42)
    torch.random.manual_seed(42)

    tlm = LlamaForCausalLM(config)
    print(f"num parameters: {sum(p.numel() for p in tlm.parameters()):,}")

    with tempfile.TemporaryDirectory() as temp_dir:
        fp = os.path.join(temp_dir, 'planck-llama')
        print(f'saving model to {fp!r}')
        tlm.save_pretrained(fp)
        svd_cls = create_svd_model_class(tlm.__class__)
        print(f'loading pretrained model from {fp!r}')
        svd_lm = svd_cls.from_pretrained(
            fp,
            config=config,
            initialize_svd=True,
            output_dtype=output_dtype,
            upcast_dtype=upcast_dtype
        )

    # for n, p in svd_lm.named_parameters():
    #     print(f"{n} -> {p.shape}")

    # enable training
    svd_lm.train()

    # run a forward pass -- single input sample
    input_ids = torch.randint(0, 4, size=((1, 10)))
    labels = torch.randint(0, 4, size=((1, 10)))

    output = svd_lm(input_ids=input_ids, labels=labels)
    loss = output.loss.float()
    loss *= scaling_factor
    loss.backward()
    print(f"loss: {loss.item()}")

    total_checked = 0
    U_correct_orthogonal_pieces = 0
    U_correct_low_value_pieces = 0
    V_correct_orthogonal_pieces = 0
    V_correct_low_value_pieces = 0

    # now that we have access to the SVD params, we can just extract the SVD dict
    with torch.no_grad():
        for safe_name in svd_lm.svd_params.keys():
            total_checked += 1
            svd_dict = svd_lm.get_svd_dict(safe_name)
            # print(f"{safe_name}")
            # print(svd_dict)

            U_full = torch.cat((svd_dict['U_high'], svd_dict['U_low']), dim=1)
            Vt_full = torch.cat((svd_dict['V_high'], svd_dict['V_low']), dim=0)
            S_full = torch.cat((svd_dict['S_high'], svd_dict['S_low']), dim=0)

            # convert this into a diagonal to make the computations easier
            # since these are just small tests it doesn't matter a whole lot
            S_diag = torch.diag(S_full)

            # okay so for all intents and purposes, this does recover the intended values
            W_reconstructed = U_full @ (S_diag @ Vt_full)
            # print(W_reconstructed)

            W_svd_reconstructed = reconstruct_weight_matrix(svd_dict, upcast_dtype=upcast_dtype, output_dtype=output_dtype)

            # assert W_svd_reconstructed.allclose(W_reconstructed)

            # now we want to look at the gradients
            # in particular, if we compute the inner product of the gradients against the basis (U / Vt)
            # we should see that when they are projected correctly, the dot product between any of the gradients and the
            # top-K basis singular vectors is 0
            assert svd_dict['U_low'].grad is not None
            U_low = svd_dict['U_low']
            U_high = svd_dict['U_high']
            Vt_low = svd_dict['V_low']
            Vt_high = svd_dict['V_high']
            S_low = svd_dict['S_low']
            S_high = svd_dict['S_high']
            top_k = svd_dict['rank_high']



            # ================================================================================
            # ||   CHECKING THE LEFT SINGULAR VECTORS                                       ||
            # ================================================================================
            # here, we have an (n x m)  matrix, where each entry i is the i-th basis vector, and each column j is the j-th column from v
            # in our case, it's one of the gradients from U_low, and each entry is the dot product between them.
            # Since this needs to be orthogonal, we would expect the subspace within top-K to be zero
            before_proj = U_full.T @ U_low.grad.data   # map projections back into the basis vectors
            # actual_projected = project_onto(U_full, U_low.grad.data, top_k=top_k)
            # after_proj = U_full.T @ actual_projected  # map projections back into the basis vectors
            after_proj = before_proj

            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE NOT ORTHOGONAL TO HIGH COMPONENTS BEFORE PROJECTION
            # --------------------------------------------------------------------------------
            # first check that the actual gradients are not orthogonal to the high singular vector components
            zeroed_entries = zero_small_values(before_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            assert not zeros.equal(zeroed_entries)

            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE ORTHOGONAL TO HIGH COMPONENTS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # check that projecting the gradients makes them orthogonal to the left high singular vector componetns
            zeroed_entries = zero_small_values(after_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            U_correct_orthogonal_pieces += 1 if zeros.equal(zeroed_entries) else 0


            # --------------------------------------------------------------------------------
            # CHECK THAT THE GRADIENTS ARE NOT ORTHOGONAL TO LOW COMPONENTS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # now check the left-singular vectors corresponding to the low
            # singular values
            zeroed_entries = zero_small_values(after_proj[top_k:,:])
            zeros = torch.zeros_like(zeroed_entries)
            U_correct_low_value_pieces += 1 if not zeros.equal(zeroed_entries) else 0



            # Now check right sinular vectors (Vt, dVt)
            # ================================================================================
            # ||   CHECKING THE RIGHT SINGULAR VECTORS                                      ||
            # ================================================================================

            before_proj = Vt_full @ Vt_low.grad.data.T
            # dV = Vt_low.grad.data 
            # projected = project_onto(Vt_full.T, dV.T, top_k=top_k)
            # after_proj = Vt_full @ projected
            after_proj = before_proj
            assert before_proj.shape == after_proj.shape 

            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE NOT ORTHOGONAL TO HIGH COMPONENTS OF THE RIGHT
            # SINGULAR VECTORS BEFORE PROJECTION
            # --------------------------------------------------------------------------------
            zeroed_entries = zero_small_values(before_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            assert not zeros.equal(zeroed_entries)



            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE ORTHOGONAL TO HIGH COMPONENTS OF THE RIGHT
            # SINGULAR VECTORS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # check that that the gradients are actually orthogonal to the
            # right singular vectors corresponding to high singular values
            zeroed_entries = zero_small_values(after_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            V_correct_orthogonal_pieces += 1 if zeros.equal(zeroed_entries) else 0

            # --------------------------------------------------------------------------------
            # CHECK THAT THE GRADIENTS ARE NOT ORTHOGONAL TO LOW COMPONENTS OF
            # RIGHT SINGULAR VECTORS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # ensure that the gradients are not orthogonal to the right singular
            # vectors corresponding to the low singular values 
            zeroed_entries = zero_small_values(after_proj[top_k:,:]) # here we check after the top_k values
            zeros = torch.zeros_like(zeroed_entries)
            V_correct_low_value_pieces += 1 if not zeros.equal(zeroed_entries) else 0


    print("BEFORE VECTOR PROJECTION")
    print(f"U correct orthogonal pieces: {U_correct_orthogonal_pieces/total_checked*100:.1f}%")
    print(f"U correct low value pieces: {U_correct_low_value_pieces/total_checked*100:.1f}%")
    print(f"V correct orthogonal pieces: {V_correct_orthogonal_pieces/total_checked*100:.1f}%")
    print(f"V correct low value pieces: {V_correct_low_value_pieces/total_checked*100:.1f}%")
    print(f"total counted: {total_checked}")

    
    svd_lm.project_gradients()
    
    total_checked = 0
    U_correct_orthogonal_pieces = 0
    U_correct_low_value_pieces = 0
    V_correct_orthogonal_pieces = 0
    V_correct_low_value_pieces = 0

    # now that we have access to the SVD params, we can just extract the SVD dict
    with torch.no_grad():
        for safe_name in svd_lm.svd_params.keys():
            total_checked += 1
            svd_dict = svd_lm.get_svd_dict(safe_name)
            # print(f"{safe_name}")
            # print(svd_dict)

            U_full = torch.cat((svd_dict['U_high'], svd_dict['U_low']), dim=1)
            Vt_full = torch.cat((svd_dict['V_high'], svd_dict['V_low']), dim=0)
            S_full = torch.cat((svd_dict['S_high'], svd_dict['S_low']), dim=0)

            # convert this into a diagonal to make the computations easier
            # since these are just small tests it doesn't matter a whole lot
            S_diag = torch.diag(S_full)

            # okay so for all intents and purposes, this does recover the intended values
            W_reconstructed = U_full @ (S_diag @ Vt_full)
            # print(W_reconstructed)

            W_svd_reconstructed = reconstruct_weight_matrix(svd_dict, upcast_dtype=upcast_dtype, output_dtype=output_dtype)

            # assert W_svd_reconstructed.allclose(W_reconstructed)

            # now we want to look at the gradients
            # in particular, if we compute the inner product of the gradients against the basis (U / Vt)
            # we should see that when they are projected correctly, the dot product between any of the gradients and the
            # top-K basis singular vectors is 0
            assert svd_dict['U_low'].grad is not None
            U_low = svd_dict['U_low']
            U_high = svd_dict['U_high']
            Vt_low = svd_dict['V_low']
            Vt_high = svd_dict['V_high']
            S_low = svd_dict['S_low']
            S_high = svd_dict['S_high']
            top_k = svd_dict['rank_high']



            # ================================================================================
            # ||   CHECKING THE LEFT SINGULAR VECTORS                                       ||
            # ================================================================================
            # here, we have an (n x m)  matrix, where each entry i is the i-th basis vector, and each column j is the j-th column from v
            # in our case, it's one of the gradients from U_low, and each entry is the dot product between them.
            # Since this needs to be orthogonal, we would expect the subspace within top-K to be zero
            # before_proj = U_full.T @ U_low.grad.data   # map projections back into the basis vectors
            # actual_projected = project_onto(U_full, U_low.grad.data, top_k=top_k)
            # after_proj = U_full.T @ actual_projected  # map projections back into the basis vectors

            # we already projected, so this is now our projection
            after_proj = U_full.T @ U_low.grad.data

            # we already projected so cant check now
            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE NOT ORTHOGONAL TO HIGH COMPONENTS BEFORE PROJECTION
            # --------------------------------------------------------------------------------
            # first check that the actual gradients are not orthogonal to the high singular vector components
            # zeroed_entries = zero_small_values(before_proj[:top_k,:])
            # zeros = torch.zeros_like(zeroed_entries)
            # assert not zeros.equal(zeroed_entries)

            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE ORTHOGONAL TO HIGH COMPONENTS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # check that projecting the gradients makes them orthogonal to the left high singular vector componetns
            zeroed_entries = zero_small_values(after_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            U_correct_orthogonal_pieces += 1 if zeros.equal(zeroed_entries) else 0


            # --------------------------------------------------------------------------------
            # CHECK THAT THE GRADIENTS ARE NOT ORTHOGONAL TO LOW COMPONENTS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # now check the left-singular vectors corresponding to the low
            # singular values
            zeroed_entries = zero_small_values(after_proj[top_k:,:])
            zeros = torch.zeros_like(zeroed_entries)
            U_correct_low_value_pieces += 1 if not zeros.equal(zeroed_entries) else 0



            # Now check right sinular vectors (Vt, dVt)
            # ================================================================================
            # ||   CHECKING THE RIGHT SINGULAR VECTORS                                      ||
            # ================================================================================

            # before_proj = Vt_full @ Vt_low.grad.data.T
            # dV = Vt_low.grad.data 
            # projected = project_onto(Vt_full.T, dV.T, top_k=top_k)
            after_proj = Vt_full @ Vt_low.grad.data.T
            # assert before_proj.shape == after_proj.shape 

            # we already projected, so we cannot check this now
            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE NOT ORTHOGONAL TO HIGH COMPONENTS OF THE RIGHT
            # SINGULAR VECTORS BEFORE PROJECTION
            # --------------------------------------------------------------------------------
            # zeroed_entries = zero_small_values(before_proj[:top_k,:])
            # zeros = torch.zeros_like(zeroed_entries)
            # assert not zeros.equal(zeroed_entries)



            # --------------------------------------------------------------------------------
            # CHECKING THE GRADIENTS ARE ORTHOGONAL TO HIGH COMPONENTS OF THE RIGHT
            # SINGULAR VECTORS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # check that that the gradients are actually orthogonal to the
            # right singular vectors corresponding to high singular values
            zeroed_entries = zero_small_values(after_proj[:top_k,:])
            zeros = torch.zeros_like(zeroed_entries)
            V_correct_orthogonal_pieces += 1 if zeros.equal(zeroed_entries) else 0

            # --------------------------------------------------------------------------------
            # CHECK THAT THE GRADIENTS ARE NOT ORTHOGONAL TO LOW COMPONENTS OF
            # RIGHT SINGULAR VECTORS AFTER PROJECTION
            # --------------------------------------------------------------------------------
            # ensure that the gradients are not orthogonal to the right singular
            # vectors corresponding to the low singular values 
            zeroed_entries = zero_small_values(after_proj[top_k:,:]) # here we check after the top_k values
            zeros = torch.zeros_like(zeroed_entries)
            V_correct_low_value_pieces += 1 if not zeros.equal(zeroed_entries) else 0


    print("AFTER VECTOR PROJECTION")
    print(f"U correct orthogonal pieces: {U_correct_orthogonal_pieces/total_checked*100:.1f}%")
    print(f"U correct low value pieces: {U_correct_low_value_pieces/total_checked*100:.1f}%")
    print(f"V correct orthogonal pieces: {V_correct_orthogonal_pieces/total_checked*100:.1f}%")
    print(f"V correct low value pieces: {V_correct_low_value_pieces/total_checked*100:.1f}%")
    print(f"total counted: {total_checked}")
