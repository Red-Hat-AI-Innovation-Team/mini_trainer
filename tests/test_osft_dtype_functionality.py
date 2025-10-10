"""Tests for OSFT dtype functionality and parameter propagation."""

import pytest
import torch
import torch.nn as nn
import tempfile
from unittest.mock import patch, MagicMock

from mini_trainer.osft_utils import (
    create_svd_dict,
    reconstruct_weight_matrix,
    create_osft_model_class,
    auto_generate_target_osft_config
)
from mini_trainer.setup_model_for_training import setup_model


class TestOSFTDtypeFunctions:
    """Test the core SVD functions handle dtypes correctly."""
    
    def test_create_svd_dict_default_dtypes(self):
        """Test create_svd_dict with default dtype behavior."""
        # Create a test weight matrix
        weight = torch.randn(64, 32, dtype=torch.bfloat16)
        top_k = 16
        
        # Test with default upcast_dtype=float32, output_dtype=None
        svd_dict = create_svd_dict(
            weight, 
            top_k, 
            decompose_existing=True,
            upcast_dtype=torch.float32,
            output_dtype=None
        )
        
        # Should default output_dtype to upcast_dtype when None
        assert svd_dict["U_high"].dtype == torch.float32
        assert svd_dict["S_high"].dtype == torch.float32
        assert svd_dict["V_high"].dtype == torch.float32
        assert svd_dict["U_low"].dtype == torch.float32
        assert svd_dict["S_low"].dtype == torch.float32
        assert svd_dict["V_low"].dtype == torch.float32
    
    def test_create_svd_dict_custom_dtypes(self):
        """Test create_svd_dict with custom upcast and output dtypes."""
        weight = torch.randn(64, 32, dtype=torch.bfloat16)
        top_k = 16
        
        # Test with upcast_dtype=float32, output_dtype=bfloat16
        svd_dict = create_svd_dict(
            weight, 
            top_k, 
            decompose_existing=True,
            upcast_dtype=torch.float32,
            output_dtype=torch.bfloat16
        )
        
        # All components should be in output_dtype
        assert svd_dict["U_high"].dtype == torch.bfloat16
        assert svd_dict["S_high"].dtype == torch.bfloat16
        assert svd_dict["V_high"].dtype == torch.bfloat16
        assert svd_dict["U_low"].dtype == torch.bfloat16
        assert svd_dict["S_low"].dtype == torch.bfloat16
        assert svd_dict["V_low"].dtype == torch.bfloat16
    
    def test_create_svd_dict_without_decomposition(self):
        """Test create_svd_dict creates dummy parameters with correct dtypes."""
        weight = torch.randn(64, 32, dtype=torch.bfloat16)
        top_k = 16
        
        # Test without decomposing existing weights
        svd_dict = create_svd_dict(
            weight, 
            top_k, 
            decompose_existing=False,
            upcast_dtype=torch.float32,
            output_dtype=torch.float16
        )
        
        # Should create zeros with output_dtype
        assert svd_dict["U_high"].dtype == torch.float16
        assert svd_dict["S_high"].dtype == torch.float16
        assert svd_dict["V_high"].dtype == torch.float16
        assert torch.allclose(svd_dict["U_high"], torch.zeros_like(svd_dict["U_high"]))
    
    def test_reconstruct_weight_matrix_dtypes(self):
        """Test reconstruct_weight_matrix handles dtypes correctly."""
        weight = torch.randn(32, 16, dtype=torch.bfloat16)
        top_k = 8
        
        # Create SVD dict
        svd_dict = create_svd_dict(
            weight, 
            top_k, 
            decompose_existing=True,
            upcast_dtype=torch.float32,
            output_dtype=torch.bfloat16
        )
        
        # Test reconstruction with different dtypes
        reconstructed = reconstruct_weight_matrix(
            svd_dict,
            upcast_dtype=torch.float32,
            output_dtype=torch.float16
        )
        
        # Should return tensor in output_dtype
        assert reconstructed.dtype == torch.float16
        assert reconstructed.shape == weight.shape
    
    def test_reconstruct_weight_matrix_no_output_dtype(self):
        """Test reconstruct_weight_matrix when output_dtype is None."""
        weight = torch.randn(32, 16, dtype=torch.bfloat16)
        top_k = 8
        
        svd_dict = create_svd_dict(weight, top_k, decompose_existing=True)
        
        # Reconstruct without specifying output_dtype
        reconstructed = reconstruct_weight_matrix(
            svd_dict,
            upcast_dtype=torch.float32,
            output_dtype=None
        )
        
        # Should stay in upcast_dtype
        assert reconstructed.dtype == torch.float32
    
    def test_svd_reconstruction_accuracy(self):
        """Test that SVD reconstruction preserves the original matrix with high precision."""
        # Use a small matrix for exact reconstruction
        weight = torch.randn(16, 8, dtype=torch.float64)
        top_k = min(weight.shape) - 1  # Almost full rank
        
        svd_dict = create_svd_dict(
            weight, 
            top_k, 
            decompose_existing=True,
            upcast_dtype=torch.float64,
            output_dtype=torch.float64
        )
        
        reconstructed = reconstruct_weight_matrix(
            svd_dict,
            upcast_dtype=torch.float64,
            output_dtype=torch.float64
        )
        
        # Should be very close to original (within numerical precision)
        assert torch.allclose(weight, reconstructed, atol=1e-6)


class TestOSFTModelDtypeIntegration:
    """Test SVD model integration with dtype parameters."""
    
    def test_osft_model_dtype_initialization(self):
        """Test that OSFT model properly initializes with dtype parameters."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.linear = nn.Linear(32, 16)
                self.config = config
        
        # Create OSFT model class
        config = MagicMock()
        config.vocab_size = 1000
        OSFTModelClass = create_osft_model_class(SimpleModel)
        
        # Initialize with custom dtypes
        osft_config = {"linear.weight": 8}
        model = OSFTModelClass(
            config=config,
            osft_config=osft_config,
            initialize_osft=False,  # Don't initialize to avoid needing real weights
            upcast_dtype=torch.float32,
            output_dtype=torch.bfloat16
        )
        
        # Check that dtypes are stored correctly
        assert model.upcast_dtype == torch.float32
        assert model.output_dtype == torch.bfloat16
    
    def test_osft_model_reconstruction_uses_dtypes(self):
        """Test that OSFT model reconstruction uses the stored dtypes."""
        class SimpleModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.linear = nn.Linear(32, 16)
                self.config = config
        
        config = MagicMock()
        config.vocab_size = 1000
        OSFTModelClass = create_osft_model_class(SimpleModel)
        
        osft_config = {"linear.weight": 8}
        model = OSFTModelClass(
            config=config,
            osft_config=osft_config,
            initialize_osft=True,
            upcast_dtype=torch.float32,
            output_dtype=torch.bfloat16
        )
        
        # Test that reconstruction uses the model's dtypes
        if "linear_weight" in model.name_mapping:
            safe_name = model.name_mapping["linear.weight"]
            reconstructed = model._reconstruct_weight_by_safe_name(safe_name)
            
            # Should use model's output_dtype
            assert reconstructed.dtype == torch.bfloat16
    
    @patch('mini_trainer.setup_model_for_training.create_osft_model_class')
    @patch('mini_trainer.setup_model_for_training.align_model_and_tokenizer')
    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.setup_model_for_training.get_model_class_from_config')
    @patch('transformers.AutoModelForCausalLM')
    @patch('mini_trainer.setup_model_for_training.AutoTokenizer')
    @patch('transformers.AutoConfig')
    @patch('mini_trainer.setup_model_for_training.AutoConfig')
    def test_setup_model_assigns_dtype_attributes(self, mock_setup_auto_config, mock_trans_auto_config, mock_tokenizer, mock_model_class,
                                                 mock_get_model_class, mock_get_rank, mock_is_initialized,
                                                 mock_align, mock_create_osft):
        """Test that setup_model correctly assigns dtype attributes to OSFT model."""
        # Create a real object to verify attribute assignment
        class MockOSFTModel:
            def __init__(self):
                self.config = MagicMock() 
                self.config.use_cache = True
                self.__class__.__name__ = "LlamaForCausalLM"  # Supported model name
                # Make reinitialize_osft a MagicMock so we can track calls
                self.reinitialize_osft = MagicMock()
                # Add some dummy parameters for OSFT config generation
                self._parameters = {
                    'model.layers.0.self_attn.q_proj.weight': torch.nn.Parameter(torch.randn(512, 512)),
                    'model.layers.0.self_attn.v_proj.weight': torch.nn.Parameter(torch.randn(512, 512)),
                }
            
            def named_parameters(self):
                """Return dummy parameters for OSFT config generation."""
                return self._parameters.items()
            
            def parameters(self):
                """Return just the parameter values."""
                return self._parameters.values()
            
            def to(self, device):
                return self
        
        # Create the mock OSFT model instance
        mock_osft_model = MockOSFTModel()
        
        # Create a proper base model class with from_pretrained
        class MockBaseModelClass:
            __name__ = "LlamaForCausalLM"
            
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                # This is what super().from_pretrained() will call
                return mock_osft_model
        
        # Mock the temporary base model that gets created and deleted
        mock_temp_model = MagicMock()
        mock_temp_model.config = MagicMock()
        mock_temp_model.__class__ = MockBaseModelClass
        mock_model_class.from_pretrained.return_value = mock_temp_model
        
        # Mock get_model_class_from_config to return the base model class
        mock_get_model_class.return_value = MockBaseModelClass
        
        # Mock AutoConfig to return a non-GPT-OSS config
        mock_osft_config = MagicMock()
        mock_osft_config.model_type = "llama"  # Not GPT-OSS
        mock_setup_auto_config.from_pretrained.return_value = mock_osft_config
        mock_trans_auto_config.from_pretrained.return_value = mock_osft_config
        
        # Mock tokenizer
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.__len__ = lambda: 1000
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
        
        # Mock align_model_and_tokenizer to return the models as-is
        mock_align.side_effect = lambda model, tokenizer: model
        
        # Mock the OSFT class creation to return a function that creates the proper class
        def create_mock_osft_class(base_cls):
            # Create a class that inherits from the base class
            class MockOSFTClass(base_cls):
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    # Set the proper attributes on the model
                    mock_osft_model.upcast_dtype = kwargs.get('upcast_dtype', torch.float32)
                    mock_osft_model.output_dtype = kwargs.get('output_dtype', None)
                    mock_osft_model.reinitialize_osft = MagicMock()
                    return mock_osft_model
            return MockOSFTClass
            
        mock_create_osft.side_effect = create_mock_osft_class
        
        # Call setup_model
        result = setup_model(
            model_name_or_path="test-model",
            osft=True,
            local_rank=0,
            osft_upcast_dtype=torch.float32,
            osft_output_dtype=torch.bfloat16,
            osft_rank_ratio=0.5
        )
        
        # Verify the attributes were set correctly on the returned model
        assert result is mock_osft_model
        assert result.upcast_dtype == torch.float32
        assert result.output_dtype == torch.bfloat16
        
        # Verify the OSFT model was created from the base model class
        mock_create_osft.assert_called_once_with(MockBaseModelClass)
        
        # Verify reinitialize_osft was called
        assert hasattr(result, 'reinitialize_osft')
        result.reinitialize_osft.assert_called_once_with(decompose_existing_weights=True)


class TestOSFTParameterFlow:
    """Test end-to-end parameter flow for dtype parameters."""
    
    @patch('torch.distributed.get_world_size', return_value=1)
    @patch('mini_trainer.train.get_node_rank', return_value=0)
    @patch('mini_trainer.train.setup_model')
    @patch('mini_trainer.train.setup_training_components')
    @patch('mini_trainer.train.train')
    @patch('mini_trainer.train.get_data_loader')
    @patch('mini_trainer.train.calculate_num_training_steps')
    @patch('mini_trainer.train.destroy_distributed_environment')
    @patch('mini_trainer.train.init_distributed_environment')
    @patch('torch.distributed.get_rank')
    def test_dtype_flow_from_main_to_svd_model(self, mock_get_rank, mock_init_dist, mock_destroy_dist, mock_calc_steps, mock_data_loader,
                                               mock_train_fn, mock_setup_components, mock_setup_model, mock_get_node_rank, mock_world_size):
        """Test that dtype parameters flow correctly from main to SVD model creation."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_calc_steps.return_value = 1000
        mock_data_loader.return_value = (MagicMock(), MagicMock())
        
        mock_model = MagicMock()
        mock_model.upcast_dtype = torch.bfloat16
        mock_model.output_dtype = torch.float16
        mock_setup_model.return_value = mock_model
        mock_setup_components.return_value = (mock_model, MagicMock(), MagicMock())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from mini_trainer.train import main
            
            try:
                main(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=1,
                    max_tokens_per_gpu=100,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    osft=True,
                    osft_unfreeze_rank_ratio=0.5,
                    osft_upcast_dtype="bfloat16",
                    osft_output_dtype="float16",
                )
            except SystemExit:
                pass
            
            # Verify setup_model was called with correct dtype parameters
            mock_setup_model.assert_called_once()
            call_kwargs = mock_setup_model.call_args.kwargs
            
            assert call_kwargs['osft_upcast_dtype'] == torch.bfloat16
            assert call_kwargs['osft_output_dtype'] == torch.float16


class TestOSFTDtypeEdgeCases:
    """Test edge cases and error conditions for dtype handling."""
    
    def test_svd_dict_preserves_device(self):
        """Test that SVD dict creation preserves the device of input tensor."""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            weight = torch.randn(32, 16, dtype=torch.bfloat16, device=device)
            
            svd_dict = create_svd_dict(
                weight, 
                top_k=8, 
                decompose_existing=True,
                upcast_dtype=torch.float32,
                output_dtype=torch.bfloat16
            )
            
            # All components should be on the same device as input
            assert svd_dict["U_high"].device == device
            assert svd_dict["S_high"].device == device
            assert svd_dict["V_high"].device == device
            assert svd_dict["U_low"].device == device
    
    def test_reconstruct_with_mismatched_dtypes(self):
        """Test reconstruction when SVD components have different dtypes."""
        weight = torch.randn(32, 16, dtype=torch.float32)
        
        # Create SVD dict with one dtype
        svd_dict = create_svd_dict(
            weight, 
            top_k=8, 
            decompose_existing=True,
            upcast_dtype=torch.float32,
            output_dtype=torch.float32
        )
        
        # Manually change some component dtypes to simulate mixed precision
        svd_dict["U_high"] = svd_dict["U_high"].to(torch.bfloat16)
        svd_dict["S_high"] = svd_dict["S_high"].to(torch.bfloat16)
        
        # Reconstruction should still work by upcasting everything
        reconstructed = reconstruct_weight_matrix(
            svd_dict,
            upcast_dtype=torch.float32,
            output_dtype=torch.bfloat16
        )
        
        assert reconstructed.dtype == torch.bfloat16
        assert reconstructed.shape == weight.shape
    
    def test_svd_dict_invalid_tensor_dimensions(self):
        """Test that create_svd_dict raises error for non-2D tensors."""
        # 1D tensor should raise error
        weight_1d = torch.randn(32)
        with pytest.raises(ValueError, match="non-2D tensor"):
            create_svd_dict(weight_1d, top_k=8)
        
        # 3D tensor should raise error
        weight_3d = torch.randn(32, 16, 8)
        with pytest.raises(ValueError, match="non-2D tensor"):
            create_svd_dict(weight_3d, top_k=8)
