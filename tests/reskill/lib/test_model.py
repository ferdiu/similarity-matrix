import os
import torch
import pytest
import psutil
from unittest.mock import Mock, patch, MagicMock
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from unittest.mock import Mock, patch, MagicMock

from similarity_matrix.lib.model import get_model_size, initialize_model
from similarity_matrix.lib.model import get_memory_capacity, get_total_memory_capacity


class TestGetModelSize:
    """Test cases for get_model_size function"""

    def test_get_model_size_default_float_bit(self):
        """Test model size calculation with default 32-bit floats"""
        # Create a mock model with known parameters
        mock_model = Mock()

        # Create mock parameters
        param1 = Mock()
        param1.numel.return_value = 1000
        param1.requires_grad = True

        param2 = Mock()
        param2.numel.return_value = 2000
        param2.requires_grad = True

        param3 = Mock()
        param3.numel.return_value = 500
        param3.requires_grad = False  # This should be excluded

        mock_model.parameters.return_value = [param1, param2, param3]

        from similarity_matrix.lib.model import get_model_size

        # Expected: (1000 + 2000) * 32 / 8 = 3000 * 4 = 12000 bytes
        result = get_model_size(mock_model)
        assert result == 12000

    def test_get_model_size_custom_float_bit(self):
        """Test model size calculation with custom float bit size"""
        mock_model = Mock()

        param1 = Mock()
        param1.numel.return_value = 1000
        param1.requires_grad = True

        mock_model.parameters.return_value = [param1]

        from similarity_matrix.lib.model import get_model_size

        # Expected: 1000 * 16 / 8 = 2000 bytes
        result = get_model_size(mock_model, float_bit=16)
        assert result == 2000

    def test_get_model_size_no_trainable_params(self):
        """Test model size calculation when no parameters require gradients"""
        mock_model = Mock()

        param1 = Mock()
        param1.numel.return_value = 1000
        param1.requires_grad = False

        mock_model.parameters.return_value = [param1]

        from similarity_matrix.lib.model import get_model_size

        result = get_model_size(mock_model)
        assert result == 0

    def test_get_model_size_empty_model(self):
        """Test model size calculation with no parameters"""
        mock_model = Mock()
        mock_model.parameters.return_value = []

        from similarity_matrix.lib.model import get_model_size

        result = get_model_size(mock_model)
        assert result == 0


class TestInitializeModel:
    """Test cases for initialize_model function"""

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.torch.cuda.empty_cache')
    @patch('similarity_matrix.lib.model.SentenceTransformer')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_model_cuda_available_sentence_transformer(
            self,
            mock_logger,
            mock_sentence_transformer,
            mock_empty_cache,
            mock_cuda_available):
        """Test model initialization with CUDA available using SentenceTransformer"""
        mock_cuda_available.return_value = True
        mock_model_instance = Mock()
        mock_sentence_transformer.return_value = mock_model_instance

        from similarity_matrix.lib.model import initialize_model

        result = initialize_model("test-model", use_cuda=True)

        # Verify CUDA setup
        mock_empty_cache.assert_called_once()
        mock_logger.info.assert_called_with("Using GPU for model inference.")

        # Verify model creation
        mock_sentence_transformer.assert_called_once_with(
            "test-model",
            device=torch.device("cuda"),
            trust_remote_code=True
        )
        assert result == mock_model_instance

        # Verify environment variable is set
        assert os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.torch.cuda.empty_cache')
    @patch('similarity_matrix.lib.model.AutoModel')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_model_cuda_available_automodel(
            self,
            mock_logger,
            mock_automodel,
            mock_empty_cache,
            mock_cuda_available):
        """Test model initialization with CUDA available using AutoModel (model with /)"""
        mock_cuda_available.return_value = True
        mock_model_instance = Mock()
        mock_automodel.from_pretrained.return_value = mock_model_instance

        from similarity_matrix.lib.model import initialize_model

        result = initialize_model("org/test-model", use_cuda=True)

        # Verify CUDA setup
        mock_empty_cache.assert_called_once()
        mock_logger.info.assert_called_with("Using GPU for model inference.")

        # Verify model creation
        mock_automodel.from_pretrained.assert_called_once_with(
            "org/test-model",
            device_map='cuda',
            trust_remote_code=True
        )
        assert result == mock_model_instance

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.torch.cuda.empty_cache')
    @patch('similarity_matrix.lib.model.SentenceTransformer')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_model_force_sentence_transformer(
            self,
            mock_logger,
            mock_sentence_transformer,
            mock_empty_cache,
            mock_cuda_available):
        """Test model initialization forcing SentenceTransformer even with / in name"""
        mock_cuda_available.return_value = True
        mock_model_instance = Mock()
        mock_sentence_transformer.return_value = mock_model_instance

        from similarity_matrix.lib.model import initialize_model

        result = initialize_model(
            "org/test-model",
            use_cuda=True,
            force_sentence_transformer=True)

        # Verify SentenceTransformer is used despite / in name
        mock_sentence_transformer.assert_called_once_with(
            "org/test-model",
            device=torch.device("cuda"),
            trust_remote_code=True
        )
        assert result == mock_model_instance

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.SentenceTransformer')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_model_cuda_not_available(
        self, mock_logger, mock_sentence_transformer, mock_cuda_available
    ):
        """Test model initialization when CUDA is not available"""
        mock_cuda_available.return_value = False
        mock_model_instance = Mock()
        mock_sentence_transformer.return_value = mock_model_instance

        from similarity_matrix.lib.model import initialize_model

        result = initialize_model("test-model", use_cuda=True)

        # Verify CPU warning
        mock_logger.warning.assert_called_with(
            "Warning! Using CPU for model inference.")

        # Verify model creation without device specification
        mock_sentence_transformer.assert_called_once_with(
            "test-model",
            trust_remote_code=True
        )
        assert result == mock_model_instance

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.SentenceTransformer')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_model_use_cuda_false(
        self, mock_logger, mock_sentence_transformer, mock_cuda_available
    ):
        """Test model initialization when use_cuda is False"""
        mock_cuda_available.return_value = True  # CUDA available but not requested
        mock_model_instance = Mock()
        mock_sentence_transformer.return_value = mock_model_instance

        from similarity_matrix.lib.model import initialize_model

        result = initialize_model("test-model", use_cuda=False)

        # Verify CPU warning was not called
        mock_logger.warning.assert_not_called()

        # Verify model creation without device specification
        mock_sentence_transformer.assert_called_once_with(
            "test-model",
            trust_remote_code=True
        )
        assert result == mock_model_instance

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.AutoModel')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_model_automodel_cpu(
        self, mock_logger, mock_automodel, mock_cuda_available
    ):
        """Test AutoModel initialization on CPU"""
        mock_cuda_available.return_value = False
        mock_model_instance = Mock()
        mock_automodel.from_pretrained.return_value = mock_model_instance

        from similarity_matrix.lib.model import initialize_model

        result = initialize_model("org/test-model", use_cuda=True)

        # Verify CPU warning
        mock_logger.warning.assert_called_with(
            "Warning! Using CPU for model inference.")

        # Verify model creation without device_map
        mock_automodel.from_pretrained.assert_called_once_with(
            "org/test-model",
            trust_remote_code=True
        )
        assert result == mock_model_instance

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.torch.cuda.empty_cache')
    @patch('similarity_matrix.lib.model.SentenceTransformer')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_model_with_kwargs(
            self,
            mock_logger,
            mock_sentence_transformer,
            mock_empty_cache,
            mock_cuda_available):
        """Test model initialization with additional kwargs"""
        mock_cuda_available.return_value = True
        mock_model_instance = Mock()
        mock_sentence_transformer.return_value = mock_model_instance

        from similarity_matrix.lib.model import initialize_model

        result = initialize_model(
            "test-model",
            use_cuda=True,
            custom_param="test_value",
            another_param=42
        )

        # Verify model creation with kwargs
        mock_sentence_transformer.assert_called_once_with(
            "test-model",
            device=torch.device("cuda"),
            trust_remote_code=True,
            custom_param="test_value",
            another_param=42
        )
        assert result == mock_model_instance

    def test_initialize_model_environment_variable_cleanup(self):
        """Test that environment variable is properly set"""
        # Clean up any existing value
        if "CUDA_LAUNCH_BLOCKING" in os.environ:
            del os.environ["CUDA_LAUNCH_BLOCKING"]

        with patch('similarity_matrix.lib.model.torch.cuda.is_available', return_value=True), \
                patch('similarity_matrix.lib.model.torch.cuda.empty_cache'), \
                patch('similarity_matrix.lib.model.SentenceTransformer'), \
                patch('similarity_matrix.lib.model.logger'):

            from similarity_matrix.lib.model import initialize_model
            initialize_model("test-model", use_cuda=True)

            # Verify environment variable is set
            assert os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"


class TestIntegration:
    """Integration tests for both functions working together"""

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.torch.cuda.empty_cache')
    @patch('similarity_matrix.lib.model.SentenceTransformer')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_and_get_size(
            self,
            mock_logger,
            mock_sentence_transformer,
            mock_empty_cache,
            mock_cuda_available):
        """Test initializing a model and then getting its size"""
        mock_cuda_available.return_value = True

        # Create a mock model with parameters
        mock_model = Mock()
        param1 = Mock()
        param1.numel.return_value = 1000
        param1.requires_grad = True
        mock_model.parameters.return_value = [param1]

        mock_sentence_transformer.return_value = mock_model

        from similarity_matrix.lib.model import initialize_model, get_model_size

        # Initialize model
        model = initialize_model("test-model", use_cuda=True)

        # Get model size
        size = get_model_size(model)

        # Verify results
        assert model == mock_model
        assert size == 4000  # 1000 * 32 / 8


# Fixtures for common test data
@pytest.fixture
def mock_model_with_params():
    """Fixture that creates a mock model with predefined parameters"""
    mock_model = Mock()

    param1 = Mock()
    param1.numel.return_value = 1000
    param1.requires_grad = True

    param2 = Mock()
    param2.numel.return_value = 2000
    param2.requires_grad = True

    mock_model.parameters.return_value = [param1, param2]
    return mock_model


@pytest.fixture
def mock_cuda_environment():
    """Fixture that sets up CUDA mocking"""
    with patch('similarity_matrix.lib.model.torch.cuda.is_available') as mock_available, \
            patch('similarity_matrix.lib.model.torch.cuda.empty_cache') as mock_empty:
        mock_available.return_value = True
        yield mock_available, mock_empty


# Additional test for error handling
class TestErrorHandling:
    """Test error handling scenarios"""

    @patch('similarity_matrix.lib.model.torch.cuda.is_available')
    @patch('similarity_matrix.lib.model.SentenceTransformer')
    @patch('similarity_matrix.lib.model.logger')
    def test_initialize_model_exception_handling(
        self, mock_logger, mock_sentence_transformer, mock_cuda_available
    ):
        """Test that exceptions during model initialization are properly handled"""
        mock_cuda_available.return_value = False
        mock_sentence_transformer.side_effect = Exception(
            "Model loading failed")

        from similarity_matrix.lib.model import initialize_model

        with pytest.raises(Exception, match="Model loading failed"):
            initialize_model("test-model")

    def test_get_model_size_with_none_parameters(self):
        """Test get_model_size with model that has None parameters"""
        mock_model = Mock()
        mock_model.parameters.return_value = [None]

        from similarity_matrix.lib.model import get_model_size

        # This should handle the None parameter gracefully
        with pytest.raises(AttributeError):
            get_model_size(mock_model)


class TestGetMemoryCapacity:
    """Tests for get_memory_capacity function"""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_cuda_available_success(
            self,
            mock_memory_reserved,
            mock_memory_allocated,
            mock_get_device_properties,
            mock_cuda_available):
        """Test successful VRAM detection when CUDA is available"""
        # Setup mocks
        mock_cuda_available.return_value = True

        mock_device_properties = Mock()
        mock_device_properties.name = "NVIDIA GeForce RTX 3080"
        mock_device_properties.total_memory = 10 * 1024**3  # 10 GB
        mock_get_device_properties.return_value = mock_device_properties

        mock_memory_allocated.return_value = 1 * 1024**3  # 1 GB allocated
        mock_memory_reserved.return_value = 2 * 1024**3   # 2 GB reserved

        # Call function
        result = get_memory_capacity()

        # Expected: 10 GB total - 2 GB reserved = 8 GB available
        expected = 8.0
        assert abs(result - expected) < 0.01

        # Verify function calls
        mock_cuda_available.assert_called_once()
        mock_get_device_properties.assert_called_once_with(0)
        mock_memory_allocated.assert_called_once_with(0)
        mock_memory_reserved.assert_called_once_with(0)

    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_cuda_not_available_fallback_to_ram(
            self, mock_virtual_memory, mock_cuda_available):
        """Test fallback to RAM when CUDA is not available"""
        # Setup mocks
        mock_cuda_available.return_value = False

        mock_memory_info = Mock()
        mock_memory_info.total = 16 * 1024**3      # 16 GB total
        mock_memory_info.available = 12 * 1024**3  # 12 GB available
        mock_virtual_memory.return_value = mock_memory_info

        # Call function
        result = get_memory_capacity()

        # Expected: 12 GB available RAM
        expected = 12.0
        assert abs(result - expected) < 0.01

        # Verify function calls
        mock_cuda_available.assert_called_once()
        mock_virtual_memory.assert_called_once()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('psutil.virtual_memory')
    def test_cuda_exception_fallback_to_ram(
            self,
            mock_virtual_memory,
            mock_get_device_properties,
            mock_cuda_available):
        """Test fallback to RAM when CUDA operations raise exception"""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_get_device_properties.side_effect = RuntimeError("CUDA error")

        mock_memory_info = Mock()
        mock_memory_info.total = 8 * 1024**3     # 8 GB total
        mock_memory_info.available = 6 * 1024**3  # 6 GB available
        mock_virtual_memory.return_value = mock_memory_info

        # Call function
        result = get_memory_capacity()

        # Expected: 6 GB available RAM (fallback)
        expected = 6.0
        assert abs(result - expected) < 0.01

    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_device_parameter_cpu(
            self,
            mock_virtual_memory,
            mock_cuda_available):
        """Test explicit CPU device parameter"""
        # Setup mocks
        mock_cuda_available.return_value = True  # CUDA available but we force CPU

        mock_memory_info = Mock()
        mock_memory_info.total = 32 * 1024**3     # 32 GB total
        mock_memory_info.available = 24 * 1024**3  # 24 GB available
        mock_virtual_memory.return_value = mock_memory_info

        # Call function with device='cpu'
        result = get_memory_capacity(device='cpu')

        # Expected: 24 GB available RAM
        expected = 24.0
        assert abs(result - expected) < 0.01

        # CUDA functions should not be called
        mock_virtual_memory.assert_called_once()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_device_parameter_cuda(
            self,
            mock_memory_reserved,
            mock_memory_allocated,
            mock_get_device_properties,
            mock_cuda_available):
        """Test explicit CUDA device parameter"""
        # Setup mocks
        mock_cuda_available.return_value = True

        mock_device_properties = Mock()
        mock_device_properties.name = "Tesla V100"
        mock_device_properties.total_memory = 16 * 1024**3  # 16 GB
        mock_get_device_properties.return_value = mock_device_properties

        mock_memory_allocated.return_value = 2 * 1024**3  # 2 GB allocated
        mock_memory_reserved.return_value = 4 * 1024**3   # 4 GB reserved

        # Call function with device='cuda'
        result = get_memory_capacity(device='cuda')

        # Expected: 16 GB total - 4 GB reserved = 12 GB available
        expected = 12.0
        assert abs(result - expected) < 0.01

    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_device_parameter_cuda_not_available(
            self, mock_virtual_memory, mock_cuda_available):
        """Test CUDA device parameter when CUDA is not available"""
        # Setup mocks
        mock_cuda_available.return_value = False

        mock_memory_info = Mock()
        mock_memory_info.total = 8 * 1024**3     # 8 GB total
        mock_memory_info.available = 5 * 1024**3  # 5 GB available
        mock_virtual_memory.return_value = mock_memory_info

        # Call function with device='cuda' but CUDA not available
        result = get_memory_capacity(device='cuda')

        # Expected: fallback to RAM
        expected = 5.0
        assert abs(result - expected) < 0.01

    def test_invalid_device_parameter(self):
        """Test invalid device parameter raises ValueError"""
        with pytest.raises(ValueError, match="Invalid device: invalid"):
            get_memory_capacity(device='invalid')

    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_case_insensitive_device_parameter(
            self, mock_virtual_memory, mock_cuda_available):
        """Test device parameter is case insensitive"""
        # Setup mocks
        mock_cuda_available.return_value = False

        mock_memory_info = Mock()
        mock_memory_info.total = 16 * 1024**3     # 16 GB total
        mock_memory_info.available = 10 * 1024**3  # 10 GB available
        mock_virtual_memory.return_value = mock_memory_info

        # Test various case combinations
        for device in ['CPU', 'Cpu', 'cPu']:
            result = get_memory_capacity(device=device)
            expected = 10.0
            assert abs(result - expected) < 0.01


class TestGetTotalMemoryCapacity:
    """Tests for get_total_memory_capacity function"""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_cuda_available_success(
            self,
            mock_get_device_properties,
            mock_cuda_available):
        """Test successful total VRAM detection when CUDA is available"""
        # Setup mocks
        mock_cuda_available.return_value = True

        mock_device_properties = Mock()
        mock_device_properties.name = "NVIDIA A100"
        mock_device_properties.total_memory = 40 * 1024**3  # 40 GB
        mock_get_device_properties.return_value = mock_device_properties

        # Call function
        result = get_total_memory_capacity()

        # Expected: 40 GB total VRAM
        expected = 40.0
        assert abs(result - expected) < 0.01

        # Verify function calls
        mock_cuda_available.assert_called_once()
        mock_get_device_properties.assert_called_once_with(0)

    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_cuda_not_available_fallback_to_ram(
            self, mock_virtual_memory, mock_cuda_available):
        """Test fallback to total RAM when CUDA is not available"""
        # Setup mocks
        mock_cuda_available.return_value = False

        mock_memory_info = Mock()
        mock_memory_info.total = 64 * 1024**3  # 64 GB total
        mock_virtual_memory.return_value = mock_memory_info

        # Call function
        result = get_total_memory_capacity()

        # Expected: 64 GB total RAM
        expected = 64.0
        assert abs(result - expected) < 0.01

        # Verify function calls
        mock_cuda_available.assert_called_once()
        mock_virtual_memory.assert_called_once()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('psutil.virtual_memory')
    def test_cuda_exception_fallback_to_ram(
            self,
            mock_virtual_memory,
            mock_get_device_properties,
            mock_cuda_available):
        """Test fallback to RAM when CUDA operations raise exception"""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_get_device_properties.side_effect = Exception(
            "GPU not accessible")

        mock_memory_info = Mock()
        mock_memory_info.total = 128 * 1024**3  # 128 GB total
        mock_virtual_memory.return_value = mock_memory_info

        # Call function
        result = get_total_memory_capacity()

        # Expected: 128 GB total RAM (fallback)
        expected = 128.0
        assert abs(result - expected) < 0.01

    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_device_parameter_cpu(
            self,
            mock_virtual_memory,
            mock_cuda_available):
        """Test explicit CPU device parameter"""
        # Setup mocks
        mock_cuda_available.return_value = True  # CUDA available but we force CPU

        mock_memory_info = Mock()
        mock_memory_info.total = 32 * 1024**3  # 32 GB total
        mock_virtual_memory.return_value = mock_memory_info

        # Call function with device='cpu'
        result = get_total_memory_capacity(device='cpu')

        # Expected: 32 GB total RAM
        expected = 32.0
        assert abs(result - expected) < 0.01

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_device_parameter_cuda(
            self,
            mock_get_device_properties,
            mock_cuda_available):
        """Test explicit CUDA device parameter"""
        # Setup mocks
        mock_cuda_available.return_value = True

        mock_device_properties = Mock()
        mock_device_properties.name = "RTX 4090"
        mock_device_properties.total_memory = 24 * 1024**3  # 24 GB
        mock_get_device_properties.return_value = mock_device_properties

        # Call function with device='cuda'
        result = get_total_memory_capacity(device='cuda')

        # Expected: 24 GB total VRAM
        expected = 24.0
        assert abs(result - expected) < 0.01

    def test_invalid_device_parameter(self):
        """Test invalid device parameter raises ValueError"""
        with pytest.raises(ValueError, match="Invalid device: gpu"):
            get_total_memory_capacity(device='gpu')


class TestMemoryCapacityIntegration:
    """Integration tests comparing both functions"""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_available_vs_total_vram_relationship(
            self,
            mock_memory_reserved,
            mock_memory_allocated,
            mock_get_device_properties,
            mock_cuda_available):
        """Test that available VRAM is always <= total VRAM"""
        # Setup mocks
        mock_cuda_available.return_value = True

        mock_device_properties = Mock()
        mock_device_properties.name = "Test GPU"
        mock_device_properties.total_memory = 12 * 1024**3  # 12 GB
        mock_get_device_properties.return_value = mock_device_properties

        mock_memory_allocated.return_value = 3 * 1024**3  # 3 GB allocated
        mock_memory_reserved.return_value = 5 * 1024**3   # 5 GB reserved

        # Call both functions
        available = get_memory_capacity()
        total = get_total_memory_capacity()

        # Available should be less than or equal to total
        assert available <= total

        # Specific values check
        assert abs(available - 7.0) < 0.01  # 12 - 5 = 7
        assert abs(total - 12.0) < 0.01

    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_available_vs_total_ram_relationship(
            self, mock_virtual_memory, mock_cuda_available):
        """Test that available RAM is always <= total RAM"""
        # Setup mocks
        mock_cuda_available.return_value = False

        mock_memory_info = Mock()
        mock_memory_info.total = 16 * 1024**3      # 16 GB total
        mock_memory_info.available = 10 * 1024**3  # 10 GB available
        mock_virtual_memory.return_value = mock_memory_info

        # Call both functions
        available = get_memory_capacity()
        total = get_total_memory_capacity()

        # Available should be less than or equal to total
        assert available <= total

        # Specific values check
        assert abs(available - 10.0) < 0.01
        assert abs(total - 16.0) < 0.01


@pytest.fixture
def mock_logger():
    """Fixture to mock the logger"""
    with patch('similarity_matrix.lib.model.logger') as mock_log:
        yield mock_log


class TestLogging:
    """Test logging behavior in memory capacity functions"""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_cuda_logging(
            self,
            mock_memory_reserved,
            mock_memory_allocated,
            mock_get_device_properties,
            mock_cuda_available,
            mock_logger):
        """Test that CUDA information is properly logged"""
        # Setup mocks
        mock_cuda_available.return_value = True

        mock_device_properties = Mock()
        mock_device_properties.name = "Test GPU"
        mock_device_properties.total_memory = 8 * 1024**3
        mock_get_device_properties.return_value = mock_device_properties

        mock_memory_allocated.return_value = 1 * 1024**3
        mock_memory_reserved.return_value = 2 * 1024**3

        # Call function
        get_memory_capacity()

        # Verify logging calls
        # At least device name, total, available
        assert mock_logger.info.call_count >= 3

    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_ram_logging(
            self,
            mock_virtual_memory,
            mock_cuda_available,
            mock_logger):
        """Test that RAM information is properly logged"""
        # Setup mocks
        mock_cuda_available.return_value = False

        mock_memory_info = Mock()
        mock_memory_info.total = 16 * 1024**3
        mock_memory_info.available = 12 * 1024**3
        mock_virtual_memory.return_value = mock_memory_info

        # Call function
        get_memory_capacity()

        # Verify logging calls
        assert mock_logger.info.call_count >= 2  # At least total and available RAM
