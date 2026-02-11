import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock, call
from similarity_matrix.lib.similarity import cos_sim_mem


class TestCosSimMem:
    """Tests for cos_sim_mem function"""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = Mock()
        return model

    @pytest.fixture
    def sample_texts(self):
        """Sample text data for testing"""
        row_texts = ["text 1", "text 2", "text 3"]
        column_texts = ["col 1", "col 2"]
        return row_texts, column_texts

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings with known dimensions"""
        # Create embeddings with dimension 384 (common for sentence
        # transformers)
        row_embeddings = torch.randn(3, 384)  # 3 texts, 384 dimensions
        col_embeddings = torch.randn(2, 384)  # 2 texts, 384 dimensions
        test_embedding = torch.randn(1, 384)  # Test embedding
        return row_embeddings, col_embeddings, test_embedding

    @patch('similarity_matrix.lib.similarity.get_memory_capacity')
    @patch('similarity_matrix.lib.similarity.torch.cuda.is_available')
    @patch('similarity_matrix.lib.similarity.torch.cuda.empty_cache')
    @patch('similarity_matrix.lib.similarity.get_model_size')
    @patch('similarity_matrix.lib.similarity.encode')
    @patch('similarity_matrix.lib.similarity.cos_sim')
    def test_basic_functionality_cpu(
            self,
            mock_cos_sim,
            mock_encode,
            mock_get_model_size,
            mock_empty_cache,
            mock_cuda_available,
            mock_get_memory_capacity,
            mock_model,
            sample_texts,
            mock_embeddings):
        """Test basic functionality with CPU (no CUDA)"""
        # Setup
        row_texts, column_texts = sample_texts
        row_embeddings, col_embeddings, test_embedding = mock_embeddings

        mock_cuda_available.return_value = False
        mock_get_memory_capacity.return_value = 16.0  # 16 GB RAM
        mock_get_model_size.return_value = 1024**3  # 1 GB model

        # Mock encode to return different embeddings for different calls
        def encode_side_effect(model, texts):
            if texts == ["test"]:
                return test_embedding
            elif len(texts) == 3:  # row batch
                return row_embeddings
            elif len(texts) == 2:  # column batch
                return col_embeddings
            else:
                return torch.randn(len(texts), 384)

        mock_encode.side_effect = encode_side_effect

        # Mock cosine similarity result
        expected_similarity = torch.randn(3, 2)
        mock_cos_sim.return_value = expected_similarity

        # Execute
        result = cos_sim_mem(mock_model, row_texts, column_texts)

        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(
            result, expected_similarity.cpu().numpy().astype(float))

        # Verify function calls
        mock_get_memory_capacity.assert_called_once()
        mock_get_model_size.assert_called_once_with(mock_model)
        assert mock_encode.call_count >= 3  # test + row + column embeddings
        mock_cos_sim.assert_called_once_with(row_embeddings, col_embeddings)

    @patch('similarity_matrix.lib.similarity.get_memory_capacity')
    @patch('similarity_matrix.lib.similarity.torch.cuda.is_available')
    @patch('similarity_matrix.lib.similarity.torch.cuda.get_device_properties')
    @patch('similarity_matrix.lib.similarity.torch.cuda.memory_allocated')
    @patch('similarity_matrix.lib.similarity.torch.cuda.memory_reserved')
    @patch('similarity_matrix.lib.similarity.torch.cuda.empty_cache')
    @patch('similarity_matrix.lib.similarity.get_model_size')
    @patch('similarity_matrix.lib.similarity.encode')
    @patch('similarity_matrix.lib.similarity.cos_sim')
    def test_cuda_memory_management(
            self,
            mock_cos_sim,
            mock_encode,
            mock_get_model_size,
            mock_empty_cache,
            mock_memory_reserved,
            mock_memory_allocated,
            mock_get_device_properties,
            mock_cuda_available,
            mock_get_memory_capacity,
            mock_model,
            sample_texts,
            mock_embeddings):
        """Test CUDA memory management functionality"""
        # Setup
        row_texts, column_texts = sample_texts
        row_embeddings, col_embeddings, test_embedding = mock_embeddings

        mock_cuda_available.return_value = True
        mock_get_memory_capacity.return_value = 8.0  # 8 GB VRAM
        mock_get_model_size.return_value = 2 * 1024**3  # 2 GB model

        # Mock CUDA memory properties
        mock_device_props = Mock()
        mock_device_props.total_memory = 8 * 1024**3  # 8 GB
        mock_get_device_properties.return_value = mock_device_props
        mock_memory_allocated.return_value = 1 * 1024**3  # 1 GB allocated
        mock_memory_reserved.return_value = 2 * 1024**3   # 2 GB reserved

        # Mock encode function
        def encode_side_effect(model, texts):
            if texts == ["test"]:
                return test_embedding
            elif len(texts) == 3:
                return row_embeddings
            elif len(texts) == 2:
                return col_embeddings
            else:
                return torch.randn(len(texts), 384)

        mock_encode.side_effect = encode_side_effect
        mock_cos_sim.return_value = torch.randn(3, 2)

        # Execute
        result = cos_sim_mem(mock_model, row_texts, column_texts)

        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

        # Verify CUDA functions were called
        mock_get_device_properties.assert_called_with(0)
        mock_memory_allocated.assert_called_with(0)
        mock_memory_reserved.assert_called_with(0)
        assert mock_empty_cache.call_count > 0  # Should be called multiple times

    @patch('similarity_matrix.lib.similarity.get_memory_capacity')
    @patch('similarity_matrix.lib.similarity.torch.cuda.is_available')
    @patch('similarity_matrix.lib.similarity.torch.cuda.empty_cache')
    @patch('similarity_matrix.lib.similarity.get_model_size')
    @patch('similarity_matrix.lib.similarity.encode')
    def test_max_retries_exceeded(
            self,
            mock_encode,
            mock_get_model_size,
            mock_empty_cache,
            mock_cuda_available,
            mock_get_memory_capacity,
            mock_model,
            sample_texts):
        """Test that function raises exception when max retries exceeded"""
        # Setup
        row_texts, column_texts = sample_texts

        mock_cuda_available.return_value = True
        mock_get_memory_capacity.return_value = 4.0
        mock_get_model_size.return_value = 1024**3

        # Test embedding succeeds, but all other encode calls fail
        def encode_side_effect(model, texts):
            if texts == ["test"]:
                return torch.randn(1, 384)
            else:
                raise torch.cuda.OutOfMemoryError("Persistent CUDA OOM")

        mock_encode.side_effect = encode_side_effect

        # Execute and verify exception is raised (RuntimeError added
        # for compatibility with systems without CUDA device that raise
        # RuntimeError)
        with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError)):
            cos_sim_mem(mock_model, row_texts, column_texts)

    @patch('similarity_matrix.lib.similarity.get_memory_capacity')
    @patch('similarity_matrix.lib.similarity.torch.cuda.is_available')
    @patch('similarity_matrix.lib.similarity.get_model_size')
    @patch('similarity_matrix.lib.similarity.encode')
    @patch('similarity_matrix.lib.similarity.cos_sim')
    def test_custom_memory_limit(
            self,
            mock_cos_sim,
            mock_encode,
            mock_get_model_size,
            mock_cuda_available,
            mock_get_memory_capacity,
            mock_model,
            sample_texts,
            mock_embeddings):
        """Test custom memory limit parameter"""
        # Setup
        row_texts, column_texts = sample_texts
        row_embeddings, col_embeddings, test_embedding = mock_embeddings

        mock_cuda_available.return_value = False
        mock_get_model_size.return_value = 1024**3

        # Mock encode
        def encode_side_effect(model, texts):
            if texts == ["test"]:
                return test_embedding
            elif len(texts) == 3:
                return row_embeddings
            else:
                return col_embeddings

        mock_encode.side_effect = encode_side_effect
        mock_cos_sim.return_value = torch.randn(3, 2)

        # Execute with custom memory limit
        custom_memory = 4.0  # 4 GB
        result = cos_sim_mem(
            mock_model,
            row_texts,
            column_texts,
            max_memory_gb=custom_memory)

        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

        # get_memory_capacity should not be called when custom limit is
        # provided
        mock_get_memory_capacity.assert_not_called()

    @patch('similarity_matrix.lib.similarity.get_memory_capacity')
    @patch('similarity_matrix.lib.similarity.torch.cuda.is_available')
    @patch('similarity_matrix.lib.similarity.get_model_size')
    @patch('similarity_matrix.lib.similarity.encode')
    @patch('similarity_matrix.lib.similarity.cos_sim')
    def test_empty_input_lists(
            self,
            mock_cos_sim,
            mock_encode,
            mock_get_model_size,
            mock_cuda_available,
            mock_get_memory_capacity,
            mock_model):
        """Test handling of empty input lists"""
        # Setup
        row_texts = []
        column_texts = []

        mock_cuda_available.return_value = False
        mock_get_memory_capacity.return_value = 8.0
        mock_get_model_size.return_value = 1024**3

        # Test embedding still needed for dimension detection
        test_embedding = torch.randn(1, 384)
        mock_encode.side_effect = lambda model, texts: test_embedding if texts == [
            "test"] else torch.randn(0, 384)

        # Execute
        result = cos_sim_mem(mock_model, row_texts, column_texts)

        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 0)

    @patch('similarity_matrix.lib.similarity.get_memory_capacity')
    @patch('similarity_matrix.lib.similarity.torch.cuda.is_available')
    @patch('similarity_matrix.lib.similarity.get_model_size')
    @patch('similarity_matrix.lib.similarity.encode')
    @patch('similarity_matrix.lib.similarity.cos_sim')
    def test_single_item_lists(
            self,
            mock_cos_sim,
            mock_encode,
            mock_get_model_size,
            mock_cuda_available,
            mock_get_memory_capacity,
            mock_model):
        """Test handling of single-item input lists"""
        # Setup
        row_texts = ["single row"]
        column_texts = ["single col"]

        mock_cuda_available.return_value = False
        mock_get_memory_capacity.return_value = 8.0
        mock_get_model_size.return_value = 1024**3

        # Mock embeddings
        test_embedding = torch.randn(1, 384)
        single_embedding = torch.randn(1, 384)

        def encode_side_effect(model, texts):
            if texts == ["test"]:
                return test_embedding
            else:
                return single_embedding

        mock_encode.side_effect = encode_side_effect
        mock_cos_sim.return_value = torch.tensor(
            [[0.8]])  # Single similarity value

        # Execute
        result = cos_sim_mem(mock_model, row_texts, column_texts)

        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 0.8)


class TestCosSimMemIntegration:
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = Mock()
        return model

    """Integration tests for cos_sim_mem function"""

    @patch('similarity_matrix.lib.similarity.get_memory_capacity')
    @patch('similarity_matrix.lib.similarity.torch.cuda.is_available')
    @patch('similarity_matrix.lib.similarity.get_model_size')
    @patch('similarity_matrix.lib.similarity.encode')
    @patch('similarity_matrix.lib.similarity.cos_sim')
    def test_matrix_values_consistency(
            self,
            mock_cos_sim,
            mock_encode,
            mock_get_model_size,
            mock_cuda_available,
            mock_get_memory_capacity,
            mock_model):
        """Test that the resulting matrix values are consistent and in expected range"""
        # Setup
        row_texts = ["hello world", "goodbye world", "test text"]
        column_texts = ["hello", "world"]

        mock_cuda_available.return_value = False
        mock_get_memory_capacity.return_value = 8.0
        mock_get_model_size.return_value = 1024**3

        # Create realistic embeddings
        test_embedding = torch.randn(1, 384)
        row_embeddings = torch.randn(3, 384)
        col_embeddings = torch.randn(2, 384)

        def encode_side_effect(model, texts):
            if texts == ["test"]:
                return test_embedding
            elif len(texts) == 3:
                return row_embeddings
            else:
                return col_embeddings

        mock_encode.side_effect = encode_side_effect

        # Create realistic similarity values
        realistic_similarities = torch.tensor(
            [[0.8, 0.6], [0.7, 0.9], [0.5, 0.4]])
        mock_cos_sim.return_value = realistic_similarities

        # Execute
        result = cos_sim_mem(mock_model, row_texts, column_texts)

        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

        # Check that values are in reasonable similarity range
        assert np.all(result >= -1.0), "Similarity values should be >= -1"
        assert np.all(result <= 1.0), "Similarity values should be <= 1"

        # Check specific values match expected
        np.testing.assert_array_almost_equal(
            result, realistic_similarities.numpy(), decimal=5)


@pytest.fixture
def mock_logger():
    """Fixture to mock the logger"""
    with patch('similarity_matrix.lib.similarity.logger') as mock_log:
        yield mock_log


class TestCosSimMemLogging:
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = Mock()
        return model

    """Test logging behavior in cos_sim_mem function"""

    @patch('similarity_matrix.lib.similarity.get_memory_capacity')
    @patch('similarity_matrix.lib.similarity.torch.cuda.is_available')
    @patch('similarity_matrix.lib.similarity.get_model_size')
    @patch('similarity_matrix.lib.similarity.encode')
    @patch('similarity_matrix.lib.similarity.cos_sim')
    def test_logging_output(
            self,
            mock_cos_sim,
            mock_encode,
            mock_get_model_size,
            mock_cuda_available,
            mock_get_memory_capacity,
            mock_model,
            mock_logger):
        """Test that appropriate logging messages are generated"""
        # Setup
        row_texts = ["test1", "test2"]
        column_texts = ["col1"]

        mock_cuda_available.return_value = False
        mock_get_memory_capacity.return_value = 8.0
        mock_get_model_size.return_value = 1024**3

        # Mock embeddings
        test_embedding = torch.randn(1, 384)
        mock_encode.side_effect = lambda model, texts: test_embedding if texts == [
            "test"] else torch.randn(len(texts), 384)
        mock_cos_sim.return_value = torch.randn(2, 1)

        # Execute
        cos_sim_mem(mock_model, row_texts, column_texts)

        # Verify logging calls
        # Should log model size, embedding dim, batch sizes, completion
        assert mock_logger.info.call_count >= 4

        # Check for specific log messages
        log_messages = [call.args[0]
                        for call in mock_logger.info.call_args_list]
        assert any("Model size:" in msg for msg in log_messages)
        assert any("Embedding dimension:" in msg for msg in log_messages)
        assert any("Initial batch sizes:" in msg for msg in log_messages)
        assert any(
            "Completed similarity matrix computation" in msg for msg in log_messages)
