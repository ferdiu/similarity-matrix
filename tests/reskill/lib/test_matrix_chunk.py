import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

from similarity_matrix.lib.matrix import SimilarityMatrix
from similarity_matrix.lib.matrix_chunk import ChunkedSimilarityMatrix


class TestChunkedSimilarityMatrix:
    """Test suite for ChunkedSimilarityMatrix class."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        row_ids = [f"row_{i}" for i in range(10)]
        column_ids = [f"col_{i}" for i in range(8)]
        row_texts = [f"row text {i}" for i in range(10)]
        column_texts = [f"column text {i}" for i in range(8)]

        return {
            'row_ids': row_ids,
            'column_ids': column_ids,
            'row_texts': row_texts,
            'column_texts': column_texts
        }

    @pytest.fixture
    def mock_functions(self, sample_data):
        """Create mock functions for loading data."""
        def mock_row_load():
            return sample_data['row_texts']

        def mock_column_load():
            return sample_data['column_texts']

        return mock_row_load, mock_column_load

    @pytest.fixture
    def chunked_matrix(self, sample_data, mock_functions):
        """Create a ChunkedSimilarityMatrix instance for testing."""
        row_load_fn, col_load_fn = mock_functions
        return ChunkedSimilarityMatrix(
            row_ids=sample_data['row_ids'],
            column_ids=sample_data['column_ids'],
            name="test_matrix",
            row_load_function=row_load_fn,
            column_load_function=col_load_fn,
            row_chunk_size=3,
            column_chunk_size=3
        )

    def test_initialization(self, sample_data, mock_functions):
        """Test ChunkedSimilarityMatrix initialization."""
        row_load_fn, col_load_fn = mock_functions

        matrix = ChunkedSimilarityMatrix(
            row_ids=sample_data['row_ids'],
            column_ids=sample_data['column_ids'],
            name="test_matrix",
            row_load_function=row_load_fn,
            column_load_function=col_load_fn,
            row_chunk_size=5,
            column_chunk_size=4
        )

        assert matrix.row_chunk_size == 5
        assert matrix.column_chunk_size == 4
        assert matrix.name == "test_matrix"
        assert len(matrix.row_ids) == 10
        assert len(matrix.column_ids) == 8
        assert matrix.temp_dir is None

    def test_initialization_with_temp_dir(self, sample_data, mock_functions):
        """Test initialization with custom temp directory."""
        row_load_fn, col_load_fn = mock_functions
        temp_dir = "/custom/temp"

        matrix = ChunkedSimilarityMatrix(
            row_ids=sample_data['row_ids'],
            column_ids=sample_data['column_ids'],
            name="test_matrix",
            row_load_function=row_load_fn,
            column_load_function=col_load_fn,
            temp_dir=temp_dir
        )

        assert matrix.temp_dir == Path(temp_dir)

    def test_set_chunk_sizes(self, chunked_matrix):
        """Test updating chunk sizes."""
        chunked_matrix.set_chunk_sizes(20, 15)

        assert chunked_matrix.row_chunk_size == 20
        assert chunked_matrix.column_chunk_size == 15

    def test_estimate_memory_usage(self, chunked_matrix):
        """Test memory usage estimation."""
        memory_info = chunked_matrix.estimate_memory_usage()

        assert 'chunk_matrix_mb' in memory_info
        assert 'embeddings_mb' in memory_info
        assert 'full_matrix_mb' in memory_info
        assert 'peak_chunk_memory_mb' in memory_info

        # Test with custom parameters
        memory_info_custom = chunked_matrix.estimate_memory_usage(
            embedding_dim=512, dtype_size=8
        )

        # Custom parameters should give different results
        assert memory_info_custom['embeddings_mb'] != memory_info['embeddings_mb']

    @patch('similarity_matrix.lib.model.initialize_model')
    @patch('similarity_matrix.lib.matrix_chunk.cos_sim_mem')
    @patch('similarity_matrix.lib.windowing.compute_aggregation_tail_weight')
    def test_calculate_fake(
            self,
            mock_aggregation,
            mock_cos_sim,
            mock_model,
            chunked_matrix):
        """Test calculate method with fake data."""
        # Test fake calculation (should use parent's implementation)
        chunked_matrix.calculate(fake=42)

        # Should not call the mocked functions for fake calculation
        mock_model.assert_not_called()
        mock_cos_sim.assert_not_called()

        # Matrix should be initialized with fake data
        assert chunked_matrix.matrix.shape == (10, 8)
        assert np.all(chunked_matrix.matrix >= -1)
        assert np.all(chunked_matrix.matrix <= 1)

    @patch('similarity_matrix.lib.matrix_chunk.initialize_model')
    @patch('similarity_matrix.lib.matrix_chunk.cos_sim_mem')
    @patch('similarity_matrix.lib.matrix_chunk.compute_aggregation_tail_weight')
    def test_calculate_real(
            self,
            mock_aggregation,
            mock_cos_sim,
            mock_model,
            chunked_matrix):
        """Test calculate method with real computation."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        # Mock cos_sim_mem to return predictable results
        def mock_cos_sim_side_effect(model, rows, cols):
            # Values between 0.1 and 0.9
            return np.random.default_rng(
                42).random((len(rows), len(cols))) * 0.8 + 0.1

        mock_cos_sim.side_effect = mock_cos_sim_side_effect

        # Mock aggregation to return the same matrix
        mock_aggregation.side_effect = lambda matrix, *args, **kwargs: matrix

        # Run calculation
        chunked_matrix.calculate()

        # Verify mocks were called
        mock_model.assert_called_once()
        assert mock_cos_sim.call_count > 1  # Should be called multiple times for chunks

        # Verify matrix properties
        assert chunked_matrix.matrix.shape == (10, 8)
        assert np.all(chunked_matrix.matrix >= -1)
        assert np.all(chunked_matrix.matrix <= 1)

    @patch('similarity_matrix.lib.matrix_chunk.initialize_model')
    @patch('similarity_matrix.lib.matrix_chunk.cos_sim_mem')
    def test_calculate_with_pointers(
            self,
            mock_cos_sim,
            mock_model,
            sample_data):
        """Test calculate method when load functions return pointers."""
        # Create mock functions that return tuples with pointers
        def mock_row_load_with_pointers():
            return [
                "row text 0",
                "row text 1",
                "row text 2-1", "row text 2-2",
                "row text 3",
                "row text 4-1", "row text 4-2", "row text 4-3",
                "row text 5",
                "row text 6",
                "row text 7",
                "row text 8-1", "row text 8-2",
                "row text 9"
            ], [[1], [1], [1, 1], [1], [1, 2, 2], [1], [1], [1], [1, 2], [1]]

        def mock_column_load_with_pointers():
            return [
                "column text 0-1", "column text 0-2",
                "column text 1",
                "column text 2-1", "column text 2-2",
                "column text 3",
                "column text 4-1", "column text 4-2",
                "column text 5",
                "column text 6",
                "column text 7",
            ], [[1, 1], [1], [1, 2], [1], [1, 1], [1], [1], [1]]

        chunked_matrix = ChunkedSimilarityMatrix(
            row_ids=sample_data['row_ids'],
            column_ids=sample_data['column_ids'],
            name="test_matrix_pointers",
            row_load_function=mock_row_load_with_pointers,
            column_load_function=mock_column_load_with_pointers,
            row_chunk_size=3,
            column_chunk_size=3
        )

        # Setup mocks
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_cos_sim.return_value = np.random.default_rng(
            42).random((3, 3)) * 0.8 + 0.1

        def mock_compute_aggregation_tail_weight(
                cos_sim_windows, row_pointers, column_pointers, tail_weight):
            return np.random.default_rng(42).random(
                (len(row_pointers), len(column_pointers))) * 0.8 + 0.1

        with patch('similarity_matrix.lib.matrix_chunk.compute_aggregation_tail_weight', side_effect=mock_compute_aggregation_tail_weight) as mock_agg:
            chunked_matrix.calculate()

            # Verify that aggregation was called with pointers
            mock_agg.assert_called()
            args = mock_agg.call_args[0]
            assert len(args) >= 3  # matrix, row_pointers, column_pointers

    def test_inheritance(self, chunked_matrix):
        """Test that ChunkedSimilarityMatrix properly inherits from SimilarityMatrix."""
        # Test that inherited methods work
        assert chunked_matrix.shape == (10, 8)
        assert chunked_matrix.num_rows == 10
        assert chunked_matrix.num_columns == 8

        # Test that we can use inherited methods like get_value, set_value
        # (after initializing the matrix)
        chunked_matrix.matrix = np.random.default_rng(42).random((10, 8))

        # Test get_value and set_value
        chunked_matrix.set_value("row_0", "col_0", 0.75)
        assert np.isclose(chunked_matrix.get_value("row_0", "col_0"), 0.75)

        # Test get_row and get_column
        row_data = chunked_matrix.get_row("row_0")
        assert len(row_data) == 8

        col_data = chunked_matrix.get_column("col_0")
        assert len(col_data) == 10

    def test_different_chunk_sizes(self, sample_data, mock_functions):
        """Test with different chunk sizes to ensure robustness."""
        row_load_fn, col_load_fn = mock_functions

        test_cases = [
            (1, 1),    # Very small chunks
            (5, 4),    # Medium chunks
            (20, 20),  # Larger than data size
        ]

        for row_chunk, col_chunk in test_cases:
            matrix = ChunkedSimilarityMatrix(
                row_ids=sample_data['row_ids'],
                column_ids=sample_data['column_ids'],
                name=f"test_matrix_{row_chunk}_{col_chunk}",
                row_load_function=row_load_fn,
                column_load_function=col_load_fn,
                row_chunk_size=row_chunk,
                column_chunk_size=col_chunk
            )

            # Test with fake calculation
            matrix.calculate(fake=42)
            assert matrix.matrix.shape == (10, 8)

    @patch('similarity_matrix.lib.matrix_chunk.logger')
    def test_logging(self, mock_logger, chunked_matrix):
        """Test that appropriate logging messages are generated."""
        with patch('similarity_matrix.lib.matrix_chunk.initialize_model'), \
                patch('similarity_matrix.lib.matrix_chunk.cos_sim_mem') as mock_cos_sim:

            # Mock the cos_sim_mem to return a predictable matrix with the
            # shape from the passed chunk sizes
            def mock_cos_sim_side_effect(model, rows, cols):
                return np.random.default_rng(42).random(
                    (len(rows), len(cols))) * 0.8 + 0.1
            mock_cos_sim.side_effect = mock_cos_sim_side_effect

            chunked_matrix.calculate()

            # Verify that info logs were called
            info_calls = [call for call in mock_logger.info.call_args_list]
            assert len(info_calls) > 0, mock_logger.info

            # Check for specific log messages
            log_messages = [str(call) for call in info_calls]
            assert any("Matrix shape (final)" in msg for msg in log_messages)
            assert any("Processing in chunks" in msg for msg in log_messages)


# Pytest fixtures for integration testing
@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestChunkedSimilarityMatrixIntegration:
    """Integration tests for ChunkedSimilarityMatrix."""

    def test_save_and_load(self, temp_directory):
        """Test saving and loading chunked matrices."""
        # Create a matrix with fake data
        row_ids = [f"row_{i}" for i in range(5)]
        col_ids = [f"col_{i}" for i in range(4)]

        matrix = ChunkedSimilarityMatrix(
            row_ids=row_ids,
            column_ids=col_ids,
            name="integration_test",
            row_chunk_size=2,
            column_chunk_size=2
        )

        # Calculate with fake data
        matrix.calculate(fake=123)
        original_matrix = matrix.matrix.copy()

        # Save the matrix
        matrix.save(temp_directory)

        # Load the matrix back
        loaded_matrix = ChunkedSimilarityMatrix.load(
            temp_directory, "integration_test")

        # Verify the loaded matrix matches the original
        assert np.array_equal(loaded_matrix.matrix, original_matrix)
        assert loaded_matrix.row_ids == row_ids
        assert loaded_matrix.column_ids == col_ids
        assert loaded_matrix.name == "integration_test"

    def test_end_to_end_workflow(self, temp_directory):
        """Test a complete end-to-end workflow."""
        # Create sample data
        row_ids = [f"document_{i}" for i in range(6)]
        col_ids = [f"query_{i}" for i in range(4)]

        def load_documents():
            return [f"This is document {i} content" for i in range(6)]

        def load_queries():
            return [f"Query {i} text" for i in range(4)]

        # Create and configure matrix
        matrix = ChunkedSimilarityMatrix(
            row_ids=row_ids,
            column_ids=col_ids,
            name="end_to_end_test",
            row_load_function=load_documents,
            column_load_function=load_queries,
            row_chunk_size=2,
            column_chunk_size=2,
            temp_dir=temp_directory
        )

        # Calculate similarity (using fake for reproducibility)
        matrix.calculate(fake=456)

        # Test matrix operations
        assert matrix.shape == (6, 4)

        # Test specific value access
        value = matrix.get_value("document_0", "query_0")
        assert -1 <= value <= 1

        # Test row and column access
        row_values = matrix.get_row("document_0")
        assert len(row_values) == 4

        col_values = matrix.get_column("query_0")
        assert len(col_values) == 6

        # Test normalization
        normalized = matrix.normalized()
        assert normalized.shape == matrix.shape

        # Test iteration
        similarities = list(matrix)
        assert len(similarities) == 6 * 4  # Total number of elements

        # Test save/load
        matrix.save(temp_directory)
        loaded = ChunkedSimilarityMatrix.load(
            temp_directory, "end_to_end_test")
        assert np.array_equal(loaded.matrix, matrix.matrix)
