import pytest
from unittest.mock import Mock, MagicMock, call
from mysql.connector.cursor import MySQLCursor

from similarity_matrix.lib.utils import _batch_iterator, update_db_batches, total_length_lists


class TestBatchIterator:
    """Test cases for _batch_iterator function."""

    def test_batch_iterator_exact_batches(self):
        """Test batching when items divide evenly into batches."""
        items = [1, 2, 3, 4, 5, 6]
        batches = list(_batch_iterator(items, batch_size=2))

        expected = [[1, 2], [3, 4], [5, 6]]
        assert batches == expected

    def test_batch_iterator_partial_last_batch(self):
        """Test batching when last batch is partial."""
        items = [1, 2, 3, 4, 5]
        batches = list(_batch_iterator(items, batch_size=2))

        expected = [[1, 2], [3, 4], [5]]
        assert batches == expected

    def test_batch_iterator_single_item_batches(self):
        """Test batching with batch size of 1."""
        items = [1, 2, 3]
        batches = list(_batch_iterator(items, batch_size=1))

        expected = [[1], [2], [3]]
        assert batches == expected

    def test_batch_iterator_empty_iterable(self):
        """Test batching with empty iterable."""
        items = []
        batches = list(_batch_iterator(items, batch_size=3))

        assert batches == []

    def test_batch_iterator_batch_size_larger_than_items(self):
        """Test when batch size is larger than number of items."""
        items = [1, 2]
        batches = list(_batch_iterator(items, batch_size=5))

        expected = [[1, 2]]
        assert batches == expected

    def test_batch_iterator_with_generator(self):
        """Test batching with a generator input."""
        def number_generator():
            for i in range(5):
                yield i

        batches = list(_batch_iterator(number_generator(), batch_size=2))
        expected = [[0, 1], [2, 3], [4]]
        assert batches == expected

    def test_batch_iterator_with_different_types(self):
        """Test batching with different data types."""
        items = ['a', 'b', 'c', 'd']
        batches = list(_batch_iterator(items, batch_size=3))

        expected = [['a', 'b', 'c'], ['d']]
        assert batches == expected


class TestUpdateDbBatches:
    """Test cases for update_db_batches function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create a mock MySQL cursor."""
        cursor = Mock(spec=MySQLCursor)
        return cursor

    @pytest.fixture
    def mock_matrix(self):
        """Create a mock similarity matrix."""
        matrix = Mock()
        # Mock the iteration behavior of the matrix
        matrix.__iter__ = Mock(return_value=iter([
            (1, 2, 0.5),
            (2, 3, 0.8),
            (3, 4, 0.3),
            (4, 5, 0.9)
        ]))
        return matrix

    def test_update_db_batches_successful_execution(
            self, mock_cursor, mock_matrix):
        """Test successful batch execution."""
        sql = "INSERT INTO similarity (row_id, col_id, value) VALUES (%s, %s, %s)"

        update_db_batches(mock_cursor, sql, mock_matrix, batch_size=2)

        # Verify transaction management
        expected_calls = [
            call("START TRANSACTION"),
            call("COMMIT"),
            call("START TRANSACTION"),
            call("COMMIT")
        ]
        mock_cursor.execute.assert_has_calls(expected_calls, any_order=False)

        # Verify executemany was called with correct batches
        assert mock_cursor.executemany.call_count == 2

        # Check the first batch
        first_call_args = mock_cursor.executemany.call_args_list[0]
        assert first_call_args[0][0] == sql
        assert first_call_args[0][1] == [(1, 2, 0.5), (2, 3, 0.8)]

        # Check the second batch
        second_call_args = mock_cursor.executemany.call_args_list[1]
        assert second_call_args[0][0] == sql
        assert second_call_args[0][1] == [(3, 4, 0.3), (4, 5, 0.9)]

    def test_update_db_batches_with_exception_rollback(
            self, mock_cursor, mock_matrix):
        """Test rollback behavior when executemany raises an exception."""
        sql = "INSERT INTO similarity (row_id, col_id, value) VALUES (%s, %s, %s)"

        # Make executemany raise an exception on first call
        mock_cursor.executemany.side_effect = [
            Exception("Database error"), None]

        with pytest.raises(Exception, match="Database error"):
            update_db_batches(mock_cursor, sql, mock_matrix, batch_size=2)

        # Verify rollback was called
        mock_cursor.execute.assert_any_call("ROLLBACK")
        mock_cursor.execute.assert_any_call("START TRANSACTION")

    def test_update_db_batches_custom_batch_size(self, mock_cursor):
        """Test with custom batch size."""
        # Create a matrix with more data points
        matrix = Mock()
        matrix.__iter__ = Mock(return_value=iter([
            (i, i + 1, 0.1 * i) for i in range(1, 8)  # 7 items
        ]))

        sql = "INSERT INTO similarity VALUES (%s, %s, %s)"

        update_db_batches(mock_cursor, sql, matrix, batch_size=3)

        # Should have 3 batches: [3 items], [3 items], [1 item]
        assert mock_cursor.executemany.call_count == 3

        # Verify transaction calls (START TRANSACTION + COMMIT for each batch)
        assert mock_cursor.execute.call_count == 6  # 3 * (START + COMMIT)

    def test_update_db_batches_empty_matrix(self, mock_cursor):
        """Test with empty matrix."""
        matrix = Mock()
        matrix.__iter__ = Mock(return_value=iter([]))

        sql = "INSERT INTO similarity VALUES (%s, %s, %s)"

        update_db_batches(mock_cursor, sql, matrix, batch_size=1000)

        # No batches should be processed
        mock_cursor.executemany.assert_not_called()
        mock_cursor.execute.assert_not_called()

    def test_update_db_batches_single_item(self, mock_cursor):
        """Test with matrix containing single item."""
        matrix = Mock()
        matrix.__iter__ = Mock(return_value=iter([(1, 2, 0.5)]))

        sql = "INSERT INTO similarity VALUES (%s, %s, %s)"

        update_db_batches(mock_cursor, sql, matrix, batch_size=1000)

        # Should process one batch
        mock_cursor.executemany.assert_called_once_with(sql, [(1, 2, 0.5)])
        mock_cursor.execute.assert_has_calls([
            call("START TRANSACTION"),
            call("COMMIT")
        ])

    def test_update_db_batches_float_conversion(self, mock_cursor):
        """Test that similarity values are converted to float."""
        matrix = Mock()
        # Use different numeric types that should be converted to float
        matrix.__iter__ = Mock(return_value=iter([
            (1, 2, 0.5),      # already float
            (2, 3, 1),        # int
            (3, 4, "0.8"),    # string
        ]))

        sql = "INSERT INTO similarity VALUES (%s, %s, %s)"

        update_db_batches(mock_cursor, sql, matrix, batch_size=10)

        # Verify the values were converted to float
        call_args = mock_cursor.executemany.call_args[0][1]
        expected = [(1, 2, 0.5), (2, 3, 1.0), (3, 4, 0.8)]
        assert call_args == expected

        # Verify all similarity values are floats
        for row_id, col_id, similarity in call_args:
            assert isinstance(similarity, float)

    def test_update_db_batches_exception_during_commit(
            self, mock_cursor, mock_matrix):
        """Test rollback when commit fails."""
        sql = "INSERT INTO similarity VALUES (%s, %s, %s)"

        # Make the execute method raise exception only for COMMIT
        def side_effect(query):
            if query == "COMMIT":
                raise Exception("Commit failed")

        mock_cursor.execute.side_effect = side_effect

        with pytest.raises(Exception, match="Commit failed"):
            update_db_batches(mock_cursor, sql, mock_matrix, batch_size=2)

        # Verify START TRANSACTION was called but ROLLBACK should also be
        # called
        mock_cursor.execute.assert_any_call("START TRANSACTION")
        mock_cursor.execute.assert_any_call("ROLLBACK")


# Integration-style tests
class TestBatchIteratorIntegration:
    """Integration tests combining both functions."""

    def test_batch_processing_flow(self):
        """Test the flow of data from matrix through batch_iterator to database updates."""
        # This test verifies that the batching logic works correctly
        # with the actual data transformation in update_db_batches

        mock_cursor = Mock(spec=MySQLCursor)

        # Create a matrix that yields specific data
        matrix = Mock()
        matrix.__iter__ = Mock(return_value=iter([
            (1, 2, 0.1),
            (2, 3, 0.2),
            (3, 4, 0.3),
            (4, 5, 0.4),
            (5, 6, 0.5)
        ]))

        sql = "INSERT INTO test VALUES (%s, %s, %s)"

        update_db_batches(mock_cursor, sql, matrix, batch_size=2)

        # Verify the correct number of batches
        assert mock_cursor.executemany.call_count == 3

        # Verify batch contents
        call_args_list = mock_cursor.executemany.call_args_list

        # First batch
        assert call_args_list[0][0][1] == [(1, 2, 0.1), (2, 3, 0.2)]
        # Second batch
        assert call_args_list[1][0][1] == [(3, 4, 0.3), (4, 5, 0.4)]
        # Third batch (partial)
        assert call_args_list[2][0][1] == [(5, 6, 0.5)]


class TestTotalLengthLists:
    """Test cases for total_length_lists function."""

    def test_total_length_lists_empty(self):
        """Test with empty list of lists."""
        assert total_length_lists([]) == 0

    def test_total_length_lists_single_empty_list(self):
        """Test with a single empty list."""
        assert total_length_lists([[]]) == 0

    def test_total_length_lists_multiple_empty_lists(self):
        """Test with a single empty list."""
        assert total_length_lists([[], []]) == 0

    def test_total_length_lists_single_non_empty_list(self):
        """Test with a single non-empty list."""
        assert total_length_lists([[1, 2, 3]]) == 3

    def test_total_length_lists_multiple_lists(self):
        """Test with multiple lists of varying lengths."""
        assert total_length_lists([[1, 2], [3], [4, 5, 6]]) == 6

    def test_total_length_lists_nested_empty_and_non_empty(self):
        """Test with nested empty and non-empty lists."""
        assert total_length_lists([[], [1, 2], [], [3]]) == 3


if __name__ == "__main__":
    pytest.main([__file__])
