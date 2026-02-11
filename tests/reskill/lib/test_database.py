import pytest
from unittest.mock import patch, MagicMock, call
from mysql.connector import Error

from similarity_matrix.lib.database import Database


class TestDatabase:
    """Test suite for Database class."""

    @pytest.fixture
    def db_config(self):
        """Fixture providing sample database configuration."""
        return {
            'host': 'localhost',
            'port': 3306,
            'user': 'testuser',
            'password': 'testpass',
            'database': 'testdb'
        }

    @pytest.fixture
    def database_instance(self, db_config):
        """Fixture providing a Database instance."""
        return Database(**db_config)

    def test_init(self, db_config):
        """Test Database initialization."""
        db = Database(**db_config)

        assert db.config == db_config
        assert db.conn is None
        assert db.cursor is None

    @patch('similarity_matrix.lib.database.mysql.connector.connect')
    def test_enter_success(self, mock_connect, database_instance):
        """Test successful context manager entry."""
        # Setup mocks
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor

        # Test __enter__
        result = database_instance.__enter__()

        # Verify connection was established
        mock_connect.assert_called_once_with(**database_instance.config)
        mock_connection.cursor.assert_called_once_with(dictionary=True)

        # Verify instance variables are set
        assert database_instance.conn == mock_connection
        assert database_instance.cursor == mock_cursor

        # Verify cursor is returned
        assert result == mock_cursor

    @patch('similarity_matrix.lib.database.mysql.connector.connect')
    def test_enter_connection_error(self, mock_connect, database_instance):
        """Test context manager entry with connection error."""
        # Setup mock to raise error
        connection_error = Error("Connection failed")
        mock_connect.side_effect = connection_error

        # Test that __enter__ raises the error
        with pytest.raises(Error, match="Connection failed"):
            database_instance.__enter__()

        # Verify connection attempt was made
        mock_connect.assert_called_once_with(**database_instance.config)

    def test_exit_success_no_exception(self, database_instance):
        """Test context manager exit with successful execution (no exception)."""
        # Setup mock objects
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        database_instance.cursor = mock_cursor
        database_instance.conn = mock_connection

        # Test __exit__ with no exception
        database_instance.__exit__(None, None, None)

        # Verify cleanup sequence
        mock_cursor.close.assert_called_once()
        mock_connection.commit.assert_called_once()
        mock_connection.rollback.assert_not_called()
        mock_connection.close.assert_called_once()

    def test_exit_with_exception(self, database_instance):
        """Test context manager exit with exception (rollback scenario)."""
        # Setup mock objects
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        database_instance.cursor = mock_cursor
        database_instance.conn = mock_connection

        # Test __exit__ with exception
        exc_type = Exception
        exc_val = Exception("Test exception")
        exc_tb = None
        database_instance.__exit__(exc_type, exc_val, exc_tb)

        # Verify cleanup sequence with rollback
        mock_cursor.close.assert_called_once()
        mock_connection.rollback.assert_called_once()
        mock_connection.commit.assert_not_called()
        mock_connection.close.assert_called_once()

    def test_exit_cursor_none(self, database_instance):
        """Test context manager exit when cursor is None."""
        # Setup mock connection only
        mock_connection = MagicMock()
        database_instance.cursor = None
        database_instance.conn = mock_connection

        # Test __exit__
        database_instance.__exit__(None, None, None)

        # Verify connection is still handled properly
        mock_connection.commit.assert_called_once()
        mock_connection.close.assert_called_once()

    def test_exit_connection_none(self, database_instance):
        """Test context manager exit when connection is None."""
        # Setup cursor only
        mock_cursor = MagicMock()
        database_instance.cursor = mock_cursor
        database_instance.conn = None

        # Test __exit__
        database_instance.__exit__(None, None, None)

        # Verify cursor is closed but no connection operations
        mock_cursor.close.assert_called_once()

    def test_exit_both_none(self, database_instance):
        """Test context manager exit when both cursor and connection are None."""
        database_instance.cursor = None
        database_instance.conn = None

        # Test __exit__ - should not raise any exceptions
        database_instance.__exit__(None, None, None)

        # No assertions needed - just ensuring no exceptions are raised

    @patch('similarity_matrix.lib.database.logger')
    def test_test_connection_success(self, mock_logger, database_instance):
        """Test successful connection test."""
        # Mock the context manager behavior
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {'1': 1}  # Non-empty result

        with patch.object(Database, '__enter__', return_value=mock_cursor), \
                patch.object(Database, '__exit__', return_value=None):

            result = database_instance.test_connection()

            # Verify query execution
            mock_cursor.execute.assert_called_once_with("SELECT 1")
            mock_cursor.fetchone.assert_called_once()

            # Verify success logging
            mock_logger.debug.assert_called_once_with(
                "Connection to MySQL database successful.")

            # Verify no error returned
            assert result is None

    @patch('similarity_matrix.lib.database.logger')
    def test_test_connection_empty_result(
            self, mock_logger, database_instance):
        """Test connection test with empty result."""
        # Mock the context manager behavior
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # Empty result

        with patch.object(Database, '__enter__', return_value=mock_cursor), \
                patch.object(Database, '__exit__', return_value=None):

            result = database_instance.test_connection()

            # Verify query execution
            mock_cursor.execute.assert_called_once_with("SELECT 1")
            mock_cursor.fetchone.assert_called_once()

            # Verify error logging
            mock_logger.error.assert_called_once_with(
                "Connection to MySQL database failed.")

            # Verify no error returned (method doesn't return error for empty
            # result)
            assert result is None

    @patch('similarity_matrix.lib.database.logger')
    def test_test_connection_error(self, mock_logger, database_instance):
        """Test connection test with database error."""
        # Create a test error
        test_error = Error("Database connection failed")

        # Mock the context manager to raise an error
        with patch.object(Database, '__enter__', side_effect=test_error), \
                patch.object(Database, '__exit__', return_value=None):

            result = database_instance.test_connection()

            # Verify error logging
            mock_logger.error.assert_called_once_with(
                "Error connecting to MySQL: Database connection failed")

            # Verify error is returned
            assert result == test_error

    @patch('similarity_matrix.lib.database.mysql.connector.connect')
    @patch('similarity_matrix.lib.database.logger')
    def test_context_manager_full_workflow(
            self, mock_logger, mock_connect, database_instance):
        """Test complete context manager workflow."""
        # Setup mocks
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [{'id': 1, 'name': 'test'}]

        # Test context manager usage
        with database_instance as cursor:
            cursor.execute("SELECT * FROM test_table")
            result = cursor.fetchall()

        # Verify connection setup
        mock_connect.assert_called_once_with(**database_instance.config)
        mock_connection.cursor.assert_called_once_with(dictionary=True)

        # Verify query execution
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table")
        mock_cursor.fetchall.assert_called_once()

        # Verify cleanup
        mock_cursor.close.assert_called_once()
        mock_connection.commit.assert_called_once()
        mock_connection.close.assert_called_once()

        # Verify result
        assert result == [{'id': 1, 'name': 'test'}]

    @patch('similarity_matrix.lib.database.mysql.connector.connect')
    def test_context_manager_with_exception_in_block(
            self, mock_connect, database_instance):
        """Test context manager when exception occurs in the with block."""
        # Setup mocks
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor

        # Test context manager with exception
        with pytest.raises(ValueError, match="Test exception"):
            with database_instance as cursor:
                cursor.execute("SELECT * FROM test_table")
                raise ValueError("Test exception")

        # Verify rollback was called instead of commit
        mock_cursor.close.assert_called_once()
        mock_connection.rollback.assert_called_once()
        mock_connection.commit.assert_not_called()
        mock_connection.close.assert_called_once()

    def test_config_immutability(self, db_config):
        """Test that modifying original config doesn't affect Database instance."""
        db = Database(**db_config)
        original_config = db.config.copy()

        # Modify original config
        db_config['host'] = 'modified_host'

        # Verify Database config is unchanged
        assert db.config == original_config
        assert db.config['host'] == 'localhost'

    def test_multiple_context_entries(self, database_instance):
        """Test that Database can be used multiple times as context manager."""
        mock_cursor1 = MagicMock()
        mock_cursor2 = MagicMock()

        with patch.object(Database, '__enter__', side_effect=[mock_cursor1, mock_cursor2]), \
                patch.object(Database, '__exit__', return_value=None):

            # First usage
            with database_instance as cursor1:
                assert cursor1 == mock_cursor1

            # Second usage
            with database_instance as cursor2:
                assert cursor2 == mock_cursor2


class TestDatabaseIntegration:
    """Integration-style tests for Database class."""

    @pytest.fixture
    def db_config(self):
        """Database configuration for integration tests."""
        return {
            'host': 'localhost',
            'port': 3306,
            'user': 'test_user',
            'password': 'test_password',
            'database': 'test_database'
        }

    @patch('similarity_matrix.lib.database.mysql.connector.connect')
    @patch('similarity_matrix.lib.database.logger')
    def test_complete_database_workflow(
            self, mock_logger, mock_connect, db_config):
        """Test complete database workflow with realistic scenario."""
        # Setup mocks for realistic database interaction
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor

        # Mock database responses
        mock_cursor.fetchone.side_effect = [
            {'1': 1},  # For test_connection
            {'count': 5}  # For actual query
        ]
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'name': 'Item 1'},
            {'id': 2, 'name': 'Item 2'}
        ]

        # Create Database instance
        db = Database(**db_config)

        # Test connection
        error = db.test_connection()
        assert error is None
        mock_logger.debug.assert_called_with(
            "Connection to MySQL database successful.")

        # Use database for actual work
        with db as cursor:
            # Execute multiple queries
            cursor.execute("SELECT COUNT(*) as count FROM items")
            count_result = cursor.fetchone()

            cursor.execute("SELECT * FROM items LIMIT 2")
            items = cursor.fetchall()

            # Insert operation
            cursor.execute(
                "INSERT INTO items (name) VALUES (%s)", ("New Item",))

        # Verify all interactions
        assert mock_connect.call_count == 2  # Once for test, once for actual usage
        assert mock_cursor.execute.call_count == 4  # SELECT 1, COUNT, SELECT, INSERT
        assert count_result == {'count': 5}
        assert items == [{'id': 1, 'name': 'Item 1'},
                         {'id': 2, 'name': 'Item 2'}]

        # Verify proper cleanup (commit called twice - once for each context)
        assert mock_connection.commit.call_count == 2
        assert mock_connection.close.call_count == 2


# Edge cases and error scenarios
class TestDatabaseEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    def db_config(self):
        """Fixture providing sample database configuration."""
        return {
            'host': 'localhost',
            'port': 3306,
            'user': 'testuser',
            'password': 'testpass',
            'database': 'testdb'
        }

    @pytest.fixture
    def database_instance(self, db_config):
        """Fixture providing a Database instance."""
        return Database(**db_config)

    def test_empty_config_values(self):
        """Test Database with empty configuration values."""
        db = Database('', 0, '', '', '')

        assert db.config['host'] == ''
        assert db.config['port'] == 0
        assert db.config['user'] == ''
        assert db.config['password'] == ''
        assert db.config['database'] == ''

    @patch('similarity_matrix.lib.database.mysql.connector.connect')
    def test_connection_error_types(self, mock_connect):
        """Test different types of connection errors."""
        db = Database('localhost', 3306, 'user', 'pass', 'db')

        # Test different error types
        errors_to_test = [
            Error("Authentication failed"),
            Error("Host not found"),
            Error("Database does not exist"),
            Error("Connection timeout")
        ]

        for error in errors_to_test:
            mock_connect.side_effect = error
            with pytest.raises(Error):
                db.__enter__()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
