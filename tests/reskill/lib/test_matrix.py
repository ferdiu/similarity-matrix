import pytest
import numpy as np
import json
import uuid
import tempfile
import shutil
from pathlib import Path

from similarity_matrix.lib.matrix import SimilarityMatrix


class TestSymilarityMatrix:
    """Test suite for SymilarityMatrix class."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample test data."""
        rows = ["SUA_001", "SUA_002", "SUA_003"]
        columns = [str(uuid.uuid4()) for _ in range(4)]
        name = "test_matrix"
        return rows, columns, name

    @pytest.fixture
    def sample_matrix(self, sample_data):
        """Fixture providing a sample SymilarityMatrix instance."""
        rows, columns, name = sample_data
        return SimilarityMatrix.create_empty(rows, columns, name)

    @pytest.fixture
    def temp_dir(self):
        """Fixture providing a temporary directory for file operations."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_init_with_empty_matrix(self, sample_data):
        """Test initialization with empty matrix."""
        rows, columns, name = sample_data
        matrix = SimilarityMatrix(rows, columns, name)

        assert matrix.row_ids == rows
        assert matrix.column_ids == columns
        assert matrix.name == name
        assert matrix.matrix.shape == (len(rows), len(columns))
        assert np.all(matrix.matrix == 0)
        assert matrix.matrix.dtype == float

    def test_init_with_provided_matrix(self, sample_data):
        """Test initialization with provided matrix."""
        rows, columns, name = sample_data
        test_matrix = np.random.default_rng(
            42).random((len(rows), len(columns)))

        matrix = SimilarityMatrix(rows, columns, name, matrix=test_matrix)

        assert matrix.row_ids == rows
        assert matrix.column_ids == columns
        assert matrix.name == name
        assert np.array_equal(matrix.matrix, test_matrix.astype(float))

    def test_init_with_wrong_matrix_dimensions(self, sample_data):
        """Test initialization with incorrectly sized matrix raises ValueError."""
        rows, columns, name = sample_data
        wrong_matrix = np.random.default_rng(
            42).random((2, 2))  # Wrong dimensions

        with pytest.raises(ValueError, match="Matrix shape .* doesn't match dimensions"):
            SimilarityMatrix(rows, columns, name, matrix=wrong_matrix)

    def test_create_empty(self, sample_data):
        """Test create_empty class method."""
        rows, columns, name = sample_data
        matrix = SimilarityMatrix.create_empty(rows, columns, name)

        assert matrix.row_ids == rows
        assert matrix.column_ids == columns
        assert matrix.name == name
        assert matrix.matrix.shape == (len(rows), len(columns))
        assert np.all(matrix.matrix == 0)

    def test_calculate(self, sample_matrix):
        """Test calculate method generates random values between -1 and 1."""
        # Set seed for reproducible testing
        np.random.seed(42)

        sample_matrix.calculate(fake=42)

        # Check that matrix is no longer all zeros
        assert not np.all(sample_matrix.matrix == 0)

        # Check that all values are between -1 and 1
        assert np.all(sample_matrix.matrix >= -1)
        assert np.all(sample_matrix.matrix <= 1)

        # Check matrix shape is preserved
        assert sample_matrix.matrix.shape == (
            sample_matrix.num_rows, sample_matrix.num_columns)

    def test_get_set_value(self, sample_matrix):
        """Test getting and setting individual values."""
        row_id = sample_matrix.row_ids[0]
        column_id = sample_matrix.column_ids[0]
        test_value = 0.75

        # Test setting value
        sample_matrix.set_value(row_id, column_id, test_value)

        # Test getting value
        retrieved_value = sample_matrix.get_value(row_id, column_id)
        assert np.isclose(retrieved_value, test_value, rtol=1e-09, atol=1e-09)

    def test_get_set_value_invalid_ids(self, sample_matrix):
        """Test getting/setting values with invalid IDs raises ValueError."""
        with pytest.raises(ValueError):
            sample_matrix.get_value("invalid_row", sample_matrix.column_ids[0])

        with pytest.raises(ValueError):
            sample_matrix.get_value(sample_matrix.row_ids[0], "invalid_column")

        with pytest.raises(ValueError):
            sample_matrix.set_value(
                "invalid_row", sample_matrix.column_ids[0], 0.5)

    def test_get_row(self, sample_matrix):
        """Test getting entire SUA row."""
        # Set some test values
        sample_matrix.matrix[0, :] = [0.1, 0.2, 0.3, 0.4]

        row = sample_matrix.get_row(sample_matrix.row_ids[0])
        expected = np.array([0.1, 0.2, 0.3, 0.4])

        assert np.array_equal(row, expected)
        assert len(row) == sample_matrix.num_columns

    def test_get_column(self, sample_matrix):
        """Test getting entire column column."""
        # Set some test values
        sample_matrix.matrix[:, 0] = [0.1, 0.2, 0.3]

        column = sample_matrix.get_column(sample_matrix.column_ids[0])
        expected = np.array([0.1, 0.2, 0.3])

        assert np.array_equal(column, expected)
        assert len(column) == sample_matrix.num_rows

    def test_properties(self, sample_matrix):
        """Test class properties."""
        assert sample_matrix.shape == (3, 4)
        assert sample_matrix.num_rows == 3
        assert sample_matrix.num_columns == 4

    def test_iterable_yields_correct_tuples(self):
        # Prepare test matrix
        row_ids = ['row1', 'row2']
        col_ids = ['colA', 'colB']
        name = 'test_matrix'

        matrix = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ], dtype=float)

        sim_matrix = SimilarityMatrix(row_ids, col_ids, name, matrix=matrix)

        expected = [
            ('row1', 'colA', 0.1),
            ('row1', 'colB', 0.2),
            ('row2', 'colA', 0.3),
            ('row2', 'colB', 0.4)
        ]

        # Collect from iterator
        actual = list(sim_matrix)

        # Assert all elements match
        assert len(actual) == len(expected)
        for act, exp in zip(actual, expected):
            assert act[0] == exp[0]
            assert act[1] == exp[1]
            assert np.isclose(
                act[2], exp[2]), f"Expected {exp[2]}" + \
                f" but got {act[2]}"

    def test_iterable_on_empty_matrix(self):
        sim_matrix = SimilarityMatrix.create_empty(['r1'], ['c1'], 'empty')

        result = list(sim_matrix)
        assert result == [('r1', 'c1', 0.0)]

    def test_str_repr(self, sample_matrix):
        """Test string representations."""
        str_repr = str(sample_matrix)
        assert "test_matrix" in str_repr
        assert "3 rows" in str_repr
        assert "4 columns" in str_repr

        repr_str = repr(sample_matrix)
        assert "name='test_matrix'" in repr_str
        assert "matrix.shape=(3, 4)" in repr_str

    def test_save_creates_files(self, sample_matrix, temp_dir):
        """Test that save method creates the expected files."""
        sample_matrix.calculate(fake=42)
        sample_matrix.save(temp_dir)

        # Check files exist
        matrix_file = Path(temp_dir) / f"{sample_matrix.name}.npy"
        json_file = Path(temp_dir) / f"{sample_matrix.name}.json"

        assert matrix_file.exists()
        assert json_file.exists()

    def test_save_creates_directory(self, sample_matrix):
        """Test that save method creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = Path(temp_dir) / "new_directory"
            assert not non_existent_dir.exists()

            sample_matrix.save(non_existent_dir)

            assert non_existent_dir.exists()
            assert (non_existent_dir / f"{sample_matrix.name}.npy").exists()
            assert (non_existent_dir / f"{sample_matrix.name}.json").exists()

    def test_save_json_content(self, sample_matrix, temp_dir):
        """Test that JSON file contains correct data."""
        sample_matrix.save(temp_dir)

        json_file = Path(temp_dir) / f"{sample_matrix.name}.json"
        with open(json_file, 'r') as f:
            data = json.load(f)

        assert data['name'] == sample_matrix.name
        assert data['row_ids'] == sample_matrix.row_ids
        assert data['column_ids'] == sample_matrix.column_ids

    def test_save_matrix_content(self, sample_matrix, temp_dir):
        """Test that numpy file contains correct matrix data."""
        # Set known values
        sample_matrix.matrix[0, 0] = 0.123
        sample_matrix.matrix[1, 1] = 0.456

        sample_matrix.save(temp_dir)

        matrix_file = Path(temp_dir) / f"{sample_matrix.name}.npy"
        loaded_matrix = np.load(matrix_file)

        assert np.array_equal(loaded_matrix, sample_matrix.matrix)

    def test_load_success(self, sample_matrix, temp_dir):
        """Test successful loading of saved matrix."""
        # Save original matrix
        sample_matrix.calculate(fake=42)
        original_matrix = sample_matrix.matrix.copy()
        sample_matrix.save(temp_dir)

        # Load matrix
        loaded_matrix = SimilarityMatrix.load(temp_dir, sample_matrix.name)

        assert loaded_matrix.name == sample_matrix.name
        assert loaded_matrix.row_ids == sample_matrix.row_ids
        assert loaded_matrix.column_ids == sample_matrix.column_ids
        assert np.array_equal(loaded_matrix.matrix, original_matrix)

    def test_load_missing_matrix_file(self, temp_dir):
        """Test loading with missing matrix file raises FileNotFoundError."""
        # Create only JSON file
        json_data = {
            'name': 'test',
            'rows': ['SUA_001'],
            'columns': ['column_001']
        }
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f)

        with pytest.raises(FileNotFoundError, match="Matrix file not found"):
            SimilarityMatrix.load(temp_dir, "test")

    def test_load_missing_json_file(self, temp_dir):
        """Test loading with missing JSON file raises FileNotFoundError."""
        # Create only matrix file
        matrix_file = Path(temp_dir) / "test.npy"
        np.save(matrix_file, np.zeros((2, 2)))

        with pytest.raises(FileNotFoundError, match="Indices file not found"):
            SimilarityMatrix.load(temp_dir, "test")

    def test_load_fallback_name(self, sample_matrix, temp_dir):
        """Test loading with fallback name when JSON doesn't contain name."""
        # Save matrix
        sample_matrix.save(temp_dir)

        # Modify JSON to remove name
        json_file = Path(temp_dir) / f"{sample_matrix.name}.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
        del data['name']
        with open(json_file, 'w') as f:
            json.dump(data, f)

        # Load should use provided name as fallback
        loaded_matrix = SimilarityMatrix.load(temp_dir, sample_matrix.name)
        assert loaded_matrix.name == sample_matrix.name

    def test_roundtrip_save_load(self, sample_data, temp_dir):
        """Test complete save/load roundtrip preserves all data."""
        rows, columns, name = sample_data

        # Create matrix with calculated values
        original = SimilarityMatrix.create_empty(rows, columns, name)
        original.calculate(fake=42)

        # Save and load
        original.save(temp_dir)
        loaded = SimilarityMatrix.load(temp_dir, name)

        # Verify everything matches
        assert loaded.name == original.name
        assert loaded.row_ids == original.row_ids
        assert loaded.column_ids == original.column_ids
        assert loaded.shape == original.shape
        assert np.array_equal(loaded.matrix, original.matrix)

    def test_calculate_deterministic_with_seed(self, sample_matrix):
        """Test that calculate produces deterministic results with same seed."""
        sample_matrix.calculate(fake=123)
        matrix1 = sample_matrix.matrix.copy()

        sample_matrix.calculate(fake=123)
        matrix2 = sample_matrix.matrix.copy()

        assert np.array_equal(matrix1, matrix2)

    def test_calculate_different_with_different_seeds(self, sample_matrix):
        """Test that calculate produces different results with different seeds."""
        sample_matrix.calculate(fake=123)
        matrix1 = sample_matrix.matrix.copy()

        sample_matrix.calculate(fake=456)
        matrix2 = sample_matrix.matrix.copy()

        assert not np.array_equal(matrix1, matrix2)

    def test_empty_lists(self):
        """Test handling of empty SUA and column lists."""
        matrix = SimilarityMatrix([], [], "empty_matrix")
        assert matrix.shape == (0, 0)
        assert matrix.num_rows == 0
        assert matrix.num_columns == 0

    def test_single_element_lists(self):
        """Test handling of single-element lists."""
        matrix = SimilarityMatrix(["SUA_001"], ["column_001"], "single_matrix")
        assert matrix.shape == (1, 1)
        assert matrix.num_rows == 1
        assert matrix.num_columns == 1

        matrix.set_value("SUA_001", "column_001", 0.5)
        assert np.isclose(matrix.get_value("SUA_001", "column_001"), 0.5,
                          rtol=1e-09, atol=1e-09)


class TestSimilarityMatrixEquality:
    """Test cases for __eq__ and __ne__ methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.row_ids = ['row1', 'row2', 'row3']
        self.column_ids = ['col1', 'col2', 'col3', 'col4']
        self.name = 'test_matrix'
        self.matrix = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ])

        self.sim_matrix = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name=self.name,
            matrix=self.matrix
        )

    def test_eq_identical_matrices(self):
        """Test that identical matrices are equal."""
        other_matrix = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name=self.name,
            matrix=self.matrix.copy()
        )

        assert self.sim_matrix == other_matrix
        assert not (self.sim_matrix != other_matrix)

    def test_eq_different_names(self):
        """Test that matrices with different names are not equal."""
        other_matrix = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name='different_name',
            matrix=self.matrix.copy()
        )

        assert self.sim_matrix != other_matrix
        assert not (self.sim_matrix == other_matrix)

    def test_eq_different_row_ids(self):
        """Test that matrices with different row IDs are not equal."""
        different_row_ids = ['rowA', 'rowB', 'rowC']
        other_matrix = SimilarityMatrix(
            row_ids=different_row_ids,
            column_ids=self.column_ids,
            name=self.name,
            matrix=self.matrix.copy()
        )

        assert self.sim_matrix != other_matrix
        assert not (self.sim_matrix == other_matrix)

    def test_eq_different_column_ids(self):
        """Test that matrices with different column IDs are not equal."""
        different_column_ids = ['colA', 'colB', 'colC', 'colD']
        other_matrix = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=different_column_ids,
            name=self.name,
            matrix=self.matrix.copy()
        )

        assert self.sim_matrix != other_matrix
        assert not (self.sim_matrix == other_matrix)

    def test_eq_different_matrix_values(self):
        """Test that matrices with different values are not equal."""
        different_matrix = self.matrix.copy()
        different_matrix[0, 0] = 999.0

        other_matrix = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name=self.name,
            matrix=different_matrix
        )

        assert self.sim_matrix != other_matrix
        assert not (self.sim_matrix == other_matrix)

    def test_eq_different_matrix_shapes(self):
        """Test that matrices with different shapes are not equal."""
        different_matrix = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        different_row_ids = ['row1', 'row2']
        different_column_ids = ['col1', 'col2']

        other_matrix = SimilarityMatrix(
            row_ids=different_row_ids,
            column_ids=different_column_ids,
            name=self.name,
            matrix=different_matrix
        )

        assert self.sim_matrix != other_matrix
        assert not (self.sim_matrix == other_matrix)

    def test_eq_with_nan_values(self):
        """Test equality with NaN values in matrices."""
        matrix_with_nan = self.matrix.copy()
        matrix_with_nan[0, 0] = np.nan

        matrix1 = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name=self.name,
            matrix=matrix_with_nan
        )

        matrix2 = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name=self.name,
            matrix=matrix_with_nan.copy()
        )

        # NaN != NaN, so matrices with NaN should not be equal
        assert matrix1 != matrix2
        assert not (matrix1 == matrix2)

    def test_eq_with_non_similarity_matrix(self):
        """Test equality with non-SimilarityMatrix objects."""
        assert self.sim_matrix != "not a matrix"
        assert self.sim_matrix != 42
        assert self.sim_matrix is not None
        assert self.sim_matrix != self.matrix
        assert self.sim_matrix != ['list', 'of', 'values']

        # Test __ne__ as well
        assert self.sim_matrix != "not a matrix"
        assert not (self.sim_matrix == "not a matrix")

    def test_ne_explicit(self):
        """Test explicit __ne__ method calls."""
        other_matrix = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name='different_name',
            matrix=self.matrix.copy()
        )

        # Test __ne__ directly
        assert self.sim_matrix.__ne__(other_matrix)
        assert not self.sim_matrix.__ne__(self.sim_matrix)


class TestSimilarityMatrixGetItem:
    """Test cases for __getitem__ method."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.row_ids = ['row1', 'row2', 'row3']
        self.column_ids = ['col1', 'col2', 'col3', 'col4']
        self.name = 'test_matrix'
        self.matrix = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ])

        self.sim_matrix = SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name=self.name,
            matrix=self.matrix
        )

    def test_getitem_with_int_index(self):
        """Test __getitem__ with integer index (returns matrix row)."""
        # Test valid indices
        row_0 = self.sim_matrix[0]
        expected_row_0 = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(row_0, expected_row_0)

        row_1 = self.sim_matrix[1]
        expected_row_1 = np.array([5.0, 6.0, 7.0, 8.0])
        np.testing.assert_array_equal(row_1, expected_row_1)

        row_2 = self.sim_matrix[2]
        expected_row_2 = np.array([9.0, 10.0, 11.0, 12.0])
        np.testing.assert_array_equal(row_2, expected_row_2)

    def test_getitem_with_negative_int_index(self):
        """Test __getitem__ with negative integer index."""
        # Test negative indices
        row_neg1 = self.sim_matrix[-1]
        expected_row_neg1 = np.array([9.0, 10.0, 11.0, 12.0])
        np.testing.assert_array_equal(row_neg1, expected_row_neg1)

        row_neg2 = self.sim_matrix[-2]
        expected_row_neg2 = np.array([5.0, 6.0, 7.0, 8.0])
        np.testing.assert_array_equal(row_neg2, expected_row_neg2)

    def test_getitem_with_out_of_bounds_int_index(self):
        """Test __getitem__ with out-of-bounds integer index."""
        with pytest.raises(IndexError):
            _ = self.sim_matrix[3]  # Only indices 0, 1, 2 are valid

        with pytest.raises(IndexError):
            _ = self.sim_matrix[-4]  # Out of bounds negative index

    def test_getitem_with_string_row_id(self):
        """Test __getitem__ with string row ID (returns row data)."""
        # Test valid row IDs
        row_data_1 = self.sim_matrix['row1']
        expected_row_1 = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(row_data_1, expected_row_1)

        row_data_2 = self.sim_matrix['row2']
        expected_row_2 = np.array([5.0, 6.0, 7.0, 8.0])
        np.testing.assert_array_equal(row_data_2, expected_row_2)

        row_data_3 = self.sim_matrix['row3']
        expected_row_3 = np.array([9.0, 10.0, 11.0, 12.0])
        np.testing.assert_array_equal(row_data_3, expected_row_3)

    def test_getitem_with_invalid_string_row_id(self):
        """Test __getitem__ with invalid string row ID."""
        with pytest.raises(ValueError, match="'invalid_row' is not in list"):
            _ = self.sim_matrix['invalid_row']

        with pytest.raises(ValueError, match="'nonexistent' is not in list"):
            _ = self.sim_matrix['nonexistent']

    def test_getitem_with_invalid_type(self):
        """Test __getitem__ with invalid key types."""
        with pytest.raises(TypeError, match="Invalid key type. Expected int or str."):
            _ = self.sim_matrix[1.5]  # float

        with pytest.raises(TypeError, match="Invalid key type. Expected int or str."):
            _ = self.sim_matrix[['list']]  # list

        with pytest.raises(TypeError, match="Invalid key type. Expected int or str."):
            _ = self.sim_matrix[{'dict': 'value'}]  # dict

        with pytest.raises(TypeError, match="Invalid key type. Expected int or str."):
            _ = self.sim_matrix[None]  # None

    def test_getitem_return_types(self):
        """Test that __getitem__ returns the correct types."""
        # Integer index should return numpy array
        result_int = self.sim_matrix[0]
        assert isinstance(result_int, np.ndarray)
        assert result_int.shape == (4,)  # Should be 1D array with 4 elements

        # String index should return numpy array
        result_str = self.sim_matrix['row1']
        assert isinstance(result_str, np.ndarray)
        assert result_str.shape == (4,)  # Should be 1D array with 4 elements

        # Both should be equivalent
        np.testing.assert_array_equal(result_int, result_str)

    def test_getitem_with_empty_matrix(self):
        """Test __getitem__ with an empty matrix."""
        empty_matrix = SimilarityMatrix(
            row_ids=[],
            column_ids=[],
            name='empty',
            matrix=np.array([]).reshape(0, 0)
        )

        # Should raise IndexError for any integer index
        with pytest.raises(IndexError):
            _ = empty_matrix[0]

        # Should raise ValueError for any string index
        with pytest.raises(ValueError):
            _ = empty_matrix['any_string']

    def test_getitem_single_element_matrix(self):
        """Test __getitem__ with a single-element matrix."""
        single_matrix = SimilarityMatrix(
            row_ids=['single_row'],
            column_ids=['single_col'],
            name='single',
            matrix=np.array([[42.0]])
        )

        # Test integer index
        result_int = single_matrix[0]
        expected = np.array([42.0])
        np.testing.assert_array_equal(result_int, expected)

        # Test string index
        result_str = single_matrix['single_row']
        np.testing.assert_array_equal(result_str, expected)


# Integration tests
class TestSymilarityMatrixIntegration:
    """Integration tests for SymilarityMatrix."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: create, calculate, save, load, verify."""
        # Setup
        rows = [f"SUA_{i:03d}" for i in range(10)]
        columns = [str(uuid.uuid4()) for _ in range(5)]
        name = "integration_test_matrix"

        # Create and calculate
        matrix = SimilarityMatrix.create_empty(rows, columns, name)
        matrix.calculate(fake=42)

        # Verify calculation worked
        assert not np.all(matrix.matrix == 0)
        assert np.all(matrix.matrix >= -1)
        assert np.all(matrix.matrix <= 1)

        # Save
        save_dir = tmp_path / "matrices"
        matrix.save(save_dir)

        # Verify files exist
        assert (save_dir / f"{name}.npy").exists()
        assert (save_dir / f"{name}.json").exists()

        # Load
        loaded_matrix = SimilarityMatrix.load(save_dir, name)

        # Verify loaded matrix matches original
        assert loaded_matrix.name == matrix.name
        assert loaded_matrix.row_ids == matrix.row_ids
        assert loaded_matrix.column_ids == matrix.column_ids
        assert np.array_equal(loaded_matrix.matrix, matrix.matrix)

        # Test operations on loaded matrix
        test_value = 0.999
        loaded_matrix.set_value(rows[0], columns[0], test_value)
        assert np.isclose(loaded_matrix.get_value(rows[0], columns[0]),
                          test_value, rtol=1e-09, atol=1e-09)

    def test_equality_and_getitem_consistency(self):
        """Test that equal matrices return same values with __getitem__."""
        row_ids = ['a', 'b']
        column_ids = ['x', 'y', 'z']
        matrix = np.array([[1, 2, 3], [4, 5, 6]])

        matrix1 = SimilarityMatrix(
            row_ids, column_ids, 'test', matrix=matrix.copy())
        matrix2 = SimilarityMatrix(
            row_ids, column_ids, 'test', matrix=matrix.copy())

        # Matrices should be equal
        assert matrix1 == matrix2

        # __getitem__ should return same values
        np.testing.assert_array_equal(matrix1[0], matrix2[0])
        np.testing.assert_array_equal(matrix1['a'], matrix2['a'])
        np.testing.assert_array_equal(matrix1[1], matrix2[1])
        np.testing.assert_array_equal(matrix1['b'], matrix2['b'])


if __name__ == "__main__":
    pytest.main([__file__])
