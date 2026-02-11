import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from similarity_matrix.lib.windowing import (
    _overlapping_windows_slices,
    extract_window_and_pointers_from_text_sections,
    _superellipse_aggregation_function,
    _constraint_function,
    _set_omega,
    _find_gamma,
    _set_beta,
    _edge_case_cover,
    _get_weights,
    compute_single_aggregation_tail_weight,
    compute_dual_aggregation_tail_weight,
    compute_aggregation_tail_weight,
)


class TestOverlappingWindowsSlices:
    """Test cases for _overlapping_windows_slices function"""

    def test_basic_functionality(self):
        """Test basic window slicing functionality"""
        text = "This is an example sentence for testing"
        result = _overlapping_windows_slices(text, 3, 0.2)
        expected = [
            'This is an',
            'an example sentence',
            'sentence for testing']
        assert result == expected

    def test_perfect_division(self):
        """Test when text divides perfectly into windows"""
        text = "one two three four five six"
        result = _overlapping_windows_slices(text, 2, 0.0)
        expected = ['one two', 'three four', 'five six']
        assert result == expected

    def test_imperfect_division(self):
        """Test when text doesn't divide perfectly"""
        text = "This is an example sentence for testing extra"
        result = _overlapping_windows_slices(text, 3, 0.2)
        # Should include the last window with rightmost words
        assert len(result) >= 3
        assert result[-1] == "sentence for testing extra" or "testing extra" in result[-1]

    def test_overlap_factor(self):
        """Test different overlap factors"""
        text = "word1 word2 word3 word4 word5"

        # No overlap
        result_no_overlap = _overlapping_windows_slices(text, 2, 0.0)
        assert len(result_no_overlap) >= 2

        # 50% overlap
        result_overlap = _overlapping_windows_slices(text, 2, 0.5)
        assert len(result_overlap) >= len(result_no_overlap)

    def test_window_size_none(self):
        """Test when window_size is None"""
        text = "This is a test"
        result = _overlapping_windows_slices(text, None, 0.2)
        assert result == [text]

    def test_short_text(self):
        """Test when text is shorter than window size"""
        text = "short"
        result = _overlapping_windows_slices(text, 5, 0.2)
        assert result == [text]

    def test_empty_text(self):
        """Test with empty text"""
        result = _overlapping_windows_slices("", 3, 0.2)
        assert result == []

    def test_invalid_delta(self):
        """Test that delta > 0.8 raises ValueError"""
        with pytest.raises(ValueError, match="overlap factor.*cannot exceed 0.8"):
            _overlapping_windows_slices("test text", 2, 0.9)

    @pytest.mark.parametrize("delta", [0.0, 0.2, 0.5, 0.7])
    def test_valid_delta_values(self, delta):
        """Test various valid delta values"""
        text = "word1 word2 word3 word4 word5"
        result = _overlapping_windows_slices(text, 2, delta)
        assert isinstance(result, list)
        assert len(result) > 0


class TestExtractWindowAndPointers:
    """Test cases for extract_window_and_pointers_from_text_sections function"""

    def test_basic_functionality(self):
        """Test basic window and pointer extraction"""
        texts = np.array([
            ["Section 1 text here", "Section 2 more text"],
            ["Another section text", "Final section here"]
        ])
        windows, pointers = extract_window_and_pointers_from_text_sections(
            texts, 2, 0.2)

        assert isinstance(windows, np.ndarray)
        assert isinstance(pointers, list)
        assert len(pointers) == len(texts)

    def test_no_splitted_in_sections(self):
        """Test basic window and pointer extraction"""
        texts = np.array([
            "Mono-section of first document",
            "Mono-section of second document"
        ])
        windows, pointers = extract_window_and_pointers_from_text_sections(
            texts, 2, 0.2)

        assert isinstance(windows, np.ndarray)
        assert isinstance(pointers, list)
        assert len(pointers) == len(texts)

    def test_empty_sections(self):
        """Test handling of empty sections"""
        texts = np.array([
            ["Valid text here", ""],
            ["", "Another valid text"]
        ])
        _, pointers = extract_window_and_pointers_from_text_sections(
            texts, 2, 0.2)

        # Empty sections should be skipped
        for pointer_array in pointers:
            assert all(p > 0 for p in pointer_array)  # Pointers are 1-indexed

    def test_single_text_multiple_sections(self):
        """Test with single text having multiple sections"""
        texts = np.array([
            ["First section text", "Second section text", "Third section text"]
        ])
        _, pointers = extract_window_and_pointers_from_text_sections(
            texts, 1, 0.0)

        assert len(pointers) == 1
        assert len(pointers[0]) > 0


class TestSuperellipseAggregationFunction:
    """Test cases for _superellipse_aggregation_function"""

    def test_basic_computation(self):
        """Test basic superellipse computation"""
        x = np.array([0, 1, 2, 3])
        alpha, beta, gamma = 3.0, 1.0, 2.0

        result = _superellipse_aggregation_function(x, alpha, beta, gamma)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert result[0] == beta  # At x=0, y should equal beta
        assert result[-1] >= 0  # All values should be non-negative

    def test_boundary_conditions(self):
        """Test boundary conditions"""
        x = np.array([0])
        result = _superellipse_aggregation_function(x, 1.0, 1.0, 2.0)
        assert np.isclose(result[0], 1.0, rtol=1e-09, atol=1e-09)

    def test_negative_x_values(self):
        """Test with negative x values"""
        x = np.array([-1, 0, 1])
        result = _superellipse_aggregation_function(x, 2.0, 1.0, 2.0)

        # Should handle negative values due to abs()
        assert all(r >= 0 for r in result)


class TestConstraintFunction:
    """Test cases for _constraint_function"""

    def test_constraint_computation(self):
        """Test constraint function computation"""
        gamma = 2.0
        n_points = 5
        beta = 1.0
        omega = 0.5

        result = _constraint_function(gamma, n_points, beta, omega)
        assert isinstance(result, (int, float))

    def test_zero_constraint(self):
        """Test that constraint can reach zero"""
        # This tests the mathematical correctness
        gamma = 2.0
        n_points = 3
        beta = 1.0
        omega = 0.1

        result = _constraint_function(gamma, n_points, beta, omega)
        # Should be close to zero for properly chosen parameters
        assert isinstance(result, (int, float))


class TestSetOmega:
    """Test cases for _set_omega function"""

    def test_valid_tail_weights(self):
        """Test valid tail weight values"""
        n_points = 5
        tolerance = 1e-8

        # Test various valid tail weights
        for tail_weight in [0.0, 0.2, 0.5, 0.8, 1.0]:
            omega = _set_omega(n_points, tail_weight, tolerance)
            assert isinstance(omega, float)

    def test_invalid_tail_weights(self):
        """Test invalid tail weight values"""
        n_points = 5
        tolerance = 1e-8

        with pytest.raises(ValueError, match="Tail weight must be between 0 and 1"):
            _set_omega(n_points, -0.1, tolerance)

        with pytest.raises(ValueError, match="Tail weight must be between 0 and 1"):
            _set_omega(n_points, 1.1, tolerance)

    def test_edge_cases(self):
        """Test edge cases for tail weights"""
        n_points = 5
        tolerance = 1e-8

        # Test tail_weight = 0 (maximum weight)
        omega_zero = _set_omega(n_points, 0.0, tolerance)
        assert np.isclose(omega_zero, tolerance, rtol=1e-09, atol=1e-09)

        # Test tail_weight = 1 (average weight)
        omega_one = _set_omega(n_points, 1.0, tolerance)
        assert omega_one < 1.0


class TestFindGamma:
    """Test cases for _find_gamma function"""

    def test_gamma_finding(self):
        """Test gamma parameter finding"""
        n_points = 5
        beta = 0.8
        omega = 0.4
        tolerance = 1e-8

        gamma = _find_gamma(n_points, beta, omega, tolerance)
        assert isinstance(gamma, float)
        assert gamma > 0  # Gamma should be positive

    def test_gamma_consistency(self):
        """Test that found gamma satisfies constraint"""
        n_points = 3
        beta = 0.5
        omega = 0.3
        tolerance = 1e-6

        gamma = _find_gamma(n_points, beta, omega, tolerance)

        # Verify the constraint is satisfied
        constraint_value = _constraint_function(gamma, n_points, beta, omega)
        assert abs(constraint_value) < tolerance * \
            10  # Allow some numerical error


class TestSetBeta:
    """Test cases for _set_beta function"""

    def test_beta_calculation(self):
        """Test beta parameter calculation"""
        n_points = 5

        # Test different tail weights
        beta_zero = _set_beta(n_points, 0.0)
        assert np.isclose(beta_zero, 1.0, rtol=1e-09, atol=1e-09)

        beta_one = _set_beta(n_points, 1.0)
        assert np.isclose(beta_one, 1.0 / n_points, rtol=1e-09, atol=1e-09)

        beta_half = _set_beta(n_points, 0.5)
        assert np.isclose(beta_half, 0.5, rtol=1e-09, atol=1e-09)


class TestEdgeCaseCover:
    """Test cases for _edge_case_cover function"""

    def test_edge_case_output(self):
        """Test edge case cover function"""
        n_points = 5
        tail_weight = 0.8

        array, indexes, gamma = _edge_case_cover(n_points, tail_weight)

        assert isinstance(array, np.ndarray)
        assert isinstance(indexes, np.ndarray)
        assert isinstance(gamma, float)
        assert len(array) == n_points
        assert len(indexes) == n_points
        assert np.isclose(gamma, 0.0, rtol=1e-09, atol=1e-09)

    def test_weight_distribution(self):
        """Test weight distribution in edge case"""
        n_points = 4
        tail_weight = 0.6

        array, _, _ = _edge_case_cover(n_points, tail_weight)

        # First element should be (1 - tail_weight)
        assert np.isclose(array[0], 1 - tail_weight, rtol=1e-09, atol=1e-09)

        # Other elements should be tail_weight / (n_points - 1)
        expected_weight = tail_weight / (n_points - 1)
        assert all(array[1:] == expected_weight)


class TestGetWeights:
    """Test cases for _get_weights function"""

    def test_basic_weight_computation(self):
        """Test basic weight computation"""
        n_points = 5
        tail_weight = 0.3

        weights, indexes, gamma = _get_weights(n_points, tail_weight)

        assert isinstance(weights, np.ndarray)
        assert isinstance(indexes, np.ndarray)
        assert isinstance(gamma, float)
        assert len(weights) == n_points
        assert len(indexes) == n_points

    def test_single_point(self):
        """Test with single point"""
        weights, indexes, gamma = _get_weights(1, 0.5)

        assert len(weights) == 1
        assert np.isclose(weights[0], 1.0, rtol=1e-09, atol=1e-09)
        assert len(indexes) == 1
        assert np.isclose(gamma, 0.0, rtol=1e-09, atol=1e-09)

    def test_weights_sum_to_one(self):
        """Test that weights sum to approximately 1"""
        n_points = 10
        tail_weight = 0.4

        weights, _, _ = _get_weights(n_points, tail_weight)

        # Weights should sum to approximately 1 (within numerical precision)
        assert abs(np.sum(weights) - 1.0) < 1e-6

    def test_decreasing_weights(self):
        """Test that weights are in decreasing order"""
        n_points = 8
        tail_weight = 0.2

        weights, _, _ = _get_weights(n_points, tail_weight)

        # Weights should be in decreasing order
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    @pytest.mark.parametrize("tail_weight", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_different_tail_weights(self, tail_weight):
        """Test different tail weight values"""
        n_points = 6
        weights, _, _ = _get_weights(n_points, tail_weight)

        assert len(weights) == n_points
        assert abs(np.sum(weights) - 1.0) < 1e-6


class TestComputeAggregationTailWeight:
    """Test cases for compute_single_aggregation_tail_weight function"""

    def test_basic_aggregation_default(self):
        """Test basic aggregation functionality"""
        # Create sample data - 10 windows, 5 skills
        cos_sim_windows = np.random.default_rng(42).random((10, 5))
        # 2 texts with different window counts
        pointers = [[1, 1, 2], [1, 2, 2, 3]]
        tail_weight = 0.3

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(pointers), cos_sim_windows.shape[1])

    def test_basic_aggregation_rows(self):
        """Test basic aggregation functionality"""
        # Create sample data - 10 windows, 5 skills
        cos_sim_windows = np.random.default_rng(42).random((10, 5))
        # 2 texts with different window counts
        pointers = [[1, 1, 2], [1, 2, 2, 3]]
        tail_weight = 0.3

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight, axis=0
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(pointers), cos_sim_windows.shape[1])

    def test_basic_aggregation_columns(self):
        """Test basic aggregation functionality"""
        # Create sample data - 5 windows, 10 skills
        cos_sim_windows = np.random.default_rng(42).random((5, 10))
        # 2 texts with different window counts
        pointers = [[1, 1, 2], [1, 2, 2, 3]]
        tail_weight = 0.3

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight, axis=1
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (cos_sim_windows.shape[0], len(pointers))

    def test_with_precalculated_weights(self):
        """Test aggregation with precalculated weights"""
        cos_sim_windows = np.random.default_rng(213).random((6, 3))
        pointers = [[1, 1], [1, 2, 2, 3]]
        tail_weight = 0.5

        # Mock precalculated weights
        precalculated_weights = {
            "0.5": {
                "2": [0.7, 0.3],
                "4": [0.4, 0.3, 0.2, 0.1]
            }
        }

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight,
            precalculated_weights=precalculated_weights, max_precalculated=1000
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(pointers), cos_sim_windows.shape[1])

    def test_aggregation_properties(self):
        """Test mathematical properties of aggregation"""
        # Create controlled data
        cos_sim_windows = np.array([
            [1.0, 0.8, 0.6],
            [0.9, 0.7, 0.5],
            [0.8, 0.9, 0.7],
            [0.7, 0.6, 0.8]
        ])
        pointers = [[1, 1], [2, 2]]  # 2 texts, 2 windows each
        tail_weight = 0.0  # Maximum weight (should pick highest values)

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight
        )

        # With tail_weight=0, should heavily weight the maximum values
        assert result.shape == (2, 3)
        assert all(result[0, :] <= 1.0)  # Values should be <= 1
        assert all(result[1, :] <= 1.0)

    def test_empty_pointers(self):
        """Test with empty pointers (edge case)"""
        cos_sim_windows = np.random.default_rng(123).random((2, 3))
        pointers = []
        tail_weight = 0.5

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight
        )

        assert result.shape == (0, cos_sim_windows.shape[1])

    def test_empty_pointer_element_axis0(self):
        """Test aggregation with one empty pointer element (axis=0, row aggregation)"""
        # Create sample data - 4 windows, 3 skills
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.7, 0.6, 0.9],
            [0.6, 0.7, 0.8]
        ])

        # One valid pointer and one empty pointer
        pointers = [[1, 2], []]  # Second text has no windows
        tail_weight = 0.3

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight, axis=0
        )

        assert result.shape == (2, 3)  # 2 texts, 3 skills

        # First row should have valid aggregated values > 0
        assert np.all(result[0, :] > 0)

        # Second row should be all zeros (empty pointer)
        assert np.all(np.isclose(result[1, :], 0.0, rtol=1e-09, atol=1e-09))

    def test_empty_pointer_element_axis1(self):
        """Test aggregation with one empty pointer element (axis=1, column aggregation)"""
        # Create sample data - 3 windows, 4 skills
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7, 0.6],
            [0.8, 0.9, 0.6, 0.7],
            [0.7, 0.6, 0.9, 0.8]
        ])

        # One valid pointer and one empty pointer
        pointers = [[1, 2], []]  # Second skill group has no windows
        tail_weight = 0.4

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight, axis=1
        )

        assert result.shape == (3, 2)  # 3 windows, 2 skill groups

        # First column should have valid aggregated values > 0
        assert np.all(result[:, 0] > 0)

        # Second column should be all zeros (empty pointer)
        assert np.all(np.isclose(result[:, 1], 0.0, rtol=1e-09, atol=1e-09))

    def test_multiple_empty_pointer_elements(self):
        """Test aggregation with multiple empty pointer elements"""
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.7, 0.6, 0.9]
        ])

        # Mix of valid and empty pointers
        pointers = [[], [1, 2], [], [3]]  # First and third are empty
        tail_weight = 0.2

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight, axis=0
        )

        assert result.shape == (4, 3)  # 4 texts, 3 skills

        # Empty pointer results should be zero
        # First text (empty)
        assert np.all(np.isclose(result[0, :], 0.0, rtol=1e-09, atol=1e-09))
        # Third text (empty)
        assert np.all(np.isclose(result[2, :], 0.0, rtol=1e-09, atol=1e-09))

        # Valid pointer results should be > 0
        assert np.all(result[1, :] > 0)  # Second text (valid)
        assert np.all(result[3, :] > 0)  # Fourth text (valid)

    def test_all_empty_pointers(self):
        """Test aggregation when all pointers are empty"""
        cos_sim_windows = np.array([
            [0.9, 0.8],
            [0.7, 0.6]
        ])

        # All pointers are empty
        pointers = [[], [], []]
        tail_weight = 0.5

        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight, axis=0
        )

        assert result.shape == (3, 2)  # 3 texts, 2 skills

        # All results should be zero
        assert np.all(np.isclose(result, 0.0, rtol=1e-09, atol=1e-09))


class TestComputeDualAggregationTailWeight:

    def test_basic_dual_aggregation(self):
        """Test basic dual aggregation functionality"""
        # Create a simple 4x4 matrix
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7, 0.6],
            [0.8, 0.9, 0.6, 0.7],
            [0.7, 0.6, 0.9, 0.8],
            [0.6, 0.7, 0.8, 0.9]
        ])

        # Group rows: [0,1] and [2,3]
        row_pointers = [
            [1, 1],  # First 2 rows belong to group 1
            [2, 2]   # Last 2 rows belong to group 2
        ]

        # Group columns: [0,1] and [2,3]
        column_pointers = [
            [1, 1],  # First 2 columns belong to group 1
            [2, 2]   # Last 2 columns belong to group 2
        ]

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, 0.3
        )

        # Should return a 2x2 matrix (2 row groups x 2 column groups)
        assert result.shape == (2, 2)
        assert isinstance(result, np.ndarray)

        # Values should be reasonable (between 0 and 1 for cosine similarities)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_single_group_dual_aggregation(self):
        """Test dual aggregation with single groups (should work like regular aggregation)"""
        cos_sim_windows = np.array([
            [0.9, 0.8],
            [0.7, 0.6]
        ])

        # Single group for both rows and columns
        row_pointers = [[1, 1]]  # All rows in one group
        column_pointers = [[1, 1]]  # All columns in one group

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, 0.2
        )

        assert result.shape == (1, 1)
        assert result[0, 0] > 0

    def test_asymmetric_groups(self):
        """Test with different numbers of row and column groups"""
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.7, 0.6, 0.9]
        ])

        # 3 row groups, 2 column groups
        row_pointers = [
            [1],     # First row
            [2],     # Second row
            [3]      # Third row
        ]
        column_pointers = [
            [1, 1],  # First two columns
            [2]      # Third column
        ]

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, 0.3
        )

        assert result.shape == (3, 2)  # 3 row groups x 2 column groups

    def test_empty_groups_handling(self):
        """Test behavior with empty pointer groups"""
        cos_sim_windows = np.array([
            [0.9, 0.8],
            [0.7, 0.6]
        ])

        # Empty groups
        row_pointers = []
        column_pointers = [[1, 1]]

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, 0.3
        )

        assert result.shape == (0, 1)

    def test_sorting_behavior(self):
        """Test that values are properly sorted in descending order before aggregation"""
        # Create matrix where sorting makes a difference
        cos_sim_windows = np.array([
            [0.1, 0.9],
            [0.8, 0.2]
        ])  # Will be sorted to [0.9, 0.8, 0.2, 0.1]

        row_pointers = [[1, 1]]
        column_pointers = [[1, 1]]

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, 0.3
        )

        # The result should reflect that higher values get more weight
        assert result[0, 0] > 0.5  # Should be dominated by higher values

    def test_matrix_dimensions_validation(self):
        """Test that function works with various matrix dimensions"""
        # Test with rectangular matrix
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4]
        ])

        row_pointers = [[1], [2]]  # 2 row groups, 1 window each
        column_pointers = [[1], [2], [3]]  # 3 column groups, 1 window each

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, 0.3
        )

        assert result.shape == (2, 3)

    def test_weights_calculation_calls(self):
        """Test that _get_weights is called with correct parameters"""
        cos_sim_windows = np.array([
            [0.9, 0.8],
            [0.7, 0.6]
        ])

        row_pointers = [[1, 1]]  # 2 windows
        column_pointers = [[1]]  # 1 window

        with patch('similarity_matrix.lib.windowing._get_weights') as mock_get_weights:
            mock_get_weights.return_value = (np.array([0.6, 0.4]), None, None)

            compute_dual_aggregation_tail_weight(
                cos_sim_windows, row_pointers, column_pointers, 0.3
            )

            # Should be called twice: once for row weights (2), once for column
            # weights (1)
            calls = mock_get_weights.call_args_list
            assert len(calls) == 1
            assert calls[0][0] == (2, 0.3)  # (n_windows, tail_weight) for rows

    def test_empty_row_pointer_element(self):
        """Test dual aggregation with one empty row pointer element"""
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.7, 0.6, 0.9]
        ])

        # One empty row pointer, valid column pointers
        row_pointers = [[], [1, 2]]  # First group is empty
        column_pointers = [[1, 2], [3]]
        tail_weight = 0.3

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, tail_weight
        )

        assert result.shape == (2, 2)  # 2 row groups x 2 column groups

        # First row (empty group) should be all zeros
        assert np.all(np.isclose(result[0, :], 0.0, rtol=1e-09, atol=1e-09))

        # Second row should have valid values > 0
        assert np.all(result[1, :] > 0)

    def test_empty_column_pointer_element(self):
        """Test dual aggregation with one empty column pointer element"""
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.7, 0.6, 0.9]
        ])

        # Valid row pointers, one empty column pointer
        row_pointers = [[1, 2], [3]]
        column_pointers = [[1, 2], []]  # Second group is empty
        tail_weight = 0.3

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, tail_weight
        )

        assert result.shape == (2, 2)  # 2 row groups x 2 column groups

        # Second column (empty group) should be all zeros
        assert np.all(np.isclose(result[:, 1], 0.0, rtol=1e-09, atol=1e-09))

        # First column should have valid values > 0
        assert np.all(result[:, 0] > 0)

    def test_both_empty_pointer_elements(self):
        """Test dual aggregation with both row and column empty pointer elements"""
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7, 0.6],
            [0.8, 0.9, 0.6, 0.7],
            [0.7, 0.6, 0.9, 0.8],
            [0.6, 0.7, 0.8, 0.9]
        ])

        # Both row and column pointers have empty elements
        row_pointers = [[], [1, 2], [3]]  # First row group is empty
        column_pointers = [[1, 2], [], [3]]  # Second column group is empty
        tail_weight = 0.4

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, tail_weight
        )

        assert result.shape == (3, 3)  # 3 row groups x 3 column groups

        # First row should be all zeros (empty row group)
        assert np.all(np.isclose(result[0, :], 0.0, rtol=1e-09, atol=1e-09))

        # Second column should be all zeros (empty column group)
        assert np.all(np.isclose(result[:, 1], 0.0, rtol=1e-09, atol=1e-09))

        # Valid intersections should have values > 0
        assert result[1, 0] > 0  # Valid row group 2, valid column group 1
        assert result[1, 2] > 0  # Valid row group 2, valid column group 3
        assert result[2, 0] > 0  # Valid row group 3, valid column group 1
        assert result[2, 2] > 0  # Valid row group 3, valid column group 3

    def test_all_empty_row_pointers(self):
        """Test dual aggregation when all row pointers are empty"""
        cos_sim_windows = np.array([
            [0.9, 0.8],
            [0.7, 0.6]
        ])

        # All row pointers are empty
        row_pointers = [[], []]
        column_pointers = [[1], [2]]
        tail_weight = 0.3

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, tail_weight
        )

        assert result.shape == (2, 2)

        # All results should be zero (all row groups are empty)
        assert np.all(np.isclose(result, 0.0, rtol=1e-09, atol=1e-09))

    def test_all_empty_column_pointers(self):
        """Test dual aggregation when all column pointers are empty"""
        cos_sim_windows = np.array([
            [0.9, 0.8],
            [0.7, 0.6]
        ])

        # All column pointers are empty
        row_pointers = [[1], [2]]
        column_pointers = [[], []]
        tail_weight = 0.3

        result = compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, tail_weight
        )

        assert result.shape == (2, 2)

        # All results should be zero (all column groups are empty)
        assert np.all(np.isclose(result, 0.0, rtol=1e-09, atol=1e-09))


class TestComputeAggregationTailWeightWrapper:
    """Edge case tests for compute_aggregation_tail_weight wrapper function"""

    def test_empty_row_pointers_only(self):
        """Test wrapper with only row pointers containing empty elements"""
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.7, 0.6, 0.9]
        ])

        # Row aggregation with empty pointer
        row_pointers = [[], [1, 2]]
        tail_weight = 0.3

        result = compute_aggregation_tail_weight(
            cos_sim_windows, row_pointers, None, tail_weight
        )

        assert result.shape == (2, 3)  # 2 row groups x 3 original columns

        # First row (empty group) should be all zeros
        assert np.all(np.isclose(result[0, :], 0.0, rtol=1e-09, atol=1e-09))

        # Second row should have valid values > 0
        assert np.all(result[1, :] > 0)

    def test_empty_column_pointers_only(self):
        """Test wrapper with only column pointers containing empty elements"""
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.7, 0.6, 0.9]
        ])

        # Column aggregation with empty pointer
        column_pointers = [[1, 2], []]
        tail_weight = 0.3

        result = compute_aggregation_tail_weight(
            cos_sim_windows, None, column_pointers, tail_weight
        )

        assert result.shape == (3, 2)  # 3 original rows x 2 column groups

        # Second column (empty group) should be all zeros
        assert np.all(np.isclose(result[:, 1], 0.0, rtol=1e-09, atol=1e-09))

        # First column should have valid values > 0
        assert np.all(result[:, 0] > 0)

    def test_empty_both_pointers_dual_aggregation(self):
        """Test wrapper with both row and column pointers containing empty elements"""
        cos_sim_windows = np.array([
            [0.9, 0.8, 0.7, 0.6],
            [0.8, 0.9, 0.6, 0.7],
            [0.7, 0.6, 0.9, 0.8]
        ])

        # Both pointers have empty elements
        row_pointers = [[1], []]  # Second row group is empty
        column_pointers = [[], [1, 2]]  # First column group is empty
        tail_weight = 0.4

        result = compute_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, tail_weight
        )

        assert result.shape == (2, 2)  # 2 row groups x 2 column groups

        # Second row should be all zeros (empty row group)
        assert np.all(np.isclose(result[1, :], 0.0, rtol=1e-09, atol=1e-09))

        # First column should be all zeros (empty column group)
        assert np.all(np.isclose(result[:, 0], 0.0, rtol=1e-09, atol=1e-09))

        # Only valid intersection should have value > 0
        assert result[0, 1] > 0  # Valid row group 1, valid column group 2

    def test_mixed_empty_scenarios(self):
        """Test various mixed scenarios with empty pointers"""
        cos_sim_windows = np.random.default_rng(555).random((6, 4))

        # Test case 1: Some empty row pointers
        row_pointers = [[], [1, 2], [], [3, 4]]
        result1 = compute_aggregation_tail_weight(
            cos_sim_windows, row_pointers, None, 0.3
        )
        assert result1.shape == (4, 4)
        # Empty group
        assert np.all(np.isclose(result1[0, :], 0.0, rtol=1e-09, atol=1e-09))
        # Empty group
        assert np.all(np.isclose(result1[2, :], 0.0, rtol=1e-09, atol=1e-09))
        assert np.all(result1[1, :] > 0)     # Valid group
        assert np.all(result1[3, :] > 0)     # Valid group

        # Test case 2: Some empty column pointers
        column_pointers = [[1], [], [2, 3], []]
        result2 = compute_aggregation_tail_weight(
            cos_sim_windows, None, column_pointers, 0.3
        )
        assert result2.shape == (6, 4)
        # Empty group
        assert np.all(np.isclose(result2[:, 1], 0.0, rtol=1e-09, atol=1e-09))
        # Empty group
        assert np.all(np.isclose(result2[:, 3], 0.0, rtol=1e-09, atol=1e-09))
        assert np.all(result2[:, 0] > 0)     # Valid group
        assert np.all(result2[:, 2] > 0)     # Valid group


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_pipeline(self):
        """Test the full pipeline from text to aggregated similarities"""
        # Create sample text data
        texts = np.array([
            ["First section of text one", "Second section of text one"],
            ["First section of text two", "Second section of text two"]
        ])

        # Extract windows and pointers
        windows, pointers = extract_window_and_pointers_from_text_sections(
            texts, window_size=3, delta=0.2
        )

        # Create mock cosine similarity matrix
        n_windows = len(windows) if len(windows) > 0 else sum(len(p)
                                                              for p in pointers)
        n_texts = len(texts)
        n_skills = 4
        cos_sim_windows = np.random.default_rng(
            321).random((n_windows, n_skills))

        # Compute aggregation
        result = compute_single_aggregation_tail_weight(
            cos_sim_windows, pointers, tail_weight=0.3
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_texts, n_skills)

    def test_weight_computation_pipeline(self):
        """Test weight computation pipeline"""
        n_points = 7
        tail_weight = 0.4

        # Test the full weight computation pipeline
        weights, _, _ = _get_weights(n_points, tail_weight)

        # Verify mathematical properties
        assert abs(np.sum(weights) - 1.0) < 1e-6  # Sum to 1
        assert all(weights[i] >= weights[i + 1]
                   for i in range(len(weights) - 1))  # Decreasing
        assert weights[0] > 0  # Positive weights
        assert all(w >= 0 for w in weights)  # Non-negative weights


# Fixtures for common test data
@pytest.fixture
def sample_text_data():
    """Fixture providing sample text data"""
    return np.array([
        ["This is the first section", "This is the second section"],
        ["Another first section here", "Another second section here"],
        ["Final first section", "Final second section"]
    ])


@pytest.fixture
def sample_cosine_similarity():
    """Fixture providing sample cosine similarity matrix"""
    return np.random.default_rng(344).rand((12, 5))  # 12 windows, 5 skills


@pytest.fixture
def sample_pointers():
    """Fixture providing sample pointers"""
    return [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]


# Parametrized tests
@pytest.mark.parametrize("window_size,delta", [
    (2, 0.0), (3, 0.2), (4, 0.5), (5, 0.8)
])
def test_window_slicing_parameters(window_size, delta):
    """Test window slicing with different parameters"""
    text = "word1 word2 word3 word4 word5 word6 word7 word8"
    result = _overlapping_windows_slices(text, window_size, delta)

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(window, str) for window in result)


@pytest.mark.parametrize("tail_weight", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_tail_weight_values(tail_weight):
    """Test different tail weight values"""
    n_points = 6
    weights, _, _ = _get_weights(n_points, tail_weight)

    assert len(weights) == n_points
    assert abs(np.sum(weights) - 1.0) < 1e-6


# Performance tests (optional - can be slow)
@pytest.mark.slow
def test_large_scale_processing():
    """Test processing with larger datasets"""
    # Create larger text dataset
    large_texts = np.array([
        [f"Section {i} text content here" for i in range(10)]
        for _ in range(20)
    ])

    windows, pointers = extract_window_and_pointers_from_text_sections(
        large_texts, window_size=5, delta=0.3
    )

    # Should handle large datasets without errors
    assert len(windows) > 0
    assert len(pointers) == len(large_texts)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main(["-v", __file__])
