import numpy as np
import pytest
from similarity_matrix.lib.math import normalize_array


# -----------------------------------------------------------------------
# Normalization (copy) tests

class TestMathLibrary:
    def test_normalize_default(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = normalize_array(arr)
        expected = (arr - 1) / (5 - 1)
        np.testing.assert_allclose(result, expected)

    def test_normalize_with_min_v(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = normalize_array(arr, min_v=0)
        expected = (arr - 0) / (5 - 0)
        np.testing.assert_allclose(result, expected)

    def test_normalize_with_max_v(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = normalize_array(arr, max_v=10)
        expected = (arr - 1) / (10 - 1)
        np.testing.assert_allclose(result, expected)

    def test_normalize_with_min_and_max_v(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = normalize_array(arr, min_v=0, max_v=10)
        expected = (arr - 0) / (10 - 0)
        np.testing.assert_allclose(result, expected)

    def test_normalize_constant_array(self):
        arr = np.array([3, 3, 3])
        result = normalize_array(arr)
        expected = np.zeros_like(arr, dtype=float)
        np.testing.assert_allclose(result, expected)

    def test_normalize_negative_values(self):
        arr = np.array([-5, 0, 5])
        result = normalize_array(arr)
        expected = (arr - (-5)) / (5 - (-5))
        np.testing.assert_allclose(result, expected)

    def test_normalize_single_element(self):
        arr = np.array([42])
        result = normalize_array(arr)
        expected = np.zeros_like(arr, dtype=float)
        np.testing.assert_allclose(result, expected)

    def test_normalize_min_greater_than_max(self):
        arr = np.array([1, 2, 3])
        result = normalize_array(arr, min_v=5, max_v=1)
        expected = (arr - 5) / (1 - 5)
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
