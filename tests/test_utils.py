import pytest
import numpy as np
from numpy.testing import assert_array_equal

from protein_universe_annotate.utils import get_top_k_values_indices


def test_get_top_k_values_indices():
    # Test case 1: Regular input with num_top_values = 3
    arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_array_equal(get_top_k_values_indices(arr1), np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))

    # Test case 2: Regular input with num_top_values = 2
    arr2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])
    assert_array_equal(get_top_k_values_indices(arr2, num_top_values=2), np.array([[2, 3], [2, 3], [2, 3]]))

    # Test case 3: Input with all the same values
    arr3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert_array_equal(get_top_k_values_indices(arr3), np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))

    # Test case 4: Input with negative values
    arr4 = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])
    assert_array_equal(get_top_k_values_indices(arr4), np.array([[1, 0, 2], [2, 0, 1], [1, 0, 2]]))

    # Test case 5: Edge case with empty array
    arr5 = np.array([])
    assert_array_equal(get_top_k_values_indices(arr5), np.array([]))

    # Test case 6: Array containing repeated values
    arr = np.array([[1, 2, 2, 3], [5, 2, 5, 6], [0, 8, 8, 7], [9, 9, 9, 0]])
    expected_result = np.array([[1, 2, 3], [0, 2, 3], [3, 1, 2], [0, 1, 2]])
    assert_array_equal(get_top_k_values_indices(arr, num_top_values=3), expected_result)


