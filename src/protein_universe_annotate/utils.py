import numpy as np


def get_top_k_values_indices(arr, num_top_values=3):
    """
    Given a 2D numpy array,
    returns a 2D numpy array containing the indices of the topK largest values in each row,
    sorted in ascending order based on the corresponding values in the input array.
    """
    if arr.size == 0: return np.array([])

    # Find the indices of the topK values in each row
    top_indices = np.argsort(arr, axis=1)[:, -num_top_values:]

    return top_indices
