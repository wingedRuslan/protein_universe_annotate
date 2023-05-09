import numpy as np
import json
from sklearn.preprocessing import LabelEncoder


def get_top_k_values_indices(arr: np.ndarray,
                             k: int = 3) -> np.ndarray:
    """
    Returns a 2D numpy array containing the indices of the top k largest values in each row
    of the input 2D numpy array, sorted in ascending order based on the corresponding values in the input array.

    Args:
        arr (np.ndarray): A 2D numpy array.
        k (int, optional): The number of indices to return for each row. Default is 3.

    Returns:
        np.ndarray: A 2D numpy array containing the indices of the top k largest values in each row.
    """
    if arr.size == 0: return np.array([])

    # Find the indices of the top k values in each row
    top_indices = np.argsort(arr, axis=1)[:, -k:]

    return top_indices


def save_label_encoder(label_encoder: LabelEncoder,
                       save_path: str) -> None:
    """
    Saves a scikit-learn LabelEncoder mapping to a JSON file.

    Args:
        label_encoder (LabelEncoder): LabelEncoder object containing label to index mapping
        save_path (str): Path to the file to save the JSON
    """

    # Get the mapping of labels to indices from the LabelEncoder object
    label_index_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Convert NumPy data types to Python data types for JSON serialization (json does not recognize NumPy data types)
    label_index_map_converted = {str(label): int(index) for label, index in label_index_map.items()}

    # Verify that the mapping was converted correctly
    assert len(label_index_map) == len(label_index_map_converted)

    # Save the label index mapping (dict) to a JSON file
    with open(f"{save_path}/label_encoder_mapping.json", "w") as f:
        json.dump(label_index_map_converted, f, indent=4)

    return

