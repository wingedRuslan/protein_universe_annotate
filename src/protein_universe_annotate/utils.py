import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import os


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


def get_array_size_in_mb(arr: np.ndarray) -> float:
    """
    Calculate the size of a NumPy array in megabytes (MB).

    Args:
        arr (numpy.ndarray): The NumPy array.

    Returns:
        float: The size of the array in megabytes.
    """
    # Get the size of the array in bytes
    size_in_bytes = arr.nbytes

    # Convert to megabytes (MB)
    size_in_mb = size_in_bytes / (1024 * 1024)

    return size_in_mb


def save_count_vectorizer_params(count_vectorizer: CountVectorizer,
                                 output_dir: str) -> None:
    """
    Save the parameters of the CountVectorizer as a JSON file.

    Args:
        count_vectorizer (sklearn.feature_extraction.text.CountVectorizer): The CountVectorizer object.
        output_dir (str): Directory path to save the JSON file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the parameters of the CountVectorizer
    count_vectorizer_params = count_vectorizer.get_params()

    # Convert non-serializable values to JSON serializable types
    for key, value in count_vectorizer_params.items():
        if isinstance(value, type):
            count_vectorizer_params[key] = str(value)

    # Save the parameters to a JSON file
    output_path = os.path.join(output_dir, 'count_vectorizer_params.json')
    with open(output_path, 'w') as file:
        json.dump(count_vectorizer_params, file, indent=4)
    print(f"CountVectorizer parameters saved at: {output_path}")

    return


def save_count_vectorizer_vocab(count_vectorizer: CountVectorizer,
                                output_dir: str) -> None:
    """
    Save the vocabulary of the CountVectorizer as a JSON file.

    Args:
        count_vectorizer (sklearn.feature_extraction.text.CountVectorizer): The CountVectorizer object.
        output_dir (str): Directory path to save the JSON file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the vocabulary
    vocabulary = count_vectorizer.vocabulary_

    # Convert the values to integers to allow json serialization (no support for numpy.int64)
    vocabulary = {key: int(value) for key, value in vocabulary.items()}

    output_path = os.path.join(output_dir, 'count_vectorizer_vocab.json')
    with open(output_path, 'w') as file:
        json.dump(vocabulary, file, indent=4)
    print(f"CountVectorizer vocabulary saved at: {output_path}")

    return


def load_count_vectorizer(cv_params_path: str,
                          cv_vocab_path: str) -> CountVectorizer:
    """
    Load CountVectorizer object from saved parameters and vocabulary.

    Args:
        cv_params_path (str): Path to the JSON file containing CountVectorizer parameters.
        cv_vocab_path (str): Path to the JSON file containing CountVectorizer vocabulary.

    Returns:
        sklearn.feature_extraction.text.CountVectorizer: Loaded CountVectorizer object.
    """

    # Load CountVectorizer parameters from JSON file
    with open(cv_params_path, 'r') as f:
        cv_params = json.load(f)

    # Load CountVectorizer vocabulary from JSON file
    with open(cv_vocab_path, 'r') as file:
        cv_vocab = json.load(file)

    # Instantiate CountVectorizer object with loaded parameters and vocabulary
    loaded_cv = CountVectorizer(lowercase=cv_params['lowercase'],
                                analyzer=cv_params['analyzer'],
                                ngram_range=cv_params['ngram_range'],
                                max_features=cv_params['max_features'],
                                vocabulary=cv_vocab)

    assert len(loaded_cv.get_feature_names_out()) == cv_params['max_features']
    assert len(loaded_cv.vocabulary_) == cv_params['max_features']

    return loaded_cv

