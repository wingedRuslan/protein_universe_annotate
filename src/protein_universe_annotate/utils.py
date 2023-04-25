import numpy as np
import json


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


def save_label_encoder(label_encoder, save_path):
    """
    Saves a scikit-learn LabelEncoder mapping to a JSON file.

    Args:
        label_encoder_mapping (dict): a dictionary with label to index mapping
        save_path (str): path to the file to save the JSON
    """

    label_encoder_mapping = dict(zip(label_encoder.classes_,
                                     label_encoder.transform(label_encoder.classes_)
                                     )
                                 )

    # Convert NumPy data types to Python data types for JSON serialization (json does not recognize NumPy data types)
    le_name_mapping_save = {}
    for label, index in label_encoder_mapping.items():
        le_name_mapping_save[str(label)] = int(index)

    # Make sure that nothing is missing
    assert len(label_encoder_mapping) == len(le_name_mapping_save)

    # Save to a file
    with open(save_path + '/label_encoder_mapping.json', 'w') as f:
        # Write the dictionary to the file in JSON format
        json.dump(le_name_mapping_save, f, indent=4)

    return 0

