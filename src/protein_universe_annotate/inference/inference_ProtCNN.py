import json
import numpy as np
import tensorflow.compat.v1 as tf
import tqdm
import argparse

# Suppress noisy log messages
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from protein_universe_annotate.utils import get_top_k_values_indices
from protein_universe_annotate.config import AMINO_ACID_VOCABULARY, _PFAM_GAP_CHARACTER
from protein_universe_annotate.data_processing import read_pfam_dataset
from protein_universe_annotate.inference.inference_misc import infer_predictions


def residues_to_one_hot(amino_acid_residues):
    """
    Converts an amino acid sequence to a one-hot encoded numpy array.

    Args:
        amino_acid_residues (str): A string consisting of characters from the AMINO_ACID_VOCABULARY.

    Returns:
        numpy.ndarray: A numpy array of shape (len(sequence), len(AMINO_ACID_VOCABULARY)).

    Raises:
        ValueError: If the input sequence contains a character not in the vocabulary + X.

    :param amino_acid_residues:
    :return:
    """
    # Initialize an empty list to store one-hot encoded vectors
    one_hot_encoded = []

    normalized_residues = amino_acid_residues.replace('U', 'C').replace('O', 'X')

    # Convert each character in the sequence to a one-hot encoded vector
    for char in normalized_residues:
        if char in AMINO_ACID_VOCABULARY:
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index(char)] = 1.
            one_hot_encoded.append(to_append)
        elif char == 'B':  # Asparagine or aspartic acid
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index('D')] = .5
            to_append[AMINO_ACID_VOCABULARY.index('N')] = .5
            one_hot_encoded.append(to_append)
        elif char == 'Z':  # Glutamine or glutamic acid
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index('E')] = .5
            to_append[AMINO_ACID_VOCABULARY.index('Q')] = .5
            one_hot_encoded.append(to_append)
        elif char == 'X':
            one_hot_encoded.append(
                np.full(len(AMINO_ACID_VOCABULARY), 1. / len(AMINO_ACID_VOCABULARY)))
        elif char == _PFAM_GAP_CHARACTER:
            one_hot_encoded.append(np.zeros(len(AMINO_ACID_VOCABULARY)))
        else:
            raise ValueError(f'Could not one-hot code character {char}, Character {char} not in the vocabulary + X.')

    return np.array(one_hot_encoded)


def pad_one_hot_sequence(sequence: np.ndarray,
                         target_length: int) -> np.ndarray:
    """
    Pads a one hot encoded amino acid sequence (aas) in the seq_len dimension with zeros.

    Args:
        sequence: numpy.ndarray of shape (seq_len, num_aas). The one hot encoded amino acid sequence to be padded.
        target_length: int. The desired length of the padded sequence.

    Returns:
        numpy.ndarray of shape (target_length, num_aas). The padded sequence.

    Raises:
        ValueError: if target_length is less than the length of the input sequence.
    """

    # Get the length of the input sequence
    sequence_length = sequence.shape[0]

    # Calculate the amount of padding required
    pad_length = target_length - sequence_length

    # Check if padding is required
    if pad_length < 0:
        raise ValueError(
            'Cannot set a negative amount of padding. Sequence length was {}, target_length was {}.'
                .format(sequence_length, target_length))

    # Define the padding values
    pad_values = [[0, pad_length], [0, 0]]

    # Pad the sequence with zeros
    return np.pad(sequence, pad_values, mode='constant')


def batch_iterable(iterable, batch_size):
    """
    Yields batches of a specified size from an iterable.

    If the number of elements in the iterable is not a multiple of the batch size,
    the last batch will have fewer elements.

    Args:
        iterable: An iterable containing the data to be batched.
        batch_size: An integer specifying the size of the batches.

    Yields:
        array of length batch_size, containing elements, in order, from iterable.

    Raises:
        ValueError: If batch_size < 1.

    """
    if batch_size < 1:
        raise ValueError(f'batch_size must be >= 1. Received: {batch_size}')

    batch = []
    for item in iterable:
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)

    # Yield the last batch if it has any remaining items
    if batch:
        yield batch


def batch_inference_ProtCNN(model_path='../../../models/trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760',
                            vocab_path='../../../models/trained_model_pfam_32.0_vocab.json',
                            data_path='../../../data/random_split/',
                            pred_confidence=1,
                            save_path='../../../output/test_dataset_predictions.csv',
                            batch_size=8):
    """
    Performs batch inference using the trained ProtCNN model on Pfam dataset.

    Batch prediction is preferred when dealing with large datasets,
    as it is more efficient to make predictions for many samples at once rather than individually.
    (because the computation can be parallelized and optimized for larger batch sizes, leading to faster predictions.)

    Args:
        model_path (str): Path to the ProtCNN model
        vocab_path (str): Path to the dict with target classes being encoded
        data_path (str): Path to the data directory containing test dataset
        pred_confidence (int): Confidence level for predictions, (1 equals to argmax())
        save_path (str): Path to save the predictions
        batch_size (int): Size of the batches to use for prediction

    Returns:
        pd.DataFrame: predictions added as columns to the test dataset (pd.DataFrame)
    """

    # Load the model into TensorFlow
    sess = tf.Session()
    graph = tf.Graph()

    with graph.as_default():
        backup_model = tf.saved_model.load(sess, ['serve'], model_path)

    # Load tensors for class confidence prediction
    class_confidence_signature = backup_model.signature_def['confidences']
    class_confidence_signature_tensor_name = class_confidence_signature.outputs['output'].name

    sequence_input_tensor_name = backup_model.signature_def['confidences'].inputs['sequence'].name
    sequence_lengths_input_tensor_name = backup_model.signature_def['confidences'].inputs['sequence_length'].name

    # Read the test dataset
    test_df = read_pfam_dataset(partition='test', data_dir=data_path)

    # Sort test_df by sequence length so that batches have as little padding as possible -> faster inference.
    test_df = test_df.sort_values('sequence', key=lambda col: [len(c) for c in col])

    # Keep the predictions
    inference_results_topKclasses = []

    batches = list(batch_iterable(test_df.sequence, batch_size))
    for seq_batch in tqdm.tqdm(batches, position=0):

        # Infers class confidence for a batch of amino acid sequences
        # batch: List of strings containing amino acid sequences
        seq_lens = [len(seq) for seq in seq_batch]
        one_hots = [residues_to_one_hot(seq) for seq in seq_batch]
        max_sequence_length = max(seq_lens)
        padded_sequence_inputs = [pad_one_hot_sequence(seq, max_sequence_length) for seq in one_hots]

        with graph.as_default():
            preds = sess.run(
                class_confidence_signature_tensor_name,
                {
                    sequence_input_tensor_name: padded_sequence_inputs,
                    sequence_lengths_input_tensor_name: seq_lens,
                })

        # Instead of returning the most probable class as pred, get TOP5 most probable classes
        # inference_results_class.extend([np.argmax(pred) for pred in preds])

        batch_topK_classes = get_top_k_values_indices(preds, num_top_values=pred_confidence)
        inference_results_topKclasses.extend(batch_topK_classes)

    # Convert true labels from PF00001.21 to PF00001
    test_df['true_label'] = test_df.family_accession.apply(lambda s: s.split('.')[0])

    # Load vocab
    with open(vocab_path) as f:
        vocab = json.loads(f.read())

    # Keep predictions from (pred_confidence) most probable classes (not only np.argmax())
    for conf in range(1, pred_confidence+1):
        conf_preds = infer_predictions(predictions=inference_results_topKclasses, confidence_level=conf)
        test_df[f'predicted_label_{conf}'] = [vocab[pred] for pred in conf_preds]

    # Save predictions
    test_df.to_csv(save_path, index=True)

    return test_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perform batch inference using ProtCNN')
    parser.add_argument('--model_path', type=str,
                        default='../../../models/trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760',
                        help='Path to the trained ProtCNN model')
    parser.add_argument('--vocab_path', type=str,
                        default='../../../models/trained_model_pfam_32.0_vocab.json',
                        help='Path to the ProtCNN vocabulary')
    parser.add_argument('--data_path', type=str,
                        default='../../data/random_split/',
                        help='Path to the data directory')
    parser.add_argument('--pred_confidence', type=int,
                        default=1,
                        help='Confidence level for predictions')
    parser.add_argument('--save_path', type=str,
                        default='../../../output/test_dataset_ProtCNN_predictions.csv',
                        help='Path to save the predictions')
    parser.add_argument('--batch_size', type=int,
                        default=8,
                        help='Number of samples to include in each batch')

    args = parser.parse_args()

    test_df = batch_inference_ProtCNN(model_path=args.model_path,
                                      vocab_path=args.vocab_path,
                                      data_path=args.data_path,
                                      pred_confidence=args.pred_confidence,
                                      save_path=args.save_path,
                                      batch_size=args.batch_size)

