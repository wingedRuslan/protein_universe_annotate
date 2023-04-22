import numpy as np
from protein_universe_annotate.inference.inference_ProtCNN import residues_to_one_hot, \
    pad_one_hot_sequence, batch_iterable


def _test_residues_to_one_hot():
    expected = np.zeros((3, 20))
    expected[0, 0] = 1.  # Amino acid A
    expected[1, 1] = 1.  # Amino acid C
    expected[2, :] = .05  # Amino acid X

    actual = residues_to_one_hot('ACX')
    np.testing.assert_allclose(actual, expected)


def _test_pad_one_hot():
    input_one_hot = residues_to_one_hot('ACX')
    expected = np.array(input_one_hot.tolist() + np.zeros((4, 20)).tolist())
    actual = pad_one_hot_sequence(input_one_hot, 7)

    np.testing.assert_allclose(expected, actual)


def _test_batch_iterable():
    itr = [1, 2, 3]
    batched_itr = list(batch_iterable(itr, 2))
    assert batched_itr == [[1, 2], [3]]
