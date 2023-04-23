import pandas as pd
import numpy as np
from collections import Counter
from protein_universe_annotate.data_exploration import compute_overlap, get_amino_acid_freq


def test_compute_overlap():
    # Create two DataFrames with overlapping and non-overlapping values
    df1 = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
    df2 = pd.DataFrame({'col1': [4, 5, 6, 7, 8]})

    # Test the function on the overlapping values
    overlap = compute_overlap(df1, df2, 'col1')
    assert overlap == {4, 5}

    # Create two DataFrames with no overlapping values
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})

    # Test the function on the non-overlapping values
    overlap = compute_overlap(df1, df2, 'col1')
    assert overlap == set()

    # Test the function on an empty DataFrame
    df1 = pd.DataFrame({'col1': []})
    df2 = pd.DataFrame({'col1': []})

    # Test the function on an empty DataFrame
    overlap = compute_overlap(df1, df2, 'col1')
    assert overlap == set()


def test_get_amino_acid_freq():
    # Create a test DataFrame
    data = {'sequence': ['AAGGTT', 'CCCCGG', 'AAATTT', 'GGGCCC']}
    test_df = pd.DataFrame(data)

    # Call the function on the test DataFrame
    result_df = get_amino_acid_freq(test_df, 'Test')

    # Check that the output DataFrame has the expected columns
    assert all(x in result_df.columns for x in ['amino_acid', 'frequency'])

    # Check that the output DataFrame has the expected number of rows
    assert len(result_df) == 4

    # Check that the amino acid labels are correct
    assert np.array_equal(result_df['amino_acid'].values, np.array(['G', 'C', 'A', 'T']))

