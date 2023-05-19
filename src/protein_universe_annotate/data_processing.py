import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import logging


def read_pfam_dataset(partition='dev', data_dir='../../data/random_split'):
    """
    Reads all CSV files in a specified partition directory and returns them as a single pandas DataFrame.

    Args:
        partition (str): The name of the partition directory to read CSV files from.
        data_dir (str): The directory containing all partition directories.
    
    Returns:
        pandas.DataFrame: A single DataFrame containing all CSV data from the specified partition directory.
    """

    # Initialize an empty list to store each data file as DataFrame
    data_blocks = []

    # Iterate over each CSV file in the specified partition directory
    for filename in os.listdir(os.path.join(data_dir, partition)):
        with open(os.path.join(data_dir, partition, filename)) as file:
            data_blocks.append(pd.read_csv(file, index_col=None))

    # Concatenate all DataFrame blocks into a single DataFrame and return it
    return pd.concat(data_blocks)


def select_long_tail(data_df, col_name):
    """
    Selects a subset of the input data with a focus on rare values in col_name.
    If the number of label occurrences is greater than or equal to 50, sample random 50 entries,
    otherwise include all samples.

    Args:
        data_df: pandas DataFrame containing the data
        col_name: name of the column that contains the target values

    Returns:
        A pandas DataFrame containing the selected subset of the input data
    """

    def select_rows(x):
        """
        A rule to select samples for a given label:
            if number of label occurrences > 50, sample random 50 entries, else include all samples
        """
        return x.sample(n=50) if len(x) >= 50 else x

    # Group the input data by col_name and apply the select_rows rule to each group
    data_selected = data_df.groupby(col_name).apply(select_rows).reset_index(drop=True)

    # Shuffle
    data_selected = data_selected.sample(frac=1).reset_index(drop=True)

    return data_selected


def fit_count_vectorizer(ngram_range, max_features, input_sequences):
    """
    Fit a CountVectorizer for character n-grams.

    Tokenize the protein sequences into individual amino acids (analyzer='char').

    Args:
        ngram_range (tuple): The range of n-grams to include, e.g., (1, 3) for 1-3 grams.
        max_features (int): The maximum number of features (vocabulary size) to limit the feature representation.
        input_sequences (list): List of input sequences to fit the CountVectorizer on.

    Returns:
        sklearn.feature_extraction.text.CountVectorizer: Fitted CountVectorizer object.
    """
    # Log the message about fitting the CountVectorizer
    logging.info(f"Fitting a CountVectorizer with ngram_range={ngram_range}, max_features={max_features} "
                 f"on {len(input_sequences)} input sequences.")

    # Initialize and configure the CountVectorizer
    vectorizer = CountVectorizer(lowercase=False, analyzer='char', ngram_range=ngram_range, max_features=max_features)

    # Fit the CountVectorizer on the input sequences
    vectorizer.fit(input_sequences)

    # assert len(vectorizer.vocabulary_) == max_features
    # print(vectorizer.vocabulary_)

    return vectorizer
