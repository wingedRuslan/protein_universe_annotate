import os
import pandas as pd


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

