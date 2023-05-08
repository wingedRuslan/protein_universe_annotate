import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from protein_universe_annotate.data_processing import read_pfam_dataset


def get_partitions_info(data_frames, split_names):
    """
    Create a summary DataFrame containing basic statistics about each partition.

    Args:
        data_frames (list of pd.DataFrames): List of DataFrames containing data for each partition.
        split_names (list of str): List of partition names.

    Returns:
        A pandas DataFrame containing information about each partition.
    """
    return pd.DataFrame([
        {
            'partition': name,
            'num_of_samples': frame.shape[0],
            'num_of_true_class': frame['true_label'].unique().size,
            'max_samples_per_class': frame.groupby('true_label').size().max(),
            'min_samples_per_class': frame.groupby('true_label').size().min(),
            'min_seq_length': frame['sequence_len'].min(),
            'max_seq_length': frame['sequence_len'].max(),
            'avg_seq_length': frame['sequence_len'].mean()
        } for name, frame in zip(split_names, data_frames)]
    )


def is_bijection_mapping(data_df: pd.DataFrame,
                         col1: str,
                         col2: str):
    """
    Check if there is a 1:1 mapping between 'col1' and 'col2' in the given dataset.

    "bijection" describes a 1:1 mapping between two sets.
    A bijection is a function between two sets
    where each element of one set is paired with exactly one element of the other set.

    Args:
        dataset (pandas.DataFrame): The dataset containing col1 and col2
        col1 (str): The column name of feature
        col2 (str): The column name of feature

    Returns:
        bool: True if there is a 1:1 mapping, False otherwise.
    """
    # Group the rows by 'col1' and count the number of unique 'col2' values in each group
    mapping_counts = data_df.groupby(col1)[col2].nunique()

    # Get the list of base values that occur more than once
    not_bijection_mapping = mapping_counts[mapping_counts > 1].index.tolist()

    return False if not_bijection_mapping else True


def compute_overlap(df1: pd.DataFrame,
                    df2: pd.DataFrame,
                    col_name: str):
    """
    Computes the overlap between unique values in a given column of two DataFrames.

    Args:
        df1 (pandas DataFrame): The first DataFrame.
        df2 (pandas DataFrame): The second DataFrame.
        col_name (st): Name of the column to compute the overlap for.

    Returns:
        set. A set containing the overlapping values.
    """
    # Find unique values in the specified column of both DataFrames
    df1_values = set(df1[col_name])
    df2_values = set(df2[col_name])

    # compute the overlap
    overlap = df1_values & df2_values

    return overlap


def detect_outliers_zscore(data_df: pd.DataFrame,
                           column: str,
                           threshold: int = 3):
    """
    Detects outliers in a pandas DataFrame column using the Z-score method.

    Args:
        data_df (pd.DataFrame): The DataFrame to detect outliers in.
        column (str): The name of the column to detect outliers in.
        threshold (int): The number of standard deviations from the mean at which to consider a value an outlier.
            Default value is 3.

    Returns:
        A DataFrame containing the outliers.
    """

    # Calculate the z-scores for each value in the column
    z_scores = (data_df[column] - data_df[column].mean()) / data_df[column].std()

    # Identify values with a z-score greater than the specified threshold
    outliers = data_df[abs(z_scores) > threshold]

    return outliers


def plot_sequence_len_dist(data_df: pd.DataFrame,
                           split_name: str = 'train',
                           bins: int = 50):
    """
    Plot the distribution of sequence lengths for a given dataset.

    Args:
        data_df (pd.DataFrame): DataFrame containing the dataset.
        split_name (str): Name of the dataset split (e.g. 'train', 'dev', 'test').
        bins (int): Number of bins for the histogram. Default is 50.

    Returns:
        None.
    """
    sns.histplot(data_df['sequence_len'].values, stat='frequency', bins=bins)
    plt.title(f'Sequence Length Distribution: {split_name}')
    plt.xlabel(f'Sequence Length')
    plt.grid(True)


def get_amino_acid_freq(data_df: pd.DataFrame,
                        split_name: str = 'train'):
    """
    Get the frequency of each amino acid in the sequences of a given DataFrame.

    Args:
        data_df (pd.DataFrame): DataFrame containing the protein sequence data
        split_name (str): Name of the dataset split (e.g. 'Train', 'Dev', 'Test')

    Returns:
        pandas DataFrame containing the amino acid frequency data.
    """

    # Extract the sequences from the DataFrame
    sequences = data_df['sequence']

    # Count the occurrence of each amino acid in the sequences
    amino_acids_counter = Counter()
    for seq in sequences:
        amino_acids_counter.update(seq)

    # Print the dataset split name and the number of unique amino acids
    print(f'\t##### {split_name}')
    print(f'Total unique Amino Acids: {len(amino_acids_counter.keys())}')

    # Create a DataFrame to store the amino acid frequency data
    df = pd.DataFrame({'amino_acid': list(amino_acids_counter.keys()),
                       'frequency': list(amino_acids_counter.values())
                       })

    # Calculate the frequency of each amino acid as a percentage of the total
    df['frequency'] = (df['frequency'] / sum(amino_acids_counter.values())) * 100

    # Sort the DataFrame by frequency in descending order
    df = df.sort_values('frequency', ascending=False).reset_index()[['amino_acid', 'frequency']]

    return df


def plot_code_freq(data_df: pd.DataFrame,
                   split_name: str = 'train'):
    """
    Plot the frequency of amino acid codes in a given dataset.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data
        split_name (str): Name of the dataset (e.g. train, dev, test)

    Returns:
        None
    """
    # Set the color palette
    palette_colors = sns.color_palette("viridis", len(data_df))

    # Set the plot title
    plt.title(f'Amino Acid Code Frequency: {split_name}')

    # Plot the bar chart
    sns.barplot(x='amino_acid', y='frequency', data=data_df, palette=np.array(palette_colors))


def plot_class_distribution_hist(data_df: pd.DataFrame,
                                 split_name: str,
                                 column_name: str,
                                 num_bins: int = 50):
    """
    Plot the distribution of class label occurrences for a given split of the dataset as a histogram.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        split_name (str): Name of the dataset split (e.g. 'Train', 'Test', etc.)
        col_name (str): The name of the column containing the class values
        num_bins (int): Number of bins to use for the histogram

    Returns:
        None.
    """
    # Group the data by the true_label column and count the occurrences
    class_label_counts = data_df.groupby(column_name).size()

    # Set the y-scale to log for a better visualization of the distribution
    plt.yscale('log')

    # Plot the histogram of class label counts
    sns.histplot(class_label_counts, stat='frequency', bins=num_bins)

    # Set the title and labels for the plot
    plt.title(f'Distribution of class label occurrences for {split_name}')
    plt.xlabel('Class label occurrence')
    plt.ylabel('Frequency (log scale)')


def plot_class_distribution_line(data_df: pd.DataFrame,
                                 column_name: str = 'true_label'):
    """Plot the distribution of classes in the given DataFrame as a line on the logarithmic scale

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        column_name (str): str representing the name of the column containing the class labels.

    Returns:
        None.
    """
    # Group the data by the class column and count the occurrences
    class_counts = data_df.groupby(column_name).size().sort_values(ascending=False)

    # Extract the values of the class counts
    class_counts_values = class_counts.values

    # Set the y-scale to logarithmic for a better visualization of the distribution
    plt.yscale('log')

    # Set the y-tick labels to use a more human-readable format
    plt.yticks(
        ticks=[1, 10, 100, 1000, 10000],
        labels=[1, '10', '100', '1K', '10K']
    )

    # Plot the class counts as a line plot
    plt.plot(class_counts_values)

    # Set the x- and y-axis labels
    plt.xlabel('Target Class')
    plt.ylabel('Samples per class')

    # Set the x-tick labels to the first and last class labels in the DataFrame
    plt.xticks(
        ticks=[0, class_counts_values.size-1],
        labels=[class_counts.index[0], class_counts.index[-1]]
    )


def plot_class_percentage_distribution_line(data_df: pd.DataFrame,
                                            column_name: str = 'true_label'):
    """Plot the percentage of samples in each class relative to the full size of the dataset.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        column_name (str): str representing the name of the column containing the class labels.

    Returns:
        None.
    """
    # Group the data by the class column and count the occurrences
    class_counts = data_df.groupby(column_name).size().sort_values(ascending=False)

    # Calculate the percentage of samples in each class relative to the full size of the dataset
    class_percentages = 100 * class_counts.values / class_counts.values.sum()

    # Set the y-scale to logarithmic for a better visualization of the distribution
    plt.yscale('log')

    # Set the y-tick labels to use a more human-readable format
    ticks = list(10.**(-np.arange(1,5))) + [class_percentages[0]]
    labels = [f'{p:.5f}%' for p in ticks]
    plt.yticks(
        ticks=ticks,
        labels=labels
    )

    # Plot the class percentages as a line plot
    plt.plot(class_percentages)

    # Set the x- and y-axis labels
    plt.xlabel('Class')
    plt.ylabel('Dataset %')

    # Set the x-tick labels to the first and last class labels in the DataFrame
    plt.xticks(
        ticks=[0, class_percentages.size-1],
        labels=[class_counts.index[0], class_counts.index[-1]]
    )


def explore_pfam_dataset(data_partitions_dirpath):
    """
    Perform data analysis to explore statistics and potential issues of the dataset.

    Args:
    data_partitions_dirpath (str): The directory path where the dataset partitions are stored.
    """
    # Print the available dataset partitions
    print('Available dataset partitions: ', os.listdir(data_partitions_dirpath))

    # Read each dataset partition into a pandas DataFrame
    dev_df = read_pfam_dataset('dev', data_partitions_dirpath)
    test_df = read_pfam_dataset('test', data_partitions_dirpath)
    train_df = read_pfam_dataset('train', data_partitions_dirpath)

    # Assign the dataset type to each sample by adding a new 'split' column
    dev_df['split'] = 'dev'
    test_df['split'] = 'test'
    train_df['split'] = 'train'

    # Convert true labels from PF00001.21 to PF00001
    dev_df['true_label'] = dev_df.family_accession.apply(lambda s: s.split('.')[0])
    test_df['true_label'] = test_df.family_accession.apply(lambda s: s.split('.')[0])
    train_df['true_label'] = train_df.family_accession.apply(lambda s: s.split('.')[0])

    # Extract length of sequence
    dev_df['sequence_len'] = dev_df['sequence'].apply(lambda x: len(x))
    test_df['sequence_len'] = test_df['sequence'].apply(lambda x: len(x))
    train_df['sequence_len'] = train_df['sequence'].apply(lambda x: len(x))

    # Concatenate the datasets in one DataFrame
    total_data_df = pd.concat([test_df, dev_df, train_df], ignore_index=True)

    # Basic info
    print(f"Number of samples from dev-test-train: {total_data_df.shape[0]}")
    print(f"Number of features: {total_data_df.shape[1]}")
    print(f"Features: {total_data_df.columns.tolist()}")

    print(f"Dev size: {dev_df.shape[0]}")
    print(f"Test size: {test_df.shape[0]}")
    print(f"Train size: {train_df.shape[0]}")

    # Check for missing values in each column
    print('Missing values in dataset: \n', total_data_df.isnull().sum())

    # checking for duplicates on the sample level
    print(f'Duplicated entries in dataset: {total_data_df.duplicated().any()}')

    # Check for duplicates on 'sequence' level in each dataset split
    train_duplicates = train_df[train_df.duplicated(subset=['sequence'], keep='first')]
    print(f'Num. duplicates in Train: {train_duplicates.shape[0]}')

    test_duplicates = test_df[test_df.duplicated(subset=['sequence'], keep='first')]
    print(f'Num. duplicates in Test: {test_duplicates.shape[0]}')

    dev_duplicates = dev_df[dev_df.duplicated(subset=['sequence'], keep='first')]
    print(f'Num. duplicates in Dev: {dev_duplicates.shape[0]}')

    # Does the target label like PF00001 have multiple versions?
    print('The target variable \'family_accession\' contains only unique mapping to its version: {}'.
          format(is_bijection_mapping(total_data_df, col1='true_label', col2='family_accession')))

    # Does the 'family_id' and 'true_label' have 1:1 mapping?
    print('The \'family_id\' feature contains only unique mapping to true_label feature: {}'.
          format(is_bijection_mapping(total_data_df, col1='family_id', col2='true_label')))

    # Distinct target label
    num_distinct_labels = total_data_df['true_label'].nunique()
    print(f"There are {num_distinct_labels} distinct values in the 'true_label' column.")
    print('Number of unique classes in Train: ', train_df['true_label'].nunique())
    print('Number of unique classes in Test: ', test_df['true_label'].nunique())
    print('Number of unique classes in Dev: ', dev_df['true_label'].nunique())

    # Are there overlaps between sets in 'true_label'?
    overlapping_vals = compute_overlap(train_df, test_df, 'true_label')
    print(f'Number of overlapping labels Train -- Test: {len(overlapping_vals)}, '
          f'ratio: {len(overlapping_vals) / test_df["true_label"].nunique()}')

    overlapping_vals = compute_overlap(train_df, dev_df, 'true_label')
    print(f'Number of overlapping labels Train -- Dev: {len(overlapping_vals)}, '
          f'ratio: {len(overlapping_vals) / dev_df["true_label"].nunique()}')

    overlapping_vals = compute_overlap(dev_df, test_df, 'true_label')
    print(f'Number of overlapping labels Dev -- Test: {len(overlapping_vals)}, '
          f'ratio: {len(overlapping_vals) / test_df["true_label"].nunique()}')

    # Are there overlaps between sets in 'sequence'?
    overlapping_vals = compute_overlap(train_df, test_df, 'sequence')
    print(f'Number of overlapping sequences Train -- Test: {len(overlapping_vals)}')

    overlapping_vals = compute_overlap(train_df, dev_df, 'sequence')
    print(f'Number of overlapping sequences Train -- Dev: {len(overlapping_vals)}')

    overlapping_vals = compute_overlap(test_df, dev_df, 'sequence')
    print(f'Number of overlapping sequences Test -- Dev: {len(overlapping_vals)}')

    # Plot the distribution of sequence lengths
    plt.subplot(1, 3, 1)
    plot_sequence_len_dist(train_df, 'Train')

    plt.subplot(1, 3, 2)
    plot_sequence_len_dist(dev_df, 'Dev')

    plt.subplot(1, 3, 3)
    plot_sequence_len_dist(test_df, 'Test')

    plt.subplots_adjust(right=3.0)
    plt.savefig('../../output/sequence_len_dist.png', dpi=300, bbox_inches='tight')

    # Get the frequency of each amino acid in the sequences of a given DataFrame
    dev_amino_acid_freq = get_amino_acid_freq(dev_df, 'Dev')
    test_amino_acid_freq = get_amino_acid_freq(test_df, 'Test')
    train_amino_acid_freq = get_amino_acid_freq(train_df, 'Train')

    # Are amino acids same across sets?
    print('Are all test amino acids covered in train: ',
          set(train_amino_acid_freq['amino_acid'].values).issuperset(set(test_amino_acid_freq['amino_acid'].values)))
    print('Are all dev amino acids covered in train: ',
          set(train_amino_acid_freq['amino_acid'].values).issuperset(set(dev_amino_acid_freq['amino_acid'].values)))
    print('Are all dev amino acids covered in test: ',
          set(test_amino_acid_freq['amino_acid'].values).issuperset(set(dev_amino_acid_freq['amino_acid'].values)))

    # Plot the frequency of amino acid codes in a given dataset.
    plt.subplot(1, 3, 1)
    plot_code_freq(train_amino_acid_freq, 'Train')

    plt.subplot(1, 3, 2)
    plot_code_freq(dev_amino_acid_freq, 'Dev')

    plt.subplot(1, 3, 3)
    plot_code_freq(test_amino_acid_freq, 'Test')

    plt.subplots_adjust(right=3.0)
    plt.savefig('../../output/amino_acid_codes_dist.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    """
    Interactive Data Exploration is performed in the jupyter notebook in ../notebooks.
    """

    parser = argparse.ArgumentParser(description='Perform Data Exploration on the Pfam random dataset')
    parser.add_argument('--data_path', type=str,
                        default='../../data/random_split/',
                        help='Path to the data directory')
    args = parser.parse_args()

    explore_pfam_dataset(data_partitions_dirpath=args.data_path)

