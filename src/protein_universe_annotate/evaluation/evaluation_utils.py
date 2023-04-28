import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections


TRUE_LABEL_KEY = 'true_label'
PREDICTED_LABEL_KEY = 'predicted_label'
ACCURACY_KEY = 'accuracy'
NUM_EXAMPLES_KEY = 'num_samples'


def calc_mean_accuracy_per_class(predictions_dataframe):
    """Compute accuracy of predictions per each target label, 
            giving equal weight to all classes.
    
    Args:
    predictions_dataframe: pandas DataFrame with 2 columns,
        true_label and predicted_label
    
    Returns:
    dict. The average accuracy per each class.
    """
    grouped_predictions = collections.defaultdict(list)
    for row in predictions_dataframe.itertuples():
        grouped_predictions[row.true_label].append(row.predicted_label)

    accuracy_per_class = {
        true_label: np.mean(predicted_label == np.array(true_label))
        for true_label, predicted_label in grouped_predictions.items()
    }

    return accuracy_per_class


def number_correct(predictions_dataframe,
                   true_label=TRUE_LABEL_KEY,
                   predicted_label=PREDICTED_LABEL_KEY):
    """Computes the number of correct predictions.

    Args:
        predictions_dataframe: pandas DataFrame with at least 2 columns, true_label
            and predicted_label.
        true_label: str. Column name of true labels.
        predicted_label: str. Column name of predicted labels.

    Returns:
        int.
    """
    return (predictions_dataframe[true_label] ==
            predictions_dataframe[predicted_label]).sum()


def raw_unweighted_accuracy(
                predictions_dataframe,
                true_label=TRUE_LABEL_KEY,
                predicted_label=PREDICTED_LABEL_KEY):
    """Compute accuracy, regardless of which class each prediction corresponds to.

    Args:
        predictions_dataframe: pandas DataFrame with at least 2 columns, true_label
        and predicted_label.
        true_label: str. Column name of true labels.
        predicted_label: str. Column name of predicted labels.

    Returns:
        float. Accuracy.
    """
    num_correct = number_correct(predictions_dataframe, true_label, predicted_label)

    total = len(predictions_dataframe)
    return num_correct / total


def mean_acc_per_class_for_only_large_classes(predictions_dataframe, n):
    """Compute mean per class accuracy on classes with lots of training data.

    Args:
    all_predictions_dataframe: pandas DataFrame with 2 columns, true label and predicted label
    n: int : class_minimum_size
    size_of_training_set_by_family: pandas DataFrame with two columns,
        NUM_EXAMPLES_KEY and TRUE_LABEL_KEY

    Returns:
    float.
    """
    # Init the count of each label to be 0 (dummy variable)
    predictions_dataframe[NUM_EXAMPLES_KEY] = 0

    # Count occurrences of each value in TRUE_LABEL_KEY
    value_counts = predictions_dataframe[TRUE_LABEL_KEY].value_counts()

    # Iterate over each row and add 'num_label' column
    for idx, row in predictions_dataframe.iterrows():
        true_label = row[TRUE_LABEL_KEY]
        num_label = value_counts[TRUE_LABEL_KEY]
        predictions_dataframe.at[idx, NUM_EXAMPLES_KEY] = num_label
    
    # Consider true labels with occurrences > n
    qualifying_predictions = predictions_dataframe[predictions_dataframe[NUM_EXAMPLES_KEY] > n]

    accuracy_per_class = calc_mean_accuracy_per_class(qualifying_predictions)

    return np.mean(list(accuracy_per_class.values()))
    

def accuracy_by_family(predictions_dataframe):
    """Return DataFrame that has accuracy by TRUE_LABEL_KEY.

    Args:
    predictions_dataframe: pandas DataFrame with 2 columns. The true and predicted labels

    Returns:
    pandas DataFrame with two columns, TRUE_LABEL_KEY and ACCURACY_KEY.
    """
    return predictions_dataframe.groupby([TRUE_LABEL_KEY]).apply(raw_unweighted_accuracy).reset_index(name=ACCURACY_KEY)


def accuracy_by_size_of_family(family_predictions, size_of_family):
    """Return DataFrame with the accuracy computed, segmented by family.

    Args:
    family_predictions: pandas DataFrame with 3 columns,
        PREDICTION_FILE_COLUMN_NAMES. The true and predicted label
    size_of_family: pandas DataFrame with two columns, NUM_EXAMPLES_KEY and TRUE_LABEL_KEY

    Returns:
    pandas DataFrame with two columns, NUM_EXAMPLES_KEY and ACCURACY_KEY.
    """
    return pd.merge(
        accuracy_by_family(family_predictions),
        size_of_family,
        left_on=TRUE_LABEL_KEY,
        right_on=TRUE_LABEL_KEY)[[NUM_EXAMPLES_KEY, ACCURACY_KEY]]


def accuracy_by_sequence_length(family_predictions,
                                length_of_examples_by_family):
    """Return DataFrame with accuracy computed by avg sequence length per family.

    Args:
    family_predictions: pandas DataFrame wtih TRUE_LABEL_KEY and corresponding mean(ACCURACY_KEY) accuracy scores.
    length_of_examples_by_family: pandas DataFrame with two columns,
        AVERAGE_SEQUENCE_LENGTH_KEY and FAMILY_ACCESSION_KEY

    Returns:
    pandas DataFrame with two columns, AVERAGE_SEQUENCE_LENGTH_KEY and
        ACCURACY_KEY.
    """
    return pd.merge(
        accuracy_by_family(family_predictions),
        length_of_examples_by_family,
        left_on=TRUE_LABEL_KEY,
        right_on=TRUE_LABEL_KEY)[[
            'avg_seq_len', ACCURACY_KEY
        ]]

