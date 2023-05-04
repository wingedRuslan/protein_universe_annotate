import pandas as pd
import numpy as np


TRUE_LABEL_KEY = 'true_label'
PREDICTED_LABEL_KEY = 'predicted_label'
ACCURACY_KEY = 'accuracy'
NUM_EXAMPLES_KEY = 'num_samples'


def number_correct(predictions_df: pd.DataFrame,
                   true_label: str = TRUE_LABEL_KEY,
                   predicted_label: str = PREDICTED_LABEL_KEY):
    """Computes the number of correct predictions.

    Args:
        predictions_df: pandas DataFrame with at least 2 columns, true_label and predicted_label.
        true_label: str. Column name of true labels.
        predicted_label: str. Column name of predicted labels.

    Returns:
        int.
    """
    return (predictions_df[true_label] ==
            predictions_df[predicted_label]).sum()


def raw_unweighted_accuracy(predictions_df: pd.DataFrame,
                            true_label: str = TRUE_LABEL_KEY,
                            predicted_label: str = PREDICTED_LABEL_KEY):
    """Compute accuracy

    Args:
        predictions_df: pandas DataFrame with at least 2 columns, true_label and predicted_label.
        true_label: str. Column name of true labels.
        predicted_label: str. Column name of predicted labels.

    Returns:
        float. Accuracy.
    """
    num_correct = number_correct(predictions_df, true_label, predicted_label)
    total = len(predictions_df)
    return num_correct / total


def calc_mean_accuracy_per_class(predictions_df: pd.DataFrame,
                                 true_label: str = TRUE_LABEL_KEY,
                                 predicted_label: str = PREDICTED_LABEL_KEY):
    """Calculate the mean accuracy per each true label class.

    Args:
        predictions_df (pandas.DataFrame): DataFrame with 2 columns, true_label and predicted_label
        true_label (str): The name of the column containing the true labels.
        predicted_label (str): The name of the column containing the predicted labels.

    Returns:
        pandas.DataFrame: The average accuracy per each class.
    """
    # Group the DataFrame by the true_label column and apply the raw_unweighted_accuracy function to each group
    accuracy_by_true_label = predictions_df.groupby(true_label).apply(
        lambda x: raw_unweighted_accuracy(x, true_label=true_label, predicted_label=predicted_label)
    )

    # Convert the resulting Series to a DataFrame
    avg_acc_per_true_label_df = accuracy_by_true_label.reset_index()
    avg_acc_per_true_label_df.columns = [true_label, 'avg_accuracy']

    return avg_acc_per_true_label_df


def get_misclassified_samples(predictions_df: pd.DataFrame,
                              true_label: str = TRUE_LABEL_KEY,
                              predicted_label: str = PREDICTED_LABEL_KEY):
    """
    Returns a DataFrame of misclassified samples.

    Args:
        predictions_df: The DataFrame of predictions (with at least 2 columns, true_label and predicted_label) to be analyzed.
        true_label: Column name of true labels.
        predicted_label: Column name of predicted labels.

    Returns:
        A DataFrame containing the misclassified samples.
    """
    # Select rows where the true label and predicted label are not equal
    misclassified_samples_df = predictions_df[predictions_df[true_label] != predictions_df[predicted_label]]
    return misclassified_samples_df

