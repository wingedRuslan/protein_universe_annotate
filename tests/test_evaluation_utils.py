import pandas as pd
import numpy as np
import collections
from protein_universe_annotate.evaluation.evaluation_utils import calc_mean_accuracy_per_class


def test_calc_mean_accuracy_per_class():

    # Alternative implementation of the function to test
    def legacy_calc_mean_accuracy_per_class(predictions_df):
        """Compute accuracy
        Args:
            predictions_df: pandas DataFrame with 2 columns, true_label and predicted_label
        Returns:
            dict. The average accuracy per each class.
        """
        grouped_predictions = collections.defaultdict(list)
        for row in predictions_df.itertuples():
            grouped_predictions[row.true_label].append(row.predicted_label)

        accuracy_per_class = {
            true_label: np.mean(predicted_label == np.array(true_label))
            for true_label, predicted_label in grouped_predictions.items()
        }
        return accuracy_per_class

    # Create a sample predictions DataFrame
    predictions = pd.DataFrame({
        'true_label': ['A', 'A', 'B', 'B', 'C'],
        'predicted_label': ['A', 'B', 'B', 'C', 'C']
    })

    legacy_output = legacy_calc_mean_accuracy_per_class(predictions)
    output = calc_mean_accuracy_per_class(predictions)

    # Test new implementation vs previous implementation
    assert len(output) == len(legacy_output)

    for index, row in output.iterrows():
        true_label = row['true_label']
        accuracy_score = row['avg_accuracy']
        assert legacy_output[true_label] == accuracy_score, ValueError('Failed the verification accuracy_by_true_label')

