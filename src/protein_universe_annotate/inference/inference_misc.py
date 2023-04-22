
def infer_predictions(predictions, confidence_level=1):
    """Returns the prediction for each sample based on confidence_level
    confidence_level = 1 - equivalent to np.argmax(preds, axis=1)
    confidence_level = 2 - means the second most probable class will be used as prediction

    predictions contains indices sorted in ascending order based on the corresponding values.
    e.g array([14366,  1184, 15928, 15456, 17576]), class corresponding to index 17576 has the highest probability

    Args:
      predictions: an array of probabilities (output of utils.get_top_k_values_indices applied to softmax())
      confidence_level: int, the order of the prediction level

    Returns:
      an array containing the most confident prediction for each element in the input array.

    Raises:
      ValueError: if confidence_level is less than 1.
    """
    if confidence_level < 1:
        raise ValueError('Confidence level cannot be less than 1. Received: {}'.format(confidence_level))

    # Get the prediction according to the confidence level
    final_predictions = [pred[-confidence_level] for pred in predictions]

    return final_predictions

