import numpy as np

def calculate_precision_recall(true_labels, predicted_labels):
    """
    Calculates the precision and recall from true labels and predicted labels of a classification task.

    Args:
    true_labels: A numpy array of actual labels.
    predicted_labels: A numpy array of predicted labels.

    Returns:
    precision: The precision of the predictions.
    recall: The recall of the predictions.
    """

    # True Positives (TP): Correctly predicted positive observations
    true_positives = np.sum((predicted_labels == 1) & (true_labels == 1))

    # False Positives (FP): Incorrectly predicted positive observations
    false_positives = np.sum((predicted_labels == 1) & (true_labels == 0))

    # False Negatives (FN): Incorrectly predicted negative observations
    false_negatives = np.sum((predicted_labels == 0) & (true_labels == 1))

    # Precision Calculation: TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Recall Calculation: TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def test_Precision_Recall():

    true_labels = np.array([1, 0, 1, 1, 0, 1, 0])
    predicted_labels = np.array([1, 1, 1, 0, 0, 1, 0])

    precision, recall = calculate_precision_recall(true_labels, predicted_labels)
    print(f"Precision: {precision}, Recall: {recall}")
