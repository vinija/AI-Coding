import numpy as np

"""
The `calculate_precision_recall` function computes precision and recall metrics for a binary classification task. Here's a brief analysis of its time and space complexity:

Time Complexity:
- The function's time complexity is O(n), where n is the number of elements in `true_labels` and `predicted_labels`. This complexity arises from the element-wise comparisons and sum operations that are performed once for each element in the arrays.

Space Complexity:
- The space complexity is O(1) as the function uses a constant amount of space. Additional space usage does not depend on the input size, as it only involves a few scalar variables (`true_positives`, `false_positives`, `false_negatives`, `precision`, and `recall`).

This function is efficient for computing precision and recall, with linear time complexity and constant space complexity, making it suitable for large datasets commonly used in classification tasks.
"""
import torch

def calculate_precision_recall_pytorch(true_labels, predicted_labels):
    """
    Calculates the precision and recall from true labels and predicted labels of a classification task using PyTorch.

    Args:
    true_labels (torch.Tensor): A tensor of actual labels.
    predicted_labels (torch.Tensor): A tensor of predicted labels.

    Returns:
    precision (float): The precision of the predictions.
    recall (float): The recall of the predictions.
    """

    # Convert arrays to tensors if they are not already
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels)
    if not isinstance(predicted_labels, torch.Tensor):
        predicted_labels = torch.tensor(predicted_labels)

    # True Positives (TP): Correctly predicted positive observations
    true_positives = torch.sum((predicted_labels == 1) & (true_labels == 1))

    # False Positives (FP): Incorrectly predicted positive observations
    false_positives = torch.sum((predicted_labels == 1) & (true_labels == 0))

    # False Negatives (FN): Incorrectly predicted negative observations
    false_negatives = torch.sum((predicted_labels == 0) & (true_labels == 1))

    # Precision Calculation: TP / (TP + FP)
    precision = true_positives.float() / (true_positives + false_positives).float() if (true_positives + false_positives) > 0 else 0

    # Recall Calculation: TP / (TP + FN)
    recall = true_positives.float() / (true_positives + false_negatives).float() if (true_positives + false_negatives) > 0 else 0

    return precision.item(), recall.item()

def test_torch():
    # Example usage
    true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    predicted_labels = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]

    precision, recall = calculate_precision_recall_pytorch(true_labels, predicted_labels)
    print("Precision:", precision, "Recall:", recall)


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
