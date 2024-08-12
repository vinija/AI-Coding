import numpy as np

"""
The `calculate_precision_recall` function computes precision and recall metrics for a binary classification task. Here's a brief analysis of its time and space complexity:

Time Complexity:
- The function's time complexity is O(n), where n is the number of elements in `true_labels` and `predicted_labels`. This complexity arises from the element-wise comparisons and sum operations that are performed once for each element in the arrays.

Space Complexity:
- The space complexity is O(1) as the function uses a constant amount of space. Additional space usage does not depend on the input size, as it only involves a few scalar variables (`true_positives`, `false_positives`, `false_negatives`, `precision`, and `recall`).

This function is efficient for computing precision and recall, with linear time complexity and constant space complexity, making it suitable for large datasets commonly used in classification tasks.

Edge Case:

Precision and recall are important metrics in evaluating the performance of classification models, particularly in scenarios where the class distribution is imbalanced. However, there are certain edge cases where precision and recall can behave in ways that might be misleading or require special attention. Here are some of those edge cases and how they can impact the interpretation of these metrics:

1. Perfect Precision but Low Recall: This can occur when the model is extremely conservative and only makes a positive prediction when it is very confident. This might result in very few positive predictions, but those that are predicted are almost always correct. The model might have perfect precision (i.e., no false positives), but recall could be very low if many true positives are missed. This can be problematic in scenarios where identifying all positive cases is important, such as in medical diagnoses. To address this, evaluate the trade-off between precision and recall using the F1 score or Precision-Recall curves. If recall is critical, consider adjusting the decision threshold to increase recall, even if it lowers precision.

2. High Recall but Low Precision: This occurs when the model is highly aggressive in making positive predictions, capturing almost all true positives but also predicting a large number of false positives. The model might have high recall but low precision, leading to many false alarms. This can be problematic in applications where false positives carry a high cost, such as fraud detection. Adjusting the decision threshold to balance precision and recall, or using a different metric like the F1 score, can help find a more balanced model. Additionally, consider the cost of false positives and negatives in the specific application to determine the appropriate balance.

3. Class Imbalance: In datasets with a large imbalance between classes (e.g., 95% negatives and 5% positives), precision and recall can be misleading. For example, a model that always predicts the majority class will have high precision and recall for that class but will perform poorly on the minority class. Precision and recall may not fully capture the model's effectiveness in identifying minority class instances, leading to an overestimation of model performance. Use additional metrics like the Precision-Recall AUC, F1 score, or Matthews Correlation Coefficient (MCC) to get a more nuanced view of model performance. Also, consider resampling techniques like SMOTE to balance the classes or use algorithms that handle class imbalance better.

4. High Precision and High Recall: While this is the desired outcome, it can sometimes occur due to overfitting, especially in small or overly simplified datasets. The model might perform exceptionally well on the training data but generalize poorly to unseen data, leading to a false sense of security about the model's performance. Validate the model on a separate test set or using cross-validation to ensure that high precision and recall are not just artifacts of overfitting. Monitor other metrics like the F1 score, and consider regularization techniques to improve generalization.

5. Precision or Recall of Zero: Precision or recall can be zero if the model fails to identify any positive cases correctly, or if it fails to predict any positive cases at all. A precision or recall of zero is a clear indicator that the model is either too conservative (no positive predictions) or too inaccurate (all positive predictions are wrong). This could happen in edge cases like extreme class imbalance or poor model training. If precision or recall is zero, re-evaluate the model's decision threshold, consider re-training with better features, or employ a different model that might be more suited to the problem. Ensure the training process is adequate and that the data used is representative of the problem at hand.

6. False Sense of Security with High Precision: High precision might suggest the model is performing well, but if the recall is low, the model may not be capturing enough positive cases. This can occur in situations where false positives are rare, but so are true positives. The model may appear effective based on high precision alone, but in reality, it might be missing a large portion of the positive cases, which could be critical depending on the application. Always evaluate precision alongside recall and consider the F1 score or the Precision-Recall curve to get a more comprehensive understanding of the model's performance.

7. Precision-Recall Trade-Off: Precision and recall often have a trade-off relationship; improving one may reduce the other. Adjusting the decision threshold of a classifier can shift the balance between precision and recall. Depending on the application, favoring precision over recall, or vice versa, can have significant implications. For instance, in spam detection, high recall might lead to capturing most spam messages, but at the cost of falsely flagging important emails as spam (low precision). Conversely, high precision might result in very few false positives but could miss some spam messages (low recall). Balancing this trade-off depends on the specific requirements of the task at hand.

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
