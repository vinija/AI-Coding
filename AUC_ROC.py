import numpy as np
from sklearn import metrics
import pytest

"""
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

True Positive Rate
False Positive Rate
True Positive Rate (TPR) is a synonym for recall and is therefore defined as follows:

TPR = TP/TP + FN

False Positive Rate (FPR) is defined as follows:

FPR = FP/ FP + TN

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.
- Source: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
"""


def calculate_tpr_fpr(y_true, y_scores, threshold):
    """
    Calculate True Positive Rate (TPR) and False Positive Rate (FPR).

    The TPR is the ratio of true positive predictions to the total actual positives,
    while FPR is the ratio of false positive predictions to the total actual negatives.

    Parameters:
    y_true (np.array): Array of true binary labels (1s and 0s).
    y_scores (np.array): Array of scores, typically as probability estimates from a classifier.
    threshold (float): Threshold for classifying a score as a positive (1) or negative (0).

    Returns:
    float, float: TPR and FPR values.
    """

    # Convert scores to binary predictions based on the threshold.
    # Scores equal to or higher than the threshold are predicted as 1, else as 0.
    y_pred = (y_scores >= threshold).astype(int)

    # Calculate the number of true positives, false positives, false negatives, and true negatives.
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives: Correctly identified positives
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives: Incorrectly identified positives
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives: Incorrectly identified negatives
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives: Correctly identified negatives

    # Calculate TPR: TP / (TP + FN)
    # TPR is the proportion of actual positives that are correctly identified.
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0

    # Calculate FPR: FP / (FP + TN)
    # FPR is the proportion of actual negatives that are incorrectly identified as positives.
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

    return TPR, FPR

def test_tpr_fpr_numpy():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_scores = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.9])
    threshold = 0.5
    TPR, FPR = calculate_tpr_fpr(y_true, y_scores, threshold)

    expected_TPR = 1.0  # Updated expected TPR value
    expected_FPR = 0.0  # Expected FPR value

    assert np.isclose(TPR, expected_TPR) and np.isclose(FPR, expected_FPR), "TPR or FPR calculation is incorrect"


def calculate_auc_roc_with_sklearn(y_true, y_scores):
    """
    Calculates the Area Under the Receiver Operating Characteristic (AUC-ROC) curve.

    Args:
    y_true: A numpy array of true binary labels. (1 for positive class, 0 for negative class)
    y_scores: A numpy array of scores, can either be probability estimates of the positive class,
              confidence values, or non-thresholded measure of decisions.

    Returns:
    auc: The calculated AUC-ROC value.

    The ROC curve is plotted with True Positive Rate (TPR) against the False Positive Rate (FPR)
    where TPR is on y-axis and FPR is on the x-axis.
    AUC - ROC curve is a performance measurement for classification problems at various thresholds settings.
    AUC stands for "Area under the ROC Curve." The higher the AUC, the better the model is at
    predicting 0s as 0s and 1s as 1s.
    """

    # Calculate the AUC-ROC using sklearn's roc_auc_score function
    auc = metrics.roc_auc_score(y_true, y_scores)

    return auc

# Pytest for calculate_auc_roc function
def test_calculate_auc_roc():
    y_true = np.array([1, 0, 1, 0, 1])
    y_scores = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
    expected_auc_roc = 1.0  # Expected AUC-ROC value

    calculated_auc_roc = calculate_auc_roc_with_sklearn(y_true, y_scores)
    assert calculated_auc_roc == expected_auc_roc, f"Expected AUC-ROC: {expected_auc_roc}, but got: {calculated_auc_roc}"

# Running the test
pytest.main(["-v"])

