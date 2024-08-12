import numpy as np
from sklearn import metrics
import pytest
import torch
from torchmetrics import AUROC

"""

Time and Space Complexity:

The `calculate_tpr_fpr` and `calculate_auc_roc_with_sklearn` functions in this script are used for evaluating the performance of binary classification models. The time and space complexity for these functions are as follows:

1. **`calculate_tpr_fpr` Function**:
   - Time Complexity: O(n), where n is the number of elements in `y_true` and `y_scores`. This is because the function involves a single pass through these arrays to compute True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN).
   - Space Complexity: O(1), as the space used by the function is constant and does not depend on the size of the input arrays. The primary variables are scalar values that store the counts of TP, FP, FN, and TN.

2. **`calculate_auc_roc_with_sklearn` Function**:
   - Time Complexity: O(n * log(n)), typically dominated by the sorting operation involved in computing the ROC curve, where n is the number of elements in `y_true` and `y_scores`.
   - Space Complexity: O(n), where n is the number of elements in `y_true` and `y_scores`. This is because the function may internally need to store sorted arrays or additional structures for computing the AUC-ROC.

Both functions are crucial for analyzing the performance of classification algorithms, with `calculate_tpr_fpr` providing insights into model sensitivity (TPR) and specificity (FPR), and `calculate_auc_roc_with_sklearn` offering a comprehensive measure of model performance across various thresholds.



An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

True Positive Rate
False Positive Rate
True Positive Rate (TPR) is a synonym for recall and is therefore defined as follows:

TPR = TP/TP + FN

False Positive Rate (FPR) is defined as follows:

FPR = FP/ FP + TN

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.
- Source: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

Edge Cases

- Edge cases in AUC-ROC include scenarios such as a perfect classifier (AUC = 1), indicating no misclassifications, and a random classifier (AUC = 0.5), which performs no better than guessing. An inverse classifier (AUC = 0) consistently predicts the opposite of the true class. Other edge cases include when a model predicts only a single class, making the ROC curve undefined, and when dealing with imbalanced datasets, where AUC might misleadingly suggest good performance despite poor handling of the minority class. These edge cases highlight the importance of context when interpreting AUC-ROC results.
- These edge cases in AUC-ROC typically arise due to model overfitting, class imbalance, insufficient model complexity, or data quality issues. To avoid them, employ robust model evaluation techniques, use alternative metrics when appropriate, and ensure that the model and data preprocessing steps are tailored to the specific challenges of the problem at hand.
"""

def calculate_tpr_fpr_torch(y_true, y_pred, threshold):
    # Apply the threshold to the predictions
    pred_bin = (y_pred >= threshold).float()

    # True Positive (TP): correctly predicted positive
    TP = torch.sum((y_true == 1) & (pred_bin == 1)).float()

    # False Positive (FP): incorrectly predicted positive
    FP = torch.sum((y_true == 0) & (pred_bin == 1)).float()

    # True Negative (TN): correctly predicted negative
    TN = torch.sum((y_true == 0) & (pred_bin == 0)).float()

    # False Negative (FN): incorrectly predicted negative
    FN = torch.sum((y_true == 1) & (pred_bin == 0)).float()

    # True Positive Rate (TPR) or Sensitivity
    TPR = TP / (TP + FN)

    # False Positive Rate (FPR)
    FPR = FP / (FP + TN)

    return TPR, FPR

def auc_roc(y_true, y_pred):
    # Sort predictions and corresponding true values
    sorted_indices = torch.argsort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Initialize
    tpr_values = [0.0]
    fpr_values = [0.0]

    # Compute TPR and FPR at each threshold
    for threshold in y_pred_sorted:
        TPR, FPR = calculate_tpr_fpr_torch(y_true_sorted, y_pred_sorted, threshold)
        tpr_values.append(TPR.item())
        fpr_values.append(FPR.item())

    # Add (1,1) to complete the curve
    tpr_values.append(1.0)
    fpr_values.append(1.0)

    # Compute AUC using the trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr_values)):
        auc += (fpr_values[i] - fpr_values[i - 1]) * (tpr_values[i] + tpr_values[i - 1]) / 2

    return auc

def test_full_torch():
    y_true = torch.tensor([1, 0, 1, 0])
    y_pred = torch.tensor([0.9, 0.3, 0.8, 0.4])
    print("AUC-ROC:", auc_roc(y_true, y_pred))



def auc_pytorch():
    # example predictions and labels
    y_pred = torch.tensor([0.1,0.4,0.35,0.8])
    y_true = torch.tensor([0,0,1,1])

    # Initialize AUCROC metric
    aucroc = AUROC(task="binary")

    #Compute AUC-ROC
    auc_score = aucroc(y_pred, y_true)

    print(f"AUC-ROC Score: {auc_score}")

def test_auc_pytorch():
    auc_pytorch()


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

