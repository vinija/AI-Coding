from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

def calculate_multiclass_metrics(y_true, y_pred, classes):
    """
    Calculate precision, recall, and F1-score for a multi-class classification problem.

    Args:
    y_true (list): True class labels.
    y_pred (list): Predicted class labels.
    classes (list): List of unique classes.

    Returns:
    dict: A dictionary containing precision, recall, and F1-score for each class.

    Precision is the ratio of correctly predicted positive observations to the total predicted positives. High precision indicates a low false positive rate.
    Recall (or Sensitivity) measures the ratio of correctly predicted positive observations to all observations in the actual class. High recall indicates a low false negative rate.
    F1-score is the weighted average of Precision and Recall. It takes both false positives and false negatives into account. F1-score is useful when you seek a balance between Precision and Recall and there's an uneven class distribution.


Let's analyze the time and space complexity of the `calculate_confusion_matrix` and `calculate_metrics` functions:

### `calculate_confusion_matrix` Function

1. **Time Complexity**:
   - The function iterates through each element in `y_true` and `y_pred` once, leading to O(n), where n is the number of elements in these arrays.
   - Inside the loop, it iterates through each class in `classes`, adding an additional factor of O(c), where c is the number of unique classes.
   - Thus, the overall time complexity is O(n*c).

2. **Space Complexity**:
   - The function creates a confusion matrix dictionary, `confusion_matrix`, where each key corresponds to a class and its value is a dictionary of four counters (TP, FP, FN, TN).
   - The space complexity is O(c), where c is the number of classes, as space is required to store counts for each class.

### `calculate_metrics` Function

1. **Time Complexity**:
   - The function iterates through each class in the confusion matrix once, leading to O(c), where c is the number of classes.
   - The calculations within the loop are constant time operations.
   - Therefore, the overall time complexity is O(c).

2. **Space Complexity**:
   - For each class, the function stores the calculated precision, recall, and F1-score in a dictionary.
   - The space complexity is O(c), as space is needed for storing these metrics for each class.

### Overall Complexity

- **Time Complexity**: The dominant factor in time complexity is the `calculate_confusion_matrix` function, which is O(n*c). The `calculate_metrics` function adds a smaller O(c) component.
- **Space Complexity**: The space complexity for both functions is O(c), dominated by the storage requirements for the confusion matrix and the metrics results.

These complexities indicate that the functions are efficient in terms of space, using only as much space as there are classes. The time complexity is dependent on the number of samples and the number of classes, which is typical for multi-class classification metrics calculations.
    """

    # Binarize the output for multi-class evaluation
    y_true_binarized = label_binarize(y_true, classes=classes)
    y_pred_binarized = label_binarize(y_pred, classes=classes)

    # Calculate metrics
    precision = precision_score(y_true_binarized, y_pred_binarized, average=None)
    recall = recall_score(y_true_binarized, y_pred_binarized, average=None)
    f1 = f1_score(y_true_binarized, y_pred_binarized, average=None)

    # Create a dictionary to store metrics for each class
    metrics = {"precision": precision, "recall": recall, "f1-score": f1}
    return metrics

def test_multiclass():
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 1, 2, 0]
    classes = [0, 1, 2]

    metrics = calculate_multiclass_metrics(y_true, y_pred, classes)
    print(metrics)


def calculate_confusion_matrix(y_true, y_pred, classes):
    """
    Calculate the confusion matrix for multi-class classification.

    Args:
    y_true (list): True class labels.
    y_pred (list): Predicted class labels.
    classes (list): List of unique classes.

    Returns:
    dict: A dictionary of confusion matrix components (TP, FP, FN, TN) for each class.
    """
    confusion_matrix = {cls: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for cls in classes}

    for actual, predicted in zip(y_true, y_pred):
        for cls in classes:
            if actual == cls:
                if predicted == cls:
                    confusion_matrix[cls]["TP"] += 1
                else:
                    confusion_matrix[cls]["FN"] += 1
            else:
                if predicted == cls:
                    confusion_matrix[cls]["FP"] += 1
                elif predicted != cls:
                    confusion_matrix[cls]["TN"] += 1

    return confusion_matrix

def calculate_metrics(confusion_matrix):
    """
    Calculate precision, recall, and F1-score from the confusion matrix.

    Args:
    confusion_matrix (dict): The confusion matrix for each class.

    Returns:
    dict: A dictionary containing precision, recall, and F1-score for each class.
    """
    metrics = {}
    for cls, matrix in confusion_matrix.items():
        precision = matrix["TP"] / (matrix["TP"] + matrix["FP"]) if matrix["TP"] + matrix["FP"] > 0 else 0
        recall = matrix["TP"] / (matrix["TP"] + matrix["FN"]) if matrix["TP"] + matrix["FN"] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        metrics[cls] = {"precision": precision, "recall": recall, "f1-score": f1}

    return metrics

def test_non_sklearn():
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 1, 2, 0]
    classes = [0, 1, 2]

    confusion_matrix = calculate_confusion_matrix(y_true, y_pred, classes)
    metrics = calculate_metrics(confusion_matrix)
    print(metrics)

