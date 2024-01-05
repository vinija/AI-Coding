import numpy as np
import pytest
def binary_cross_entropy_loss(y_true, y_pred):
    """
    Compute the binary cross-entropy loss, a common loss function for binary classification tasks.

    Args:
    y_true (np.array): Array of true binary labels (0 or 1).
    y_pred (np.array): Array of predicted probabilities, corresponding to the probability of the positive class.

    Returns:
    float: The average binary cross-entropy loss across all the predictions.

    The function uses a small constant 'epsilon' to avoid numerical issues with log(0). It clips the predicted
    probabilities to be within the range [epsilon, 1-epsilon] before computing the loss.
    """

    epsilon = 1e-15  # Small constant to avoid log(0) issue
    # Clip predictions to avoid log(0). Clipping within the range [epsilon, 1 - epsilon].
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate binary cross-entropy loss
    # For each true label and corresponding prediction, compute the log loss part and take the average.
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def test_binary_cross_entropy_loss():
    """
    Test the binary_cross_entropy_loss function to ensure it is calculating the loss correctly.
    """

    # Define sample true labels and predicted probabilities
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.3])

    # Assert that the calculated loss is approximately equal to the expected value.
    # pytest.approx is used to handle floating-point arithmetic inaccuracies.
    expected_loss = 0.409
    calculated_loss = binary_cross_entropy_loss(y_true, y_pred)
    assert calculated_loss == pytest.approx(expected_loss, 0.01), f"Expected Loss: {expected_loss}, but got: {calculated_loss}"

# Run the test
test_binary_cross_entropy_loss()

# Example usage
# loss = binary_cross_entropy_loss(y_true, y_pred)



