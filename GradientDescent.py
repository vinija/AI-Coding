import pytest
import numpy as np
import torch

"""
Gradient descent is an optimization algorithm used to minimize a function (usually a loss function) by iteratively moving towards the minimum value of the function.

The `gradient_descent` function in this script is used to find the best-fit line for a given set of data points by minimizing the mean squared error. It does so by iteratively adjusting the slope (`m`) and y-intercept (`c`) of the line. Here's a brief analysis of its time and space complexity:

- **Time Complexity**: O(iterations * n), where `iterations` is the number of iterations the gradient descent algorithm runs for, and `n` is the number of data points. This complexity arises because, in each iteration, the function calculates gradients and updates `m` and `c` based on all `n` data points.

- **Space Complexity**: O(1), since the space used by the function is constant and does not depend on the size of the input data. The primary variables (`m`, `c`, `dm`, `dc`, etc.) use a fixed amount of space regardless of the number of data points.

This function is a basic implementation of gradient descent for linear regression and is useful for understanding the fundamental concepts of this optimization technique.
"""

import torch


# Objective function: f(x) = (x - 3)^2
def objective_function(x):
    return (x - 3) ** 2


# Gradient descent function
def gradient_descent_torch(starting_point, learning_rate, num_iterations):
    x = torch.tensor([starting_point],dtype=torch.float32, requires_grad=True)
    for _ in range(num_iterations):
        # Calculate the function value and gradients
        loss = objective_function(x)
        loss.backward()

        # Update x without accumulating gradients
        with torch.no_grad():
            x -= learning_rate * x.grad

        # Zero the gradients after updating
        x.grad.zero_()

    return x.item()

def test_gd_torch():
    # Parameters
    starting_point = 0  # Starting point for x
    learning_rate = 0.1  # Learning rate
    num_iterations = 100  # Number of iterations

    # Perform gradient descent
    minimum = gradient_descent_torch(starting_point, learning_rate, num_iterations)
    print(f"The minimum of the function is at x = {minimum}")


def gradient_descent(x, y, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6):
    """
    Performs gradient descent to fit a line to the data points in x and y.
    This method iteratively adjusts the line (defined by slope 'm' and y-intercept 'c')
    to minimize the sum of squares of the differences (cost) between predicted and actual y-values.

    Args:
    x: Array of predictor variable values.
    y: Array of dependent variable values.
    iterations: Maximum number of iterations for the gradient descent.
    learning_rate: The step size during gradient descent.
    stopping_threshold: The threshold for stopping the iterations when the change in 'm' and 'c' becomes very small.

    Returns:
    m: The slope of the fitted line.
    c: The y-intercept of the fitted line.
    """

    # Initialize the slope (m) and y-intercept (c) with zero values.
    m, c = 0, 0
    # Number of data points in the dataset.
    n = len(x)

    for _ in range(iterations):
        # Predict the y-values using the current values of m and c.
        y_pred = m * x + c

        # Compute the cost (mean squared error) between predicted and actual y-values.
        cost = (1/n) * sum([val**2 for val in (y - y_pred)])

        # Calculate gradients for m and c to minimize the cost.
        # dm and dc represent the direction and amount by which m and c should be updated.
        dm = -(2/n) * sum(x * (y - y_pred))  # Gradient with respect to m
        dc = -(2/n) * sum(y - y_pred)        # Gradient with respect to c

        # Update m and c by moving against the gradient direction by the learning rate.
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Check if the updates to m and c are smaller than the stopping threshold.
        # If yes, it means convergence has been achieved and stop further updates.
        if max(abs(learning_rate * dm), abs(learning_rate * dc)) < stopping_threshold:
            break

    return m, c

def test_gradient_descent():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    # Perform gradient descent to find the optimal m and c
    m, c = gradient_descent(x, y)

    # Print the slope and intercept of the fitted line
    print(f"Slope: {m}, Intercept: {c}")
