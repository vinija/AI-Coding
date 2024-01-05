import pytest
import numpy as np

def knn_find_neighbors(data, query, k):
    """
    Finds the k nearest neighbors of a query point in the given dataset using the Euclidean distance.

    Args:
    data: A NumPy array of data points. Shape: [num_samples, num_features].
    query: A NumPy array representing the query point. Shape: [num_features].
    k: The number of nearest neighbors to find.

    Returns:
    A tuple containing:
    - Nearest neighbors of the query point from the dataset. Shape: [k, num_features].
    - Indices of these neighbors in the original dataset.
    """

    # Calculate the Euclidean distance from each point in the dataset to the query point.
    # This is done by squaring the difference, summing over the feature dimensions, and taking the square root.
    distances = np.sqrt(((data - query)**2).sum(axis=1))

    # Sort the distances and get the indices of the k smallest distances.
    # These indices correspond to the k nearest neighbors.
    k_indices = np.argsort(distances)[:k]

    # Return the nearest neighbors and their indices.
    # data[k_indices] fetches the rows from 'data' at positions in 'k_indices'.
    return data[k_indices], k_indices

def test_knn_find_neighbors_numpy():
    """
    Test for the knn_find_neighbors function to ensure it correctly identifies nearest neighbors.
    """

    # Define a small dataset of points and a query point.
    data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    query = np.array([2.5, 3.5])
    k = 2  # Number of neighbors to find

    # Call the KNN function with the data, query, and k.
    neighbors, indices = knn_find_neighbors(data, query, k)

    # Define the expected output for validation.
    expected_neighbors = np.array([[2, 3], [3, 4]])
    expected_indices = np.array([1, 2])

    # Validate that the function's output matches the expected output.
    # np.testing.assert_array_equal throws an AssertionError if two array_like objects are not equal.
    np.testing.assert_array_equal(neighbors, expected_neighbors)
    np.testing.assert_array_equal(indices, expected_indices)

    print("Test passed. Neighbors and indices are as expected.")


