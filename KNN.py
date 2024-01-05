import pytest
import numpy as np
import torch

"""
Time and Space Complexity:

The `knn_find_neighbors` function, implemented in both NumPy and PyTorch, finds the k nearest neighbors to a given query point within a dataset, using Euclidean distance. Here's an analysis of its time and space complexity:

Time Complexity:
- The primary operation is the computation of Euclidean distances between the query point and each point in the dataset. This operation is O(d) for each point, where d is the number of dimensions.
- Since this operation is performed for each of the n data points, the total time complexity for the distance calculation is O(n*d).
- Sorting or partitioning these distances to find the k smallest values has a complexity of O(n*log(n)) for sorting, or O(n) for partitioning.
- Therefore, the overall time complexity is O(n*d + n*log(n)) for sorting or O(n*d + n) for partitioning, depending on the implementation.

Space Complexity:
- The function stores the distances from each data point to the query point, requiring O(n) space.
- It also stores indices of the k smallest distances and the k nearest neighbors, requiring O(k) space.
- Thus, the total space complexity is O(n + k).

The NumPy and PyTorch implementations have similar complexity, but they might differ slightly in performance due to differences in how these libraries handle array operations and memory management. The PyTorch version can leverage GPU acceleration if the data is on a GPU, potentially offering faster computation for large datasets.

"""
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


## PyTorch implementation

def knn_find_neighbors_torch(data, query, k):
    # Calculate Euclidean distances between query and all data points
    distances = torch.sqrt(((data - query)**2).sum(dim=1))

    # Find the indices of the k smallest distances
    k_indices = torch.argsort(distances)[:k]

    # Return the k nearest neighbors and their indices
    return data[k_indices], k_indices


def test_knn_find_neighbors_pytorch():
    data = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    query = torch.tensor([2.5, 3.5])
    k = 2

    neighbors, indices = knn_find_neighbors_torch(data, query, k)
    expected_neighbors = torch.tensor([[2.0, 3.0], [3.0, 4.0]])
    expected_indices = torch.tensor([1, 2])

    assert torch.equal(neighbors, expected_neighbors)
    assert torch.equal(indices, expected_indices)







