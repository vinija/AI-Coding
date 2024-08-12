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

Edge Case

When using K-Nearest Neighbors (KNN), several corner cases can arise that may affect its performance or applicability:

1. **Choosing the Value of `k`**: The value of `k` (the number of neighbors to consider) is a crucial hyperparameter in KNN. Choosing the wrong `k` can lead to poor model performance. If `k` is too small (e.g., `k = 1`), the model becomes sensitive to noise and outliers, which may result in overfitting. If `k` is too large, the model may oversmooth the decision boundary, leading to underfitting and potentially capturing less distinct patterns in the data. Use cross-validation to experiment with different values of `k` and select the one that gives the best performance on validation data. Typically, odd values of `k` are preferred to avoid ties in binary classification tasks.

2. **Handling Ties**: Ties can occur when an equal number of neighbors from different classes are within the same distance from the query point, especially when `k` is an even number. A tie can make it unclear how to classify the query point, leading to potential ambiguity in predictions. Choose an odd value of `k` to reduce the likelihood of ties. If ties still occur, consider breaking the tie by using additional criteria such as the distance of the neighbors or their class probabilities.

3. **Curse of Dimensionality**: KNN relies on distance metrics to identify nearest neighbors. In high-dimensional spaces, distances between points become less meaningful because all points tend to become equidistant. The model may perform poorly because the nearest neighbors in high-dimensional space may not be representative of the true nearest neighbors, leading to incorrect classifications. Use dimensionality reduction techniques like PCA, t-SNE, or LDA to reduce the number of features before applying KNN. Feature selection techniques can also help by retaining only the most relevant features.

4. **Imbalanced Classes**: KNN can struggle with class imbalance, where one class is significantly underrepresented compared to others. The majority class can dominate the predictions, leading to poor performance on the minority class. Consider using techniques like weighted KNN, where the votes of the neighbors are weighted by their distance to the query point, giving closer neighbors more influence. Additionally, you can balance the dataset using techniques like SMOTE (Synthetic Minority Over-sampling Technique) or by using stratified sampling.

5. **Computational Cost**: KNN requires computing the distance between the query point and all points in the training set, which can be computationally expensive, especially for large datasets. The model can become slow and inefficient, making it impractical for real-time applications or very large datasets. Use techniques like KD-trees, Ball-trees, or approximate nearest neighbor algorithms to reduce the computational cost. For very large datasets, consider sampling a subset of the data or using a precomputed distance matrix.

6. **Sensitivity to Noise**: KNN is sensitive to noisy data points because they can disproportionately affect the classification, especially when `k` is small. Noisy data points can lead to incorrect classifications, reducing the model's accuracy. Increase the value of `k` to reduce the influence of noise or use a distance-weighted version of KNN, where closer neighbors have more influence. Preprocessing the data to remove or reduce noise can also help improve performance.

7. **Non-Uniform Feature Scales**: If the features in the dataset have different scales, the distance metric can be dominated by features with larger ranges, leading to biased results. The model might give more importance to features with larger scales, even if they are not the most relevant for classification. Standardize or normalize the feature values to ensure that all features contribute equally to the distance calculation. Techniques like Min-Max scaling or Z-score standardization are commonly used to address this issue.

8. **Handling Missing Data**: KNN does not inherently handle missing data, so it must be addressed before applying the algorithm. Missing data can lead to incorrect distance calculations or the inability to compute distances for certain points, leading to unreliable predictions. Impute missing values using techniques like mean, median, or mode imputation, or use more sophisticated imputation methods like KNN imputation (where missing values are filled in based on the values of the nearest neighbors).

9. **Decision Boundary Sensitivity**: KNN's decision boundary is non-parametric and can be very sensitive to the distribution of the training data. Small changes in the training data can lead to significant changes in the decision boundary, leading to inconsistent model performance. Ensure that the training data is representative and includes enough samples to smooth out the decision boundary. If necessary, use techniques like bagging or bootstrapping to improve the stability of the model.

10. **Handling Large Datasets**: KNN can become impractical with very large datasets due to its high memory usage and slow prediction time. The model may become too slow to be useful, especially in real-time applications or when dealing with large-scale datasets. Use dimensionality reduction techniques or approximate nearest neighbor methods to speed up the computation. Alternatively, consider using faster algorithms like decision trees or random forests, which can handle large datasets more efficiently.

"""

import torch
import numpy as np

def knn_pytorch(data, query, k, distance_fn):
    """
    Perform K-Nearest Neighbors classification.

    Args:
        data (torch.Tensor): Training dataset, where each row is a data point.
        query (torch.Tensor): Query dataset, where each row is a data point to classify.
        k (int): Number of nearest neighbors to consider.
        distance_fn (callable): Function to compute the distance between data points.

    Returns:
        torch.Tensor: Predicted labels for the query dataset.
    """

    # Calculate distances between each query point and all data points
    distances = distance_fn(data, query)

    # Find the indices of k smallest distances
    _, indices = torch.topk(distances, k, largest=False)

    # Gather labels of k nearest neighbors
    labels = data[indices][:, :, -1]  # Assuming the last column of data is the label

    # Predict by majority vote
    predictions, _ = torch.mode(labels, dim=1)

    return predictions

# Distance function (Euclidean)
def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, axis=1))

# Example usage
data = torch.tensor([[1, 1, 0], [2, 2, 0], [3, 3, 1], [4, 4, 1]])  # Last column is the label
query = torch.tensor([[1.5, 1.5], [3.5, 3.5]])
k = 2
predictions = knn_pytorch(data[:, :-1], query, k, euclidean_distance)
print(predictions)



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







