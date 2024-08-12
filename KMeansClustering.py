import numpy as np
import torch

"""
Time and Space Complexity

- **Time Complexity**: O(n * k * i * d), where `n` is the number of data points, `k` is the number of clusters, `i` is the number of iterations, and `d` is the number of dimensions/features in each data point. This complexity arises from calculating the distance of each data point to each centroid in every iteration.

- **Space Complexity**: O(n * d + k * d), where `n * d` is for storing the data points and `k * d` for the centroids. Additionally, space is used to store the distances from each point to each centroid and the index of the closest centroid for each point.

- Edge Case:

K-means clustering is a popular unsupervised learning algorithm used to partition a dataset into `k` clusters. Despite its simplicity and effectiveness, there are several corner cases and challenges that can arise when using K-means clustering. Here are some common corner cases and how to address them:

1. **Choosing the Number of Clusters (`k`)**
   - **When It Happens**: One of the primary challenges in K-means clustering is selecting the appropriate number of clusters (`k`). If `k` is too small, distinct groups might be merged; if `k` is too large, the clusters might be too granular or may split meaningful groups.
   - **Impact**: Incorrect selection of `k` can lead to poor clustering results that do not reflect the true underlying structure of the data.
   - **How to Avoid**: Use methods like the Elbow Method, Silhouette Analysis, or Gap Statistics to determine the optimal number of clusters. These methods provide a way to evaluate different values of `k` and choose the one that best fits the data.

2. **Empty Clusters**
   - **When It Happens**: During the K-means iteration process, it is possible for some clusters to end up with no points assigned to them, particularly when the initial centroids are poorly chosen or if the data is sparse.
   - **Impact**: Empty clusters can cause the algorithm to stop prematurely or produce meaningless clusters.
   - **How to Avoid**: If an empty cluster occurs, one common approach is to reinitialize the centroid of the empty cluster to a different position, such as choosing a random point from the dataset or a point farthest from the current centroids.

3. **Sensitivity to Initialization**
   - **When It Happens**: K-means clustering is highly sensitive to the initial placement of centroids. Poor initialization can lead to suboptimal clustering, where the algorithm converges to a local minimum rather than the global minimum.
   - **Impact**: The final clusters can vary significantly depending on the initial centroids, leading to inconsistent results.
   - **How to Avoid**: Use more sophisticated initialization techniques like K-means++ to improve the initial placement of centroids. K-means++ selects initial centroids in a way that increases the likelihood of finding better clusters. Running the algorithm multiple times with different initializations and choosing the best result can also mitigate this issue.

4. **Non-Spherical Clusters**
   - **When It Happens**: K-means clustering assumes that clusters are spherical and equally sized. If the true clusters are not spherical or have varying sizes, K-means might fail to correctly identify them.
   - **Impact**: The algorithm might merge distinct clusters or split a single cluster into multiple parts, leading to poor clustering performance.
   - **How to Avoid**: Consider using other clustering algorithms like DBSCAN, Gaussian Mixture Models (GMM), or hierarchical clustering, which can handle non-spherical and varying-sized clusters better than K-means. Alternatively, you can use a kernelized version of K-means or pre-process the data using dimensionality reduction techniques like PCA to transform it into a space where the clusters are more spherical.

5. **Imbalanced Cluster Sizes**
   - **When It Happens**: K-means tends to assign roughly the same number of data points to each cluster. If the underlying clusters in the data are of very different sizes, K-means might not perform well.
   - **Impact**: Smaller clusters might be merged into larger ones, and large clusters might be split unnaturally.
   - **How to Avoid**: Use algorithms that can handle varying cluster sizes, such as GMM or DBSCAN. Alternatively, consider using weighted K-means, where points in smaller clusters are given higher importance.

6. **Overlapping Clusters**
   - **When It Happens**: K-means assumes that data points belong entirely to one cluster. If clusters overlap significantly, K-means might not perform well, as it forces points to belong to the nearest centroid, even if that point is in the overlap region.
   - **Impact**: Points in overlapping regions might be assigned to incorrect clusters, leading to poor representation of the data.
   - **How to Avoid**: Consider using soft clustering algorithms like GMM, where a data point can belong to multiple clusters with different probabilities. This allows for a more nuanced clustering in cases where clusters overlap.

7. **Outliers and Noise**
   - **When It Happens**: K-means is sensitive to outliers and noise in the data. Outliers can disproportionately influence the position of centroids, leading to poor clustering results.
   - **Impact**: The algorithm might create a separate cluster for outliers or drag the centroid towards outliers, distorting the true clusters.
   - **How to Avoid**: Before applying K-means, consider preprocessing the data to remove or reduce the impact of outliers. You can use techniques like outlier detection (e.g., Z-score or IQR) or robust K-means variants that are less sensitive to outliers. Alternatively, using algorithms like DBSCAN, which can handle outliers better, might be a better choice.

8. **High-Dimensional Data**
   - **When It Happens**: K-means can struggle with high-dimensional data because the concept of "distance" becomes less meaningful in high dimensions due to the curse of dimensionality.
   - **Impact**: The algorithm might fail to find meaningful clusters, as all points become equidistant in high-dimensional space.
   - **How to Avoid**: Use dimensionality reduction techniques like PCA or t-SNE to reduce the data to a lower-dimensional space where K-means can work more effectively. Alternatively, consider using specialized clustering methods designed for high-dimensional data.



"""

import torch


def kmeans(data, k, n_iterations=100):
    """
    Perform K-means clustering using PyTorch.

    Args:
    data (torch.Tensor): A 2D tensor containing the data points.
    k (int): Number of clusters.
    n_iterations (int, optional): Number of iterations for the K-means algorithm. Default is 100.

    Returns:
    torch.Tensor: A tensor containing the centroid coordinates for each cluster.
    torch.Tensor: A tensor with the cluster indices for each data point.
    """

    # Number of samples in the dataset
    n_samples = data.shape[0]

    # Randomly initialize the centroids by selecting k data points
    centroids = data[torch.randperm(n_samples)[:k]]

    for _ in range(n_iterations):
        # Compute distances from each data point to each centroid
        distances = torch.cdist(data, centroids)

        # Assign each data point to the closest centroid
        labels = torch.argmin(distances, axis=1)

        # Update centroids by calculating the mean of all data points assigned to each cluster
        for i in range(k):
            centroids[i] = data[labels == i].mean(dim=0)

    return centroids, labels


# Example usage:
# Assuming `data_tensor` is a 2D tensor of your data points and `num_clusters` is the number of clusters you want
# centroids, labels = kmeans(data_tensor, num_clusters)


def kmeans_clustering(data, k, num_iterations=100):
    """
    Performs K-Means clustering which is an unsupervised machine learning algorithm used for partitioning a given dataset into a set of k groups (or clusters).
    It is widely used in data analysis for pattern recognition, image compression, and other similar tasks where natural groupings within data need to be identified.

    Args:
    data: A NumPy array of data points, shape [num_samples, num_features].
    k: The number of clusters to form.
    num_iterations: The number of iterations to run the algorithm.

    Returns:
    centroids: The final centroids of the clusters.
    closest_centroids: An array indicating the index of the centroid that each data point is closest to.


    This is the logic we plan to implement below:
    function KMeans(data, k, num_iterations):
        Initialize centroids by selecting k random data points

        for i in range(num_iterations):
            Assign each data point to the nearest centroid
            Update each centroid to be the mean of its assigned points

        return centroids, assigned_clusters
    """

    # Randomly initialize k centroids by selecting k unique data points.
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(num_iterations):
        # Calculate the distance from each data point to each centroid.
        # distances has shape [k, num_samples] and holds the squared Euclidean distance of each point from each centroid.
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))

        # For each data point, find the closest centroid.
        # closest_centroids holds the index of the closest centroid for each data point.
        closest_centroids = np.argmin(distances, axis=0)

        # Update each centroid to be the mean of the data points assigned to it.
        for i in range(k):
            centroids[i] = data[closest_centroids == i].mean(axis=0)

    return centroids, closest_centroids

def test_kmeans_clustering_numpy():
    np.random.seed(0)  # This function sets the seed for NumPy's random number generator. essentially set the starting point for the sequence of pseudo-random numbers. If you use the same seed value later, you will get the same sequence of numbers. This is useful for ensuring reproducibility, especially in scientific computing and data science, where you want your results to be repeatable and verifiable.
    data = np.random.rand(100, 2)  # Generate some random 2D data points
    k = 5  # Number of clusters to form

    # Perform K-Means clustering
    centroids, assignments = kmeans_clustering(data, k)

    # Basic assertions to check the results
    assert len(centroids) == k  # Check if the number of centroids is correct
    assert len(np.unique(assignments)) == k  # Check if the correct number of unique clusters is formed

    # Print centroids and their assigned points for visualization
    print("Centroids:\n", centroids)
    for i in range(k):
        print(f"Cluster {i} points:\n", data[assignments == i])

if __name__ == "__main__":
    test_kmeans_clustering_numpy()
