import numpy as np

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