"""

This function uses a nested loop to iterate through the start positions of the sliding window, extracts the elements of each window, and then computes the median using NumPy's `np.median()` function. It assumes the input matrix is a list of lists of integers and that the sliding window size `pq` is valid for the given matrix.

The time complexity of this function is O((n-p+1) * (m-q+1) * p * q), which comes from iterating over each sliding window and collecting the elements in the window.
The space complexity is O(p * q) for storing elements of a single window plus O(W) for storing medians of W windows.

To solve the problem of finding the median of all possible sliding windows of size `pq` in an `nm` sized matrix, you'll need to iterate over each possible window, extract the elements in the window, and compute their median. The sliding window moves horizontally first, and then down, much like reading a page of text.
"""

import numpy as np

def sliding_window_median(matrix, p, q):
    """
    Find the median of all possible sliding windows of size pq in an nm sized matrix.

    Args:
    matrix (list[list[int]]): A 2D list representing the matrix of size nm.
    p (int): The number of rows in the sliding window.
    q (int): The number of columns in the sliding window.

    Returns:
    list[float]: A list containing the median of all sliding windows.
    """
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    medians = []

    # Iterate over each possible starting position of the sliding window
    for i in range(n - p + 1):
        for j in range(m - q + 1):
            # Extract the elements in the current window
            window = [matrix[x][y] for x in range(i, i + p) for y in range(j, j + q)]
            # Compute and store the median of the current window
            medians.append(np.median(window))

    return medians

def test_case():
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    p, q = 2, 2  # Window size
    medians = sliding_window_median(matrix, p, q)
    print(medians)


