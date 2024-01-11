import numpy as np
import torch

"""
Check to see if the matrix has equal diagonals from upper left to lower right. Example: [[0,1,2],[3,0,1],[4,3,0]]

Time Complexity: O(rows * cols), where rows and cols are the dimensions of the matrix. The algorithm iterates over each element of the matrix once.

Space Complexity: O(1), as the space used by the function is constant and does not depend on the size of the matrix. The algorithm only uses a fixed number of variables for iteration and comparison.
"""


def check_diagonal_consistency_torch(matrix):
    # Convert the 2D list matrix to a PyTorch tensor
    tensor = torch.tensor(matrix)

    # Get the size of the tensor
    rows, cols = tensor.shape

    # Iterate over all possible diagonals that start from the first row and the last column
    for start_col in range(cols):
        if not torch.all(tensor.diagonal(start_col) == tensor[0, start_col]):
            return False

    # Iterate over all possible diagonals that start from the first column and the last row (excluding the first element)
    for start_row in range(1, rows):
        if not torch.all(tensor.diagonal(-start_row) == tensor[start_row, 0]):
            return False

    return True

def test_torch():
    # Example matrix
    matrix = [[1, 2, 3], [4, 1, 2], [5, 4, 1]]

    # Check diagonal consistency
    consistent = check_diagonal_consistency(matrix)
    print(consistent)

def check_diagonal_consistency(matrix):
    """
    Check if all diagonals (from upper left to lower right) in the matrix have the same values.

    Args:
    matrix (list[list[int]]): A 2D list representing the matrix.

    Returns:
    bool: True if all diagonals have the same values, False otherwise.
    """

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    for i in range(rows):
        for j in range(cols):
            # Check if the current element's diagonal element is within the bounds of the matrix
            if i + 1 < rows and j + 1 < cols:
                # Compare the current element with its diagonal successor
                if matrix[i][j] != matrix[i + 1][j + 1]:
                    return False

    return True

def test_diagonal():
    matrix1 = [[0, 1, 2], [3, 0, 1], [4, 3, 0]]
    print(check_diagonal_consistency(matrix1))  # Output: True
    assert(check_diagonal_consistency(matrix1)) == True, "Test failed for matrix 1"

    matrix2 = [[0, 1, 2], [3, 1, 1], [4, 3, 0]]
    print(check_diagonal_consistency(matrix2))  # Output: False
    assert (check_diagonal_consistency(matrix2)) == False, "Test failed for matrix 2"

