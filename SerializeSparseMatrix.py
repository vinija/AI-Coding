import numpy as np

def serialize_sparse_matrix(sparse_matrix, file_path):
    """
    Serialize a sparse matrix to a file.

    Args:
    sparse_matrix (np.array): 2D numpy array representing the sparse matrix.
    file_path (str): Path of the file where the matrix will be saved.
    """
    # Find the non-zero elements and their indices
    non_zero_indices = np.argwhere(sparse_matrix != 0)
    # ROW: sparse_matrix[non_zero_indices[:, 0], COL: non_zero_indices[:, 1]]
    non_zero_values = sparse_matrix[non_zero_indices[:, 0], non_zero_indices[:, 1]]

    print("index ", non_zero_indices)
    print("values ", non_zero_values)

    # Write the non-zero elements and their indices to the file
    with open(file_path, 'w') as file:
        for (x, y), value in zip(non_zero_indices, non_zero_values):
            file.write(f"{x},{y},{value}\n")

def deserialize_sparse_matrix(file_path, shape):
    """
    Deserialize a sparse matrix from a file.

    Args:
    file_path (str): Path of the file from which the matrix will be loaded.
    shape (tuple): Shape of the original sparse matrix.

    Returns:
    np.array: The deserialized sparse matrix.
    """
    # Initialize an empty matrix with the given shape
    sparse_matrix = np.zeros(shape)

    # Read the file and reconstruct the matrix
    with open(file_path, 'r') as file:
        for line in file:
            x, y, value = line.split(',')
            sparse_matrix[int(x), int(y)] = float(value)

    return sparse_matrix

def test_sparse_matrix():
    sparse_matrix = np.array([[0, 0, 3], [4, 0, 0], [0, 0, 0], [0, 5, 0]])
    file_path = "sparse_matrix.txt"

    serialize_sparse_matrix(sparse_matrix, file_path)
    deserialized_matrix = deserialize_sparse_matrix(file_path, sparse_matrix.shape)

    print("Original Sparse Matrix:")
    print(sparse_matrix)
    print("Deserialized Sparse Matrix:")
    print(deserialized_matrix)
