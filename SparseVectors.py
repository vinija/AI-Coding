"""
The `SparseVector` class and its `dotProduct` method represent an efficient way to handle sparse vectors and calculate their dot product. Here's the time and space complexity analysis:

1. **Initialization (`__init__`)**:
   - Time Complexity: O(n), where n is the number of elements in the input `nums` array. This is due to the iteration over each element to check and store the non-zero elements.
   - Space Complexity: O(k), where k is the number of non-zero elements in the input `nums` array. The space is used to store these non-zero elements and their indices in a dictionary.

2. **Dot Product (`dotProduct`)**:
   - Time Complexity: O(min(k1, k2)), where k1 and k2 are the numbers of non-zero elements in the two sparse vectors. The method iterates through the non-zero elements of the first vector and checks for corresponding non-zero elements in the second vector.
   - Space Complexity: O(1), as the space used for the dot product calculation is constant. It involves a few scalar variables for the calculation and does not depend on the size of the input vectors.

The `SparseVector` class is particularly efficient for vectors with a large number of elements, most of which are zeros, as it saves significant memory and computational resources compared to dense representation. The dot product operation is also optimized by only considering the non-zero elements, which is especially beneficial when the vectors are very sparse.
"""

class SparseVector:
    def __init__(self, nums):
        """
        Initialize the SparseVector.

        A sparse vector is represented efficiently by storing only its non-zero elements.
        This is useful in scenarios where the vector has a large number of elements,
        most of which are zeros, to save memory and computational resources.

        Args:
        nums (list[int]): List of integers representing the sparse vector.

        Attributes:
        non_zero_elements (dict): A dictionary mapping indices of non-zero elements to their values.
        """
        # Store only non-zero elements in a dictionary with their index as the key.
        self.non_zero_elements = {i: val for i, val in enumerate(nums) if val != 0}

    def dotProduct(self, vec):
        """
        Compute the dot product between this SparseVector and another SparseVector.

        The dot product is calculated as the sum of products of corresponding elements
        of the two vectors. For sparse vectors, we only need to consider the non-zero elements.

        Args:
        vec (SparseVector): Another SparseVector instance against which to calculate the dot product.

        Returns:
        int: The result of the dot product.
        """

        result = 0
        # Iterate through the non-zero elements of this vector.
        for i, val in self.non_zero_elements.items():
            # Multiply with corresponding element in the other vector if it's non-zero.
            # The `get` method on dictionary returns 0 if the key (index) is not found.
            result += val * vec.non_zero_elements.get(i, 0)
        return result

# NumPy test cases for SparseVector
def test_sparse_vector_numpy():
    # Test case 1
    nums1 = [1, 0, 0, 2, 3]
    nums2 = [0, 3, 0, 4, 0]
    v1 = SparseVector(nums1)
    v2 = SparseVector(nums2)
    # Expected dot product is 1*0 + 0*3 + 0*0 + 2*4 + 3*0 = 8
    assert v1.dotProduct(v2) == 8, "Dot product calculation is incorrect."

    # Test case 2
    nums1 = [0, 1, 0, 0, 0]
    nums2 = [0, 0, 0, 0, 2]
    v1 = SparseVector(nums1)
    v2 = SparseVector(nums2)
    # Expected dot product is 0*0 + 1*0 + 0*0 + 0*0 + 0*2 = 0
    assert v1.dotProduct(v2) == 0, "Dot product calculation is incorrect."


