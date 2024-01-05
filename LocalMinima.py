"""
Input: arr[] = {9, 6, 3, 14, 5, 7, 4};
Output: Index of local minima is 2
The output prints index of 3 because it is
smaller than both of its neighbors.
Note that indexes of elements 5 and 4 are
also valid outputs.

Input: arr[] = {23, 8, 15, 2, 3};
Output: Index of local minima is 1

Input: arr[] = {1, 2, 3};
Output: Index of local minima is 0

Input: arr[] = {3, 2, 1};
Output: Index of local minima is 2

Source: https://www.geeksforgeeks.org/find-local-minima-array/

# Time Complexity: O(log n) due to binary search
# Space Complexity: O(1)
"""

def find_local_minima(arr):
    """
    Find the index of a local minima in the given array. A local minima is defined as an element
    which is smaller than its neighbors. For edge elements, only one neighbor is considered.

    Args:
    arr (list[int]): The input array.

    Returns:
    int: The index of a local minima in the array.
    """
    n = len(arr)
    start, end = 0, n - 1

    # Edge cases for the first and last element
    if n == 1 or arr[0] < arr[1]:
        return 0
    if arr[n - 1] < arr[n - 2]:
        return n - 1

    # Binary search for the local minima
    while start <= end:
        mid = (start + end) // 2

        # Check if the middle element is less than its neighbors
        if arr[mid] < arr[mid - 1] and arr[mid] < arr[mid + 1]:
            return mid
        # If the left neighbor is smaller, move to the left half
        elif arr[mid - 1] < arr[mid]:
            end = mid - 1
        # If the right neighbor is smaller, move to the right half
        else:
            start = mid + 1

    # Fallback in case no local minima is found (should not happen in a well-formed input)
    return -1

def test_function():
    arr = [9, 6, 3, 14, 5, 7, 4]
    print(f"Index of local minima is {find_local_minima(arr)}")  # Output: 2

    arr = [23, 8, 15, 2, 3]
    print(f"Index of local minima is {find_local_minima(arr)}")  # Output: 1

    arr = [1, 2, 3]
    print(f"Index of local minima is {find_local_minima(arr)}")  # Output: 0

    arr = [3, 2, 1]
    print(f"Index of local minima is {find_local_minima(arr)}")  # Output: 2




