import numpy as np
import pytest
import torch

"""
Description: IoU is a measure used to evaluate object detection models. 
It calculates the ratio of the area of overlap to the area of union between the predicted bounding box and the ground truth bounding box.

The time and space complexity of the `calculate_iou` function are as follows:

1. **Time Complexity**: The time complexity of the function is O(1). This constant time complexity arises because the function performs a fixed number of arithmetic operations regardless of the size of the input. The computation involves simple arithmetic calculations (such as finding the maximum and minimum of coordinates, and calculating areas) which are executed in constant time.

2. **Space Complexity**: The space complexity is also O(1). The function uses a constant amount of space to store variables for calculations, such as `xA`, `yA`, `xB`, `yB`, `interArea`, `boxAArea`, and `boxBArea`, along with the input parameters `boxA` and `boxB`. The space used does not scale with the input size, as it only involves storing a fixed number of variables.


"""

def calculate_iou_torch(box1,box2):
    # Calculate intersection coordinates
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate area of each bbox
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate area of union
    union = area_box1 + area_box2 - intersection

    # Calculate IoU
    iou = intersection / union

    return iou


def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    IoU is a measure of the overlap between two bounding boxes. It is calculated as the area of
    intersection divided by the area of union of the two boxes.

    Parameters:
        boxA (np.array): Numpy array [x1, y1, x2, y2] representing the first box,
                         where (x1, y1) is the top left coordinate,
                         and (x2, y2) is the bottom right coordinate.
        boxB (np.array): Numpy array [x1, y1, x2, y2] for the second box.

    Returns:
        float: IoU of boxA and boxB.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    # xA and yA represent the top-left coordinate of the intersection
    # xB and yB represent the bottom-right coordinate of the intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IoU by dividing the intersection area by the union area
    # Union area is the sum of both box areas minus the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Test case for the IoU calculation
def test_iou_numpy():
    # Define two example bounding boxes
    boxA = np.array([50, 50, 150, 150])
    boxB = np.array([60, 60, 170, 160])

    # Calculate the IoU and assert it matches the expected value
    expected_iou = 0.6306450384
    calculated_iou = calculate_iou(boxA, boxB)
    assert np.isclose(calculated_iou, expected_iou), f"Expected IoU: {expected_iou}, but got: {calculated_iou}"


