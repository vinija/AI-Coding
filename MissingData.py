

"""
Design an algorithm to handle missing data in a dataset.
Discuss the pros and cons of different imputation methods like mean imputation, median imputation, or using a model to predict missing values
"""

import torch
import torch.nn.functional as F


def impute_missing_data(data, method='mean'):
    """
    Impute missing data in a PyTorch tensor.

    Args:
    data (torch.Tensor): A tensor containing missing data (NaNs).
    method (str): The imputation method ('mean' or 'median').

    Returns:
    torch.Tensor: A tensor with missing data imputed.
    """
    if method not in ['mean', 'median']:
        raise ValueError("Invalid method. Use 'mean' or 'median'.")

    # Create a mask for missing values
    mask = torch.isnan(data)

    for col in range(data.shape[1]):
        if method == 'mean':
            impute_value = torch.nanmean(data[:, col])
        elif method == 'median':
            impute_value = torch.nanmedian(data[:, col])

        data[mask[:, col], col] = impute_value

    return data

def test_torch():

    # Example data with missing values (NaNs)
    data = torch.tensor([[1., 2., float('nan')], [4., float('nan'), 6.], [float('nan'), 8., 9.]])

    # Impute missing data
    imputed_data_mean = impute_missing_data(data.clone(), method='mean')
    imputed_data_median = impute_missing_data(data.clone(), method='median')

    print("Imputed Data (Mean):", imputed_data_mean)
    print("Imputed Data (Median):", imputed_data_median)
