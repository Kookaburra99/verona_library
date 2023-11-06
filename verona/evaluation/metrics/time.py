from typing import Literal, Union

import numpy as np


def get_mae(predictions: np.array, ground_truths: np.array,
            reduction: Literal['mean', 'none'] = 'mean') -> Union[float, np.array]:
    """
    Calculates the Mean Absolute Error (MAE) between the predicted and real times.

    Args:
        predictions (np.array): NumPy Array containing the predicted times as floats.
        ground_truths (np.array): NumPy Array containing the real times as floats.
        reduction (Literal['mean', 'none'], optional): Determines the type of reduction applied to the MAE.
            If 'mean', calculates the average MAE for all pairs of prediction and ground truth.
            If 'none', returns all MAE values for the individual pairs without reduction.

    Returns:
        Union[float, np.array]: MAE as a single float if reduction is 'mean', or as a NumPy Array if reduction is 'none'.
    """

    mae = np.abs(predictions - ground_truths)

    if reduction == 'mean':
        mae = np.mean(mae).item()

    return mae


def get_mse(predictions: np.array, ground_truths: np.array,
            reduction: Literal['mean', 'none'] = 'mean') -> Union[float, np.array]:
    """
    Calculates the Mean Square Error (MSE) between the predicted and real times.

    Args:
        predictions (np.array): NumPy Array containing the predicted times as floats.
        ground_truths (np.array): NumPy Array containing the real times as floats.
        reduction (Literal['mean', 'none'], optional): Determines the type of reduction applied to the MSE.
            If 'mean', calculates the average MSE for all pairs of prediction and ground truth.
            If 'none', returns all MSE values for the individual pairs without reduction.

    Returns:
        Union[float, np.array]: MSE as a single float if reduction is 'mean', or as a NumPy Array if reduction is 'none'.
    """

    mse = np.power(predictions - ground_truths, 2)

    if reduction == 'mean':
        mse = np.mean(mse).item()

    return mse
