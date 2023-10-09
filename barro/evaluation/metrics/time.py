import numpy as np
from typing import Literal, Union


def get_mae(predictions: np.array, ground_truths: np.array,
            reduction: Literal['mean', 'none'] = 'mean') -> Union[float, np.array]:
    """
    Calculates de Mean Absolute Error (MAE) between the predicted times
    and the real values.
    :param predictions: NumPy Array with the predicted times as floats.
    :param ground_truths: NumPy Array with the real times as floats.
    :param reduction: If 'mean', calculates the mean of all the MAEs between
    each pair prediction-ground truth. If 'none', no reduction is done and the
    function returns all the MAEs of the pairs.
    :return: Float indicating the MAE if reduction = 'mean' or NumPy Array if
    reduction = 'none'.
    """

    mae = np.abs(predictions - ground_truths)

    if reduction == 'mean':
        mae = np.mean(mae).item()

    return mae


def get_mse(predictions: np.array, ground_truths: np.array,
            reduction: Literal['mean', 'none'] = 'mean') -> Union[float, np.array]:
    """
    Calculates de Mean Square Error (MSE) between the predicted times
    and the real values.
    :param predictions: NumPy Array with the predicted times as floats.
    :param ground_truths: NumPy Array with the real times as floats.
    :param reduction: If 'mean', calculates the mean of all the MAEs between
    each pair prediction-ground truth. If 'none', no reduction is done and the
    function returns all the MAEs of the pairs.
    :return: Float indicating the MSE if reduction = 'mean' or NumPy Array if
    reduction = 'none'.
    """

    mse = np.power(predictions - ground_truths, 2)

    if reduction == 'mean':
        mse = np.mean(mse).item()

    return mse
