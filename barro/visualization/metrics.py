import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal

import matplotlib
matplotlib.use('TkAgg')


def bar_plot_metric(data: dict, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                    reduction: Literal['mean', 'max', 'min', 'median'] = None,
                    y_min: float = 0.0, y_max: float = 100.0,
                    print_values: bool = False, num_decimals: int = 2) -> plt:
    """
    Generates a bar chart showing in each bar the corresponding value for each category.
    :param data: Dictionary containing the values to be represented in the graph. The
    dictionary keys correspond to the categories to be represented on the X-axis, while
    the value associated with each key is the value to be represented on the respective bar.
    If a NumPy Array is passed for each key, instead of a single value, the reduction function
    specified in the 'reduction' parameter is applied.
    :param x_label: String indicating the name of the X axis.
    :param y_label: String indicating the name of the Y axis.
    :param reduction: Reduction function to be applied if a NumPy Array is passed as value for each
    key in the 'data' dictionary. If 'mean', the average over the values in each array is calculated.
    If 'max', the maximum value of the array is used as final value. If 'min', the minimum value of
    the array is used as final value. If 'median', the median (the intermediate value) of the array
    is used as final value.
    :param y_min: Minimum value showed in the Y axis. By default, 0.
    :param y_max: Maximum value showed in the Y axis. By default, 100.
    :param print_values: If 'True', metric values are printed over each bar.
    :param num_decimals: Number of decimals to show if 'print_values' is True.
    :return: Bar chart generated from input data.
    """

    x_values = list(data.keys())

    y_values_raw = list(data.values())
    if type(y_values_raw[0]) == float:
        y_values = y_values_raw
    elif type(y_values_raw[0]) == np.ndarray:
        y_values = __apply_reduction(y_values_raw, reduction)
    else:
        raise TypeError(f'Incorrect type for values in data dictionary: {type(y_values_raw[0])}. '
                        f'Only float or NumPy Array are allowed.')

    plt.figure()
    plt.bar(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=15)

    if print_values:
        for i, v in enumerate(y_values):
            plt.text(i, v + 1, f'{v:.{num_decimals}f}', ha='center')
    plt.tight_layout()

    return plt


def line_plot_metric(data: dict, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                     reduction: Literal['mean', 'max', 'min', 'median'] = None,
                     y_min: float = 0.0, y_max: float = 100.0,
                     print_values: bool = False, num_decimals: int = 2) -> plt:
    """
    Generates a line chart showing in each bar the corresponding value for each category.
    :param data: Dictionary containing the values to be represented in the graph. The
    dictionary keys correspond to the categories to be represented on the X-axis, while
    the value associated with each key is the value to be represented on the respective category.
    If a NumPy Array is passed for each key, instead of a single value, the reduction function
    specified in the 'reduction' parameter is applied.
    :param x_label: String indicating the name of the X axis.
    :param y_label: String indicating the name of the Y axis.
    :param reduction: Reduction function to be applied if a NumPy Array is passed as value for each
    key in the 'data' dictionary. If 'mean', the average over the values in each array is calculated.
    If 'max', the maximum value of the array is used as final value. If 'min', the minimum value of
    the array is used as final value. If 'median', the median (the intermediate value) of the array
    is used as final value.
    :param y_min: Minimum value showed in the Y axis. By default, 0.
    :param y_max: Maximum value showed in the Y axis. By default, 100.
    :param print_values: If 'True', metric values are printed over each point.
    :param num_decimals: Number of decimals to show if 'print_values' is True.
    :return: Bar chart generated from input data.
    """

    x_values = list(data.keys())

    y_values_raw = list(data.values())
    if type(y_values_raw[0]) == float:
        y_values = y_values_raw
    elif type(y_values_raw[0]) == np.ndarray:
        y_values = __apply_reduction(y_values_raw, reduction)
    else:
        raise TypeError(f'Incorrect type for values in data dictionary: {type(y_values_raw[0])}. '
                        f'Only float or NumPy Array are allowed.')

    plt.figure()
    plt.plot(x_values, y_values, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=15)

    if print_values:
        for i, v in enumerate(y_values):
            plt.text(i, v + 1, f'{v:.{num_decimals}f}', ha='center')
    plt.tight_layout()

    return plt


def box_plot_metric(data: dict, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                    y_min: float = 0.0, y_max: float = 100.0) -> plt:
    """
    Generates a box plot showing the corresponding box for each category.
    :param data: Dictionary containing the values to be represented in the graph. The
    dictionary keys correspond to the categories to be represented on the X-axis, while
    the value associated with each key is a NumPy Array with the values to build the
    corresponding box.
    :param x_label: String indicating the name of the X axis.
    :param y_label: String indicating the name of the Y axis.
    :param y_min: Minimum value showed in the Y axis. By default, 0.
    :param y_max: Maximum value showed in the Y axis. By default, 100.
    :return: Box plot generated from input data.
    """
    x_values = list(data.keys())

    y_values_raw = list(data.values())
    if type(y_values_raw[0]) == np.ndarray:
        y_values = y_values_raw
    else:
        raise TypeError(f'Incorrect type for values in data dictionary: {type(y_values_raw[0])}. '
                        f'Only NumPy Array is allowed.')

    plt.figure()
    plt.boxplot(y_values, labels=x_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=15)

    return plt


def error_plot_metric(data: dict, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                      y_min: float = 0.0, y_max: float = 100.0,
                      print_values: bool = False, num_decimals: int = 2) -> plt:
    """
    Generates an error plot that represent, for each NumPy Array of values, the mean and the
    standard deviation. It's very useful to show, for example, the results of a cross-validation
    experimentation.
    :param data: Dictionary containing the values to be represented in the graph. The
    dictionary keys correspond to the categories to be represented on the X-axis, while
    the value associated with each key is a NumPy Array with the values to build the
    corresponding box.
    :param x_label: String indicating the name of the X axis.
    :param y_label: String indicating the name of the Y axis.
    :param y_min: Minimum value showed in the Y axis. By default, 0.
    :param y_max: Maximum value showed in the Y axis. By default, 100.
    :param print_values: If 'True', metric values are printed over each point.
    :param num_decimals: Number of decimals to show if 'print_values' is True.
    :return: Error plot generated from the input data.
    """

    x_values = list(data.keys())

    y_values_raw = list(data.values())
    if type(y_values_raw[0]) == np.ndarray:
        y_values = y_values_raw
    else:
        raise TypeError(f'Incorrect type for values in data dictionary: {type(y_values_raw[0])}. '
                        f'Only NumPy Array is allowed.')

    y_means = __apply_reduction(y_values, 'mean')
    y_stds = __apply_reduction(y_values, 'std')
    print(y_stds)

    plt.figure()
    plt.errorbar(x_values, y_means, y_stds, fmt='o', linewidth=2, capsize=6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=15)

    if print_values:
        for i, v in enumerate(y_means):
            plt.text(i + 0.1, v, f'{v:.{num_decimals}f}', ha='center')
        for i, v in enumerate(y_stds):
            plt.text(i - 0.1, y_means[i]+v, f'{v:.{num_decimals}f}', ha='center')
    plt.tight_layout()

    return plt


def __apply_reduction(raw_values: np.array,
                      reduction: Literal['mean', 'max', 'min', 'median', 'std']) -> np.array:
    if reduction == 'mean':
        return np.array(list(map(np.mean, raw_values)))
    if reduction == 'max':
        return np.array(list(map(np.max, raw_values)))
    if reduction == 'min':
        return np.array(list(map(np.min, raw_values)))
    if reduction == 'median':
        return np.array(list(map(np.median, raw_values)))
    if reduction == 'std':
        return np.array(list(map(np.std, raw_values)))
