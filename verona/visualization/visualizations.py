import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from verona.data.download import get_dataset
from verona.data.results import load_results_plackett_luce
from verona.evaluation.stattests.plackettluce import PlackettLuceResults, PlackettLuceRanking

import numpy as np
from typing import Literal



def plot_posteriors_plackett(plackett_results : PlackettLuceResults, save_path=None):
    """
    Plot the posteriors of the Plackett-Luce model (quantiles 95%, 05% and 50%). If two approaches do not overlap,
    they have a significative different ranking.

    Parameters
        save_path: String that indicates the path where the plot will be saved. If None, the plot will not be saved.

    Returns
        fig.figure : matplotlib figure of the aforementioned plot

    Examples:
        >>> result_matrix = pd.DataFrame([[0.75, 0.6, 0.8], [0.8, 0.7, 0.9], [0.9, 0.8, 0.7]])
        >>> plackett_ranking = PlackettLuceRanking(result_matrix, ["a1", "a2", "a3"])
        >>> results = plackett_ranking.run(n_chains=10, num_samples=300000, mode="max")
        >>> plot = plot_posteriors_plackett(results, save_path=None)
        >>> print(plot)
    """

    if plackett_results is None or plackett_results.posterior is None:
        raise ValueError("You must run the model first")

    posterior = plackett_results.posterior
    y95 = posterior.quantile(q=0.95, axis=0)
    y05 = posterior.quantile(q=0.05, axis=0)
    y50 = posterior.quantile(q=0.5, axis=0)
    df_boxplot = pd.concat([y05, y50, y95], axis=1)
    df_boxplot.columns = ["y05", "y50", "y95"]
    df_boxplot["Approaches"] = posterior.columns

    y50 = df_boxplot["y50"].tolist()
    y05 = (df_boxplot["y50"] - df_boxplot["y05"]).tolist()
    y95 = (df_boxplot["y95"] - df_boxplot["y50"]).tolist()
    sizes = [y05, y95]

    fig = df_boxplot.plot.scatter(x="Approaches", y="y50", rot=90, ylabel="Probability")
    plt.tight_layout()
    fig.errorbar(df_boxplot["Approaches"], y50, yerr=sizes, solid_capstyle="projecting", capsize=5, fmt="none")
    fig.grid(linestyle="--")
    fig.xaxis.set_label_text("")
    fig.xaxis.set_major_formatter(ticker.FixedFormatter(df_boxplot["Approaches"]))
    if save_path is not None:
        fig.figure.savefig(save_path)

    return fig.figure

def bar_plot_metric(data: dict, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                    reduction: Literal['mean', 'max', 'min', 'median'] = None,
                    y_min: float = 0.0, y_max: float = 100.0,
                    print_values: bool = False, num_decimals: int = 2) -> plt:
    """Generates a bar chart from input data.

    Args:
        data (dict): A dictionary where the keys correspond to the categories to be
            represented on the X-axis and the values are either single numerical values
            or NumPy Arrays. If arrays are used, the `reduction` parameter will be applied.
        x_label (str, optional): Label for the X axis. Defaults to 'Dataset'.
        y_label (str, optional): Label for the Y axis. Defaults to 'Accuracy'.
        reduction (Literal['mean', 'max', 'min', 'median'], optional): The reduction function
            to be applied if the values in the `data` dictionary are NumPy Arrays.
            Defaults to None.
        y_min (float, optional): The minimum value for the Y-axis. Defaults to 0.0.
        y_max (float, optional): The maximum value for the Y-axis. Defaults to 100.0.
        print_values (bool, optional): If True, metric values are printed over each bar.
            Defaults to False.
        num_decimals (int, optional): Number of decimals to display if `print_values` is True.
            Defaults to 2.

    Returns:
        plt: Matplotlib plot object representing the bar chart.
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
    Generates a line chart from input data.

    Args:
        data (dict): A dictionary where the keys correspond to the categories to be
            represented on the X-axis and the values are either single numerical values
            or NumPy Arrays. If arrays are used, the `reduction` parameter will be applied.
        x_label (str, optional): Label for the X axis. Defaults to 'Dataset'.
        y_label (str, optional): Label for the Y axis. Defaults to 'Accuracy'.
        reduction (Literal['mean', 'max', 'min', 'median'], optional): The reduction function
            to be applied if the values in the `data` dictionary are NumPy Arrays.
            Defaults to None.
        y_min (float, optional): The minimum value for the Y-axis. Defaults to 0.0.
        y_max (float, optional): The maximum value for the Y-axis. Defaults to 100.0.
        print_values (bool, optional): If True, metric values are printed over each point.
            Defaults to False.
        num_decimals (int, optional): Number of decimals to display if `print_values` is True.
            Defaults to 2.

    Returns:
        plt: Matplotlib plot object representing the line chart.
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
    Generates a box plot from input data.

    Args:
        data (dict): A dictionary where the keys correspond to the categories to be
            represented on the X-axis and the values are NumPy Arrays used to construct
            the corresponding boxes.
        x_label (str, optional): Label for the X axis. Defaults to 'Dataset'.
        y_label (str, optional): Label for the Y axis. Defaults to 'Accuracy'.
        y_min (float, optional): The minimum value for the Y-axis. Defaults to 0.0.
        y_max (float, optional): The maximum value for the Y-axis. Defaults to 100.0.

    Returns:
        plt: Matplotlib plot object representing the box plot.
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
    Generates an error plot from input data.

    This function is particularly useful for visualizing results from cross-validation
    experiments, as it shows the mean and standard deviation for each NumPy Array of values.

    Args:
        data (dict): A dictionary where the keys correspond to the categories to be
            represented on the X-axis and the values are NumPy Arrays used to construct
            the corresponding error bars.
        x_label (str, optional): Label for the X axis. Defaults to 'Dataset'.
        y_label (str, optional): Label for the Y axis. Defaults to 'Accuracy'.
        y_min (float, optional): The minimum value for the Y-axis. Defaults to 0.0.
        y_max (float, optional): The maximum value for the Y-axis. Defaults to 100.0.
        print_values (bool, optional): Whether or not to print metric values over each
            point. Defaults to False.
        num_decimals (int, optional): Number of decimal places to show if 'print_values'
            is True. Defaults to 2.

    Returns:
        plt: Matplotlib plot object representing the error plot.
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

