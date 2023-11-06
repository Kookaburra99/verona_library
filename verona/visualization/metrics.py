import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
import plotly.express as px
from typing import Literal


def bar_plot_metric(data: pd.DataFrame, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                    reduction: Literal['mean', 'max', 'min', 'median'] = None,
                    y_min: float = 0.0, y_max: float = 100.0, font_size: int = 15,
                    print_values: bool = False, num_decimals: int = 2) -> Figure:
    """
    Generates a bar chart from input data.

    Args:
        data (pd.DataFrame): Pandas DataFrame where the keys correspond to the categories to be
            represented on the X-axis and the values are either single numerical values
            or NumPy Arrays. If arrays are used, the `reduction` parameter will be applied.
        x_label (str, optional): Label for the X axis. Defaults to 'Dataset'.
        y_label (str, optional): Label for the Y axis. Defaults to 'Accuracy'.
        reduction (Literal['mean', 'max', 'min', 'median'], optional): The reduction function
            to be applied if the values in the `data` dictionary are NumPy Arrays.
            Defaults to None.
        y_min (float, optional): The minimum value for the Y-axis. Defaults to 0.0.
        y_max (float, optional): The maximum value for the Y-axis. Defaults to 100.0.
        font_size (int, optional): Font size of the text in the plot. Defaults to 15.
        print_values (bool, optional): If True, metric values are printed over each bar.
            Defaults to False.
        num_decimals (int, optional): Number of decimals to display if `print_values` is True.
            Defaults to 2.

    Returns:
        fig: Plotly Figure object representing the bar chart.
    """

    x_values = data.columns.tolist()
    y_values_raw = data.T.values

    if y_values_raw.ndim == 2 and y_values_raw.shape[1] == 1:
        y_values = y_values_raw
    elif y_values_raw.ndim == 2 and y_values_raw.shape[1] > 1:
        y_values = __apply_reduction(y_values_raw, reduction)
    else:
        raise TypeError(f'Incorrect format for values in data DataFrame: {y_values_raw}. '
                        f'Only two dimension DataFrames with one or more values per column are allowed.')

    fig = px.bar(x=x_values, y=y_values, labels={x_label, y_label})
    fig.update_yaxes(range=[y_min, y_max])

    if print_values:
        for i, v in enumerate(y_values):
            fig.add_annotation(
                x=x_values[i],
                y=v + 1,
                text=f'{v:.{num_decimals}f}',
                showarrow=False,
                font=dict(size=font_size)
            )

    fig.update_xaxes(title_text=x_label, tickangle=15, tickfont=dict(size=font_size))
    fig.update_yaxes(title_text=y_label, tickfont=dict(size=font_size))
    return fig


def line_plot_metric(data: pd.DataFrame, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                     reduction: Literal['mean', 'max', 'min', 'median'] = None,
                     y_min: float = 0.0, y_max: float = 100.0, font_size: int = 15,
                     print_values: bool = False, num_decimals: int = 2) -> Figure:
    """
    Generates a line chart from input data.

    Args:
        data (pd.DataFrame): Pandas DataFrame where the keys correspond to the categories to be
            represented on the X-axis and the values are either single numerical values
            or NumPy Arrays. If arrays are used, the `reduction` parameter will be applied.
        x_label (str, optional): Label for the X axis. Defaults to 'Dataset'.
        y_label (str, optional): Label for the Y axis. Defaults to 'Accuracy'.
        reduction (Literal['mean', 'max', 'min', 'median'], optional): The reduction function
            to be applied if the values in the `data` dictionary are NumPy Arrays.
            Defaults to None.
        y_min (float, optional): The minimum value for the Y-axis. Defaults to 0.0.
        y_max (float, optional): The maximum value for the Y-axis. Defaults to 100.0.
        font_size (int, optional): Font size of the text in the plot. Defaults to 15.
        print_values (bool, optional): If True, metric values are printed over each point.
            Defaults to False.
        num_decimals (int, optional): Number of decimals to display if `print_values` is True.
            Defaults to 2.

    Returns:
        fig: Plotly Figure object representing the line chart.
    """

    x_values = data.columns.tolist()
    y_values_raw = data.T.values

    if y_values_raw.ndim == 2 and y_values_raw.shape[1] == 1:
        y_values = y_values_raw
    elif y_values_raw.ndim == 2 and y_values_raw.shape[1] > 1:
        y_values = __apply_reduction(y_values_raw, reduction)
    else:
        raise TypeError(f'Incorrect format for values in data DataFrame: {y_values_raw}. '
                        f'Only two dimension DataFrames with one or more values per column are allowed.')

    fig = px.line(x=x_values, y=y_values, labels={x_label, y_label}, markers=True)
    fig.update_yaxes(range=[y_min, y_max])

    if print_values:
        for i, v in enumerate(y_values):
            fig.add_annotation(
                x=x_values[i],
                y=v + 1,
                text=f'{v:.{num_decimals}f}',
                showarrow=False,
                font=dict(size=font_size)
            )

    fig.update_xaxes(title_text=x_label, tickangle=15, tickfont=dict(size=font_size))
    fig.update_yaxes(title_text=y_label, tickfont=dict(size=font_size))
    return fig


def box_plot_metric(data: pd.DataFrame, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                    y_min: float = 0.0, y_max: float = 100.0, font_size: int = 15) -> Figure:
    """
    Generates a box plot showing the corresponding box for each category.

    Args:
        data (pd.DataFrame): Pandas DataFrame containing the values to be represented in the graph.
            The dictionary keys correspond to the categories to be represented on the X-axis,
            while the value associated with each key is a NumPy Array with the values to build
            the corresponding box.
        x_label (str, optional): Label for the X axis. Defaults to 'Dataset'.
        y_label (str, optional): Label for the Y axis. Defaults to 'Accuracy'.
        y_min (float, optional): The minimum value for the Y-axis. Defaults to 0.0.
        y_max (float, optional): The maximum value for the Y-axis. Defaults to 100.0.
        font_size (int, optional): Font size of the text in the plot. Defaults to 15.

    Returns:
        fig: Plotly Figure object representing the error plot.
    """

    fig = px.box(data, title='Box Plot',
                 labels={y_label, x_label}, range_y=[y_min, y_max])

    fig.update_xaxes(title_text=x_label, tickangle=15, tickfont=dict(size=font_size))
    fig.update_yaxes(title_text=y_label, tickfont=dict(size=font_size))

    return fig


def error_plot_metric(data: pd.DataFrame, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                      y_min: float = 0.0, y_max: float = 100.0, font_size: int = 15,
                      print_values: bool = False, num_decimals: int = 2) -> Figure:
    """
    Generates an error plot from input data.

    This function is particularly useful for visualizing results from cross-validation
    experiments, as it shows the mean and standard deviation for each NumPy Array of values.

    Args:
        data (pd.DataFrame): Pandas DataFrame where the keys correspond to the categories to be
            represented on the X-axis and the values are NumPy Arrays used to construct
            the corresponding error bars.
        x_label (str, optional): Label for the X axis. Defaults to 'Dataset'.
        y_label (str, optional): Label for the Y axis. Defaults to 'Accuracy'.
        y_min (float, optional): The minimum value for the Y-axis. Defaults to 0.0.
        y_max (float, optional): The maximum value for the Y-axis. Defaults to 100.0.
        font_size (int, optional): Font size of the text in the plot. Defaults to 15.
        print_values (bool, optional): Whether to print metric values over each
            point. Defaults to False.
        num_decimals (int, optional): Number of decimal places to show if 'print_values'
            is True. Defaults to 2.

    Returns:
        fig: Plotly Figure object representing the error plot.
    """

    x_values = data.columns.tolist()
    y_values_raw = data.T.values

    if y_values_raw.ndim == 2 and y_values_raw.shape[1] > 1:
        y_values = y_values_raw
    else:
        raise TypeError(f'Incorrect format for values in data DataFrame: {y_values_raw}. '
                        f'Only two dimension DataFrames with two or more values per column are allowed.')

    y_means = __apply_reduction(y_values, 'mean')
    y_stds = __apply_reduction(y_values, 'std')

    fig = px.scatter(x=x_values, y=y_means, error_y=y_stds, labels={x_label, y_label})
    fig.update_yaxes(range=[y_min, y_max])

    if print_values:
        for i, (mean_val, std_val) in enumerate(zip(y_means, y_stds)):
            fig.add_annotation(
                x=x_values[i],
                y=mean_val + std_val + 1,
                text=f'Mean: {mean_val:.{num_decimals}f}<br>Std: {std_val:.{num_decimals}f}',
                showarrow=False,
                font=dict(size=font_size)
            )

    fig.update_xaxes(title_text=x_label, tickangle=15, tickfont=dict(size=font_size))
    fig.update_yaxes(title_text=y_label, tickfont=dict(size=font_size))
    return fig


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
