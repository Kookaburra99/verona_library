from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def bar_plot_metric(data: pd.DataFrame, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                    reduction: Literal['mean', 'max', 'min', 'median'] = None,
                    y_min: float = 0.0, y_max: float = 100.0, font_size: int = 15,
                    print_values: bool = False, num_decimals: int = 2) -> Figure:
    """
    Generates a bar chart from input data.

    Args:
        data (pd.DataFrame): Pandas DataFrame where the columns name correspond to the categories to be
            represented on the X-axis and the values are either single numerical values
            or NumPy Arrays. If arrays are used, the `reduction` parameter will be applied.
        x_label (str, optional): Label for the X axis. Default is ``'Dataset'``.
        y_label (str, optional): Label for the Y axis. Defaults to ``'Accuracy'``.
        reduction (Literal['mean', 'max', 'min', 'median'], optional): The reduction function
            to be applied if the values in the `data` dictionary are NumPy Arrays.
        y_min (float, optional): The minimum value for the Y-axis. Default is ``0.0``.
        y_max (float, optional): The maximum value for the Y-axis. Defaults is ``100.0``.
        font_size (int, optional): Font size of the text in the plot. Default is ``15``.
        print_values (bool, optional): If True, metric values are printed over each bar.
            Default is ``False``.
        num_decimals (int, optional): Number of decimals to display if `print_values` is ``True``.
            Default is ``2``.

    Returns:
        Plotly Figure: ``Plotly Figure`` object representing the bar chart.
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

    fig.update_layout(font_size=font_size)
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
        data (pd.DataFrame): Pandas DataFrame where the columns name correspond to the categories to be
            represented on the X-axis and the values are either single numerical values
            or NumPy Arrays. If arrays are used, the `reduction` parameter will be applied.
        x_label (str, optional): Label for the X axis. Default is ``'Dataset'``.
        y_label (str, optional): Label for the Y axis. Default is ``'Accuracy'``.
        reduction (Literal['mean', 'max', 'min', 'median'], optional): The reduction function
            to be applied if the values in the `data` dictionary are NumPy Arrays.
        y_min (float, optional): The minimum value for the Y-axis. Default is ``0.0``.
        y_max (float, optional): The maximum value for the Y-axis. Default is ``100.0``.
        font_size (int, optional): Font size of the text in the plot. Default is ``15` .
        print_values (bool, optional): If ``True``, metric values are printed over each point.
            Default is ``False``.
        num_decimals (int, optional): Number of decimals to display if `print_values` is ``True``.
            Default is ``2``.

    Returns:
        Plotly Figure: ``Plotly Figure`` object representing the line chart.
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
    fig.update_traces(line={'width': font_size/5}, marker={'size': font_size/2})
    fig.update_yaxes(range=[y_min, y_max])

    if print_values:
        for i, v in enumerate(y_values):
            fig.add_annotation(
                x=x_values[i],
                y=v + 2,
                text=f'{v:.{num_decimals}f}',
                showarrow=False,
                font=dict(size=font_size)
            )

    fig.update_layout(font_size=font_size)
    fig.update_xaxes(title_text=x_label, tickangle=15, tickfont=dict(size=font_size))
    fig.update_yaxes(title_text=y_label, tickfont=dict(size=font_size))
    return fig


def box_plot_metric(data: pd.DataFrame, x_label: str = 'Dataset', y_label: str = 'Accuracy',
                    y_min: float = 0.0, y_max: float = 100.0, font_size: int = 15) -> Figure:
    """
    Generates a box plot showing the corresponding box for each category.

    Args:
        data (pd.DataFrame): Pandas DataFrame containing the values to be represented in the graph.
            The columns name correspond to the categories to be represented on the X-axis,
            while the values associated are used to build the corresponding box.
        x_label (str, optional): Label for the X axis. Default is ``'Dataset'``.
        y_label (str, optional): Label for the Y axis. Default is ``'Accuracy'``.
        y_min (float, optional): The minimum value for the Y-axis. Defaults is ``0.0``.
        y_max (float, optional): The maximum value for the Y-axis. Default is ``100.0``.
        font_size (int, optional): Font size of the text in the plot. Default is ``15``.

    Returns:
        Plotly Figure: ``Plotly Figure`` object representing the error plot.
    """

    fig = px.box(data, title='Box Plot',
                 labels={y_label, x_label}, range_y=[y_min, y_max])

    fig.update_layout(font_size=font_size)
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
        data (pd.DataFrame): Pandas DataFrame where the columns name correspond to the categories to be
            represented on the X-axis and the values are used to construct the corresponding error bars.
        x_label (str, optional): Label for the X axis. Default is ``'Dataset'``.
        y_label (str, optional): Label for the Y axis. Default is ``'Accuracy'``.
        y_min (float, optional): The minimum value for the Y-axis. Default is ``0.0``.
        y_max (float, optional): The maximum value for the Y-axis. Default is ``100.0``.
        font_size (int, optional): Font size of the text in the plot. Default is ``15``.
        print_values (bool, optional): Whether to print metric values over each
            point. Default is ``False``.
        num_decimals (int, optional): Number of decimal places to show if `print_values`
            is ``True``. Default is ``2``.

    Returns:
        Plotly Figure: ``Plotly Figure``  object representing the error plot.
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
    fig.update_traces(error_y={'thickness': font_size / 10}, marker={'size': font_size / 2})

    if print_values:
        for i, (mean_val, std_val) in enumerate(zip(y_means, y_stds)):
            fig.add_annotation(
                x=x_values[i],
                y=mean_val + std_val + 1,
                text=f'Mean: {mean_val:.{num_decimals}f}<br>Std: {std_val:.{num_decimals}f}',
                showarrow=False,
                font=dict(size=font_size)
            )

    fig.update_layout(font_size=font_size)
    fig.update_xaxes(title_text=x_label, tickangle=15, tickfont=dict(size=font_size))
    fig.update_yaxes(title_text=y_label, tickfont=dict(size=font_size))
    return fig


def plot_metric_by_prefixes_len(data: pd.DataFrame, metric_label: str = 'Accuracy',
                                font_size: int = 15, print_values: bool = False,
                                num_decimals: int = 2) -> Figure:
    """
    Generates a mixed plot, where the bar chart indicates the number of prefixes of each length and the
    line chart indicates the value of the chosen metric for each prefix length.

    Args:
        data (pd.DataFrame): Pandas DataFrame where the column names indicate the length of the prefixes and
            the associated values indicate 1- the value of the metric and 2- the number of prefixes of that length.
        metric_label (str, optional): Label for the right Y-axis. Default is ``'Accuracy'``.
        font_size (int, optional): Font size of the text in the plot. Default is ``15``.
        print_values (bool, optional): Whether to print metric values over each
            point. Defaults is ``False``.
        num_decimals (int, optional): Number of decimal places to show if 'print_values'
            is True. Default is ``2``.

    Returns:
        PLotly Figure: ``Plotly Figure`` object representing the bar chart and the line chart.
    """

    x_values = data.columns.tolist()
    y_values_metric = data.loc[0].values
    y_values_counts = data.loc[1].values

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar chart trace
    bar_trace = go.Bar(x=x_values, y=y_values_counts, name='Counts')
    fig.add_trace(bar_trace, secondary_y=False)

    # Add line chart trace
    line_trace = go.Scatter(x=x_values, y=y_values_metric, mode='lines+markers', name=metric_label)
    fig.add_trace(line_trace, secondary_y=True)

    # Update line chart properties (color and thickness)
    fig.update_traces(selector=dict(type='scatter'), line=dict(color='red', width=font_size/5), secondary_y=True)

    if print_values:
        for i, v in enumerate(y_values_metric):
            fig.add_annotation(
                x=x_values[i],
                y=v * max(y_values_counts) + max(y_values_counts) / 50,
                text=f'{v:.{num_decimals}f}',
                showarrow=False,
                font=dict(size=font_size, color='black')
            )

    fig.update_layout(title=f'{metric_label} by prefix length', font_size=font_size)
    fig.update_xaxes(title_text='Prefix Length',  tickangle=15, tickfont=dict(size=font_size))
    fig.update_yaxes(range=[0, max(y_values_counts)*1.03], title_text='Counts',
                     secondary_y=False, tickfont=dict(size=font_size))
    fig.update_yaxes(range=[0, max(y_values_metric)*1.03], title_text='Metric',
                     secondary_y=True, tickfont=dict(size=font_size))

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
