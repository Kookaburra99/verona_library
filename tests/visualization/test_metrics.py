import numpy as np
import pandas as pd

from verona.visualization import metrics


def test_bar_plot_metric():
    data = pd.DataFrame({
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })

    plt = metrics.bar_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                  reduction='mean', print_values=True)
    plt.show()

    data = pd.DataFrame({
        'Tax': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'Camargo': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'Di Mauro': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })
    plt = metrics.bar_plot_metric(data, x_label='Author', y_label='Accuracy',
                                  reduction='median', y_min=50, y_max=90, print_values=True)
    plt.show()


def test_line_plot_metric():
    data = pd.DataFrame({
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })

    plt = metrics.line_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                   reduction='mean', print_values=True)
    plt.show()


def test_box_plot_metric():
    data = pd.DataFrame({
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })

    plt = metrics.box_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                  y_min=60, y_max=85)
    plt.show()


def test_error_plot_metric():
    data = pd.DataFrame({
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })

    plt = metrics.error_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                    y_min=60, y_max=85, print_values=True)
    plt.show()


def test_plot_metric_by_prefixes_len():
    data = pd.DataFrame({
        '1-prefix': [0.33, 3],
        '2-prefix': [0.25, 4],
        '3-prefix': [0.8, 5],
        '4-prefix': [0.5, 4],
        '5-prefix': [1, 1]
    })

    plt = metrics.plot_metric_by_prefixes_len(data, 'Accuracy', print_values=True)
    plt.show()
