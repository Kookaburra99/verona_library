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
                                  reduction='mean', font_size=30, print_values=True)
    plt.show()

    data = pd.DataFrame({
        'Tax': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'Camargo': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'Di Mauro': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })
    plt = metrics.bar_plot_metric(data, x_label='Author', y_label='Accuracy',
                                  reduction='median', y_min=50, y_max=90, font_size=30, print_values=True)
    plt.show()


def test_line_plot_metric():
    data = pd.DataFrame({
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })

    plt = metrics.line_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                   reduction='mean', font_size=30, print_values=True)
    plt.show()


def test_box_plot_metric():
    data = pd.DataFrame({
        'Helpdesk': np.array([77.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 64.2, 73.8, 74.5, 67.5]),
        'Sepsis': np.array([56.7, 65.8, 55.7, 56.4, 64.3])
    })

    plt = metrics.box_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                  y_min=50, y_max=85, font_size=40)
    plt.show()


def test_error_plot_metric():
    data = pd.DataFrame({
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })

    plt = metrics.error_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                    y_min=60, y_max=85, font_size=30, print_values=True)
    plt.show()


def test_plot_metric_by_prefixes_len():
    data = pd.DataFrame({
        '1-prefix': [0.4921, 1414],
        '2-prefix': [0.5169, 1414],
        '3-prefix': [0.5525, 1250],
        '4-prefix': [0.6021, 967],
        '5-prefix': [0.7079, 791],
        '6-prefix': [0.8353, 423],
        '7-prefix': [0.8736, 102],
        '8-prefix': [1, 6]
    })

    plt = metrics.plot_metric_by_prefixes_len(data, 'Accuracy', font_size=40, print_values=False)
    plt.show()
