from verona.visualization import metrics
import numpy as np


def test_bar_plot_metric():
    data = {
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    }

    plt = metrics.bar_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                  reduction='mean', print_values=True)
    plt.show()

    data = {
        'Tax': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'Camargo': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'Di Mauro': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    }
    plt = metrics.bar_plot_metric(data, x_label='Author', y_label='Accuracy',
                                  reduction='median', y_min=50, y_max=90, print_values=True)
    plt.show()


def test_line_plot_metric():
    data = {
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    }

    plt = metrics.line_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                   reduction='mean', print_values=True)
    plt.show()


def test_box_plot_metric():
    data = {
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    }

    plt = metrics.box_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                  y_min=60, y_max=85)
    plt.show()


def test_error_plot_metric():
    data = {
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    }

    plt = metrics.error_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                    y_min=60, y_max=85, print_values=True)
    plt.show()
