from verona.evaluation.metrics import time
from verona.data.extractor import get_prefixes_and_targets
import numpy as np
import pandas as pd


def test_get_mae():
    dataset = pd.read_csv('../../../Helpdesk.csv')
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    _, targets = get_prefixes_and_targets(dataset, 'remaining_time', prefix_size=2,
                                          case_id='CaseID', timestamp_id='Timestamp')
    targets = np.array(list(targets.values()))[:4]
    predictions = np.array([1292014, 100, 200000, 2165098])

    mae = time.get_mae(predictions, targets, reduction='mean')
    print(mae)

    mae = time.get_mae(predictions, targets, reduction='none')
    print(mae)


def test_get_mse():
    dataset = pd.read_csv('../../../Helpdesk.csv')
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    _, targets = get_prefixes_and_targets(dataset, 'remaining_time', prefix_size=2,
                                          case_id='CaseID', timestamp_id='Timestamp')
    targets = np.array(list(targets.values()))[:4]
    predictions = np.array([1292014, 100, 200000, 2165098])

    mse = time.get_mse(predictions, targets, reduction='mean')
    print(mse)

    mse = time.get_mse(predictions, targets, reduction='none')
    print(mse)
