import pandas as pd
import numpy as np
from typing import Literal
from barro.data.utils import DataFrameFields


def get_prefixes_and_targets(dataset: pd.DataFrame,
                             prediction_task: Literal['next_activity', 'activity_suffix',
                                                      'next_timestamp', 'remaining_time',
                                                      'next_attribute', 'attribute_suffix'],
                             prefix_size: int = None,
                             case_id: str = DataFrameFields.CASE_COLUMN,
                             activity_id: str = None,
                             timestamp_id: str = None,
                             attribute_id: str = None) -> (dict[int: pd.DataFrame], dict[int: np.array]):
    """
    Extracts all prefixes of the specified size from the dataset and their corresponding target
    depending on the selected prediction task. If the prefix size is not specified, prefixes of
    all possible sizes are extracted. For the next activity prediction, the target is the activity
    of the next event to the last one in the prefix. For activity suffix prediction, the target is
    the sequence of activities from the next event to the end of the case. For the next timestamp
    prediction, the target is the time difference between the timestamp of the last event in the
    prefix and the timestamp of the next event. For the remaining time prediction, the target is
    the difference between the timestamp of the last event of the prefix and the timestamp of the
    last event of the case. For the next attribute prediction, the target is the attribute value
    for the next event to the last event in the prefix. For attribute suffix prediction, the target
    is the sequence of attribute values from the next event to the end of the case.
    :param dataset: DataFrame containing the dataset from which the prefixes will be extracted.
    :param prediction_task: Prediction task to be performed, which determines the value of the targets.
    Possible values are: 'next_activity' (next activity prediction), 'activity_suffix' (activity suffix
    prediction), 'next_timestamp' (next timestamp prediction), 'remaining_time' (remaining time prediction),
    'next_attribute' (next attribute prediction) and 'attribute_suffix' (attribute prediction suffix').
    :param prefix_size: Size of the extracted prefixes. If not specified, prefixes of all possible sizes
    are extracted.
    :param case_id: Name of the case column in the DataFrame.
    :param activity_id: Name of the activity column in the DataFrame. Only needed if 'next_activity' or
    'activity_suffix' is selected as prediction task.
    :param timestamp_id: Name of the timestamp column in the DataFrame. Only needed if 'next_timestamp' or
    'remaining_time' is selected as prediction task.
    :param attribute_id: Name of the attribute column in the DataFrame. Only needed if 'next_attribute' or
    'attribute_suffix' is selected as prediction task.
    :return: Two dictionaries. The first containing the identifier of the prefixes as keys and the prefixes
    in Pandas DataFrame format as values. The latter containing the identifier of the prefixes as keys and
    corresponding targets as values.
    """

    cases = dataset.groupby(case_id)

    prefixes = dict()
    targets = dict()
    counter = 0
    for _, case in cases:
        case = case.drop(case_id, axis=1)
        case = case.reset_index(drop=True)

        for i in range(1, case.shape[0]):
            if prefix_size and i >= prefix_size:
                prefix = case.iloc[i-prefix_size:i]
                prefixes[counter] = prefix
            elif not prefix_size:
                prefix = case.iloc[:i]
                prefixes[counter] = prefix
            else:
                continue

            if prediction_task == 'next_activity':
                target = __get_next_value(case, i, activity_id)
            elif prediction_task == 'activity_suffix':
                target = __get_value_suffix(case, i, activity_id)
            elif prediction_task == 'next_timestamp':
                target = __get_next_value(case, i, timestamp_id)
            elif prediction_task == 'remaining_time':
                target = __get_remaining_time(case, i, timestamp_id)
            elif prediction_task == 'next_attribute':
                target = __get_next_value(case, i, attribute_id)
            elif prediction_task == 'attribute_suffix':
                target = __get_value_suffix(case, i, attribute_id)
            else:
                target = []
            targets[counter] = target

            counter += 1

    return prefixes, targets


def __get_next_value(case: pd.DataFrame, idx: int, column_id: str) -> np.array:
    next_value = case.loc[idx, column_id]
    return np.array([next_value])


def __get_value_suffix(case: pd.DataFrame, idx: int, column_id: str) -> np.array:
    value_suffix = case.loc[idx:, column_id].values
    return value_suffix


def __get_remaining_time(case: pd.DataFrame, idx: int, timestamp_id) -> np.array:
    if case[timestamp_id].dtype == 'O':
        case[timestamp_id] = pd.to_datetime(case[timestamp_id])

    remaining_time = case.loc[len(case)-1, timestamp_id] - case.loc[idx, timestamp_id]
    return remaining_time.total_seconds()


