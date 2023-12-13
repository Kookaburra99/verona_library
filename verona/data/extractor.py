from typing import Literal

import numpy as np
import pandas as pd

from verona.data.utils import DataFrameFields


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
    Extract prefixes and corresponding targets from a given dataset based on the prediction task.

    The function extracts prefixes of the specified or all possible sizes from the dataset,
    and returns targets corresponding to the selected prediction task.

    Args:
        dataset (pd.DataFrame): DataFrame containing the event log.
        prediction_task (Literal['next_activity', 'activity_suffix', 'next_timestamp', 'remaining_time', 'next_attribute', 'attribute_suffix']):
            Specifies the type of prediction task.

            - ``'next_activity'``: Predict the next activity.
            - ``'activity_suffix'``: Predict the remaining sequence of activities.
            - ``'next_timestamp'``: Predict the next event timestamp.
            - ``'remaining_time'``: Predict the remaining time for the case to complete.
            - ``'next_attribute'``: Predict the next attribute.
            - ``'attribute_suffix'``: Predict the remaining sequence of attributes.

        prefix_size (int, optional): Length of the prefix to be used.
            If ``None``, uses all possible sizes.
        case_id (str, optional): Column name for the case identifier. Default is ``DataFrameFields.CASE_COLUMN``.
        activity_id (str, optional): Column name for the activity.
            Needed for 'next_activity' and 'activity_suffix'.
        timestamp_id (str, optional): Column name for the timestamp.
            Needed for 'next_timestamp' and 'remaining_time'.
        attribute_id (str, optional): Column name for the attribute.
            Needed for 'next_attribute' and 'attribute_suffix'.

    Tip:
        Leaving the default values for **prefix_size** reproduces the expermiental setup of [1].

        [1]  Rama-Maneiro, E., Vidal, J. C., & Lama, M. (2023). Deep Learning for Predictive Business Process
        Monitoring: Review and Benchmark. IEEE Transactions on Services Computing, 16(1), 739-756.
        doi:10.1109/TSC.2021.3139807

    Returns:
        Tuple[Dict[int, pd.DataFrame], Dict[int, np.array]]: Returns two dictionaries:
            1. Mapping from prefix size to the DataFrame of prefixes.
            2. Mapping from prefix size to the corresponding targets in NumPy array format.

    Raises:
        ValueError: If the required column for a prediction task is not specified.

    Examples:
        >>> prefixes, targets = get_prefixes_and_targets(df_dataset, 'next_activity', prefix_size=5)
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


