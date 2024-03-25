from pathlib import Path
from typing import Literal, Tuple, List, Union

import pandas as pd
from sklearn.model_selection import KFold

from verona.data.download import DEFAULT_PATH
from verona.data.utils import XesFields, read_eventlog


def make_temporal_split(dataset: Union[str, pd.DataFrame], dataset_name: str = 'Dataset', store_path: str = None,
                        test_offset: pd.Timedelta = pd.Timedelta(365, 'D'),
                        val_offset: pd.Timedelta = None,
                        timestamp_column: str = XesFields.TIMESTAMP_COLUMN,
                        case_column: str = XesFields.CASE_COLUMN) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a given dataset following a temporal scheme. Traces starting on a date equal to or greater than the date
    of the first trace plus **test_offset** form the test partition. Optionally, traces starting on a date equal to or
    greater than the date of the first trace plus **val_offset** but less than the date of the first trace plus
    **test_offset** form the validation partition. The remaining traces form the training partition.

    Args:
        dataset (str | pd.DataFrame): If string, full path to the dataset to be split. Only csv, xes, and xes.gz
            datasets are allowed. If Pandas DataFrame, the DataFrame containing the dataset.
        dataset_name (str): Name of the dataset.
            Default is ``Dataset``.
        store_path (str, optional): Path where the splits will be stored. Defaults to the DEFAULT_PATH
        test_offset (pd.Timedelta, optional): Time difference with respect to the starting timestamp of the first
            trace, from which any trace with the same or a later starting timestamp is added to the test partition.
        val_offset (pd.Timedelta, optional): Time difference with respect to the start timestamp of the first trace,
            from which any trace with a start timestamp equal to or later, but less than the start timestamp plus
            test_offset, is added to the validation partition.
        timestamp_column (str, optional): Name of the timestamp column in the original dataset file.
            Default is ``XesFields.TIMESTAMP_COLUMN``.
        case_column (str, optional): Name of the case identifier in the original dataset file.
            Default is ``XesFields.CASE_COLUMN``.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns a tuple containing the DataFrames for the train,
        validation, and test splits.

    Raises:
        ValueError: If an invalid value for **test_offset** or **val_offset** is provided.

    Examples:
        >>> train_df, _, test_df = make_temporal_split('path/to/dataset.csv', test_offset=pd.Timedelta(days=730))
    """

    df_log = read_eventlog(dataset, sort_events_in_trace=True, sort_traces=True,
                           timestamp_column=timestamp_column, case_column=case_column)

    start_timestamp = df_log.loc[0, timestamp_column]
    val_timestamp = start_timestamp + val_offset if val_offset else None
    test_timestamp = start_timestamp + test_offset if test_offset else None

    df_groupby = df_log.groupby(case_column)

    if val_timestamp and start_timestamp < val_timestamp < test_timestamp:
        train_cases = df_groupby.filter(lambda case: case[timestamp_column].iloc[0] < val_timestamp)
        val_cases = df_groupby.filter(lambda case: val_timestamp <= case[timestamp_column].iloc[0] < test_timestamp)
        test_cases = df_groupby.filter(lambda case: case[timestamp_column].iloc[0] >= test_timestamp)

    elif val_timestamp is None and start_timestamp < test_timestamp:
        train_cases = df_groupby.filter(lambda case: case[timestamp_column].iloc[0] < test_timestamp)
        val_cases = None
        test_cases = df_groupby.filter(lambda case: case[timestamp_column].iloc[0] >= test_timestamp)

    else:
        raise ValueError(f'Wrong offset values: val_offset={val_offset}, test_offset={test_offset}. '
                         f'Offset values should be positive and the validation offset (if provided) '
                         f'should be lower than the test offset.')

    if not store_path:
        store_path = DEFAULT_PATH

    train_df = __save_split_to_file(train_cases, store_path, dataset_name, 'train')

    if val_offset:
        val_df = __save_split_to_file(val_cases, store_path, dataset_name, 'val')
    else:
        val_df = None

    test_df = __save_split_to_file(test_cases, store_path, dataset_name, 'test')

    return train_df, val_df, test_df


def make_holdout(dataset: Union[str, pd.DataFrame], dataset_name: str = 'Dataset', store_path: str = None,
                 test_size: float = 0.2, val_from_train: float = 0.2,
                 case_column: str = XesFields.CASE_COLUMN) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a given dataset following a holdout scheme (train-validation-test).

    Args:
        dataset (str | pd.DataFrame): If string, full path to the dataset to be split. Only csv, xes, and xes.gz
            datasets are allowed. If Pandas DataFrame, the DataFrame containing the dataset.
        dataset_name (str): Name of the dataset.
            Default is ``Dataset``.
        store_path (str, optional): Path where the splits will be stored. Defaults to the DEFAULT_PATH
        test_size (float, optional): Float value between 0 and 1 (both excluded), indicating the percentage of traces
            reserved for the test partition.
            Default is ``0.2``.
        val_from_train (float, optional): Float value between 0 and 1 (0 included, 1 excluded), indicating the
            percentage of traces reserved for the validation partition within the cases of the training partition.
            Default is ``0.2``.
        case_column (str, optional): Name of the case identifier in the original dataset file.
            Default is ``XesFields.CASE_COLUMN``.

    Note:
        The default values for **test_size** and **val_from_train** are based on the experimental setup from the first
        version of [1].

        [1] Rama-Maneiro, E., Vidal, J. C., & Lama, M. (2021). Deep Learning for Predictive Business Process Monitoring:
        Review and Benchmark. https://arxiv.org/abs/2009.13251v1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns a tuple containing the DataFrames for the train,
        validation, and test splits.

    Raises:
        ValueError: If an invalid value for test_size or val_from_train is provided.

    Examples:
        >>> train_df, val_df, test_df = make_holdout('path/to/dataset.csv', test_size=0.3, val_from_train=0.1)
    """

    df_log = read_eventlog(dataset)

    df_groupby = df_log.groupby(case_column)
    cases = [case for _, case in df_groupby]

    if (0 < val_from_train < 1) and (0 < test_size < 1):
        first_cut = round(len(cases) * (1 - test_size) * (1 - val_from_train))
        second_cut = round(len(cases) * (1 - test_size))

        train_cases = cases[:first_cut]
        val_cases = cases[first_cut:second_cut]
        test_cases = cases[second_cut:]

    elif val_from_train == 0 and (0 < test_size < 1):
        unique_cut = round(len(cases) * (1 - test_size))
        train_cases = cases[:unique_cut]
        val_cases = None
        test_cases = cases[unique_cut]

    else:
        raise ValueError(f'Wrong split percentages: val_from_train={val_from_train}, test_size={test_size}. '
                         f'val_from_train should be a number between 0 and 1 (0 included, 1 excluded) and '
                         f'test_size should be a number between 0 and 1 (both excluded).')

    if not store_path:
        store_path = DEFAULT_PATH

    train_df = __save_split_to_file(train_cases, store_path, dataset_name, 'train')

    if val_from_train != 0:
        val_df = __save_split_to_file(val_cases, store_path, dataset_name, 'val')
    else:
        val_df = None

    test_df = __save_split_to_file(test_cases, store_path, dataset_name, 'test')

    return train_df, val_df, test_df


def make_crossvalidation(dataset: Union[str, pd.DataFrame], dataset_name: str = 'Dataset', store_path: str = None,
                         cv_folds: int = 5, val_from_train: float = 0.2, case_column: str = XesFields.CASE_COLUMN,
                         seed: int = 42) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Split a given dataset following a cross-validation scheme.

    Args:
        dataset (str | pd.DataFrame): If string, full path to the dataset to be split. Only csv, xes, and xes.gz
            datasets are allowed. If Pandas DataFrame, the DataFrame containing the dataset.
        dataset_name (str): Name of the dataset.
            Default is ``Dataset``.
        store_path (str, optional): Path where the splits will be stored. Defaults to the current working directory.
        cv_folds (int, optional): Number of folds for the cross-validation split. Default is ``5``.
        val_from_train (float, optional): Float value between 0 and 1 (0 included, 1 excluded), indicating the
            percentage of traces reserved for the validation partition within the cases of the training partition.
            Default is ``0.2``.
        case_column (str, optional): Name of the case identifier in the original dataset file.
            Default is ``XesFields.CASE_COLUMN``.
        seed (int, optional): Set a seed for reproducibility.
            Default is ``42``.

    Returns:
        Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]: Returns a tuple containing the lists of
        DataFrames for the train, validation, and test splits.

    Tip:
        Leaving the default values for **cv_folds**, **val_from_train** and **seed** reproduces the expermiental
        setup of [1].

        [1] Rama-Maneiro, E., Vidal, J. C., & Lama, M. (2023). Deep Learning for Predictive Business Process Monitoring:
        Review and Benchmark. IEEE Transactions on Services Computing, 16(1), 739-756. doi:10.1109/TSC.2021.3139807

    Raises:
        ValueError: If an invalid value for cv_folds or val_from_train is provided.

    Examples:
        >>> splits_paths = make_crossvalidation('path/to/dataset.csv')
    """

    df_log = read_eventlog(dataset)

    unique_case_ids = list(df_log[case_column].unique())
    kfold = KFold(n_splits=cv_folds, random_state=seed, shuffle=True)
    indexes = sorted(unique_case_ids)
    splits = kfold.split(indexes)

    train_folds = []
    val_folds = []
    test_folds = []

    fold = 0
    for train_index, test_index in splits:
        if (0 < val_from_train < 1):
            val_cut = round(len(train_index) * (1 - val_from_train))

            val_index = train_index[val_cut:]
            train_index = train_index[:val_cut]

            train_cases = [df_log[df_log[case_column] == train_g] for train_g in train_index]
            val_cases = [df_log[df_log[case_column] == val_g] for val_g in val_index]
            test_cases = [df_log[df_log[case_column] == test_g] for test_g in test_index]

        elif val_from_train == 0:
            train_cases = [df_log[df_log[case_column] == train_g] for train_g in train_index]
            val_cases = None
            test_cases = [df_log[df_log[case_column] == test_g] for test_g in test_index]

        else:
            raise ValueError(f'Wrong split percentage: val_from_train={val_from_train}. '
                             f'val_from_train should be a number between 0 and 1 (0 included, 1 excluded).')

        train_path = __save_split_to_file(train_cases, store_path, dataset_name, 'train', fold)
        train_folds.append(train_path)

        if val_from_train != 0:
            val_path = __save_split_to_file(val_cases, store_path, dataset_name, 'val', fold)
            val_folds.append(val_path)

        test_path = __save_split_to_file(test_cases, store_path, dataset_name, 'test', fold)
        test_folds.append(test_path)

        fold += 1

    return train_folds, val_folds, test_folds


def __save_split_to_file(cases: Union[list, pd.DataFrame], store_path: str, dataset_name: str,
                         split: Literal['train', 'val', 'test'], fold: int = None) -> pd.DataFrame:
    if type(cases) is list:
        df_split = pd.concat(cases)
    else:
        df_split = cases

    if fold is not None:
        filename = f'fold{int(fold)}_{split}_{dataset_name}'
    else:
        filename = f'{split}_{dataset_name}'

    Path(store_path).mkdir(parents=True, exist_ok=True)

    full_path = store_path + filename + '.csv'
    df_split.to_csv(full_path, index=False)

    return df_split

