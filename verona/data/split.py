from pathlib import Path
from typing import Literal, Tuple, List, Union

import pandas as pd
import pm4py
from sklearn.model_selection import KFold

from verona.data.download import DEFAULT_PATH
from verona.data.utils import XesFields


def make_holdout(dataset: Union[str, pd.DataFrame], store_path: str = None, test_size: float = 0.2,
                 val_from_train: float = 0.2,
                 case_column: str = XesFields.CASE_COLUMN) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a given dataset following a holdout scheme (train-validation-test).

    Parameters:
        dataset (str | pd.DataFrame): If string, full path to the dataset to be split. Only csv, xes, and xes.gz
            datasets are allowed. If Pandas DataFrame, the DataFrame containing the dataset.
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

    if type(dataset) == str:
        dataset_name = Path(dataset).stem
        if len(dataset_name.split('.')) == 1:
            dataset_name += '.csv'
        dataset_name, input_extension = dataset_name.split('.')

        if input_extension == "xes":
            df_log = pm4py.read_xes(dataset)
        elif input_extension == "csv":
            df_log = pd.read_csv(dataset)
        else:
            raise ValueError(f'Wrong dataset extension: {input_extension}. '
                             f'Only .csv, .xes and .xes.gz datasets are allowed.')

    elif type(dataset) == pd.DataFrame:
        df_log = dataset

    else:
        raise TypeError(f'Wrong type for parameter dataset: {type(dataset)}. '
                        f'Only str and pd.DataFrame types are allowed.')

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


def make_crossvalidation(dataset: Union[str, pd.DataFrame], store_path: str = None, cv_folds: int = 5,
                         val_from_train: float = 0.2, case_column: str = XesFields.CASE_COLUMN,
                         seed: int = 42) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Split a given dataset following a cross-validation scheme.

    Parameters:
        dataset (str | pd.DataFrame): If string, full path to the dataset to be split. Only csv, xes, and xes.gz
            datasets are allowed. If Pandas DataFrame, the DataFrame containing the dataset.
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

    if type(dataset) == str:
        dataset_name = Path(dataset).stem
        if len(dataset_name.split('.')) == 1:
            dataset_name += '.csv'
        dataset_name, input_extension = dataset_name.split('.')

        if input_extension == "xes":
            df_log = pm4py.read_xes(dataset)
        elif input_extension == "csv":
            df_log = pd.read_csv(dataset)
        else:
            raise ValueError(f'Wrong dataset extension: {input_extension}. '
                             f'Only .csv, .xes and .xes.gz datasets are allowed.')

    elif type(dataset) == pd.DataFrame:
        df_log = dataset

    else:
        raise TypeError(f'Wrong type for parameter dataset: {type(dataset)}. '
                        f'Only str and pd.DataFrame types are allowed.')

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


def __save_split_to_file(cases: list, store_path: str, dataset_name: str,
                         split: Literal['train', 'val', 'test'], fold: int = None) -> pd.DataFrame:
    df_split = pd.concat(cases)

    if fold is not None:
        filename = f'fold{int(fold)}_{split}_{dataset_name}'
    else:
        filename = f'{split}_{dataset_name}'

    full_path = store_path + filename + '.csv'
    df_split.to_csv(full_path)

    return df_split

