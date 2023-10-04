import os
import pm4py
import pandas as pd
from typing import Literal
from pathlib import Path
from sklearn.model_selection import KFold


class XesFields:
    """
    Common xes fields that may be present in a xes log.
    """
    CASE_COLUMN = "case:concept:name"
    ACTIVITY_COLUMN = "concept:name"
    TIMESTAMP_COLUMN = "time:timestamp"
    LIFECYCLE_COLUMN = "lifecycle:transition"
    RESOURCE_COLUMN = "org:resource"


class DataFrameFields:
    """
    Common column names that may be present in a csv log.
    """
    CASE_COLUMN = "CaseID"
    ACTIVITY_COLUMN = "Activity"
    TIMESTAMP_COLUMN = "Timestamp"
    RESOURCE_COLUMN = "Resource"


def make_holdout(dataset_path: str, store_path: str = None, test_size: float = 0.2,
                 val_from_train: float = 0.2, case_column: str = XesFields.CASE_COLUMN) -> list[str]:
    """
    Split a given dataset following a holdout scheme (train-validation-test).
    :param dataset_path: Full path to the dataset to be splitted.
    Only csv, xes and xes.gz datasets are allowed.
    :param store_path: Path where the splits will be stored.
    If not specified, it is stored in the current working directory.
    :param test_size: Float value, between 0 and 1 (both excluded), indicating the percentage
    of traces reserved for the test partition.
    Default: 0.2.
    :param val_from_train: Float value, between 0 and 1 (0 included, 1 excluded), indicating the percentage
    of traces reserved for the validation partition within the cases of the train partition.
    Default: 0.2.
    :param case_column: Name of the case identifier in the original dataset file.
    Default: 'case:concept:name'.
    :return: List containing the paths to the stored splits.
    """

    dataset_name = Path(dataset_path).stem
    if len(dataset_name.split('.')) == 1:
        dataset_name += '.csv'
    dataset_name, input_extension = dataset_name.split('.')

    if input_extension == "xes":
        df_log = pm4py.read_xes(dataset_path)
    elif input_extension == "csv":
        df_log = pd.read_csv(dataset_path)
    else:
        raise ValueError(f'Wrong dataset extension: {input_extension}. '
                         f'Only .csv, .xes and .xes.gz datasets are allowed.')

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
        store_path = os.getcwd()

    return_paths = []

    train_path = __save_split_to_file(train_cases, store_path, dataset_name, 'train')
    return_paths.append(train_path)

    if val_from_train != 0:
        val_path = __save_split_to_file(val_cases, store_path, dataset_name, 'val')
        return_paths.append(val_path)

    test_path = __save_split_to_file(test_cases, store_path, dataset_name, 'test')
    return_paths.append(test_path)

    return return_paths


def make_crossvalidation(dataset_path: str, store_path: str = None, cv_folds: int = 5,
                         val_from_train: float = 0.2, case_column: str = XesFields.CASE_COLUMN,
                         seed: int = 42) -> list[str]:
    """
    Split a given dataset following a cross-validation scheme.
    :param dataset_path: Full path to the dataset to be splitted.
    Only csv, xes and xes.gz datasets are allowed.
    :param store_path: Path where the splits will be stored.
    If not specified, it is stored in the current working directory.
    :param cv_folds: Number of folds for the cross-validation split.
    :param val_from_train: Float value, between 0 and 1 (0 included, 1 excluded), indicating the percentage
    of traces reserved for the validation partition within the cases of the train partition.
    Default: 0.2.
    :param case_column: Name of the case identifier in the original dataset file.
    Default: 'case:concept:name'.
    :param seed: Set a seed for reproducibility.
    :return: List containing the paths to the stored splits.
    """

    dataset_name = Path(dataset_path).stem
    if len(dataset_name.split('.')) == 1:
        dataset_name += '.csv'
    dataset_name, input_extension = dataset_name.split('.')

    if input_extension == "xes":
        df_log = pm4py.read_xes(dataset_path)
    elif input_extension == "csv":
        df_log = pd.read_csv(dataset_path)
    else:
        raise ValueError(f'Wrong dataset extension: {input_extension}. '
                         f'Only .csv, .xes and .xes.gz datasets are allowed.')

    unique_case_ids = list(df_log[case_column].unique())
    kfold = KFold(n_splits=cv_folds, random_state=42, shuffle=True)
    indexes = sorted(unique_case_ids)
    splits = kfold.split(indexes)

    return_paths = []
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
        return_paths.append(train_path)

        if val_from_train != 0:
            val_path = __save_split_to_file(val_cases, store_path, dataset_name, 'val', fold)
            return_paths.append(val_path)

        test_path = __save_split_to_file(test_cases, store_path, dataset_name, 'test', fold)
        return_paths.append(test_path)

        fold += 1

    return return_paths


def __save_split_to_file(cases: list, store_path: str, dataset_name: str,
                         split: Literal['train', 'val', 'test'], fold: int = None) -> str:
    df_split = pd.concat(cases)

    if fold is not None:
        filename = f'fold{int(fold)}_{split}_{dataset_name}'
    else:
        filename = f'{split}_{dataset_name}'

    full_path = store_path + filename + '.csv'
    df_split.to_csv(full_path)

    return full_path

