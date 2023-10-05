import pandas as pd
from barro.data.utils import DataFrameFields


def get_num_activities(dataset: pd.DataFrame,
                       activity_id: str = DataFrameFields.ACTIVITY_COLUMN) -> int:
    """
    Returns the number of unique activities in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param activity_id: Name of the activity column in the DataFrame.
    :return: The number of unique activities in the dataset.
    """

    return get_num_values(dataset, activity_id)


def get_activities_list(dataset: pd.DataFrame,
                        activity_id: str = DataFrameFields.ACTIVITY_COLUMN) -> list:
    """
    Returns the list of unique activities in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param activity_id: Name of the activity column in the DataFrame.
    :return: The list of unique activities in the dataset.
    """

    return get_values_list(dataset, activity_id)


def get_num_values(dataset: pd.DataFrame,
                   attribute_id: str) -> int:
    """
    Returns the number of unique values for the specified attribute in the dataset,
    passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param attribute_id: Name of the attribute column in the DataFrame.
    :return: The list of unique values for the attribute in the dataset.
    """

    return dataset[attribute_id].nunique()


def get_values_list(dataset: pd.DataFrame,
                    attribute_id: str = DataFrameFields.ACTIVITY_COLUMN) -> list:
    """
    Returns the list of unique values for the specified attribute in the dataset,
    passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param attribute_id: Name of the attribute column in the DataFrame.
    :return: The list of unique values for the attribute in the dataset.
    """

    return dataset[attribute_id].unique().tolist()


def get_num_cases(dataset: pd.DataFrame,
                  case_id: str = DataFrameFields.CASE_COLUMN) -> int:
    """
    Returns the number of cases in the dataset, passed as Pandas DataFrame
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame
    :return: The number of cases in the dataset
    """

    return dataset[case_id].nunique()


def get_max_len_case(dataset: pd.DataFrame,
                     case_id: str = DataFrameFields.CASE_COLUMN) -> int:
    """
    Returns the maximum case length in the dataset, passed as Pandas DataFrame
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :return: The maximum case length in the dataset.
    """

    cases = dataset.groupby(case_id)
    return cases[case_id].count().max()


def get_min_len_case(dataset: pd.DataFrame,
                     case_id: str = DataFrameFields.CASE_COLUMN) -> int:
    """
    Returns the minimum case length in the dataset, passed as Pandas DataFrame
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :return: The minimum case length in the dataset.
    """

    cases = dataset.groupby(case_id)
    return cases[case_id].count().min()


def get_avg_len_case(dataset: pd.DataFrame,
                     case_id: str = DataFrameFields.CASE_COLUMN) -> float:
    """
    Returns the average case length in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :return: The average case length in the dataset.
    """

    cases = dataset.groupby(case_id)
    return cases[case_id].count().mean()


def get_max_duration_case(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                          timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the maximum case temporal duration in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :param timestamp_id: Name of the timestamp column in the DataFrame.
    :return: The maximum temporal duration of a case in the dataset, in Pandas Timedelta.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    first_and_last_timestamp_per_case = cases[timestamp_id].agg(["first", "last"])
    return (first_and_last_timestamp_per_case["last"] - first_and_last_timestamp_per_case["first"]).max()


def get_min_duration_case(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                          timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the minimum case temporal duration in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :param timestamp_id: Name of the timestamp column in the DataFrame.
    :return: The minimum temporal duration of a case in the dataset, in Pandas Timedelta.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    first_and_last_timestamp_per_case = cases[timestamp_id].agg(["first", "last"])
    return (first_and_last_timestamp_per_case["last"] - first_and_last_timestamp_per_case["first"]).min()


def get_avg_duration_case(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                          timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the average case temporal duration in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :param timestamp_id: Name of the timestamp column in the DataFrame.
    :return: The average temporal duration of a case in the dataset, in Pandas Timedelta.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    first_and_last_timestamp_per_case = cases[timestamp_id].agg(["first", "last"])
    return (first_and_last_timestamp_per_case["last"] - first_and_last_timestamp_per_case["first"]).mean()


def get_max_duration_event(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                           timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the maximum event temporal duration in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :param timestamp_id: Name of the timestamp column in the DataFrame.
    :return: The maximum temporal duration of an event in the dataset, in Pandas Timedelta.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    return cases[timestamp_id].diff().max()


def get_min_duration_event(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                           timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the minimum event temporal duration in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :param timestamp_id: Name of the timestamp column in the DataFrame.
    :return: The minimum temporal duration of an event in the dataset, in Pandas Timedelta.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    return cases[timestamp_id].diff().min()


def get_avg_duration_event(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                           timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the average event temporal duration in the dataset, passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :param timestamp_id: Name of the timestamp column in the DataFrame.
    :return: The average temporal duration of an event in the dataset, in Pandas Timedelta.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    return cases[timestamp_id].diff().mean()


def get_num_variants(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                     activity_id: str = DataFrameFields.ACTIVITY_COLUMN) -> int:
    """
    Returns the number of unique cases (different sequences of activities) in the dataset,
    passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :param activity_id: Name of the activity column in the DataFrame.
    :return: The number of variants (cases with different sequences of activities).
    """

    dataset[activity_id] = dataset[activity_id].astype(str)

    cases = dataset.groupby(case_id)
    return cases[activity_id].agg("->".join).nunique()


def get_count_variants(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                       activity_id: str = DataFrameFields.ACTIVITY_COLUMN) -> dict:
    """
    Returns the number of times each variant appears in the dataset,
    passed as Pandas DataFrame.
    :param dataset: DataFrame containing the dataset to be analyzed.
    :param case_id: Name of the case column in the DataFrame.
    :param activity_id: Name of the activity column in the DataFrame.
    :return: Dictionary where the keys are the variants and the values are the
     count of occurrences of each variant in the dataset.
    """

    dataset[activity_id] = dataset[activity_id].astype(str)

    cases = dataset.groupby(case_id)
    return cases[activity_id].agg("->".join).value_counts().to_dict()
