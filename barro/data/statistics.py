import pandas as pd
from barro.data.utils import DataFrameFields


def get_num_activities(dataset: pd.DataFrame,
                       activity_id: str = DataFrameFields.ACTIVITY_COLUMN) -> int:
    
    """
    Returns the number of unique activities in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        activity_id (str, optional): Name of the activity column in the DataFrame. Default is specified by `DataFrameFields.ACTIVITY_COLUMN`.

    Returns:
        int: The number of unique activities in the dataset.

    Raises:
        ValueError: If the dataset is empty or the activity column does not exist.

    Examples:
        >>> df = pd.DataFrame({'activity': ['A', 'B', 'A', 'C']})
        >>> num_activities = get_num_activities(df)
        >>> print(num_activities)
        3
    """

    return get_num_values(dataset, activity_id)


def get_activities_list(dataset: pd.DataFrame,
                        activity_id: str = DataFrameFields.ACTIVITY_COLUMN) -> list:
    """
    Returns the list of unique activities in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        activity_id (str, optional): Name of the activity column in the DataFrame. Default is specified by `DataFrameFields.ACTIVITY_COLUMN`.

    Returns:
        list: A list containing unique activities in the dataset.

    Raises:
        ValueError: If the dataset is empty or the activity column does not exist.

    Examples:
        >>> df = pd.DataFrame({'activity': ['A', 'B', 'A', 'C']})
        >>> activities_list = get_activities_list(df)
        >>> print(activities_list)
        ['A', 'B', 'C']
    """

    return get_values_list(dataset, activity_id)


def get_num_values(dataset: pd.DataFrame,
                   attribute_id: str) -> int:
    """
    Returns the number of unique values for the specified attribute in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        attribute_id (str): Name of the attribute column in the DataFrame.

    Returns:
        int: The number of unique values for the specified attribute in the dataset.

    Raises:
        ValueError: If the dataset is empty or the attribute column does not exist.

    Examples:
        >>> df = pd.DataFrame({'attribute': [1, 2, 2, 3]})
        >>> num_values = get_num_values(df, 'attribute')
        >>> print(num_values)
        3
    """

    return dataset[attribute_id].nunique()


def get_values_list(dataset: pd.DataFrame,
                    attribute_id: str = DataFrameFields.ACTIVITY_COLUMN) -> list:
    """
    Returns the list of unique values for the specified attribute in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        attribute_id (str, optional): Name of the attribute column in the DataFrame. Default is specified by `DataFrameFields.ACTIVITY_COLUMN`.

    Returns:
        list: The list of unique values for the specified attribute in the dataset.

    Raises:
        ValueError: If the dataset is empty or the attribute column does not exist.

    Examples:
        >>> df = pd.DataFrame({'attribute': [1, 2, 2, 3]})
        >>> values_list = get_values_list(df, 'attribute')
        >>> print(values_list)
        [1, 2, 3]
    """

    return dataset[attribute_id].unique().tolist()


def get_num_cases(dataset: pd.DataFrame,
                  case_id: str = DataFrameFields.CASE_COLUMN) -> int:
    """
    Returns the number of unique cases in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.

    Returns:
        int: The number of unique cases in the dataset.

    Raises:
        ValueError: If the dataset is empty or the case identifier column does not exist.

    Examples:
        >>> df = pd.DataFrame({'case': [1, 1, 2, 2, 3]})
        >>> num_cases = get_num_cases(df, 'case')
        >>> print(num_cases)
        3
    """

    return dataset[case_id].nunique()


def get_max_len_case(dataset: pd.DataFrame,
                     case_id: str = DataFrameFields.CASE_COLUMN) -> int:
    """
    Returns the maximum case length in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.

    Returns:
        int: The maximum case length in the dataset.

    Raises:
        ValueError: If the dataset is empty or the case identifier column does not exist.
    """


    cases = dataset.groupby(case_id)
    return cases[case_id].count().max()


def get_min_len_case(dataset: pd.DataFrame,
                     case_id: str = DataFrameFields.CASE_COLUMN) -> int:
    """
    Returns the minimum case length in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.

    Returns:
        int: The minimum case length in the dataset.

    Raises:
        ValueError: If the dataset is empty or the case identifier column does not exist.
    """
    cases = dataset.groupby(case_id)
    return cases[case_id].count().min()


def get_avg_len_case(dataset: pd.DataFrame,
                     case_id: str = DataFrameFields.CASE_COLUMN) -> float:
    """
    Returns the average case length in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.

    Returns:
        float: The average case length in the dataset.

    Raises:
        ValueError: If the dataset is empty or the case identifier column does not exist.
    """

    cases = dataset.groupby(case_id)
    return cases[case_id].count().mean()


def get_max_duration_case(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                          timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the maximum case temporal duration in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.
        timestamp_id (str, optional): Name of the timestamp column in the DataFrame. Default is specified by `DataFrameFields.TIMESTAMP_COLUMN`.

    Returns:
        pd.Timedelta: The maximum temporal duration of a case in the dataset.

    Raises:
        ValueError: If the dataset is empty, or the case identifier or timestamp columns do not exist.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    first_and_last_timestamp_per_case = cases[timestamp_id].agg(["first", "last"])
    return (first_and_last_timestamp_per_case["last"] - first_and_last_timestamp_per_case["first"]).max()


def get_min_duration_case(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                          timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the minimum case temporal duration in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.
        timestamp_id (str, optional): Name of the timestamp column in the DataFrame. Default is specified by `DataFrameFields.TIMESTAMP_COLUMN`.

    Returns:
        pd.Timedelta: The minimum temporal duration of a case in the dataset.

    Raises:
        ValueError: If the dataset is empty, or the case identifier or timestamp columns do not exist.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    first_and_last_timestamp_per_case = cases[timestamp_id].agg(["first", "last"])
    return (first_and_last_timestamp_per_case["last"] - first_and_last_timestamp_per_case["first"]).min()


def get_avg_duration_case(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                          timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the average case temporal duration in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.
        timestamp_id (str, optional): Name of the timestamp column in the DataFrame. Default is specified by `DataFrameFields.TIMESTAMP_COLUMN`.

    Returns:
        pd.Timedelta: The average temporal duration of a case in the dataset.

    Raises:
        ValueError: If the dataset is empty, or the case identifier or timestamp columns do not exist.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    first_and_last_timestamp_per_case = cases[timestamp_id].agg(["first", "last"])
    return (first_and_last_timestamp_per_case["last"] - first_and_last_timestamp_per_case["first"]).mean()


def get_max_duration_event(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                           timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the maximum event temporal duration in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.
        timestamp_id (str, optional): Name of the timestamp column in the DataFrame. Default is specified by `DataFrameFields.TIMESTAMP_COLUMN`.

    Returns:
        pd.Timedelta: The maximum temporal duration of an event in the dataset.

    Raises:
        ValueError: If the dataset is empty, or the case identifier or timestamp columns do not exist.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    return cases[timestamp_id].diff().max()


def get_min_duration_event(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                           timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the minimum event temporal duration in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.
        timestamp_id (str, optional): Name of the timestamp column in the DataFrame. Default is specified by `DataFrameFields.TIMESTAMP_COLUMN`.

    Returns:
        pd.Timedelta: The minimum temporal duration of an event in the dataset.

    Raises:
        ValueError: If the dataset is empty, or the case identifier or timestamp columns do not exist.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    return cases[timestamp_id].diff().min()


def get_avg_duration_event(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                           timestamp_id: str = DataFrameFields.TIMESTAMP_COLUMN) -> pd.Timedelta:
    """
    Returns the average event temporal duration in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.
        timestamp_id (str, optional): Name of the timestamp column in the DataFrame. Default is specified by `DataFrameFields.TIMESTAMP_COLUMN`.

    Returns:
        pd.Timedelta: The average temporal duration of an event in the dataset.

    Raises:
        ValueError: If the dataset is empty, or the case identifier or timestamp columns do not exist.
    """

    dataset[timestamp_id] = pd.to_datetime(dataset[timestamp_id], format='ISO8601')

    cases = dataset.groupby(case_id)
    return cases[timestamp_id].diff().mean()


def get_num_variants(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                     activity_id: str = DataFrameFields.ACTIVITY_COLUMN) -> int:
    """
    Returns the number of unique cases (different sequences of activities) in the dataset.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
        case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.
        activity_id (str, optional): Name of the activity column in the DataFrame. Default is specified by `DataFrameFields.ACTIVITY_COLUMN`.

    Returns:
        int: The number of variants (cases with different sequences of activities).

    Raises:
        ValueError: If the dataset is empty, or the case identifier or activity columns do not exist.
    """

    dataset[activity_id] = dataset[activity_id].astype(str)

    cases = dataset.groupby(case_id)
    return cases[activity_id].agg("->".join).nunique()


def get_count_variants(dataset: pd.DataFrame, case_id: str = DataFrameFields.CASE_COLUMN,
                       activity_id: str = DataFrameFields.ACTIVITY_COLUMN) -> dict:
    """
     Returns the number of times each variant appears in the dataset.

     Parameters:
         dataset (pd.DataFrame): DataFrame containing the dataset to be analyzed.
         case_id (str, optional): Name of the case identifier column in the DataFrame. Default is specified by `DataFrameFields.CASE_COLUMN`.
         activity_id (str, optional): Name of the activity column in the DataFrame. Default is specified by `DataFrameFields.ACTIVITY_COLUMN`.

     Returns:
         dict: Dictionary where the keys are the variants and the values are the count of occurrences of each variant in the dataset.

     Raises:
         ValueError: If the dataset is empty, or the case identifier or activity columns do not exist.
    """

    dataset[activity_id] = dataset[activity_id].astype(str)

    cases = dataset.groupby(case_id)
    return cases[activity_id].agg("->".join).value_counts().to_dict()
