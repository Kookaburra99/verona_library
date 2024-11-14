from typing import Literal, Tuple, List, Union
import numpy as np
import pandas as pd
import pm4py


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


def read_eventlog(dataset: Union[str, pd.DataFrame], sort_events_in_trace: bool = False,
                  sort_traces: bool = False, timestamp_column: str = XesFields.TIMESTAMP_COLUMN,
                  case_column: str = XesFields.CASE_COLUMN) -> pd.DataFrame:
    """
    Reads the event log and returns it as a Pandas DataFrame. Optionally, temporally sorts the
    events within a case and the cases within the eventlog by their start timestamp.

    Args:
        dataset (str | pd.DataFrame): If string, full path to the dataset to be split. Only csv, xes, and xes.gz
            datasets are allowed. If Pandas DataFrame, the DataFrame containing the dataset.
        sort_events_in_trace (bool, optional): If True, sort the events within each case by their timestamp.
            Default is ``False``.
        sort_traces (bool, optional): If True, sort cases by their start timestamp (the timestamp of their first event).
            Default is ``False``.
        timestamp_column (str, optional): Name of the timestamp column in the eventlog.
            Default is ``XesFields.TIMESTAMP_COLUMN``.
        case_column (str, optional): Name of the case identifier in the eventlog.
            Default is ``XesFields.CASE_COLUMN``.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the eventlog.

    Raises:
        ValueError: If an invalid extension is provided when calling the function with **dataset** as a string.
        TypeError: If **dataset** is neither a string nor a Pandas DataFrame
    """

    if type(dataset) is str:
        filename = dataset.split('/')[-1]
        if len(filename.split('.')) == 1:
            filename += '.csv'
        input_extension = filename.split('.')[-1]
        if input_extension == 'gz':
            input_extension = '.'.join(filename.split('.')[-2:])

        if input_extension == "xes" or input_extension == "xes.gz":
            df_log = pm4py.read_xes(dataset)
        elif input_extension == "csv":
            df_log = pd.read_csv(dataset)
        else:
            raise ValueError(f'Wrong dataset extension: {input_extension}. '
                             f'Only .csv, .xes and .xes.gz datasets are allowed.')

    elif type(dataset) is pd.DataFrame:
        df_log = dataset

    else:
        raise TypeError(f'Wrong type for parameter dataset: {type(dataset)}. '
                        f'Only str and pd.DataFrame types are allowed.')

    if sort_events_in_trace:
        df_log = sort_events(df_log, timestamp_column, case_column)

    if sort_traces:
        df_log = sort_dataset(df_log, timestamp_column, case_column)

    return df_log


def categorize_attribute(attr: pd.Series) -> (pd.Series, dict, dict):
    """
    Convert the attribute column type in the Pandas DataFrame dataset to
    categorical (integer indexes).

    Args:
        attr (pd.Series): Pandas Series of the attribute column in the dataset.

    Returns:
        pd.Series: Pandas Series representing the attribute column with the integer indexes
            instead of the original values.
        dict: A dictionary with the conversions (key: categorical index, value: original value).
        dict: The reverse dictionary (key: original value, value: categorical index).
    """

    uniq_attr = attr.unique()
    attr_dict = {idx: value for idx, value in enumerate(uniq_attr)}
    reverse_dict = {value: key for key, value in attr_dict.items()}
    attr_cat = pd.Series(map(lambda x: reverse_dict[x], attr.values))

    return attr_cat, attr_dict, reverse_dict


def unify_activity_and_lifecycle(dataset: pd.DataFrame, activity_id: str = XesFields.ACTIVITY_COLUMN,
                                 lifecycle_id: str = XesFields.LIFECYCLE_COLUMN,
                                 drop_lifecycle_column: bool = True) -> pd.DataFrame:
    """
    Gets real activities by unifying the values in the activity and lifecycle columns,
    like it's done in [1].

    Args:
        dataset (pd.DataFrame): DataFrame containing the dataset.
        activity_id (str, optional): Name of the activity column in the DataFrame.
            Default is ``XesFields.ACTIVITY_COLUMN``.
        lifecycle_id (str, optional): Name of the lifecycle column in the DataFrame.
            Default is ``XesFields.LIFECYCLE_COLUMN``.
        drop_lifecycle_column (bool, optional): Delete the lifecycle column after the conversion.
            Default is ``True``.

    Returns:
        pd.DataFrame: The dataset, as Pandas DataFrame, updated.

    References:
        [1] Rama-Maneiro, E., Vidal, J. C., & Lama, M. (2023). Deep Learning for Predictive Business Process Monitoring:
            Review and Benchmark. IEEE Transactions on Services Computing, 16(1), 739-756. doi:10.1109/TSC.2021.3139807
    """

    if lifecycle_id not in dataset:
        raise ValueError(f'Wrong lifecycle identifier: {lifecycle_id} is not a column in the dataframe.')

    dataset.loc[:, activity_id] = dataset[activity_id].astype(str) + '+' + dataset[lifecycle_id].astype(str)

    if drop_lifecycle_column:
        dataset.drop(lifecycle_id, axis=1)

    return dataset


def sort_events(dataset: pd.DataFrame, timestamp_column: str = XesFields.TIMESTAMP_COLUMN,
                case_column: str = XesFields.CASE_COLUMN) -> pd.DataFrame:
    """
    Sort events within each case by timestamp.

    Args:
        dataset (pd.DataFrame): DataFrame containing all the events.
        timestamp_column (str, optional): Name of the timestamp column in the DataFrame.
            Default is ``XesFields.TIMESTAMP_COLUMN``.
        case_column (str, optional): Name of the case identifier column in the DataFrame.
            Default is ``XesFields.CASE_COLUMN``.

    Returns:
        pd.DataFrame: The events of each case, as Pandas DataFrame, sorted by timestamp.
    """

    dataset[timestamp_column] = pd.to_datetime(dataset[timestamp_column])

    sorted_events = (dataset.groupby(case_column).apply(lambda case: case.sort_values(by=timestamp_column))
                     .reset_index(drop=True))

    return sorted_events


def sort_dataset(dataset: pd.DataFrame, timestamp_column: str = XesFields.TIMESTAMP_COLUMN,
                 case_column: str = XesFields.CASE_COLUMN) -> pd.DataFrame:
    """
    Sort the cases of the dataset by their first timestamp.

    Args:
        dataset (pd.DataFrame): DataFrame containing all the events.
        timestamp_column (str, optional): Name of the timestamp column in the DataFrame.
            Default is ``XesFields.TIMESTAMP_COLUMN``.
        case_column (str, optional): Name of the case identifier column in the DataFrame.
            Default is ``XesFields.CASE_COLUMN``.

    Returns:
        pd.DataFrame: The cases, as Pandas DataFrame, sorted by their first timestamp.
    """

    dataset[timestamp_column] = pd.to_datetime(dataset[timestamp_column])

    dataset['min_timestamp'] = dataset.groupby(case_column)[timestamp_column].transform('min')

    sorted_dataset = dataset.sort_values(by=['min_timestamp', case_column, timestamp_column])
    sorted_dataset = sorted_dataset.drop(columns='min_timestamp')

    return sorted_dataset


def get_onehot_representation(attribute: np.array, num_elements: int) -> np.array:
    """
    Gets attribute values as labels and converts them to their one-hot representation.

    Args:
        attribute (np.array): NumPy Array containing the values of the categorical attribute. Only numeric
            labels are allowed.
        num_elements (int): Integer indicating the number of unique values of the attribute, which is the
                            size of the one-hot vector. If not specified, the vector size is calculated from
                            the number of unique elements in 'attribute'.

    Returns:
        np.array: NumPy Array containing the one-hot vectors.
    """

    if not num_elements:
        num_elements = np.unique(attribute).size

    if attribute.ndim > 1:
        attribute = attribute.flatten()

    onehot_attr = np.zeros((attribute.size, num_elements))
    onehot_attr[np.arange(attribute.size), attribute] = 1

    return onehot_attr


def get_aggregation_representation(prefix: pd.DataFrame, unique_activities: np.array, numeric_columns: np.array = None,
                                   numeric_aggr_func: Literal['max', 'min', 'avg', 'sum'] = 'avg',
                                   activity_column: str = XesFields.ACTIVITY_COLUMN,
                                   relative_freq: bool = False) -> np.array:
    """
    Gets the aggregation sequence encoding described in [1]. Activities are represented by their frequency
    (absolute or relativea) of occurrence in the prefix. Numerical variables are represented by general statistics
    such as maximum, minimum, mean or sum.

    Args:
        prefix (pd.DataFrame): DataFrame containing the events of the prefix.
        unique_activities (np.array): NumPy Array of unique activities labels.
        numeric_columns (np.array, optional): NumPy Array of names of the numerical columns to be represented.
            If any columns with time data are included, make sure they are correctly converted to numeric value.
        numeric_aggr_func (Literal['max', 'min', 'avg', 'sum']): Statistical function to be used to obtain the
            representative value of the numerical variables.

            - ``'max'``: Uses the maximum value of the numerical attribute in the prefix.
            - ``'min'``: Uses the minimum value of the numerical attribute in the prefix.
            - ``'avg'``: Uses the mean value of the numerical attribute in the prefix.
            - ``'sum'``: Uses the sum of the values of the numerical attribute in the prefix.

            Default is ``'avg'``.

        activity_column (str, optional): Name of the activity column. Only numeric labels are allowed.
            Default is ``XesFields.ACTIVITY_COLUMN``.
        relative_freq (bool, optional): Whether to use absolute frequency (``False``) or relative (``True``)
            to prefix length to represent activities.
            Default is ``False``.

    Returns:
        NumPy Array containing the aggregation representation of the input prefix.

    Raises:
        ValueError: If an invalid value of ``numeric_aggr_func`` is provided.

    References:
        [1] Teinemaa, I., Dumas, M., Rosa, M. L., & Maggi, F. M. (2019). Outcome-oriented predictive process
            monitoring: Review and benchmark. ACM Transactions on Knowledge Discovery from Data (TKDD), 13(2), 1-57.
    """

    activity_counts = prefix[activity_column].value_counts()

    act_freq_array = np.zeros(len(unique_activities))
    for i, activity in enumerate(unique_activities):
        if activity in activity_counts:
            act_freq_array[i] = activity_counts[activity]

    if relative_freq:
        act_freq_array = act_freq_array / len(prefix)

    if numeric_columns:
        numeric_array = np.zeros(len(numeric_columns))
        for i, numeric_column in enumerate(numeric_columns):
            if numeric_aggr_func == 'max':
                numeric_array[i] = prefix[numeric_column].max()
            elif numeric_aggr_func == 'min':
                numeric_array[i] = prefix[numeric_column].min()
            elif numeric_aggr_func == 'avg':
                numeric_array[i] = prefix[numeric_column].mean()
            elif numeric_aggr_func == 'sum':
                numeric_array[i] = prefix[numeric_column].sum()
            else:
                raise ValueError(f'Wrong numeric aggregation function: {numeric_aggr_func}. '
                                 f'Only max, min, avg and sum are allowed.')

        aggr_representation = np.concatenate([act_freq_array, numeric_array])
    else:
        aggr_representation = act_freq_array

    return aggr_representation


def get_labels_from_onehot(onehots: np.array) -> np.array:
    """
    Gets the labels represented in the one-hot vectors passed as input.

    Args:
        onehots (np.array): NumPy Array containing the one-hot vectors.

    Returns:
        np.array: NumPy Array containing the labels extracted from the one-hot vectors.
    """

    return onehots.argmax(axis=-1)
