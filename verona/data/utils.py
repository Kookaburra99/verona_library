import numpy as np
import pandas as pd


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
            Defaults to XesFields.ACTIVITY_COLUMN.
        lifecycle_id (str, optional): Name of the lifecycle column in the DataFrame.
            Defaults to XesFields.LIFECYCLE_COLUMN.
        drop_lifecycle_column (bool, optional): Delete the lifecycle column after the conversion. Defaults to True.

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


def get_onehot_representation(attribute: np.array, num_elements: int) -> np.array:
    """
    Gets attribute values as labels and converts them to their one-hot representation.

    Args:
        attribute (np.array): Pandas Series or NumPy Array containing the categorical values of the attribute.
        num_elements (int): Integer indicating the number of unique values of the attribute, which is the
                            size of the one-hot vector. If not specified, the vector size is calculated from
                            the number of unique elements in 'attribute'.

    Returns:
        np.array: Pandas Series or NumPy Array (depending on the type of 'attribute') containing the one-hot vectors.
    """


    if not num_elements:
        num_elements = np.unique(attribute).size

    if attribute.ndim > 1:
        attribute = attribute.flatten()

    onehot_attr = np.zeros((attribute.size, num_elements))
    onehot_attr[np.arange(attribute.size), attribute] = 1

    return onehot_attr


def get_labels_from_onehot(onehots: np.array) -> np.array:
    """
    Gets the labels represented in the one-hot vectors passed as input.

    Args:
        onehots (np.array): NumPy Array containing the one-hot vectors.

    Returns:
        np.array: NumPy Array containing the labels extracted from the one-hot vectors.
    """

    return onehots.argmax(axis=-1)
