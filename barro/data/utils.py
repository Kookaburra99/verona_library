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
    Convert the attribute column type in the Pandas DataFrame dataset
    to categorical (integer indexes).
    :param attr: Pandas Series of the attribute column in the dataset.
    :return: Pandas Series representing the attribute column with the integer
    indexes instead of the original values, a dictionary with the conversions
    (key: categorical index, value: original value) and the reverse dictionary
    (key: original value, value: categorical index).
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
    Gets real activities by unifying the values in the activity and lifescycle columns,
    like it's done in Rama-Maneiro et al. (2023).
    :param dataset: DataFrame containing the dataset.
    :param activity_id: Name of the activity column in the DataFrame.
    :param lifecycle_id: Name of the lifecycle column in the DataFrame.
    :param drop_lifecycle_column: Delete the lifecycle column after the conversion.
    :return: The dataset, as Pandas DataFrame, updated.
    """

    if lifecycle_id not in dataset:
        raise ValueError(f'Wrong lifecycle identifier: {lifecycle_id} is not a column in the dataframe.')

    dataset.loc[:, activity_id] = dataset[activity_id].astype(str) + '+' + dataset[lifecycle_id].astype(str)

    if drop_lifecycle_column:
        dataset.drop(lifecycle_id, axis=1)

    return dataset
