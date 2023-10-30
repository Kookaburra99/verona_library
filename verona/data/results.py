from dataclasses import dataclass
import numpy as np
from importlib.resources import open_text

import pandas as pd

from enum import Enum

from verona.evaluation.stattests.correlated_t_test import CorrelatedBayesianTTest
from verona.evaluation.stattests.hierarchical import HierarchicalBayesianTest
from verona.evaluation.stattests.plackettluce import PlackettLuceRanking
from verona.evaluation.stattests.signed_rank import BayesianSignedRankTest


@dataclass(frozen=True)
class MetricValue:
    value: str
    parent: str


class AvailableMetrics:
    """
    Container class that holds available metrics for various predictive tasks in process mining.

    This class defines different metric types that can be calculated for different predictive
    tasks in process mining such as predicting the next activity, activity suffix, next timestamp,
    and remaining time.

    Attributes:
        NextActivity: Metrics for the 'Next Activity' prediction task.
        ActivitySuffix: Metrics for the 'Activity Suffix' prediction task.
        NextTimestamp: Metrics for the 'Next Timestamp' prediction task.
        RemainingTime: Metrics for the 'Remaining Time' prediction task.
    """

    class NextActivity:
        """
        Metrics available for the task of predicting the next activity in a process instance.
        """
        ACCURACY = MetricValue("accuracy", "next_activity")
        F1 = MetricValue("f1", "next_activity")
        PRECISION = MetricValue("precision", "next_activity")
        RECALL = MetricValue("recall", "next_activity")
        BRIER_SCORE = MetricValue("brier_score", "next_activity")
        MCC = MetricValue("mcc", "next_activity")

    class ActivitySuffix:
        """
        Metrics available for the task of predicting the suffix (sequence of remaining activities) in a process instance.
        """
        DAMERAU_LEVENSHTEIN = MetricValue("damerau_levenshtein", "suffix")

    class NextTimestamp:
        """
        Metrics available for the task of predicting the next timestamp in a process instance.
        """
        MAE = MetricValue("mae", "next_timestamp")

    class RemainingTime:
        """
        Metrics available for the task of predicting the remaining time for completion of a process instance.
        """
        MAE = MetricValue("mae", "remaining_time")


class MissingResultStrategy(Enum):
    """
    Enum for specifying the strategy to use for handling missing data (NaNs) in the dataset when applying Bayesian models.

    This enum provides options for how to deal with missing data (NaN values) in the dataset when preparing
    data for Bayesian models. Options include deleting the entire dataset associated with the missing data, 
    deleting only the approach (algorithm/method) associated with the missing data, or taking no action.

    Attributes:
        DELETE_DATASET: Delete the entire dataset associated with the missing data.
        DELETE_APPROACH: Delete only the approach (algorithm/method) associated with the missing data.
        NONE: Take no action on the missing data; it must be handled externally or downstream.
    """
    DELETE_DATASET = "delete_dataset",
    DELETE_APPROACH = "delete_approach",
    NONE = "none"


def __evenize_dataset(results: pd.DataFrame, even_strategy: MissingResultStrategy):
    """
    Handle missing data in a dataset according to the specified strategy.

    This function takes a DataFrame containing results from various approaches on different datasets,
    and applies the specified strategy to handle any missing data (NaNs).

    Args:
        results (pd.DataFrame): The DataFrame containing results from different approaches and datasets.
            Each row should represent a dataset, and each column an approach.

        even_strategy (MissingResultStrategy): Enum specifying the strategy to apply for handling missing data.
            - DELETE_DATASET: Removes any row (dataset) that contains at least one NaN value.
            - DELETE_APPROACH: Removes any column (approach) that contains at least one NaN value.
            - NONE: Does not modify the DataFrame and leaves handling of NaN values to downstream processing.

    Returns:
        pd.DataFrame: A new DataFrame with missing data handled according to the specified strategy.

    Raises:
        ValueError: If an unsupported even strategy is passed.
    """
    if even_strategy == MissingResultStrategy.DELETE_DATASET:
        results = results.dropna(how="any")
    elif even_strategy == MissingResultStrategy.DELETE_APPROACH:
        results = results.dropna(axis=1, how="any")
    elif even_strategy == MissingResultStrategy.NONE:
        pass
    else:
        raise ValueError(f"Unsupported even strategy")
    return results


def __load_csv_results(metric: MetricValue):
    if metric.parent == "next_activity":
        resource_path = f'{metric.value}_raw_results.csv'
    elif metric.parent == "suffix":
        resource_path = 'suffix_raw_results.csv'
    elif metric.parent == "next_timestamp":
        resource_path = 'nt_mae_raw_results.csv'
    elif metric.parent == "remaining_time":
        resource_path = 'remaining_time_results.csv'
    else:
        raise ValueError(f"Unsupported metric")

    with open_text('verona.data.csv', resource_path) as f:
        return pd.read_csv(f)



def load_results_hierarchical(approach_1="Tax", approach_2="TACO", metric=AvailableMetrics.NextActivity.ACCURACY,
                              even_strategy=MissingResultStrategy.DELETE_DATASET):
    """
    Load and preprocess the results of two approaches for comparison using a hierarchical test.

    This function fetches the raw results from CSV files based on the selected metric, filters the data for the two
    approaches specified, and handles missing data according to the provided even_strategy.

    Args:
        approach_1 (str): The name of the first approach for which to load results. Default is "Tax".
        approach_2 (str): The name of the second approach for which to load results. Default is "TACO".
        metric (AvailableMetrics): An enum specifying the metric on which the approaches should be compared.
        even_strategy (MissingResultStrategy): Enum specifying the strategy to apply for handling missing data.

    Returns:
        pd.DataFrame, pd.DataFrame, List[str]: Two DataFrames containing the preprocessed results of the two approaches,
        and a list of common dataset names (indices).

    Raises:
        ValueError: If an unsupported metric or even_strategy is passed.
        AssertionError: If the specified approaches are not available in the data.

    Examples:
        >>> approach_1_df, approach_2_df, common_datasets = load_results_hierarchical("Tax", "TACO", metric=AvailableMetrics.NextActivity.ACCURACY, even_strategy=EvenStrategy.DELETE_DATASET)
        >>> print(approach_1_df.head())
        >>> print(approach_2_df.head())
        >>> print(common_datasets)

    Notes:
        - The function assumes that the raw result CSVs are present in a folder named "csv".
        - For the metrics "next_activity" and "suffix", the values are multiplied by 100 so that they are consistent with
        the default rope values.
    """
    results = __load_csv_results(metric)

    available_approaches = results["approach"].unique()
    assert approach_1 in available_approaches, f"Approach {approach_1} not available, available approaches are {available_approaches}"
    assert approach_2 in available_approaches, f"Approach {approach_2} not available, available approaches are {available_approaches}"

    approach_1_df = results[results["approach"] == approach_1]
    approach_1_df = approach_1_df.pivot(index='log', columns='fold', values='accuracy')
    approach_1_df.sort_index(inplace=True)
    approach_1_df.sort_index(axis=1, inplace=True)

    approach_2_df = results[results["approach"] == approach_2]
    approach_2_df = approach_2_df.pivot(index='log', columns='fold', values='accuracy')
    approach_2_df.sort_index(inplace=True)
    approach_2_df.sort_index(axis=1, inplace=True)

    if metric.parent == "next_activity" or metric.parent == "suffix":
        approach_1_df = approach_1_df * 100
        approach_2_df = approach_2_df * 100

    if even_strategy == MissingResultStrategy.DELETE_DATASET:
        common_indices = approach_1_df.index.intersection(approach_2_df.index)
        approach_1_df = approach_1_df.loc[common_indices]
        approach_2_df = approach_2_df.loc[common_indices]
    elif even_strategy == MissingResultStrategy.DELETE_APPROACH:
        raise ValueError("Delete approach not valid for hierarchical tests.")

    return approach_1_df, approach_2_df, approach_1_df.index.to_list()


def load_results_plackett_luce(metric=AvailableMetrics.NextActivity.ACCURACY,
                               even_strategy=MissingResultStrategy.DELETE_DATASET):
    """
    Load and preprocess the results for applying the Plackett-Luce model.

    This function loads a CSV file containing the raw results based on the given metric. It then computes the mean
    result for each pair of (approach, dataset), and finally applies an evenizing strategy to handle missing data,
    if any.

    Args:
        metric (AvailableMetrics): The metric for which results should be loaded. Defaults to `AvailableMetrics.NextActivity.ACCURACY`.
        even_strategy (MissingResultStrategy): Strategy to apply when missing values are encountered. Determines whether
            rows (datasets) or columns (approaches) should be dropped. Defaults to `MissingResultStrategy.DELETE_DATASET`.

    Returns:
        pd.DataFrame: A DataFrame containing the mean results, where each row represents a dataset and each column an approach.
        list: A list of approach names.

    Examples:
        >>> mean_results, approaches = load_results_plackett_luce(AvailableMetrics.NextActivity.ACCURACY, MissingResultStrategy.DELETE_DATASET)
    """
    results = __load_csv_results(metric)

    mean_results = results.groupby(['approach', 'log']).mean().reset_index()
    mean_results.drop(columns=['fold'], inplace=True)
    mean_results = mean_results.pivot(index="log", columns="approach", values="accuracy")

    mean_results = __evenize_dataset(mean_results, even_strategy)

    return mean_results, mean_results.columns.to_list()


def load_results_non_hierarchical(approach_1="Tax", approach_2="TACO", metric=AvailableMetrics.NextActivity.ACCURACY,
                                  even_strategy=MissingResultStrategy.DELETE_DATASET):
    """
    Load and preprocess results for non-hierarchical statistical comparison of two approaches.

    This function initially loads results for all available approaches using the `load_results_plackett_luce` function.
    It then filters the results to include only the specified `approach_1` and `approach_2`, and applies an evenizing
    strategy to handle any missing values.

    Args:
        approach_1 (str): The name of the first approach to compare. Defaults to "Tax".
        approach_2 (str): The name of the second approach to compare. Defaults to "TACO".
        metric (AvailableMetrics): The metric to consider for loading results. Defaults to `AvailableMetrics.NextActivity.ACCURACY`.
        even_strategy (MissingResultStrategy): Strategy to apply when missing values are encountered. Determines whether
            rows (datasets) or columns (approaches) should be dropped. Defaults to `MissingResultStrategy.DELETE_DATASET`.

    Returns:
        np.ndarray: A NumPy array containing the filtered results for `approach_1`.
        np.ndarray: A NumPy array containing the filtered results for `approach_2`.

    Examples:
        >>> results_tax, results_taco = load_results_non_hierarchical("Tax", "TACO", AvailableMetrics.NextActivity.ACCURACY, MissingResultStrategy.DELETE_DATASET)
    """
    results, approaches = load_results_plackett_luce(metric, even_strategy=MissingResultStrategy.NONE)
    results = results.loc[:, [approach_1, approach_2]]
    results = __evenize_dataset(results, even_strategy)
    if metric.parent == "next_activity" or metric.parent == "suffix":
        results = results * 100
    return results[approach_1].to_numpy(), results[approach_2].to_numpy()

if __name__ == "__main__":
    results_1, results_2 = load_results_non_hierarchical(approach_1="Camargo", approach_2="Tax", metric = AvailableMetrics.NextActivity.ACCURACY, even_strategy = MissingResultStrategy.DELETE_DATASET)
    print("Results 1: ", results_1)
    print("Results 2: ", results_2)
    results = BayesianSignedRankTest(results_1, results_2, ["Camargo", "Tax"]).run()
    print(results.posterior_probabilities)

"""
if __name__ == "__main__":
    results_1, results_2 = get_results_non_hierarchical(approach_1="Camargo", approach_2="Tax", metric = AvailableMetrics.NextActivity.ACCURACY, even_strategy = EvenStrategy.DELETE_DATASET)
    print("Results 1: ", results_1)
    print("Results 2: ", results_2)
    dpos, qpos, posterior_probs = CorrelatedBayesianTTest(results_1, results_2, ["Camargo", "Tax"]).run(rho=0.2)
    print(dpos)
    print(qpos)
    print(posterior_probs)
"""

"""
if __name__ == "__main__":
    results, approaches = load_results_plackett_luce(metric=AvailableMetrics.NextActivity.ACCURACY, even_strategy=MissingResultStrategy.DELETE_DATASET)

    print(results)
    print(approaches)
    plackett_luce = PlackettLuceRanking(results, approaches)
    results = plackett_luce.run(n_chains=10, num_samples=30000, mode="max")
    plot = results.plot_posteriors(save_path=None)
"""

"""
if __name__ == "__main__":
    approach_1_df, approach_2_df, datasets = load_results_hierarchical(approach_1="Camargo", approach_2="Tax",
                                                                       metric=AvailableMetrics.NextActivity.ACCURACY)
    print("Camargo:")
    print(approach_1_df)
    print("Tax: ")
    print(approach_2_df)

    results = HierarchicalBayesianTest(approach_1_df, approach_2_df, ["Camargo", "Tax"], datasets).run()
    print("Global wins: ", results.global_wins)
    print("Posterior distribution: ", results.posterior_distribution)
    print("Per dataset: ", results.per_dataset)
    print("Glboal sign: ", results.global_sign)
"""