from dataclasses import dataclass
import numpy as np

import pandas as pd

from enum import Enum

from barro.evaluation.stattests.hierarchical import HierarchicalBayesianTest
from barro.evaluation.stattests.plackettluce import PlackettLuceRanking


@dataclass(frozen=True)
class MetricValue:
    value: str
    parent: str

class AvailableMetrics:
    class NextActivity:
        ACCURACY = MetricValue("accuracy", "next_activity")
        F1 = MetricValue("f1", "next_activity")
        PRECISION = MetricValue("precision", "next_activity")
        RECALL = MetricValue("recall", "next_activity")
        BRIER_SCORE = MetricValue("brier_score", "next_activity")
        MCC = MetricValue("mcc", "next_activity")

    class ActivitySuffix:
        DAMERAU_LEVENSHTEIN = MetricValue("damerau_levenshtein", "suffix")

    class NextTimestamp:
        MAE = MetricValue("mae", "next_timestamp")

    class RemainingTime:
        MAE = MetricValue("mae", "remaining_time")

class EvenStrategy(Enum):
    """
    Event dataset strategy to use for the Plackett-Luce model
    """
    DELETE_DATASET = "delete_dataset",
    DELETE_APPROACH = "delete_approach",
    NONE = "none"



def get_results_hierarchical(approach_1="Tax", approach_2="TACO", metric = AvailableMetrics.NextActivity.ACCURACY):
    if metric.parent == "next_activity":
        results = pd.read_csv(f"csv/{metric.value}_raw_results.csv")
    elif metric.parent == "suffix":
        results = pd.read_csv(f"csv/suffix_raw_results.csv")
    elif metric.parent == "next_timestamp":
        results = pd.read_csv(f"csv/nt_mae_raw_results.csv")
    elif metric.parent == "remaining_time":
        results = pd.read_csv("csv/remaining_time_results.csv")
    else:
        raise ValueError(f"Unsupported metric")

    # TODO: tratar el caso en el que los datasets no son siempre los mismos
    # TODO: tratar el caso en el que haya que multiplicar por 100

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
    return approach_1_df, approach_2_df, approach_1_df.index.to_list()


def get_results_plackett_luce(metric = AvailableMetrics.NextActivity.ACCURACY, even_strategy = EvenStrategy.DELETE_DATASET):
    if metric.parent == "next_activity":
        results = pd.read_csv(f"csv/{metric.value}_raw_results.csv")
    elif metric.parent == "suffix":
        results = pd.read_csv(f"csv/suffix_full_results.csv")
    elif metric.parent == "next_timestamp":
        results = pd.read_csv(f"csv/nt_mae_raw_results.csv")
    elif metric.parent == "remaining_time":
        results = pd.read_csv("csv/remaining_time_results.csv")
    else:
        raise ValueError(f"Unsupported metric")

    mean_results = results.groupby(['approach', 'log']).mean().reset_index()
    mean_results.drop(columns=['fold'], inplace=True)
    mean_results = mean_results.pivot(index="log", columns="approach", values="accuracy")

    if even_strategy == EvenStrategy.DELETE_DATASET:
        mean_results = mean_results.dropna(how="any")
    elif even_strategy == EvenStrategy.DELETE_APPROACH:
        mean_results = mean_results.dropna(axis=1, how="any")
    elif even_strategy == EvenStrategy.NONE:
        pass
    else:
        raise ValueError(f"Unsupported even strategy")

    return mean_results, mean_results.columns.to_list()

if __name__ == "__main__":
    #results, approaches = get_results_plackett_luce(metric=AvailableMetrics.NextActivity.ACCURACY, even_strategy=EvenStrategy.DELETE_DATASET)
    results, approaches = get_results_plackett_luce(metric=AvailableMetrics.ActivitySuffix.DAMERAU_LEVENSHTEIN, even_strategy=EvenStrategy.DELETE_DATASET)

    print(results)
    print(approaches)
    plackett_luce = PlackettLuceRanking(results, approaches)
    expected_prob, expected_rank, posterior = plackett_luce.run(n_chains=10, num_samples=30000, mode="max")
    plot = plackett_luce.plot_posteriors(save_path=None)
    plot.figure.savefig("probabilities_boxplot.png") # This saves the plot in a file



"""
if __name__ == "__main__":
    approach_1_df, approach_2_df, datasets = get_results_hierarchical(approach_1="TACO", approach_2="Hinkka", metric = AvailableMetrics.NextActivity.ACCURACY)
    print("TAX:")
    print(approach_1_df)
    print("TACO: ")
    print(approach_2_df)

    global_wins, posterior_distribution, per_dataset, global_sign, results = HierarchicalBayesianTest(approach_1_df * 100, approach_2_df * 100,
                                                                                                      ["taco", "hinkka"], datasets
                                                                                                       )._run_stan(
        approach_1_df * 100, approach_2_df * 100, rope=[-1, 1], n_chains=10, stan_samples=300000)
    print("Global wins: ", global_wins)
    print("Posterior distribution: ", posterior_distribution)
    print("Per dataset: ", per_dataset)
    print("Glboal sign: ", global_sign)
    print("Results: ", results)
"""