from dataclasses import dataclass

import pandas as pd

from enum import Enum

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



def get_results_hierarchical(approach="Tax", metric = AvailableMetrics.NextActivity.ACCURACY):
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

    available_approaches = results["approach"].unique()
    assert approach in available_approaches, f"Approach {approach} not available, available approaches are {available_approaches}"

    approach = results[results["approach"] == approach]
    print(approach)
    return approach


def get_results_plackett_luce(predictive_problem="next_activity"):
    pass

if __name__ == "__main__":
    get_results_hierarchical(approach="Tax", metric = AvailableMetrics.NextActivity.ACCURACY)
