from typing import Literal, Union

import numpy as np
import pandas as pd

from verona.evaluation.metrics import event, suffix, time


def get_metric_by_prefix_len(metric: Literal['accuracy', 'fbeta', 'f1_score', 'precision', 'recall',
                                             'mcc', 'brier_loss', 'damerau_levenshtein', 'mae', 'mse'],
                             predictions: np.array, ground_truths: np.array, prefixes: list[pd.DataFrame],
                             preds_format: Literal['labels', 'onehot'], gt_format: Literal['labels', 'onehot'],
                             average: Literal['micro', 'macro', 'weighted'] = None, beta: float = None,
                             eoc: Union[str, int] = None) -> pd.DataFrame:
    """
    Calculates the value of the specified metric individually for each prefix size.

    Generates a Pandas DataFrame in which each column represents a prefix size with: 1- its corresponding value
    for the selected metric, 2- the number of prefixes with that length.

    Args:
        metric (Literal['accuracy', 'fbeta', 'f1_score', 'precision', 'recall', 'mcc', 'brier_loss', 'damerau_levenshtein', 'mae', 'mse']): Metric to be calculated.
        predictions (np.array): Array of shape (n_samples, n_classes) containing the predictions done by the
            model as probabilities. The predictions on the array should respect the same order as their respective
            prefixes and their ground_truths.
        ground_truths (np.array): Array containing the ground truths. The grounds truths on the array should respect
            the same order as their respective prefixes and predictions.
        prefixes (list[pd.DataFrame]): List containing the prefixes as Pandas DataFrame. The prefixes on the
            list should respect the same order as their respective predicates and ground_truths.
        preds_format (Literal['labels', 'onehot'], optional): Format of the predictions. ``'label'`` for labels and
            ``'onehot'`` for one-hot vectors.
        gt_format (Literal['labels', 'onehot'], optional): Format of the ground truths. ``'label'`` for labels and
            ``'onehot'`` for one-hot vectors.
        average (Literal['micro', 'macro', 'weighted'], optional): Type of averaging to be performed on data.
            Only needed for ``'fbeta'``, ``'f1_score'``, ``'precision'`` and ``'recall'`` value in metric parameter.
        beta (float, optional): Ratio of recall importance to precision importance. Only needed for ``'fbeta'`` value in
            metric parameter.
        eoc (Union[str, int], optional): Label of the End-of-Case (EOC) which is an element that
            signifies the end of the trace/suffix. Only needed for ``'damerau_levenshtein'`` value in metric parameter.

    Returns:
        df_results: Pandas DataFrame where the columns indicate the size of the prefix and its two values indicate: 1- the value of the metric, 2- the number of prefixes with that size.
    """

    preds_by_lens = {}
    gts_by_lens = {}
    for prefix, pred, gt in zip(prefixes, predictions, ground_truths):
        prefix_len = len(prefix)
        if prefix_len in preds_by_lens:
            preds_by_lens[prefix_len].append(pred)
            gts_by_lens[prefix_len].append(gt)
        else:
            preds_by_lens[prefix_len] = [pred]
            gts_by_lens[prefix_len] = [gt]

    preds_by_lens = dict(sorted(preds_by_lens.items()))
    gts_by_lens = dict(sorted(gts_by_lens.items()))

    dict_results = {}
    for prefix_len in preds_by_lens.keys():
        result = __apply_metric(metric, np.array(preds_by_lens[prefix_len]), np.array(gts_by_lens[prefix_len]),
                                preds_format, gt_format, average, beta, eoc)
        num_prefixes = len(preds_by_lens[prefix_len])

        dict_results[f'{prefix_len}-prefix'] = [result, num_prefixes]

    df_result = pd.DataFrame(dict_results)
    return df_result


def __apply_metric(metric: Literal['accuracy', 'fbeta', 'f1_score', 'precision', 'recall',
                                   'mcc', 'brier_loss', 'damerau_levenshtein', 'mae', 'mse'],
                   predictions: np.array, ground_truths: np.array,
                   preds_format: Literal['labels', 'onehot'], gt_format: Literal['labels', 'onehot'],
                   average: Literal['micro', 'macro', 'weighted'], beta: float,
                   eoc: Union[str, int] = None) -> float:

    if metric == 'accuracy':
        result, _, _ = event.get_accuracy(predictions, ground_truths, preds_format, gt_format)
    elif metric == 'fbeta':
        result, _, _ = event.get_fbeta(predictions, ground_truths, beta, average, preds_format, gt_format)
    elif metric == 'f1_score':
        result, _, _ = event.get_f1_score(predictions, ground_truths, average, preds_format, gt_format)
    elif metric == 'precision':
        result = event.get_precision(predictions, ground_truths, average, preds_format, gt_format)
    elif metric == 'recall':
        result = event.get_recall(predictions, ground_truths, average, preds_format, gt_format)
    elif metric == 'mcc':
        result = event.get_mcc(predictions, ground_truths, preds_format, gt_format)
    elif metric == 'brier_loss':
        result = event.get_brier_loss(predictions, ground_truths, gt_format)
    elif metric == 'damerau_levenshtein':
        result = suffix.get_damerau_levenshtein_score(predictions, ground_truths, preds_format, gt_format, eoc)
    elif metric == 'mae':
        result = time.get_mae(predictions, ground_truths, reduction='mean')
    elif metric == 'mae':
        result = time.get_mse(predictions, ground_truths, reduction='mean')
    else:
        result = 0.0

    return result
