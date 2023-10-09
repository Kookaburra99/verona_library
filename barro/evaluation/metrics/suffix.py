import numpy as np
from typing import Literal, Union
from jellyfish._jellyfish import damerau_levenshtein_distance
from barro.data.utils import get_labels_from_onehot


def get_damerau_levenshtein_score(predictions: list[np.array], ground_truths: list[np.array],
                                  preds_format: Literal['labels', 'onehot'],
                                  gt_format: Literal['labels', 'onehot'],
                                  eoc: Union[str, int] = None) -> float:
    """
    Calculates the Damerau-Levenshtein score between the predictions and the real values.
    The Damerau-Levenshtein distance represents the number of insertions, deletions, substitutions
    and transpositions required to change the first sequence into the second. Here, as a score is returned,
    normalize the distance by the size of the longest sequence and subtract 1 minus the result to get the
    final metric.
    :param predictions: List containing the predicted suffixes as NumPy Arrays.
    :param ground_truths: List containing the ground truths suffixes as NumPy Arrays.
    :param preds_format: Format of the predictions. If 'label', predictions array
    contains the labels of the activities/attributes predicted. If 'onehot', predictions
    array contains the vectors of probabilities predicted, from which the labels are
    internally extracted from the highest value element to calculate the metric.
    :param gt_format: Format of the ground truth. If 'label', ground truth array contains
    the labels of the correct activities/attributes. If 'onehot', ground truths array contains
    the one-hot representation of the correct values, from which the labels are internally
    extracted from the highest value element to calculate the metric.
    :param eoc: String or Integer representing the label of the End-of-Case (element that represents
    the end of the trace/suffix).
    :return: Float between 0 and 1, where a lower value represents a worse suffix prediction
    and a higher value represents a suffix prediction closer to the actual suffix.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions)
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths)

    list_dl_scores = []
    for pred, gt in zip(predictions, ground_truths):
        dl_distance, len_preds, len_gts = __levenshtein_similarity(pred, gt, eoc)
        dl_score = 1 - (dl_distance / max(len_preds, len_gts))
        list_dl_scores.append(dl_score)

    dl_score = np.mean(np.array(list_dl_scores)).item()

    return dl_score


def __levenshtein_similarity(predictions: np.array, ground_truths: np.array,
                             code_end: Union[str, int]) -> (float, int, int):
    if code_end:
        try:
            l1 = np.where(predictions == code_end)[0].item()
        except ValueError:
            l1 = predictions.size
        try:
            l2 = np.where(ground_truths == code_end)[0].item()
        except ValueError:
            l2 = ground_truths.size
    else:
        l1 = predictions.size
        l2 = ground_truths.size

    if max(l1, l2) == 0:
        return 1.0

    matrix = [list(range(l1 + 1))] * (l2 + 1)

    for zz in list(range(l2 + 1)):
        matrix[zz] = list(range(zz, zz + l1 + 1))
    for zz in list(range(0,l2)):
        for sz in list(range(0,l1)):
            if predictions[sz] == ground_truths[zz]:
                matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz])
            else:
                matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz] + 1)

    distance = float(matrix[l2][l1])

    return distance, l1, l2
