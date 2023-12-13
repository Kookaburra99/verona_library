from typing import Literal, Union

import numpy as np

from verona.data.utils import get_labels_from_onehot


def get_damerau_levenshtein_score(predictions: list[np.array], ground_truths: list[np.array],
                                  preds_format: Literal['labels', 'onehot'],
                                  gt_format: Literal['labels', 'onehot'],
                                  eoc: Union[str, int] = None) -> float:
    """
    Calculates the Damerau-Levenshtein score between the predictions and the real values.

    The Damerau-Levenshtein distance represents the number of insertions, deletions,
    substitutions, and transpositions required to change the first sequence into the second.
    In this function, the score is normalized by the size of the longest sequence, and the
    value is obtained by subtracting the normalized distance from 1.

    Args:
        predictions (list[np.array]): List containing the predicted suffixes as NumPy Arrays.
        ground_truths (list[np.array]): List containing the ground truth suffixes as NumPy Arrays.
        preds_format (Literal['labels', 'onehot']): Format of the predictions. If ``'label'``,
            the predictions array contains the labels of the activities/attributes predicted.
            If ``'onehot'``, the predictions array contains vectors of probabilities, and the labels
            are internally extracted based on the highest value element for the metric calculation.
        gt_format (Literal['labels', 'onehot']): Format of the ground truth. If ``'label'``,
            the ground truth array contains the labels of the correct activities/attributes.
            If ``'onehot'``, the ground truth array contains the one-hot representation of the
            correct values, and the labels are internally extracted for the metric calculation.
        eoc (Union[str, int], optional): Label of the End-of-Case (EOC) which is an element that
            signifies the end of the trace/suffix.

    Returns:
        float: Damerau-Levenshtein score between 0 and 1. A lower value indicates worse suffix
        prediction, whereas a higher value indicates a prediction closer to the actual suffix.

    Examples:
        >>> ground_truths = [np.array([0, 1, 2, 3, 4])]
        >>> predictions = [np.array([0, 12, 2])]
        >>> dl_score = suffix.get_damerau_levenshtein_score(predictions, ground_truths, preds_format='labels', gt_format='labels')
        >>> print(dl_score)
        0.4
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions)
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths)

    list_dl_scores = []
    for pred, gt in zip(predictions, ground_truths):
        dl_distance, len_preds, len_gts = __damerau_levenshtein_similarity(pred, gt, eoc)
        dl_score = 1 - (dl_distance / max(len_preds, len_gts))
        list_dl_scores.append(dl_score)

    dl_score = np.mean(np.array(list_dl_scores)).item()

    return dl_score


def __damerau_levenshtein_similarity(predictions: np.array, ground_truths: np.array,
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

    for i in list(range(l2 + 1)):
        matrix[i] = list(range(i, i + l1 + 1))

    for i in range(1, l2 + 1):
        for j in range(1, l1 + 1):
            cost = 0 if predictions[j - 1] == ground_truths[i - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,         # Deletion
                               matrix[i][j - 1] + 1,         # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution

            # Check for transposition
            if i > 1 and j > 1 and predictions[j - 1] == ground_truths[i - 2] and \
                    predictions[j - 2] == ground_truths[i - 1]:
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + cost)  # Transposition

    distance = float(matrix[l2][l1])

    return distance, l1, l2
