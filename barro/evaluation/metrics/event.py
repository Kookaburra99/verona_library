import numpy as np
from typing import Literal
from sklearn import metrics
from barro.data.utils import get_labels_from_onehot, get_onehot_representation


def get_accuracy(predictions: np.array, ground_truths: np.array,
                 preds_format: Literal['labels', 'onehot'],
                 gt_format: Literal['labels', 'onehot']) -> (float, int, int):
    """
    Calculates the accuracy score (ratio of correct predictions, total number of correct predicted
    values and total number of predictions). Both predictions and ground truth can be specified as
    labels or one-hot vectors.
    :param predictions: NumPy Array containing the predictions done by the model.
    :param ground_truths: NumPy Array containing the ground truths.
    :param preds_format: Format of the predictions. If 'label', predictions array
    contains the labels of the activities/attributes predicted. If 'onehot', predictions
    array contains the vectors of probabilities predicted, from which the labels are
    internally extracted from the highest value element to calculate the accuracy.
    :param gt_format: Format of the ground truth. If 'label', ground truth array contains
    the labels of the correct activities/attributes. If 'onehot', ground truths array contains
    the one-hot representation of the correct values, from which the labels are internally
    extracted from the highest value element to calculate the accuracy.
    :return: Float between 0 and 1 indicating the ratio of well predicted activities, integer
    indicating the number of correct predictions and integer indicating the total number of predictions.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions).flatten()
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths).flatten()

    accuracy = metrics.accuracy_score(ground_truths, predictions, True)
    correct_preds = metrics.accuracy_score(ground_truths, predictions, False)
    total_preds = predictions.size

    return accuracy, correct_preds, total_preds


def get_fbeta(predictions: np.array, ground_truths: np.array,
              beta: float, average: Literal['micro', 'macro', 'weighted'],
              preds_format: Literal['labels', 'onehot'],
              gt_format: Literal['labels', 'onehot']) -> (float, float, float):
    """
    Calculates the F-beta score (weighted harmonic mean of precision and recall)
    between the predcictions and the ground truth. The 'beta' parameter represents
    the ratio of recall importance to precision importance. Returns the f-beta and
    the precision and recall used for the calculation.
    :param predictions: NumPy Array containing the predictions done by the model.
    :param ground_truths: NumPy Array containing the ground truths.
    :param beta: Float equal to or greater than zero representing the ratio of recall
    importance to precision importance. For example, beta > 1 gives more weight to recall,
    while beta < 1 favors precision.
    :param average: Determines the type of averaging performed on the data. If 'micro',
    calculate metrics globally. If 'macro', calculate metrics for each label. If 'weighted',
    calculate metrics for each label, and find their average weighted by support (the number
    of true instances for each label).
    :param preds_format: Format of the predictions. If 'label', predictions array
    contains the labels of the activities/attributes predicted. If 'onehot', predictions
    array contains the vectors of probabilities predicted, from which the labels are
    internally extracted from the highest value element to calculate the accuracy.
    :param gt_format: Format of the ground truth. If 'label', ground truth array contains
    the labels of the correct activities/attributes. If 'onehot', ground truths array contains
    the one-hot representation of the correct values, from which the labels are internally
    extracted from the highest value element to calculate the accuracy.
    :return: Float between 0 and 1 indicating the f-beta score, float between 0 and 1 indicating
    the precision and float between 0 and 1 indicating the recall.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions).flatten()
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths).flatten()

    f_beta = metrics.fbeta_score(ground_truths, predictions, beta=beta, average=average)
    precision = metrics.precision_score(ground_truths, predictions, average=average)
    recall = metrics.recall_score(ground_truths, predictions, average=average)

    return f_beta, precision, recall


def get_f1_score(predictions: np.array, ground_truths: np.array,
                 average: Literal['micro', 'macro', 'weighted'],
                 preds_format: Literal['labels', 'onehot'],
                 gt_format: Literal['labels', 'onehot']) -> (float, float, float):
    """
    Calculates the F1-Score (harmonic mean of precision and recall)
    between the predictions and the ground truth. It is the equivalent
    of f-beta with 'beta' = 1. Returns the f-score and the precision and recall used
    for the calculation.
    :param predictions: NumPy Array containing the predictions done by the model.
    :param ground_truths: NumPy Array containing the ground truths.
    :param average: Determines the type of averaging performed on the data. If 'micro',
    calculate metrics globally. If 'macro', calculate metrics for each label. If 'weighted',
    calculate metrics for each label, and find their average weighted by support (the number
    of true instances for each label).
    :param preds_format: Format of the predictions. If 'label', predictions array
    contains the labels of the activities/attributes predicted. If 'onehot', predictions
    array contains the vectors of probabilities predicted, from which the labels are
    internally extracted from the highest value element to calculate the accuracy.
    :param gt_format: Format of the ground truth. If 'label', ground truth array contains
    the labels of the correct activities/attributes. If 'onehot', ground truths array contains
    the one-hot representation of the correct values, from which the labels are internally
    extracted from the highest value element to calculate the accuracy.
    :return: Float between 0 and 1 indicating the f1-score, float between 0 and 1 indicating the
    precision and float between 0 and 1 indicating the recall.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions).flatten()
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths).flatten()

    f1_score, precision, recall, _ = metrics.precision_recall_fscore_support(ground_truths, predictions,
                                                                             average=average)

    return f1_score, precision, recall


def get_precision(predictions: np.array, ground_truths: np.array,
                  average: Literal['micro', 'macro', 'weighted'],
                  preds_format: Literal['labels', 'onehot'],
                  gt_format: Literal['labels', 'onehot']) -> float:
    """
    Calculates the precision (ratio tp / (tp + fp) where 'tp' is the number of true positives
    and 'fp' the number of false positives) between the predictions and the ground truth.
    :param predictions: NumPy Array containing the predictions done by the model.
    :param ground_truths: NumPy Array containing the ground truths.
    :param average: Determines the type of averaging performed on the data. If 'micro',
    calculate metrics globally. If 'macro', calculate metrics for each label. If 'weighted',
    calculate metrics for each label, and find their average weighted by support (the number
    of true instances for each label).
    :param preds_format: Format of the predictions. If 'label', predictions array
    contains the labels of the activities/attributes predicted. If 'onehot', predictions
    array contains the vectors of probabilities predicted, from which the labels are
    internally extracted from the highest value element to calculate the accuracy.
    :param gt_format: Format of the ground truth. If 'label', ground truth array contains
    the labels of the correct activities/attributes. If 'onehot', ground truths array contains
    the one-hot representation of the correct values, from which the labels are internally
    extracted from the highest value element to calculate the accuracy.
    :return: Float between 0 and 1 indicating the precision.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions).flatten()
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths).flatten()

    precision = metrics.precision_score(ground_truths, predictions, average=average)

    return precision


def get_recall(predictions: np.array, ground_truths: np.array,
               average: Literal['micro', 'macro', 'weighted'],
               preds_format: Literal['labels', 'onehot'],
               gt_format: Literal['labels', 'onehot']) -> float:
    """
    Calculates the recall (ratio tp / (tp + fn) where 'tp' is the number of true positives
    and 'fn' the number of false negatives) between the predictions and the ground truth.
    :param predictions: NumPy Array containing the predictions done by the model.
    :param ground_truths: NumPy Array containing the ground truths.
    :param average: Determines the type of averaging performed on the data. If 'micro',
    calculate metrics globally. If 'macro', calculate metrics for each label. If 'weighted',
    calculate metrics for each label, and find their average weighted by support (the number
    of true instances for each label).
    :param preds_format: Format of the predictions. If 'label', predictions array
    contains the labels of the activities/attributes predicted. If 'onehot', predictions
    array contains the vectors of probabilities predicted, from which the labels are
    internally extracted from the highest value element to calculate the accuracy.
    :param gt_format: Format of the ground truth. If 'label', ground truth array contains
    the labels of the correct activities/attributes. If 'onehot', ground truths array contains
    the one-hot representation of the correct values, from which the labels are internally
    extracted from the highest value element to calculate the accuracy.
    :return: Float between 0 and 1 indicating the recall.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions).flatten()
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths).flatten()

    recall = metrics.recall_score(ground_truths, predictions, average=average)

    return recall


def get_mcc(predictions: np.array, ground_truths: np.array,
            preds_format: Literal['labels', 'onehot'],
            gt_format: Literal['labels', 'onehot']) -> float:
    """
    Calculates the Matthews correlation coefficient (MCC). The MCC is in essence a
    correlation coefficient value between -1 and +1. A coefficient of +1 represents
    a perfect prediction, 0 an average random prediction and -1 an inverse prediction.
    :param predictions: NumPy Array containing the predictions done by the model.
    :param ground_truths: NumPy Array containing the ground truths.
    :param preds_format: Format of the predictions. If 'label', predictions array
    contains the labels of the activities/attributes predicted. If 'onehot', predictions
    array contains the vectors of probabilities predicted, from which the labels are
    internally extracted from the highest value element to calculate the accuracy.
    :param gt_format: Format of the ground truth. If 'label', ground truth array contains
    the labels of the correct activities/attributes. If 'onehot', ground truths array contains
    the one-hot representation of the correct values, from which the labels are internally
    extracted from the highest value element to calculate the accuracy.
    :return: Float between -1 and +1 indicating the Matthews correlation coefficient.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions).flatten()
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths).flatten()

    mcc = metrics.matthews_corrcoef(ground_truths, predictions)

    return mcc


def get_brier_score(predictions: np.array, ground_truths: np.array,
                    gt_format: Literal['labels', 'onehot']) -> float:
    """
    Calculates the Brier Score Loss adapted to multi-class predictions
    (predict one activity/attribute at a time out of all existing ones).
    As a measure of loss, the closer to 0, the better the predictions,
    while higher values indicate worse predictions.
    :param predictions: NumPy Array of shape (n_samples, n_classes) containing
    the predictions done by the model as probabilities.
    :param ground_truths: NumPy Array containing the ground truths.
    :param gt_format: Format of the ground truth. If 'label', ground truth array contains
    the labels of the correct activities/attributes, from which the one-hot vectors are
    internally extracted. If 'onehot', ground truths array contains
    the one-hot representation of the correct values.
    :return: Float equal or greater than zero indicating the Brier Score Loss.
    Smaller values (close to 0) indicate smaller error (better predictions), and larger
    values indicate larger error (worse predictions).
    """

    if gt_format == 'labels':
        ground_truths = get_onehot_representation(ground_truths, predictions.shape[-1])

    brier_loss = np.mean(np.sum((ground_truths - predictions)**2, axis=-1)).item()

    return brier_loss


