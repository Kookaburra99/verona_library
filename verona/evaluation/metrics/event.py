from typing import Literal

import numpy as np
from sklearn import metrics

from verona.data.utils import get_labels_from_onehot, get_onehot_representation


def get_accuracy(predictions: np.array, ground_truths: np.array,
                 preds_format: Literal['labels', 'onehot'],
                 gt_format: Literal['labels', 'onehot']) -> (float, int, int):
    """
    Calculates the accuracy score, including the ratio of correct predictions,
    total number of correct predicted values, and total number of predictions.
    Both predictions and ground truth can be specified as labels or one-hot vectors.

    Args:
        predictions (np.array): NumPy Array containing the model's predictions.
        ground_truths (np.array): NumPy Array containing the ground truths.
        preds_format (Literal['labels', 'onehot']): Format of the predictions. 'label' for labels and
            'onehot' for one-hot vectors.
        gt_format (Literal['labels', 'onehot']): Format of the ground truths. 'label' for labels and
            'onehot' for one-hot vectors.

    Returns:
        tuple: Float indicating the accuracy ratio, integer for the number of correct predictions,
            and integer for the total number of predictions.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions).flatten()
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths).flatten()

    accuracy = metrics.accuracy_score(ground_truths, predictions, normalize=True)
    correct_preds = metrics.accuracy_score(ground_truths, predictions, normalize=False)
    total_preds = predictions.size

    return accuracy, correct_preds, total_preds


def get_fbeta(predictions: np.array, ground_truths: np.array,
              beta: float, average: Literal['micro', 'macro', 'weighted'],
              preds_format: Literal['labels', 'onehot'],
              gt_format: Literal['labels', 'onehot']) -> (float, float, float):
    """
    Calculates the F-beta score between the predictions and the ground truth.
    The F-beta score is the weighted harmonic mean of precision and recall.

    Args:
        predictions (np.array): NumPy Array containing the model's predictions.
        ground_truths (np.array): NumPy Array containing the ground truths.
        beta (float): Ratio of recall importance to precision importance.
        average (Literal['micro', 'macro', 'weighted']): Type of averaging to be performed on data.
        preds_format (Literal['labels', 'onehot']): Format of the predictions. 'label' for labels and
            'onehot' for one-hot vectors.
        gt_format (Literal['labels', 'onehot']): Format of the ground truths. 'label' for labels and
            'onehot' for one-hot vectors.

    Returns:
        tuple: Float for the F-beta score, float for the precision, and float for the recall.
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
    Calculates the F1-Score, which is the harmonic mean of precision and recall,
    between the predictions and the ground truth. Equivalent to F-beta score with 'beta' = 1.
    Returns the F1-score, precision, and recall used for the calculation.

    Args:
        predictions (np.array): NumPy Array containing the model's predictions.
        ground_truths (np.array): NumPy Array containing the ground truths.
        average (Literal['micro', 'macro', 'weighted']): Type of averaging to be performed on data.
        preds_format (Literal['labels', 'onehot']): Format of the predictions. 'label' for labels and
            'onehot' for one-hot vectors.
        gt_format (Literal['labels', 'onehot']): Format of the ground truths. 'label' for labels and
            'onehot' for one-hot vectors.

    Returns:
        tuple: Float for the F1-score, float for the precision, and float for the recall.
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
    Calculates the precision using the formula \( \frac{{\text{{tp}}}}{{\text{{tp}} + \text{{fp}}}} \)
    where 'tp' is the number of true positives and 'fp' the number of false positives.

    Args:
        predictions (np.array): Array of predictions from the model.
        ground_truths (np.array): Array of ground truth labels.
        average (Literal['micro', 'macro', 'weighted']): Type of averaging performed on the data.
        preds_format (Literal['labels', 'onehot']): Format of the predictions.
        gt_format (Literal['labels', 'onehot']): Format of the ground truths.

    Returns:
        float: Precision score between 0 and 1.
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
    Calculates the recall using the formula \( \frac{{\text{{tp}}}}{{\text{{tp}} + \text{{fn}}}} \)
    where 'tp' is the number of true positives and 'fn' the number of false negatives.

    Args:
        predictions (np.array): Array of predictions from the model.
        ground_truths (np.array): Array of ground truth labels.
        average (Literal['micro', 'macro', 'weighted']): Type of averaging performed on the data.
        preds_format (Literal['labels', 'onehot']): Format of the predictions.
        gt_format (Literal['labels', 'onehot']): Format of the ground truths.

    Returns:
        float: Recall score between 0 and 1.
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
    Calculates the Matthews correlation coefficient (MCC), a value between -1 and +1.
    A coefficient of +1 represents a perfect prediction, 0 an average random prediction,
    and -1 an inverse prediction.

    Args:
        predictions (np.array): Array of predictions from the model.
        ground_truths (np.array): Array of ground truth labels.
        preds_format (Literal['labels', 'onehot']): Format of the predictions.
        gt_format (Literal['labels', 'onehot']): Format of the ground truths.

    Returns:
        float: Matthews Correlation Coefficient, between -1 and +1.
    """

    if preds_format == 'onehot':
        predictions = get_labels_from_onehot(predictions).flatten()
    if gt_format == 'onehot':
        ground_truths = get_labels_from_onehot(ground_truths).flatten()

    mcc = metrics.matthews_corrcoef(ground_truths, predictions)

    return mcc


def get_brier_loss(predictions: np.array, ground_truths: np.array,
                   gt_format: Literal['labels', 'onehot']) -> float:
    """
    Calculates the Brier Score Loss adapted to multi-class predictions.
    The formula for the Brier Score Loss is \[ \text{BSL} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2 \]
    where \( f_i \) is the predicted probability for the true class for observation \( i \),
    \( o_i \) is the actual outcome for observation \( i \) (1 if true class, 0 otherwise),
    and \( N \) is the total number of observations.

    As a measure of loss, the closer to 0, the better the predictions, while higher values
    indicate worse predictions.

    Args:
        predictions (np.array): Array of shape (n_samples, n_classes) containing
        the predictions done by the model as probabilities.
        ground_truths (np.array): Array containing the ground truths.
        gt_format (Literal['labels', 'onehot']): Format of the ground truth. If 'label',
        the ground truth array contains the labels of the correct activities/attributes,
        from which the one-hot vectors are internally extracted. If 'onehot',
        the ground truths array contains the one-hot representation of the correct values.

    Returns:
        float: Brier Score Loss, a value equal or greater than zero. Smaller values (close to 0)
        indicate smaller error (better predictions), and larger values indicate larger error
        (worse predictions).
    """

    if gt_format == 'labels':
        ground_truths = get_onehot_representation(ground_truths, predictions.shape[-1])

    brier_loss = np.mean(np.sum((ground_truths - predictions)**2, axis=-1)).item()

    return brier_loss
