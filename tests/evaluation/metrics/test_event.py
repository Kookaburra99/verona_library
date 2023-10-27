from verona.evaluation.metrics import event

import numpy as np


def test_get_accuracy():
    ground_truth = np.array([1, 3, 4, 0])
    preds_labels = np.array([1, 2, 0, 1])
    accuracy, correct, total = event.get_accuracy(preds_labels, ground_truth,
                                                  preds_format='labels', gt_format='labels')
    print(f'{accuracy} - {correct} - {total}')

    preds_onehot = np.array([[0.2, 0.7, 0.06, 0.04], [0.1, 0.2, 0.6, 0.1],
                             [0.9, 0.05, 0.04, 0.01], [0.1, 0.5, 0.3, 0.1]])
    accuracy, correct, total = event.get_accuracy(preds_onehot, ground_truth,
                                                  preds_format='onehot', gt_format='labels')
    print(f'{accuracy} - {correct} - {total}')


def test_get_fbeta():
    ground_truth = np.array([1, 3, 4, 0])
    preds_labels = np.array([1, 2, 0, 1])

    fbeta, precision, recall = event.get_fbeta(preds_labels, ground_truth, beta=1,
                                               average='micro', preds_format='labels', gt_format='labels')
    print(f'{fbeta} - {precision} - {recall}')

    fbeta, precision, recall = event.get_fbeta(preds_labels, ground_truth, beta=1,
                                               average='macro', preds_format='labels', gt_format='labels')
    print(f'{fbeta} - {precision} - {recall}')

    fbeta, precision, recall = event.get_fbeta(preds_labels, ground_truth, beta=1,
                                               average='weighted', preds_format='labels', gt_format='labels')
    print(f'{fbeta} - {precision} - {recall}')

    fbeta, precision, recall = event.get_fbeta(preds_labels, ground_truth, beta=0.5,
                                               average='weighted', preds_format='labels', gt_format='labels')
    print(f'{fbeta} - {precision} - {recall}')

    fbeta, precision, recall = event.get_fbeta(preds_labels, ground_truth, beta=2,
                                               average='weighted', preds_format='labels', gt_format='labels')
    print(f'{fbeta} - {precision} - {recall}')


def test_get_f1_score():
    ground_truth = np.array([1, 3, 4, 0])
    preds_labels = np.array([1, 2, 0, 1])

    f1, precision, recall = event.get_fbeta(preds_labels, ground_truth, beta=1,
                                            average='micro', preds_format='labels', gt_format='labels')
    print(f'{f1} - {precision} - {recall}')

    f1, precision, recall = event.get_fbeta(preds_labels, ground_truth, beta=1,
                                            average='macro', preds_format='labels', gt_format='labels')
    print(f'{f1} - {precision} - {recall}')

    f1, precision, recall = event.get_fbeta(preds_labels, ground_truth, beta=1,
                                            average='weighted', preds_format='labels', gt_format='labels')
    print(f'{f1} - {precision} - {recall}')


def test_get_precision():
    ground_truth = np.array([1, 3, 4, 0])
    preds_labels = np.array([1, 2, 0, 1])

    precision = event.get_precision(preds_labels, ground_truth, average='micro',
                                    preds_format='labels', gt_format='labels')
    print(f'{precision}')

    precision = event.get_precision(preds_labels, ground_truth, average='macro',
                                    preds_format='labels', gt_format='labels')
    print(f'{precision}')

    precision = event.get_precision(preds_labels, ground_truth, average='weighted',
                                    preds_format='labels', gt_format='labels')
    print(f'{precision}')


def test_get_recall():
    ground_truth = np.array([1, 3, 4, 0])
    preds_labels = np.array([1, 2, 0, 1])

    recall = event.get_recall(preds_labels, ground_truth, average='micro',
                              preds_format='labels', gt_format='labels')
    print(f'{recall}')

    recall = event.get_recall(preds_labels, ground_truth, average='macro',
                              preds_format='labels', gt_format='labels')
    print(f'{recall}')

    recall = event.get_recall(preds_labels, ground_truth, average='weighted',
                              preds_format='labels', gt_format='labels')
    print(f'{recall}')


def test_get_mcc():
    ground_truth = np.array([1, 3, 4, 0])
    preds_labels = np.array([1, 2, 0, 1])

    mcc = event.get_mcc(preds_labels, ground_truth,
                        preds_format='labels', gt_format='labels')
    print(mcc)


def test_get_brier_loss():
    ground_truth = np.array([1, 3, 2, 0])
    preds_onehot = np.array([[0.2, 0.7, 0.06, 0.04], [0.1, 0.2, 0.6, 0.1],
                             [0.9, 0.05, 0.04, 0.01], [0.1, 0.5, 0.3, 0.1]])

    brier_loss = event.get_brier_loss(preds_onehot, ground_truth,
                                      gt_format='labels')
    print(brier_loss)
