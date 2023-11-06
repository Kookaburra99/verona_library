import numpy as np

from verona.evaluation.metrics import suffix


def test_get_damerau_levenshtein_score():
    ground_truths = [np.array([0, 1, 2, 3, 4])]
    predictions = [np.array([0, 12, 2])]

    dl_score = suffix.get_damerau_levenshtein_score(predictions, ground_truths,
                                                    preds_format='labels', gt_format='labels')
    print(dl_score)

    predictions = [np.array([0, 1, 2, 4, 3])]
    dl_score = suffix.get_damerau_levenshtein_score(predictions, ground_truths,
                                                    preds_format='labels', gt_format='labels')
    print(dl_score)

    ground_truths = [np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3])]
    predictions = [np.array([0, 12, 2]), np.array([0, 4, 5, 3])]
    dl_score = suffix.get_damerau_levenshtein_score(predictions, ground_truths,
                                                    preds_format='labels', gt_format='labels')
    print(dl_score)



def test_get_damerau_levenshtein_score():
    ground_truths = [np.array([0, 1, 2, 3, 4])]
    predictions = [np.array([0, 12, 2])]

    dl_score = suffix.get_damerau_levenshtein_score(predictions, ground_truths,
                                                    preds_format='labels', gt_format='labels')
    print(dl_score)

    predictions = [np.array([0, 1, 2, 4, 3])]
    dl_score = suffix.get_damerau_levenshtein_score(predictions, ground_truths,
                                                    preds_format='labels', gt_format='labels')
    print(dl_score)

    ground_truths = [np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3])]
    predictions = [np.array([0, 12, 2]), np.array([0, 4, 5, 3])]
    dl_score = suffix.get_damerau_levenshtein_score(predictions, ground_truths,
                                                    preds_format='labels', gt_format='labels')
    print(dl_score)
