import numpy as np
import pandas as pd

from verona.data import extractor
from verona.evaluation.metrics import utils


def test_get_metric_by_prefix_len():
    dataset = pd.read_csv('../../../Helpdesk.csv')
    prefixes, targets = extractor.get_prefixes_and_targets(dataset, 'next_activity', None,
                                                           'CaseID', activity_id='Activity')

    prefixes = list(prefixes.values())[:10]
    predictions = np.array([1, 2, 1, 3, 0, 2, 3, 1, 1, 2])
    ground_truths = np.array(list(targets.values())[:10])

    df_results = utils.get_metric_by_prefix_len('accuracy', predictions, ground_truths, prefixes,
                                                preds_format='labels', gt_format='labels')

    print(df_results)
