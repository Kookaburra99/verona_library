import os.path

from verona.data import utils, download, extractor


def test_aggregation_representation():
    user_path = os.path.expanduser("~/.verona_datasets/")
    string, log = download.get_dataset('helpdesk', user_path, 'csv')
    prefixes, targets = extractor.get_prefixes_and_targets(log, 'next_activity', None,
                                                           case_id='case:concept:name', activity_id='concept:name')
    aggr_array = utils.get_aggregation_representation(list(prefixes.values())[3], log['concept:name'].unique(),
                                                      relative_freq=True)
    print(aggr_array)
