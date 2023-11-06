import pandas as pd

from verona.data import extractor


def test_get_prefixes_and_targets():
    dataset = pd.read_csv('../../Helpdesk.csv')

    prefixes, targets = extractor.get_prefixes_and_targets(dataset, 'next_activity', 2,
                                                           'CaseID', activity_id='Activity')
    print(dataset.loc[:10])
    print(prefixes[3])
    print(targets[3])

    prefixes, targets = extractor.get_prefixes_and_targets(dataset, 'activity_suffix', None,
                                                           'CaseID', activity_id='Activity')
    print(dataset.loc[:10])
    print(prefixes[2])
    print(targets[2])

    prefixes, targets = extractor.get_prefixes_and_targets(dataset, 'next_timestamp', 3,
                                                           'CaseID', timestamp_id='Timestamp')
    print(dataset.loc[:10])
    print(prefixes[2])
    print(targets[2])

    prefixes, targets = extractor.get_prefixes_and_targets(dataset, 'remaining_time', None,
                                                           'CaseID', timestamp_id='Timestamp')
    print(dataset.loc[:10])
    print(prefixes[1])
    print(targets[1])

    prefixes, targets = extractor.get_prefixes_and_targets(dataset, 'next_attribute', 3,
                                                           'CaseID', attribute_id='Resource')
    print(dataset.loc[:15])
    print(prefixes[4])
    print(targets[4])

    prefixes, targets = extractor.get_prefixes_and_targets(dataset, 'attribute_suffix', 3,
                                                           'CaseID', attribute_id='Resource')
    print(dataset.loc[:20])
    print(prefixes[4])
    print(targets[4])
