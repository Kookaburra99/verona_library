from verona.data import download


def test_get_available_datasets():
    list_datasets = download.get_available_datasets()
    print(list_datasets)


def test_get_dataset_1():
    string, log = download.get_dataset('bpi2012', '../../', 'xes')

    download.get_dataset('bpi2013cp', '../../', 'csv')

    download.get_dataset('bpi2013inc', '../../', 'both')
