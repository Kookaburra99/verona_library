import os

import pytest

from verona.data import download


def test_get_available_datasets():
    list_datasets = download.get_available_datasets()
    assert list_datasets # check not empty
    assert "bpi2012" in list_datasets

def test_get_dataset_3():
    string, log = download.get_dataset('bpi2013inc', None, 'csv')
    assert string == os.path.expanduser(os.path.join(download.DEFAULT_PATH, 'bpi2013inc.csv'))
    assert log is not None

def test_get_invalid_format():
    # Invalid format
    with pytest.raises(ValueError):
        download.get_dataset('bpi2012', '../../', 'invalid')

def test_get_invalid_dataset():
    # Invalid dataset name
    with pytest.raises(ValueError):
        download.get_dataset('invalid', '../../', 'xes')

