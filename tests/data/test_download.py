from verona.data import download
import pandas as pd
import os
import pytest

from verona.data.download import DEFAULT_PATH


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

def test_redownload_csv():
    # Redownload
    _, _ = download.get_dataset('bpi2012a', None, 'csv')
    string, log = download.get_dataset('bpi2012a', None, 'csv')
    assert string == os.path.expanduser(os.path.join(os.path.expanduser(DEFAULT_PATH), 'bpi2012a.csv'))
    assert log is not None
    assert isinstance(log, pd.DataFrame)

def test_redownload_xes():
    # Redownload
    _, _ = download.get_dataset('bpi2012a', None, 'xes')
    string, log = download.get_dataset('bpi2012a', None, 'xes')
    assert string == os.path.expanduser(os.path.join(os.path.expanduser(DEFAULT_PATH), 'bpi2012a.xes.gz'))
    assert log is not None
    assert isinstance(log, pd.DataFrame)

