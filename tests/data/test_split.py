import os.path

import pandas as pd

from verona.data import split, download


def test_split_holdout():
    string, log = download.get_dataset('bpi2013inc', None, 'csv')
    train_df, val_df, test_df = split.make_holdout(string, store_path=None)
    assert train_df is not None
    assert isinstance(train_df, pd.DataFrame)
    assert val_df is not None
    assert isinstance(val_df, pd.DataFrame)
    assert test_df is not None
    assert isinstance(test_df, pd.DataFrame)

    user_path = os.path.expanduser("~/.verona_datasets/")
    assert os.path.exists(os.path.join(user_path, 'train_bpi2013inc.csv'))
    assert os.path.exists(os.path.join(user_path, 'val_bpi2013inc.csv'))
    assert os.path.exists(os.path.join(user_path, 'test_bpi2013inc.csv'))


def test_split_crossvalidation():
    string, log = download.get_dataset('bpi2013inc', None, 'csv')
    return_paths = split.make_crossvalidation(string, store_path=None)
    print(return_paths)
