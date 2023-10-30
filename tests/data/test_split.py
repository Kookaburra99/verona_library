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


def test_split_crossvalidation_csv():
    string, log = download.get_dataset('bpi2012a', None, 'csv')
    train_dfs, val_dfs, test_dfs = split.make_crossvalidation(string, store_path=None)
    assert len(train_dfs) == 5
    assert len(val_dfs) == 5
    assert len(test_dfs) == 5

    user_path = os.path.expanduser("~/.verona_datasets/")

    for i in range(5):
        assert os.path.exists(os.path.join(user_path, f'fold{i}_train_bpi2012a.csv'))
        assert os.path.exists(os.path.join(user_path, f'fold{i}_val_bpi2012a.csv'))
        assert os.path.exists(os.path.join(user_path, f'fold{i}_test_bpi2012a.csv'))
        assert isinstance(train_dfs[i], pd.DataFrame)
        assert isinstance(val_dfs[i], pd.DataFrame)
        assert isinstance(test_dfs[i], pd.DataFrame)
def test_split_crossvalidation_xes():
    # TODO: this test hangs
    print("Download")
    string, log = download.get_dataset('bpi2012a', None, 'xes')
    print("Split")
    train_dfs, val_dfs, test_dfs = split.make_crossvalidation(string, store_path=None)
    print("Done")
    assert len(train_dfs) == 5
    assert len(val_dfs) == 5
    assert len(test_dfs) == 5

    user_path = os.path.expanduser("~/.verona_datasets/")

    for i in range(5):
        assert os.path.exists(os.path.join(user_path, f'fold{i}_train_bpi2012a.xes.gz'))
        assert os.path.exists(os.path.join(user_path, f'fold{i}_val_bpi2012a.xes.gz'))
        assert os.path.exists(os.path.join(user_path, f'fold{i}_test_bpi2012a.xes.gz'))
        assert isinstance(train_dfs[i], pd.DataFrame)
        assert isinstance(val_dfs[i], pd.DataFrame)
        assert isinstance(test_dfs[i], pd.DataFrame)


