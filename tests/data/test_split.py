from verona.data import split


def test_split_holdout():
    train_df, val_df, test_df = split.make_holdout('../../bpi2013cp.csv', store_path='../../')
    print(return_paths)


def test_split_crossvalidation():
    return_paths = split.make_crossvalidation('../../bpi2013cp.csv', store_path='../../')
    print(return_paths)
