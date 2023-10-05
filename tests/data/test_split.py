from barro.data import split


def test_split_holdout():
    return_paths = split.make_holdout('../../bpi2013cp.csv', store_path='../../')
    print(return_paths)


def test_split_crossvalidation():
    return_paths = split.make_crossvalidation('../../bpi2013cp.csv', store_path='../../')
    print(return_paths)
