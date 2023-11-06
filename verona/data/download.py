import os

import pandas as pd
import requests
import pm4py
from typing import Literal, Tuple
from tqdm import tqdm

# TODO: this should be in a yaml
DEFAULT_PATH = "~/.verona_datasets/"
DATASETS_LIST = {
    'bpi2011': {
        'name': 'BPI Challenge 2011',
        'url': 'https://data.4tu.nl/file/5ea5bb88-feaa-4e6f-a743-6460a755e05b/6f9640f9-0f1e-44d2-9495-ef9d1bd82218'
    },
    'bpi2012': {
        'name': 'BPI Challenge 2012',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/BPI_Challenge_2012.xes.gz?ref_type=heads'
    },
    'bpi2012comp': {
        'name': 'BPI Challenge 2012 Complete',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/BPI_Challenge_2012_Complete.xes.gz?ref_type=heads'
    },
    'bpi2012w': {
        'name': 'BPI Challenge 2012 W',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/BPI_Challenge_2012_W.xes.gz?ref_type=heads'
    },
    'bpi2012wcomp': {
        'name': 'BPI Challenge 2012 W Complete',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/BPI_Challenge_2012_W_Complete.xes.gz?ref_type=heads'
    },
    'bpi2012a': {
        'name': 'BPI Challenge 2012 A',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/BPI_Challenge_2012_A.xes.gz?ref_type=heads'
    },
    'bpi2012o': {
        'name': 'BPI Challenge 2012 O',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/BPI_Challenge_2012_O.xes.gz?ref_type=heads'
    },
    'bpi2013op': {
      'name': 'BPI Challenge 2013 Open Problems',
      'url': 'https://data.4tu.nl/file/7aafbf5b-97ae-48ba-bd0a-4d973a68cd35/0647ad1a-fa73-4376-bdb4-1b253576c3a1'
    },
    'bpi2013cp': {
        'name': 'BPI Challenge 2013 Closed Problems',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/BPI_Challenge_2013_closed_problems.xes.gz?ref_type=heads'
    },
    'bpi2013inc': {
        'name': 'BPI Challenge 2013 Incidents',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/bpi_challenge_2013_incidents.xes.gz?ref_type=heads'
    },
    'bpi2015_1': {
        'name': 'BPI Challenge 2015 Municipality 1',
        'url': 'https://data.4tu.nl/file/6f35269e-4ce7-4bc4-9abb-b3cea04cad00/2c8d5827-3e08-471d-98e2-6ffdec92f958'
    },
    'bpi2015_2': {
        'name': 'BPI Challenge 2015 Municipality 2',
        'url': 'https://data.4tu.nl/file/372d0cad-3fb1-4627-8ea9-51a09923d331/d653a8ec-4cd1-4029-8b61-6cfde4f4a666'
    },
    'bpi2015_3': {
        'name': 'BPI Challenge 2015 Municipality 3',
        'url': 'https://data.4tu.nl/file/d6741425-5f62-4a59-92c5-08bae64b4611/21b574ab-02ba-4dfb-badc-bb46ce0edc44'
    },
    'bpi2015_4': {
        'name': 'BPI Challenge 2015 Municipality 4',
        'url': 'https://data.4tu.nl/file/34216d8a-f054-46d4-bf03-d9352f90967e/68923819-b085-43be-abe2-e084a0f1381f'
    },
    'bpi2015_5': {
        'name': 'BPI Challenge 2015 Municipality 5',
        'url': 'https://data.4tu.nl/file/32b70553-0765-4808-b155-aa5319802c8a/d39e1365-e4b8-4cb8-83d3-0b01cbf6f8c2'
    },
    'bpi2017': {
        'name': 'BPI Challenge 2017',
        'url': 'https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c'
    },
    'bpi2018': {
        'name': 'BPI Challenge 2018',
        'url': 'https://data.4tu.nl/file/443451fd-d38a-4464-88b4-0fc641552632/cd4fd2b8-6c95-47ae-aad9-dc1a085db364'
    },
    'bpi2019': {
        'name': 'BPI Challenge 2019',
        'url': 'https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828'
    },
    'bpi2020domdec': {
        'name': 'BPI Challenge 2020 Domestic Declarations',
        'url': 'https://data.4tu.nl/file/6a0a26d2-82d0-4018-b1cd-89afb0e8627f/6eeb0328-f991-48c7-95f2-35033504036e'
    },
    'bpi2020intdec': {
        'name': 'BPI Challenge 2020 International Declarations',
        'url': 'https://data.4tu.nl/file/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/d45ee7dc-952c-4885-b950-4579a91ef426'
    },
    'bpi2020rfp': {
        'name': 'BPI Challenge 2020 Request For Payment',
        'url': 'https://data.4tu.nl/file/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc/7b1f2e56-e4a8-43ee-9a09-6e64f45a1a98'
    },
    'bpi2020tpd': {
        'name': 'BPI Challenge 2020 Travel Permit Data',
        'url': 'https://data.4tu.nl/file/db35afac-2133-40f3-a565-2dc77a9329a3/12b48cc1-18a8-4089-ae01-7078fc5e8f90'
    },
    'bpi2020ptc': {
        'name': 'BPI Challenge 2020 Prepaid Travel Cost',
        'url': 'https://data.4tu.nl/file/fb84cf2d-166f-4de2-87be-62ee317077e5/612068f6-14d0-4a82-b118-1b51db52e73a'
    },
    'helpdesk': {
        'name': 'Helpdesk',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/Helpdesk.xes.gz?ref_type=heads'
    },
    'sepsis': {
        'name': 'SEPSIS',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/SEPSIS.xes.gz?ref_type=heads'
    },
    'env_permit': {
        'name': 'env_permit',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/env_permit.xes.gz?ref_type=heads'
    },
    'nasa': {
        'name': 'nasa',
        'url': 'https://gitlab.citius.usc.es/efren.rama/pmdlcompararator/-/raw/crossvalidation/raw_datasets/nasa.xes.gz?ref_type=heads'
    }
}


def get_available_datasets():
    """
    Display the list of available datasets from the official repository and return their identifiers.

    This function prints out the list of available datasets along with their names, as defined in the
    `DATASETS_LIST` dictionary.

    Returns:
        list: List of available dataset identifiers.

    Examples:
        >>> available_datasets = get_available_datasets()
        >>> print(available_datasets)
    """

    print(f'Available datasets:')
    for dataset_id in DATASETS_LIST:
        print(f'\t- {dataset_id}: \"{DATASETS_LIST[dataset_id]["name"]}\"')

    dataset_ids_list = list(DATASETS_LIST.keys())

    return dataset_ids_list


def get_dataset(dataset_name: str, store_path: str = None,
                extension: Literal['xes', 'csv', 'both'] = 'xes') -> Tuple[str, pd.DataFrame]:
    """
    Download a specified dataset from the official repository and store it in a designated path.

    This function downloads the dataset in either 'xes.gz' or 'csv' format, based on the 'extension' argument.

    Args:
        dataset_name (str): Identifier of the dataset to download.

        store_path (Optional[str]): The directory path where the dataset will be stored.
            If not specified, the dataset will be stored in the folder ~/.verona_datasets/

        extension (Literal['xes', 'csv', 'both']): The format in which to save the dataset.
            Choose from 'xes' for 'xes.gz' format, 'csv' for 'csv' format, or 'both' to download both formats.
            Default is 'xes'.

    Returns:
        None: The function stores the downloaded dataset in the 'store_path' and returns nothing.

    Examples:
        >>> get_dataset('bpi2012a', store_path=None, extension='csv')
    """

    if extension not in ['xes', 'csv', 'both']:
        raise ValueError("Wrong extension. Choose from 'xes', 'csv', or 'both'.")

    if store_path is None:
        # By default, os does not expand the '~' character to the user home.
        store_path = os.path.expanduser(DEFAULT_PATH)
        print("Creating directory in: ", store_path)
        print(os.listdir("~"))
        if not os.path.exists(store_path):
            os.mkdir(store_path)

    # TODO: add caching mechanism to avoid downloading the same file multiple times
    if dataset_name in DATASETS_LIST:
        return __download_dataset(dataset_name, store_path, extension)
    else:
        raise ValueError(f'Wrong dataset identifier: {dataset_name} is not available. '
                         f'Check the list of available datasets with get_available_datasets()')


def __download_dataset(dataset_name: str, store_path: str,
                       extension: Literal['xes', 'csv', 'both']) -> Tuple[str, pd.DataFrame]:
    dataset_url = DATASETS_LIST[dataset_name]['url']
    response = requests.get(dataset_url, stream=True)

    store_path_xes = os.path.join(store_path, dataset_name + '.xes.gz')
    store_path_csv = None

    if response.status_code == 200:
        file_size = int(response.headers.get('content-length', 0))
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Downloading {dataset_name}') as pbar:
            with open(store_path_xes, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        if extension in ['csv', 'both']:
            store_path_csv = os.path.join(store_path, dataset_name + '.csv')

            log = pm4py.read_xes(store_path_xes)
            log.to_csv(store_path_csv)

        if extension not in ['xes', 'both']:
            os.remove(store_path_xes)

        # Dataset stored in xes. We need to load it with pm4py
        if store_path_csv is None:
            log = pm4py.read_xes(store_path_xes)
            return store_path_xes, log
        # Dataset stored in csv, the dataset is already preloaded
        else:
            return store_path_csv, log

    else:
        raise ValueError(f'Failed to download the file. Status code: {response.status_code}')

