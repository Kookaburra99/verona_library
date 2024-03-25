from setuptools import find_packages, setup

setup(
    name='verona',
    version='1.1.0',
    description='The predictive process monitoring library for Python',
    long_description='The VERONA library is a powerful Python tool designed to evaluate and compare predictive process '
                     'monitoring models fairly and under equals conditions. Leveraging the benchmark published in '
                     'Rama-Maneiro et al. (2021), this library provides comprehensive functions for assessing the '
                     'performance of predictive models in the context of business process monitoring.',
    author='Efren Rama-Maneiro, Pedro Gamallo-Fernandez',
    author_email='efren.rama.maneiro@usc.es, pedro.gamallo.fernandez@usc.es',
    url='https://gitlab.citius.usc.es/pedro.gamallo/verona_library',
    license='GPLv3',
    package_data={
        'verona.data.csv': ['*.csv'],  # Include all CSV files under the 'csv' directory
    },
    install_requires=['requests', 'pm4py', 'pandas', 'numpy', 'scikit-learn', 'tqdm', 'jellyfish', 'pytest', 'cmdstanpy',
              "matplotlib", "plotly", "sphinx", "sphinx-press-theme", "sphinxcontrib-bibtex"],
    packages=find_packages()
)
