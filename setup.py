from setuptools import find_packages, setup

setup(
    name='barro',
    version='1.0.0',
    description='The predictive process monitoring library for Python',
    long_description='The BARRO library is a powerful Python tool designed to evaluate and compare predictive process '
                     'monitoring models fairly and under equals conditions. Leveraging the benchmark published in '
                     'Rama-Maneiro et al. (2021), this library provides comprehensive functions for assessing the '
                     'performance of predictive models in the context of business process monitoring.',
    author='Efren Rama-Maneiro, Pedro Gamallo-Fernandez',
    author_email='efren.rama.maneiro@usc.es, pedro.gamallo.fernandez@usc.es',
    url='https://gitlab.citius.usc.es/pedro.gamallo/barro_library',
    license='ToDo',
    requires=[],
    packages=find_packages()
)
