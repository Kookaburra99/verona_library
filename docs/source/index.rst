.. VERONA documentation master file, created by
   sphinx-quickstart on Tue Nov 21 13:07:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VERONA: predictiVE pRocess mOnitoring beNchmArk
==================================================================================

The **VERONA** library is a Python tool for faster and more user-friendly development,
evaluation, and comparison of predictive process monitoring (PPM) approaches.

The main features of **VERONA** are:

* Allows downloading an extensive collection of real process event logs used by the process mining community.
* Provides functions to cover data preprocessing, partitioning, prefix and target extraction, model evaluation and result visualization from the PPM workflow in a clear and transparent manner for the user.
* Implements the reference benchmark of Rama-Maneiro et al. :footcite:`benchmark_rama_2023` to fairly compare deep learning-based PPM approaches.
* Ports the statistical tests of the R-implemented ``scmamp`` :footcite:`scmamp_calvo_2016` to Python, allowing statistical comparison of different machine learning approaches.

.. note::
   This project is under active development.

Contents
--------
.. toctree::
   :maxdepth: 2

   overview

References
----------
.. footbibliography::


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

