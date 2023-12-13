Software architecture
=====================

The **VERONA** library organizes its functionalities into three primary packages based on
their respective functionalities: ``data``, ``evaluation``, and ``visualization``, which are
represented in the package diagram of :numref:`fig_architecture`. The ``evaluation`` package,
further, is divided into two sub-packages: ``metrics`` and ``stattests``.

.. _fig_architecture:
.. figure:: arquitectura_2.png
    :scale: 50%

    VERONA software architecture

The ``data`` package contains the primitives to manage anything related to event logs,
including downloading public event logs from external repositories used in
:footcite:`benchmark_rama_2023`, partitioning strategies, and generating prefix and
target pairs. Additionally, this software package provides utilities to facilitate
comparing novel approaches, loading benchmark results from :footcite:`benchmark_rama_2023`,
and generating event log statistics for reporting the experimental setup in potential publications.

The `evaluation` package includes functions for the evaluation stage of the model in the
benchmarking framework. First, ``evaluation.metrics`` provides a wide range of metrics to
evaluate the predictive models in predicting the next activity, suffix, and process times.
Second, ``evaluation.stattests`` reimplements in Python the statistical tests used in
:footcite:`benchmark_rama_2023` so that the developed predictive models can be compared
in a fair and robust way.

Finally, the ``visualizations`` package offers graphic representations for metrics,
statistical tests (specifically the ranking test), and event log statistics derived
from the above packages. These graphs adapt dynamically to the number of approaches and
event logs utilized, enabling users to customize the content of the visualizations while
significantly easing the development process of novel approaches. Thus, this package
provides standardized visualizations, facilitating the simultaneous visual comparison of
multiple approaches.


.. rubric:: References
.. footbibliography::
