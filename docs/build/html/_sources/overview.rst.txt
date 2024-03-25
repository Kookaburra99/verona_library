Overview
========

Introduction to Predictive Process Monitoring
---------------------------------------------

Business processes, ranging from the assembly line of a car manufacturing plant to the
chaos of a hospital emergency department, govern the day-to-day operations of companies
and institutions. The ability to understand and analyze the process flow and, most
importantly, predict how it will evolve over time is critical to optimize resource
utilization, minimize waiting times, reduce costs, improve services and products quality,
and achieve many other benefits to both organizations and users.

Predictive process monitoring (PPM), a subfield of process mining, focuses on tackling
all these predictive problems. For this purpose, PPM techniques train a predictive model
using *event logs*,  which provide information about the event execution of the
business process, including the case identifier, the recorded activity, its timestamp,
and various attributes, such as the resource information, as shown in :numref:`tab_event_log`.
Although there are some datasets containing publicly available real-life event logs,
each author tends to utilize a distinct subset of these under widely different
experimental setups, which complicates the task of making a meaningful comparison
between approaches.

.. _tab_event_log:
.. table:: Extract from a real event log.
    :align: center

    ========    =====================   =================== ========
    CaseID      Activity                Timestamp           Resource
    ========    =====================   =================== ========
    Case3814    Assign seriousness      2012-05-04 15:23:08 Value 1
    Case3814    Take in charge ticket   2012-05-07 07:01:12 Value 6
    Case3814    Resolve ticket          2012-05-24 15:25:23 Value 12
    Case3814    Closed                  2012-06-08 15:25:48 Value 5
    Case3815    Assign seriousness      2010-09-08 10:36:14 Value 2
    Case3815    Wait                    2010-09-08 15:32:25 Value 14
    Case3815    Take in charge ticket   2010-09-13 09:01:56 Value 2
    ========    =====================   =================== ========


The PPM workflow
----------------

.. _fig_ppm_workflow:
.. figure:: predictive_monitoring.png
    :scale: 50%

    Predictive monitoring workflow

:numref:`fig_ppm_workflow` shows the usual workflow in PPM. Normally, predictive
monitoring approaches receive as inputs a sequence of events from a running case up to a
certain point (*event prefix*). This information serves as historical knowledge which is
then used to train a predictive model with the objective of forecasting any variable of
interest from the future events that will happen after that point in time (*event suffix*).
Some of the most common predictive tasks include predicting the next event or sequence of
events to the end of the case, the remaining time until the end of the case, or the future
sequence of activities to be executed (*activity suffix*). Note that the fact of extracting
the prefixes and their corresponding targets is a time-consuming task that often takes a large
part of the approach coding effort and is highly prone to software bugs.


Benchmarking deep learning-based approaches
-------------------------------------------

A plethora of machine learning techniques have been applied to the predictive monitoring problem,
such as decision trees, support vector machines, and, more recently, deep learning-based
architectures, with the latter being the most successful and popular. Despite the abundance of
new approaches, most of them rely on very different experimental setups, so it is hard to draw
definitive conclusions about their performance according to the inner characteristics of the event
log. To tackle this issue, :footcite:`benchmark_rama_2023` presents an empirical comparison of the most
relevant deep learning-based predictive monitoring approaches, standardizing data partitioning
type and performance metrics, and including statistical tests to determine statistically
significant differences between proposals. Unfortunately, this approach is difficult to replicate
and requires a significant amount of work to integrate into the usual workflow of a data scientist
or researcher who intends to develop a new predictive monitoring approach.


The importance of VERONA library
--------------------------------

The VERONA library aims to address all of the mentioned challenges by implementing the necessary
functions to develop the PPM workflow and reproduce the benchmark of :footcite:`benchmark_rama_2023`
in an easy and user-friendly manner. This library provides primitives for event log retrieval and
pre-processing, data partitioning, evaluation with benchmark metrics, and visualization of results
with plots. Furthermore, it also includes the R implementation of the Bayesian statistical comparison
proposed in :footcite:`scmamp_calvo_2016` into the Python language, which is one of the core pillars
of the benchmark. Finally, it allows importing the results of the original benchmark as well as three
additional approaches, so that any new approach can be easily compared with the ones already evaluated.
Designed for easy integration, VERONA facilitates the encoding, execution, and evaluation of common
predictive monitoring tasks with minimal code.


.. rubric:: References
.. footbibliography::
