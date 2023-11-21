Overview
========

Business processes, ranging from the assembly line of a car manufacturing plant to the
chaos of a hospital emergency department, govern the day-to-day operations of companies
and institutions. The ability to understand and analyze the process flow and, most
importantly, predict how it will evolve over time is critical to optimize resource
utilization, minimize waiting times, reduce costs, improve services and products quality,
and achieve many other benefits to both organizations and users.

Predictive process monitoring (PPM), a subfield of process mining, focuses on tackling
all these predictive problems. For this purpose, PPM techniques train a predictive model
using \textit{event logs},  which provide information about the event execution of the
business process, including the case identifier, the recorded activity, its timestamp,
and various attributes, such as the resource information, as shown in :numref:`tab_event_log`.
Although there are some datasets containing publicly available real-life event logs,
each author tends to utilize a distinct subset of these under widely different
experimental setups, which complicates the task of making a meaningful comparison
between approaches.

.. _tab_event_log:
.. table:: Extract from a real event log.

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
