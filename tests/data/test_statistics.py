from verona.data import statistics
import pandas as pd


def test_activity_stats():
    df = pd.read_csv('../../BPI_Challenge_2012_A.csv')

    num_acts = statistics.get_num_activities(df, 'Activity')
    print(num_acts)

    uniq_acts = statistics.get_activities_list(df, 'Activity')
    print(uniq_acts)


def test_attribute_stats():
    df = pd.read_csv('../../BPI_Challenge_2012_A.csv')

    num_atts = statistics.get_num_values(df, 'Resource')
    print(num_atts)

    uniq_atts = statistics.get_values_list(df, 'Resource')
    print(uniq_atts)


def test_case_stats():
    df = pd.read_csv('../../BPI_Challenge_2012_A.csv')

    num_cases = statistics.get_num_cases(df, 'CaseID')
    print(num_cases)

    max_len_case = statistics.get_max_len_case(df, 'CaseID')
    print(max_len_case)

    min_len_case = statistics.get_min_len_case(df, 'CaseID')
    print(min_len_case)

    avg_len_case = statistics.get_avg_len_case(df, 'CaseID')
    print(avg_len_case)

    max_dur_case = statistics.get_max_duration_case(df, 'CaseID', 'Timestamp')
    print(max_dur_case)

    min_dur_case = statistics.get_min_duration_case(df, 'CaseID', 'Timestamp')
    print(min_dur_case)

    avg_dur_case = statistics.get_avg_duration_case(df, 'CaseID', 'Timestamp')
    print(avg_dur_case)


def test_event_stats():
    df = pd.read_csv('../../BPI_Challenge_2012_A.csv')

    max_dur_event = statistics.get_max_duration_event(df, 'CaseID', 'Timestamp')
    print(max_dur_event)

    min_dur_event = statistics.get_min_duration_event(df, 'CaseID', 'Timestamp')
    print(min_dur_event)

    avg_dur_event = statistics.get_avg_duration_event(df, 'CaseID', 'Timestamp')
    print(avg_dur_event)


def test_variants_stats():
    df = pd.read_csv('../../BPI_Challenge_2012_A.csv')

    num_variants = statistics.get_num_variants(df, 'CaseID', 'Activity')
    print(num_variants)

    count_variants = statistics.get_count_variants(df, 'CaseID', 'Activity')
    print(count_variants)
