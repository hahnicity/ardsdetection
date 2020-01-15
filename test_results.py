from nose.tools import assert_list_equal

from results import PatientResults


def test_patient_results_sunny_day():
    patient_a = PatientResults('a', 1, 2)
    patient_a.set_results([1] * 60 + [0] * 50)
    lst, cols = patient_a.to_list()
    assert_list_equal(lst, [
        50,
        60,
        60 / 110.0,
        1,
        2,
        1,
    ])
    assert_list_equal(cols, [
        'other_votes',
        'ards_votes',
        'frac_votes',
        'prediction',
        'fold_idx',
        'ground_truth',
    ])


def test_calc_fold_stats():
    pass


def test_calc_aggregate_stats():
    pass


def test_get_summary_statistics_from_frame():
    pass


def test_get_all_patient_results_in_fold():
    pass


def test_get_all_patient_results():
    pass


def test_count_predictions():
    pass


def test_count_predictions_thresh_failure():
    pass
