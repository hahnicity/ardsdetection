from nose.tools import assert_list_equal

from results import PatientResults


def test_patient_results_sunny_day():
    patient_a = PatientResults('a', 1)
    patient_a.set_results([1] * 60 + [0] * 50)
    lst = patient_a.to_list()
    assert_list_equal(lst, [
        50,
        60,
        60 / 110.0,
        1,
        1,
    ])
