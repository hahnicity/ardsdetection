from nose.tools import assert_list_equal
import numpy as np
import pandas as pd

from results import ModelCollection, PatientResults

x_test_fold1 = pd.DataFrame([['a', 1, 1]] * 110, columns=['patient', 'fold_idx', 'ground_truth'])
x_test_fold2 = pd.DataFrame([['b', 2, 0]] * 150, columns=['patient', 'fold_idx', 'ground_truth'])
y_test_fold1 = pd.Series([1] * 110)
y_test_fold2 = pd.Series([0] * 150)


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
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, x_test_fold1, 1)
    model_collection.add_model(y_test_fold2, pred2, x_test_fold2, 2)
    model_collection.calc_fold_stats(.5, 1, print_results=False)
    results = model_collection.model_results['folds'][1]
    assert results[results.patho == 'ards'].iloc[0].recall == 1
    assert results[results.patho == 'ards'].iloc[0].prec == 1
    assert np.isnan(results[results.patho == 'other'].iloc[0].recall)
    assert results[results.patho == 'other'].iloc[0].spec == 1
    model_collection.calc_fold_stats(.5, 2, print_results=False)
    results = model_collection.model_results['folds'][2]
    assert results[results.patho == 'other'].iloc[0].recall == 1
    assert results[results.patho == 'other'].iloc[0].prec == 1
    assert np.isnan(results[results.patho == 'ards'].iloc[0].recall)
    assert results[results.patho == 'ards'].iloc[0].spec == 1


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
