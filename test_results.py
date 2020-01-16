from nose.tools import assert_list_equal
import numpy as np
import pandas as pd

from results import ModelCollection, ModelResults, PatientResults

x_test_fold1 = pd.DataFrame([['a', 1, 1]] * 110, columns=['patient', 'fold_idx', 'ground_truth'])
x_test_fold2 = pd.DataFrame([['b', 2, 0]] * 150, columns=['patient', 'fold_idx', 'ground_truth'])
y_test_fold1 = pd.Series([1] * 110)
y_test_fold2 = pd.Series([0] * 150)


def test_patient_results_sunny_day():
    patient_a = PatientResults('a', 1, 1, 0)
    patient_a.set_results([1] * 60 + [0] * 50)
    lst, cols = patient_a.to_list()
    assert_list_equal(lst, [
        50,
        60,
        60 / 110.0,
        1,
        1,
        0,
        1,
    ])
    assert_list_equal(cols, [
        'other_votes',
        'ards_votes',
        'frac_votes',
        'majority_prediction',
        'fold_idx',
        'model_idx',
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
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, x_test_fold1, 1)
    model_collection.add_model(y_test_fold2, pred2, x_test_fold2, 2)
    model_collection.calc_aggregate_stats(.5, print_results=False)
    results = model_collection.model_results['aggregate']
    assert results[results.patho == 'ards'].iloc[0].recall == 1
    assert results[results.patho == 'ards'].iloc[0].prec == 1
    assert results[results.patho == 'other'].iloc[0].spec == 1
    assert results[results.patho == 'other'].iloc[0].recall == 1
    assert results[results.patho == 'other'].iloc[0].prec == 1
    assert results[results.patho == 'ards'].iloc[0].spec == 1


def test_calc_aggregate_stats_with_failing_ards_thresh():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, x_test_fold1, 1)
    model_collection.add_model(y_test_fold2, pred2, x_test_fold2, 2)
    model_collection.calc_aggregate_stats((60 / 110.0)+.01, print_results=False)
    results = model_collection.model_results['aggregate']
    assert results[results.patho == 'ards'].iloc[0].recall == 0
    assert np.isnan(results[results.patho == 'ards'].iloc[0].prec)
    assert results[results.patho == 'ards'].iloc[0].spec == 1
    assert results[results.patho == 'other'].iloc[0].spec == 0
    assert results[results.patho == 'other'].iloc[0].recall == 1
    assert results[results.patho == 'other'].iloc[0].prec == 0.5


def test_calc_aggregate_stats_with_passing_ards_thresh():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, x_test_fold1, 1)
    model_collection.add_model(y_test_fold2, pred2, x_test_fold2, 2)
    model_collection.calc_aggregate_stats((60 / 110.0)-.01, print_results=False)
    results = model_collection.model_results['aggregate']
    assert results[results.patho == 'ards'].iloc[0].recall == 1
    assert results[results.patho == 'ards'].iloc[0].prec == 1
    assert results[results.patho == 'ards'].iloc[0].spec == 1
    assert results[results.patho == 'other'].iloc[0].spec == 1
    assert results[results.patho == 'other'].iloc[0].recall == 1
    assert results[results.patho == 'other'].iloc[0].prec == 1


def test_calc_aggregate_stats_with_passing_other_thresh():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, x_test_fold1, 1)
    model_collection.add_model(y_test_fold2, pred2, x_test_fold2, 2)
    model_collection.calc_aggregate_stats((40 / 150.0)+.01, print_results=False)
    results = model_collection.model_results['aggregate']
    assert results[results.patho == 'ards'].iloc[0].recall == 1
    assert results[results.patho == 'ards'].iloc[0].prec == 1
    assert results[results.patho == 'ards'].iloc[0].spec == 1
    assert results[results.patho == 'other'].iloc[0].recall == 1
    assert results[results.patho == 'other'].iloc[0].spec == 1
    assert results[results.patho == 'other'].iloc[0].prec == 1


def test_calc_aggregate_stats_with_failing_other_thresh():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, x_test_fold1, 1)
    model_collection.add_model(y_test_fold2, pred2, x_test_fold2, 2)
    model_collection.calc_aggregate_stats((40 / 150.0)-.01, print_results=False)
    results = model_collection.model_results['aggregate']
    assert results[results.patho == 'ards'].iloc[0].recall == 1
    assert results[results.patho == 'ards'].iloc[0].prec == 0.5
    assert results[results.patho == 'ards'].iloc[0].spec == 0
    assert results[results.patho == 'other'].iloc[0].recall == 0
    assert results[results.patho == 'other'].iloc[0].spec == 1
    assert np.isnan(results[results.patho == 'other'].iloc[0].prec)


def test_get_summary_statistics_from_frame():
    dataframe = pd.DataFrame([
        [10, 10, 0, 0],
        [8, 10, 2, 0],
        [5, 6, 5, 4],
        [10, 9, 0, 1],
    ], columns=['ards_tps_0.55', 'ards_tns_0.55', 'ards_fps_0.55', 'ards_fns_0.55'])
    expected = pd.DataFrame([
        [1, 1, 1],
        [1, 10.0/12, 8.0/10],
        [5.0/9, 6.0/11, 5.0/10],
        [10.0/11, 1, 1],
    ])
    model_collection = ModelCollection()
    res = model_collection.get_summary_statistics_from_frame(dataframe, 'ards', .55)
    assert (res == expected).all().all()


def test_get_all_patient_results_in_fold_np_array():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, x_test_fold1, 1)
    model_collection.add_model(y_test_fold2, pred2, x_test_fold2, 2)
    res = model_collection.get_all_patient_results_in_fold_np_array(1)
    assert (np.array([[50, 60, 60 / 110.0, 1, 1, 0, 1]]) == res).all()
    res = model_collection.get_all_patient_results_in_fold_np_array(2)
    assert (np.array([[110, 40, 40 / 150.0, 0, 2, 1, 0]]) == res).all()


def test_get_all_patient_results_dataframe():
    model_collection = ModelCollection()
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model_collection.add_model(y_test_fold1, pred1, x_test_fold1, 1)
    model_collection.add_model(y_test_fold2, pred2, x_test_fold2, 2)
    expected = pd.DataFrame([
        [50, 60, 60 / 110.0, 1, 1, 0, 1],
        [110, 40, 40 / 150.0, 0, 2, 1, 0],
    ], columns=['other_votes', 'ards_votes', 'frac_votes', 'majority_prediction', 'fold_idx', 'model_idx', 'ground_truth'])
    res = model_collection.get_all_patient_results_dataframe()
    assert (expected == res).all().all()


def test_count_predictions():
    model = ModelResults(1, 0)
    pred1 = pd.Series([0] * 50 + [1] * 60)
    pred2 = pd.Series([0] * 110 + [1] * 40)
    model.set_results(y_test_fold1, pred1, x_test_fold1)
    res, cols = model.count_predictions(.5)
    assert res == [0, 1, 0, 0, 1, 0, 0, 0, 1]
    res, cols = model.count_predictions(.3)
    assert res == [0, 1, 0, 0, 1, 0, 0, 0, 1]
    res, cols = model.count_predictions(.6)
    assert res == [0, 0, 1, 0, 0, 0, 0, 1, 1]


def test_auc_results():
    patient_results = pd.DataFrame([
        [50, 60, 60 / 110.0, 1, 1, 0, 1],
        [50, 60, 60 / 110.0, 1, 1, 0, 1],
        [50, 60, 60 / 110.0, 1, 1, 0, 1],
        [110, 40, 40 / 150.0, 0, 2, 0, 0],
        [110, 40, 40 / 150.0, 0, 2, 0, 0],
        [110, 40, 40 / 150.0, 0, 2, 0, 0],
        [50, 60, 60 / 110.0, 1, 1, 1, 1],
        [50, 60, 60 / 110.0, 1, 1, 1, 1],
        [50, 60, 60 / 110.0, 1, 1, 1, 1],
        [110, 40, 40 / 150.0, 0, 2, 1, 0],
        [110, 40, 40 / 150.0, 0, 2, 1, 0],
        [110, 40, 40 / 150.0, 0, 2, 1, 0],
    ], columns=['other_votes', 'ards_votes', 'frac_votes', 'majority_prediction', 'fold_idx', 'model_idx', 'ground_truth'])
    model_collection = ModelCollection()
    aucs = model_collection.get_auc_results(patient_results)
    assert (aucs == np.array([1, 1])).all()
