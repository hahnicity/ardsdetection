import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from train import ARDSDetectionModel, build_parser, NoFeaturesSelectedError


def lasso(df, model_args):
    # lasso feature coefficients are different from gini. On one run my coefficients were
    # array([ 0.53468622,  0.        , -0.        ,  0.95669967, -0.19588544,
    #    -2.03785016, -0.02198389,  0.1574506 , -1.07548971])
    importance_threshs = np.arange(-3, 1.01, .2)
    fs_results = None
    for thresh in importance_threshs:
        model_args.select_from_model_thresh = thresh
        model = ARDSDetectionModel(model_args, df)
        try:
            model.train_and_test()
        except NoFeaturesSelectedError:
            continue
        n_selected_features = len(model.selected_features)
        # XXX well the fs results don't seem determ. but what to do instead?
        if fs_results is not None and n_selected_features in fs_results.n_features.values:
            continue
        tmp = model.results.model_results['aggregate'].copy()
        tmp['n_features'] = n_selected_features
        tmp['selection_thresh'] = thresh
        if fs_results is None:
            fs_results = tmp
        else:
            fs_results = fs_results.append(tmp)

    fs_results.index = range(len(fs_results))
    return fs_results


def gini(df, model_args):
    importance_threshs = [.01, .02, .03, .04, .05] + list(np.arange(.08, .4, .03))
    fs_results = None
    for thresh in importance_threshs:
        model_args.select_from_model_thresh = thresh
        model = ARDSDetectionModel(model_args, df)
        try:
            model.train_and_test()
        except NoFeaturesSelectedError:
            continue
        n_selected_features = len(model.selected_features)
        # XXX well the fs results don't seem determ. but what to do instead?
        if fs_results is not None and n_selected_features in fs_results.n_features.values:
            continue
        tmp = model.results.model_results['aggregate'].copy()
        tmp['n_features'] = n_selected_features
        tmp['selection_thresh'] = thresh
        if fs_results is None:
            fs_results = tmp
        else:
            fs_results = fs_results.append(tmp)

    fs_results.index = range(len(fs_results))
    return fs_results


def pca(df, model_args):
    model_features = set(df.columns).difference(set(['row_time', 'patient', 'y', 'ventBN', 'set_type', 'hour']))
    fs_results = None
    for n in range(1, len(model_features)+1):
        model_args.n_new_features = n
        model = ARDSDetectionModel(model_args, df)
        model.train_and_test()
        tmp = model.results.model_results['aggregate'].copy()
        tmp['n_features'] = n
        if fs_results is None:
            fs_results = tmp
        else:
            fs_results = fs_results.append(tmp)

    fs_results.index = range(len(fs_results))
    return fs_results


def n_feature_selection(df, model_args):
    model_features = set(df.columns).difference(set(['row_time', 'patient', 'y', 'ventBN', 'set_type', 'hour']))
    fs_results = None
    for n in range(1, len(model_features)+1):
        model_args.n_new_features = n
        model = ARDSDetectionModel(model_args, df)
        model.train_and_test()
        tmp = model.results.model_results['aggregate'].copy()
        tmp['n_features'] = n
        for feature in model_features:
            tmp[feature] = 1 if feature in model.selected_features else 0

        if fs_results is None:
            fs_results = tmp
        else:
            fs_results = fs_results.append(tmp)

    fs_results.index = range(len(fs_results))
    return fs_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--from-pickle', help='load data frame from pickle', required=True)
    parser.add_argument('--algo', help='The type of algorithm you want to do ML with', choices=['RF', 'MLP', 'SVM', 'LOG_REG', 'GBC', 'NB', 'ADA', 'ATS_MODEL'], default='RF')

    parser.add_argument('-fsm', '--feature-selection-method', choices=['RFE', 'chi2', 'mutual_info', 'gini', 'lasso', 'PCA'], help='Feature selection method', required=True)
    parser.add_argument('--split-type', choices=['holdout', 'holdout_random', 'kfold', 'train_all', 'test_all', 'bootstrap'], help='All splits are performed so there is no test/train patient overlap', required=True)
    parser.add_argument('--savefig', help='save figure to specified location instead of plotting')
    parser.add_argument('--print-results-table', action='store_true')
    parser.add_argument('--save-results')
    parser.add_argument('--load-results')
    parser.add_argument('-sp', '--post-hour', type=int, required=True)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--bootstrap-n', type=int, default=80)
    parser.add_argument('--no-bootstrap-replace', action='store_false', help='Dont use replacement when sampling patients with bootstrap')
    parser.add_argument('--n-bootstraps', type=int, default=10, help='number of bootstrapped patient samplees to take')
    main_args = parser.parse_args()

    model_args = build_parser().parse_args([])
    model_args.no_copd_to_ctrl = False
    model_args.no_print_results = True
    model_args.feature_selection_method = main_args.feature_selection_method
    model_args.algo = main_args.algo
    model_args.split_type = main_args.split_type
    model_args.frame_size = 100
    model_args.n_runs = main_args.n_runs
    model_args.post_hour = main_args.post_hour
    model_args.bootstrap_n = main_args.bootstrap_n
    model_args.no_bootstrap_replace = main_args.no_bootstrap_replace
    model_args.n_bootstraps = main_args.n_bootstraps

    if not main_args.load_results:
        df = pd.read_pickle(main_args.from_pickle)

        if main_args.feature_selection_method in ['RFE', 'chi2', 'mutual_info']:
            results = n_feature_selection(df, model_args)
        elif main_args.feature_selection_method == 'PCA':
            results = pca(df, model_args)
        elif model_args.feature_selection_method == 'gini':
            results = gini(df, model_args)
        elif model_args.feature_selection_method == 'lasso':
            results = lasso(df, model_args)
    else:
        results = pd.read_pickle(main_args.load_results)

    if main_args.save_results:
        results.to_pickle(main_args.save_results)

    ards_results = results[(results.patho == 'ards')]
    plt.plot(ards_results['n_features'].values.astype(int), ards_results.auc.values, label='AUC')
    ards_results['f1'] = 2 * (ards_results['prec'] * ards_results['recall']) / (ards_results['prec'] + ards_results['recall'])
    plt.plot(ards_results['n_features'].values.astype(int), ards_results.acc.values, label='ARDS Accuracy', lw=2)
    features_min = ards_results['n_features'].min()
    features_max = ards_results['n_features'].max()
    plt.xticks(range(features_min, features_max+1))
    plt.xlabel('N Features')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.title('{} feature selection with {} {} - Status Post {}'.format(main_args.feature_selection_method, main_args.algo, main_args.split_type, main_args.post_hour))
    if main_args.savefig:
        plt.savefig(main_args.savefig)
    else:
        plt.show()
    if main_args.print_results_table:
        print(tabulate(results, headers='keys'))


if __name__ == "__main__":
    main()
