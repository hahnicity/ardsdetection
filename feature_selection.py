import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import ARDSDetectionModel, build_parser


def lasso(df):
    pass


def gini(df):
    pass


def pca(df, model_args):
    model_features = set(df.columns).difference(set(['row_time', 'patient', 'y', 'ventBN', 'set_type', 'hour']))
    fs_results = None
    for n in range(1, len(model_features)+1):
        model_args.n_new_features = n
        model = ARDSDetectionModel(model_args, df)
        model.train_and_test()
        tmp = model.aggregate_results.copy()
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
        tmp = model.aggregate_results.copy()
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
    parser.add_argument('-p', '--from-pickle', help='load data frame from pickle')
    parser.add_argument('--algo', help='The type of algorithm you want to do ML with', choices=['RF', 'MLP', 'SVM', 'LOG_REG', 'GBC', 'NB', 'ADA'], default='RF')
    parser.add_argument('-fsm', '--feature-selection-method', choices=['RFE', 'chi2', 'mutual_info', 'gini', 'lasso', 'PCA'], help='Feature selection method', required=True)
    main_args = parser.parse_args()

    model_args = build_parser().parse_args([])
    model_args.no_copd_to_ctrl = False
    model_args.no_print_results = True
    model_args.feature_selection_method = main_args.feature_selection_method
    model_args.algo = main_args.algo
    model_args.split_type = 'holdout'
    model_args.frame_size = 100

    df = pd.read_pickle(main_args.from_pickle)

    if main_args.feature_selection_method in ['RFE', 'chi2', 'mutual_info']:
        results = n_feature_selection(df, model_args)
    elif main_args.feature_selection_method == 'PCA':
        results = pca(df, model_args)

    ards_results = results[results.patho == 'ARDS']
    plt.plot(ards_results['n_features'].values, ards_results['auc'].values)
    plt.xlabel('N features')
    plt.ylabel('AUC')
    plt.grid()
    plt.title('{} with {}'.format(main_args.feature_selection_method, main_args.algo))
    plt.show()


if __name__ == "__main__":
    main()
