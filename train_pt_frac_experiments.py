import argparse

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from train import ARDSDetectionModel, build_parser, NoFeaturesSelectedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--from-pickle', help='load data frame from pickle', required=True)
    parser.add_argument('-nt', '--n-times-each-frac', type=int, help='number of times to run each fractional split.', default=20)
    main_args = parser.parse_args()

    model_args = build_parser().parse_args([])
    model_args.no_copd_to_ctrl = False
    model_args.no_print_results = True
    model_args.feature_selection_method = 'chi2'
    model_args.algo = 'RF'
    model_args.split_type = 'kfold'
    model_args.frame_size = 100
    model_args.n_runs = 5
    model_args.post_hour = 24
    model_args.n_new_features = 8

    df = pd.read_pickle(main_args.from_pickle)
    fs_results = dict()
    for frac in [.025, 0.05, 0.075, 0.1, .125, .25, .5, .75, 1]:
        for n in range(main_args.n_times_each_frac):
            model_args.train_pt_frac = frac
            model = ARDSDetectionModel(model_args, df)
            model.train_and_test()
            tmp = model.results.model_results['aggregate'].copy()
            if frac not in fs_results:
                fs_results[frac] = tmp
            else:
                fs_results[frac] = fs_results[frac].append(tmp)

	aucs = {}
    accuracies = {}
    for key in fs_results:
        aucs[key] = fs_results[key].groupby('patho').mean().auc.iloc[0]
        accuracies[key] = fs_results[key].groupby('patho').mean().acc.iloc[0]

	a = pd.DataFrame(accuracies, index=['accuracy'])
	a = a.append(aucs, ignore_index=True)
    a.index = ['accuracy', 'auc']
    a.transpose()
    pd.to_pickle(a, 'random_forest_pt_frac_results.pkl')


if __name__ == "__main__":
    main()
