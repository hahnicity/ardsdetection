import argparse

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import ks_2samp
from sklearn.feature_selection import chi2

from discretizer import Chi2discretizer


def ks_test_colinear(data):
    # test colinearity against all other features
    table = PrettyTable()
    table.field_names = ['f1', 'f2', 'D', 'p-val']
    for idx, feature in enumerate(data.columns.difference(['y'])):
        other_features = data.columns.difference(['y'])[idx+1:]
        for f2 in other_features:
            d, pval = ks_2samp(data[feature], data[f2])
            table.add_row([feature, f2, round(d, 3), round(pval,3)])

            if pval >= .95:
                print('{} and {} display colinearity'.format(feature, f2))
    print(table)


def ks_test_to_target(data):
    # test colinearity against all other features
    table = PrettyTable()
    table.field_names = ['f1', 'f2', 'D', 'p-val']
    for idx, feature in enumerate(data.columns.difference(['y'])):
        d, pval = ks_2samp(data[feature], data.y)
        table.add_row([feature, 'y', round(d, 3), round(pval,3)])

    print(table)


def ks_conditional(data, no_print=False):
    # test colinearity against all other features
    table = PrettyTable()
    table.field_names = ['feature', 'D', 'p-val']
    ds = []
    for idx, feature in enumerate(data.columns.difference(['y'])):
        ards = data[data.y == 1][feature]
        non_ards = data[data.y == 0][feature]
        d, pval = ks_2samp(ards, non_ards)
        ds.append([feature, d, pval])

    ds = sorted(ds, key=lambda x: -x[1])
    for feature, d, pval in ds:
        table.add_row([feature, round(d, 3), pval])
    if not no_print:
        print(table)
    return ds


def chimerge(data, bins, strat):
    feature_cols = data.columns.difference(['y'])
    y = data.y

    disc = Chi2discretizer(n_bins=bins, strategy=strat)
    data = disc.fit_transform(data[feature_cols], y)
    importances, pvals = chi2(data, y)
    idxs = np.argsort(importances)[::-1]
    table = PrettyTable()
    table.field_names = ['feature', 'importance', 'pval']

    for i in idxs:
        table.add_row([feature_cols[i], round(importances[i],3), round(pvals[i],3)])
    print(table)


def extended_chi2(data):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fold_file', help='cohort data file for a particular kfold')
    # XXX do chimerge
    parser.add_argument('method', choices=['chi2', 'kstest', 'chimerge'])
    parser.add_argument('--ks-method', choices=['colinear', 'to_target', 'conditional'])
    parser.add_argument('-cb',  '--chimerge-bins', type=int, default=100)
    parser.add_argument('-s', '--strategy', choices=['uniform', 'quantile', 'kmeans'], default='quantile')
    args = parser.parse_args()

    # by default we already do minmax scaling [0,1] so no need to do this

    df = pd.read_csv(args.fold_file)

    if args.method == 'kstest' and args.ks_method == 'colinear':
        ks_test_colinear(df)
    elif args.method == 'kstest' and args.ks_method == 'to_target':
        ks_test_to_target(df)
    elif args.method == 'kstest' and args.ks_method == 'conditional':
        ks_conditional(df)
    elif args.method == 'chi2':
        extended_chi2(df)
    elif args.method == 'chimerge':
        chimerge(df, args.chimerge_bins, args.strategy)


if __name__ == "__main__":
    main()
