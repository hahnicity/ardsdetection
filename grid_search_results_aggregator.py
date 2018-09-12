"""
grid_search_results_aggregator
~~~~~~~~~~~~

Ensure that we can take any results that we
get from the grid search and then put it
into a nice looking table
"""
from argparse import ArgumentParser
import csv
import os
import re

import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score

from train import ARDSDetectionModel, build_parser

# XXX for now only handle flow time data
file_pat = r'(?P<exp>experiment[\d\+]+)_flow_time_fs(?P<fs>\d+)_ff(?P<ff>\w+)_sd(?P<sd>\d+)_sp(?P<sp>\d+)'
dir_struct = "data/{exp}/training/grid_search/flow_time/{sd}-{sp}/{fs}/{ff}/None/None-None"


def run_dataset(dataset, folds):
    model_args = build_parser().parse_args([])
    model_args.no_copd_to_ctrl = False
    model_args.cross_patient_kfold = True
    model_args.folds = folds
    model_args.no_print_results = True
    model = ARDSDetectionModel(model_args, dataset)
    model.train_and_test()
    return model.aggregate_results


def main():
    parser = ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('output_file')
    args = parser.parse_args()

    table = PrettyTable()
    field_names = ['experiment', 'start_delta', 'status_post', 'frame_size', 'frame_func', 'patho', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc']
    table.field_names = field_names

    with open(args.output_file, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(field_names)

        for f in args.files:
            results = pd.read_pickle(f)
            top_auc = 0
            top_idx = -1
            top_folds = -1
            for idx, run in enumerate(results):
                for folds in [5, 10]:
                    if run[folds]['auc'] > top_auc:
                        top_idx = idx
                        top_folds = folds
                        top_auc = run[folds]['auc']
            best = results[top_idx]
            match_items = re.search(file_pat, f).groupdict()
            dataset_dir = dir_struct.format(**match_items)
            dataset_path = os.path.join(dataset_dir, "dataset-{}.pkl".format(best['idx']))
            dataset = pd.read_pickle(dataset_path)
            results = run_dataset(dataset, folds)
            for idx, patho_results in results.iterrows():
                row = [
                    match_items['exp'],
                    match_items['sd'],
                    match_items['sp'],
                    match_items['fs'],
                    match_items['ff'],
                    patho_results.patho,
                    patho_results.accuracy,
                    patho_results.sensitivity,
                    patho_results.specificity,
                    patho_results.precision,
                    patho_results.auc,
                ]
                writer.writerow(row)
                table.add_row(row)
    print(table)


if __name__ == "__main__":
    main()
