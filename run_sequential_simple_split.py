"""
run_sequential_simple_split
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because simple split yields varying model results based on the split just run
sequential models and take the average.
"""
import argparse

import pandas as pd

from train import ARDSDetectionModel, build_parser, create_df


def run_sequential(df, model_args, num_runs):
    ctrl_all_results = []
    ards_all_results = []
    cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'auc']
    for _ in range(num_runs):
        model = ARDSDetectionModel(model_args, df)
        model.train_and_test()
        ctrl_results = model.aggregate_results[model.aggregate_results.patho == 'OTHER'].iloc[0]
        ards_results = model.aggregate_results[model.aggregate_results.patho == 'ARDS'].iloc[0]
        ctrl_all_results.append(ctrl_results[cols].tolist())
        ards_all_results.append(ards_results[cols].tolist())
    ctrl_all_results = pd.DataFrame(ctrl_all_results, columns=cols)
    ards_all_results = pd.DataFrame(ards_all_results, columns=cols)
    return ctrl_all_results, ards_all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--from-pickle', required=True)
    parser.add_argument('-nr', '--num-runs', type=int, default=100)
    parser.add_argument('-sr', '--split-ratio', type=float, default=.2)
    main_args = parser.parse_args()

    model_args = build_parser().parse_args([])
    model_args.from_pickle = main_args.from_pickle
    model_args.split_type = 'simple'
    model_args.split_ratio = main_args.split_ratio
    model_args.no_print_results = True

    df = create_df(model_args)
    ctrl_all_results, ards_all_results = run_sequential(df, model_args, main_args.num_runs)

    print("Control means after {} runs".format(main_args.num_runs))
    print(ctrl_all_results.mean(axis=0))
    print("ARDS means after {} runs".format(main_args.num_runs))
    print(ards_all_results.mean(axis=0))


if __name__ == "__main__":
    main()
