import argparse

import numpy as np
import pandas as pd

from collate import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection')
    parser.add_argument('-t', '--to-pickle', required=True)
    parser.add_argument('-d', '--cohort-description', default='cohort-description.csv', help='Path to file describing the cohort')
    parser.add_argument('-e', '--experiment', default='1')
    parser.add_argument('--split-type', choices=['holdout', 'holdout_random', 'kfold', 'train_all', 'test_all'], help='All splits are performed so there is no test/train patient overlap', default='kfold')
    parser.add_argument('-sd', '--start-hour-delta', default=0, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data')
    parser.add_argument('-sp', '--post-hour', default=24, type=int)
    parser.add_argument('-tsd', '--test-start-hour-delta', default=None, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data. Only for usage in testing set')
    parser.add_argument('-tsp', '--test-post-hour', default=None, type=int)
    parser.add_argument("--feature-set", default="flow_time", choices=Dataset.vent_feature_sets)
    parser.add_argument('-vm', '--ventmode-model', help='path to ventmode model if we are utilizing ventmode')
    parser.add_argument('-vs', '--ventmode-scaler', help='path to ventmode scaler if we are utilizing ventmode')
    parser.add_argument('--use-tor', action='store_true', help='use tor in featurization')
    args = parser.parse_args()

    cls = Dataset(
        args.data_path,
        args.cohort_description,
        args.feature_set,
        20,
        True,
        args.experiment,
        args.post_hour,
        args.start_hour_delta,
        'mean',  # It doesn't matter which function we choose here.
        args.split_type,
        test_post_hour=args.test_post_hour,
        test_start_hour_delta=args.test_start_hour_delta,
        use_ventmode=True if args.ventmode_model or args.ventmode_scaler else False,
        ventmode_model_path=args.ventmode_model,
        ventmode_scaler_path=args.ventmode_scaler,
        use_tor=args.use_tor,
    )
    df = cls.get_unframed_dataset()
    pd.to_pickle(df, args.to_pickle)


if __name__ == "__main__":
    main()
