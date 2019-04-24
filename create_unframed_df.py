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
    parser.add_argument('--plot-by-hour', action='store_true')
    parser.add_argument('--split-type', choices=['holdout', 'holdout_random', 'kfold', 'train_all', 'test_all'], help='All splits are performed so there is no test/train patient overlap', default='kfold')
    parser.add_argument('-sd', '--start-hour-delta', default=0, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data')
    parser.add_argument('-sp', '--post-hour', default=24, type=int)
    args = parser.parse_args()

    cls = Dataset(
        args.data_path,
        args.cohort_description,
        'flow_time',
        20,
        True,
        args.experiment,
        args.post_hour,
        args.start_hour_delta,
        'mean',
        args.split_type
    )
    df = cls.get_unframed_dataset()
    pd.to_pickle(df, args.to_pickle)


if __name__ == "__main__":
    main()
