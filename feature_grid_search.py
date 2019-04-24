import argparse
from itertools import compress, product
import multiprocessing
import os

import pandas as pd
import pickle

from collate import Dataset
from run_sequential_simple_split import run_sequential
from train import ARDSDetectionModel, build_parser

DF_DIR = 'experiment{experiment_num}/all_data/grid_search/{feature_set}/ehr_{ehr_features}/demo_{demo_features}/{sd}-{sp}/{fs}/{ff}/{tfs}/{tsd}-{tsp}/{algo}'


def get_all_possible_features():
    """
    Get all possible feature permutations
    """
    all_possible_flow_time_features = [
        'mean_flow_from_pef', 'inst_RR', 'slope_minF_to_zero',
        'pef_+0.16_to_zero', 'iTime', 'eTime', 'I:E ratio',
        # XXX Add pressure itime eventually, altho it may only be useful for PC/PS pts.
        'dyn_compliance', 'tve:tvi ratio'
    ]
    # There is actually no way that we can do grid search on all possible broad features
    # because it will yield 8388608 possibilities, which is infeasible to search thru.
    # So I will only look through the ones that seem reasonable from my POV. Reducing #
    # features down to 17 possibilities still yields 262144 choices. 13 possibilities gives
    # 16384 choices
    all_possibilities = all_possible_flow_time_features + [
        'TVi', 'TVe', 'Maw', 'ipAUC', 'PIP', 'PEEP', 'epAUC',
        # XXX others that can be investigated in future
        # vol_at_76, min_pressure
    ]
    all_ft_combos = (set(compress(all_possible_flow_time_features, mask)) for mask in product(*[[0,1]]*len(all_possible_flow_time_features)))
    all_combos = (set(compress(all_possibilities, mask)) for mask in product(*[[0,1]]*len(all_possibilities)))
    return {'flow_time_gen': all_ft_combos, 'broad_gen': all_combos}


def run_model(model_args, main_args, combo, model_idx, out_dir, unframed_df):
    path = os.path.join(out_dir, 'dataset-{}.pkl'.format(model_idx))
    results = {'auc': 0, 'dataset_path': path, 'run_type': main_args.run_type, 'idx': model_idx}
    if not combo:
        results['features'] = []
        return results
    results['features'] = combo

    if os.path.exists(path) and main_args.load_if_exists:
        dataset = pd.read_pickle(path)
        if 'set_type' not in dataset.columns:
            dataset['set_type'] = 'train_test'
    else:
        combo = list(combo) + ['ventBN']
        data_cls = Dataset(
            main_args.data_path,
            model_args.cohort_description,
            'custom',
            main_args.frame_size,
            True,
            main_args.experiment,
            main_args.post_hour,
            main_args.start_hour_delta,
            main_args.frame_func,
            {'kfold': 'kfold', 'sequential_split': 'holdout_random'}[main_args.run_type],
            test_frame_size=main_args.test_frame_size,
            test_post_hour=main_args.test_post_hour,
            test_start_hour_delta=main_args.test_start_hour_delta,
            custom_vent_features=combo,
            use_ehr_features=main_args.use_ehr_features,
            use_demographic_features=main_args.use_demographic_features,
        )
        if unframed_df is None:
            dataset = data_cls.get()
        else:
            dataset = data_cls.get_framed_from_unframed_dataset(unframed_df)

    if len(dataset.patient.unique()) != 100 and (main_args.start_hour_delta == 0 and main_args.post_hour == 24):
        raise Exception('Unable to find 100 patients for features: {}'.format(dataset.columns))

    if main_args.run_type == 'kfold':
        model_args.split_type = 'kfold'
        # only run with 5-fold cross-validation
        model_args.folds = 5
        model = ARDSDetectionModel(model_args, dataset)
        try:
            model.train_and_test()
        except:
            dataset.to_pickle('err-dataset.pkl')
        auc = model.aggregate_results.auc.iloc[0]
        results['auc'] = auc
        del model  # paranoia
    elif main_args.run_type == 'sequential_split':
        model_args.split_type = 'holdout_random'
        model_args.split_ratio = main_args.split_ratio
        results['num_runs'] = main_args.num_runs
        ctrl_results, ards_results = run_sequential(dataset, model_args, main_args.num_runs)
        auc = ards_results.auc.mean()
        results['auc'] = auc

    if auc > main_args.auc_thresh:
        dataset.to_pickle(path)

    return results


def func_star(args):
    return run_model(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection')
    parser.add_argument('--feature-set', choices=['flow_time', 'broad'], default='flow_time')
    parser.add_argument('-sd', '--start-hour-delta', default=0, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data')
    parser.add_argument('-sp', '--post-hour', default=24, type=int)
    parser.add_argument('-e', '--experiment', help='Experiment number we wish to run. If you wish to mix patients from different experiments you can do <num>+<num>+... eg. 1+3  OR 1+2+3', default='1')
    parser.add_argument("-fs", "--frame-size", default=20, type=int)
    parser.add_argument('-ff', '--frame-func', choices=['median', 'mean', 'var', 'std', 'mean+var', 'mean+std', 'median+var', 'median+std'], default='median')
    parser.add_argument('-tfs', "--test-frame-size", default=None, type=int)
    parser.add_argument('-tsd', '--test-start-hour-delta', default=None, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data. Only for usage in testing set')
    parser.add_argument('-tsp', '--test-post-hour', default=None, type=int)
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(), help="Set number of threads to use, otherwise all cores will be occupied")
    parser.add_argument('--auc-thresh', type=float, help='save datasets to file if they have an auc above this', default=.8)
    parser.add_argument('--debug', action='store_true', help='debug whats going wrong with the script without implementing multiprocessing')
    parser.add_argument('--use-ehr-features', action='store_true')
    parser.add_argument('--use-demographic-features', action='store_true')
    parser.add_argument('--run-type', choices=['kfold', 'sequential_split'], default='kfold')
    parser.add_argument('-sr', '--split-ratio', type=float, default=.2)
    # just running 20 times is unfortunately insufficient
    parser.add_argument('-nr', '--num-runs', type=int, default=50)
    parser.add_argument('--algo', help='The type of algorithm you want to do ML with', choices=['RF', 'MLP', 'SVM', 'LOG_REG', 'GBC', 'NB', 'ADA'], default='RF')
    parser.add_argument('--load-if-exists', action='store_true', help='load previously saved intermediate datasets')
    parser.add_argument('--load-from-unframed', help='Load a new dataset from an existing unframed dataset')
    main_args = parser.parse_args()

    # We're doing this because these args are not necessary, and we can just pass them
    # easily over code because they wont be changing
    model_args = build_parser().parse_args([])
    model_args.no_copd_to_ctrl = False
    model_args.no_print_results = True
    model_args.algo = main_args.algo

    results = {}
    feature_combos = get_all_possible_features()
    out_dir = os.path.join(main_args.data_path, DF_DIR.format(
        experiment_num=main_args.experiment,
        feature_set=main_args.feature_set,
        ehr_features='on' if main_args.use_ehr_features else 'off',
        demo_features='on' if main_args.use_demographic_features else 'off',
        sp=main_args.post_hour,
        sd=main_args.start_hour_delta,
        fs=main_args.frame_size,
        ff=main_args.frame_func,
        tfs=main_args.test_frame_size,
        tsd=main_args.test_start_hour_delta,
        tsp=main_args.test_post_hour,
        algo=main_args.algo,
    ))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    feature_gen = feature_combos['{}_gen'.format(main_args.feature_set)]
    if main_args.load_from_unframed:
        unframed = pd.read_pickle(main_args.load_from_unframed)
    else:
        unframed = None

    input_gen = [(model_args, main_args, combo, idx, out_dir, unframed) for idx, combo in enumerate(feature_gen)]

    if not main_args.debug:
        pool = multiprocessing.Pool(main_args.threads)
        results = pool.map(func_star, input_gen)
        pool.close()
        pool.join()
    else:
        results = []
        for args in input_gen[:2]:
            results.append(run_model(*args))

    best = max([(features_run['idx'], features_run['auc']) for features_run in results], key=lambda x: x[1])
    print('Best AUC: {}'.format(best[1]))
    print('Best features: {}'.format(results[best[0]]))
    dict_ = pickle.dumps(results)
    results_file = 'experiment{}_{}_{}_ehr-{}_demo-{}_fs{}_ff{}_sd{}_sp{}_tfs{}_tsd{}_tsp{}_{}_grid_search_results.pkl'.format(
        main_args.experiment,
        main_args.feature_set,
        main_args.run_type,
        'on' if main_args.use_ehr_features else 'off',
        'on' if main_args.use_demographic_features else 'off',
        main_args.frame_size,
        main_args.frame_func,
        main_args.start_hour_delta,
        main_args.post_hour,
        main_args.test_frame_size,
        main_args.test_start_hour_delta,
        main_args.test_post_hour,
        main_args.algo,
    )
    with open(results_file, 'w') as f:
        f.write(dict_)


if __name__ == "__main__":
    main()
