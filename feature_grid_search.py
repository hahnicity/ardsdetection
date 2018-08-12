from itertools import compress, product
import os

import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score

from collate import Dataset
from train import ARDSDetectionModel, build_parser

DF_DIR = 'data/experiment1/training/grid_search'


def get_all_possible_features():
    """
    Get all possible feature permutations
    """
    all_possible_flow_time_features = [
        ('mean_flow_from_pef', 38), ('inst_RR', 8), ('minF_to_zero', 36),
        ('pef_+0.16_to_zero', 37), ('iTime', 6), ('eTime', 7), ('I:E ratio', 5),
        # XXX Add pressure itime eventually, altho it may only be useful for PC/PS pts.
        ('dyn_compliance', 39), ('TVratio', 11)
    ]
    all_possibilities = all_possible_flow_time_features + [
        ('TVi', 9), ('TVe', 10), ('Maw', 16), ('ipAUC', 18), ('PIP', 15), ('PEEP', 17),
        ('epAUC', 19), ('maxF', 12), ('minF', 13), ('maxP', 14), ('min_pressure', 35),
        ('vol_at_05', 40), ('vol_at_76', 41), ('vol_at_1', 42),
    ]
    all_ft_combos = (set(compress(all_possible_flow_time_features, mask)) for mask in product(*[[0,1]]*len(all_possible_flow_time_features)))
    all_combos = (set(compress(all_possibilities, mask)) for mask in product(*[[0,1]]*len(all_possibilities)))
    return {
        'flow_time_features': all_possible_flow_time_features,
        'flow_time_gen': all_ft_combos,
        'broad_features': all_possibilities,
        'broad_gen': all_combos,
    }


def main():
    args = build_parser().parse_args()
    # change this to True so we can calc AUC
    args.copd_to_ctrl = True

    results = {}
    feature_combos = get_all_possible_features()
        # XXX just work with flow_time for now
    if args.to_pickle:
        ft_dataset.to_pickle(args.to_pickle)

    possible_folds = [5, 10]
    for i, combo in enumerate(feature_combos['flow_time_gen']):
        if not combo:
            continue
        features = [k[0] for k in combo]

        path = os.path.join(DF_DIR, 'dataset-{}.pkl'.format(i))
        if os.path.exists(path):
            dataset = pd.read_pickle(path)
        else:
            dataset = Dataset(args.cohort_description, 'custom', args.stacks, True, custom_features=combo).get()
            dataset.to_pickle(path)

        results[i] = {i: dict() for i in possible_folds}
        for folds in possible_folds:
            model = ARDSDetectionModel(args, dataset)
            model.train_and_test()
            model_auc = roc_auc_score(model.results.patho.tolist(), model.results.prediction.tolist())
            results[i][folds] = {'features': features, 'auc': model_auc}

    best = max([(i, results[i][folds]['auc']) for i in results for folds in possible_folds], key=lambda x: x[1])
    print('Best AUC: {}'.format(best[1]))
    print('Best features: {}'.format(results[best[0]]))
    dict_ = pickle.dumps(results)
    with open('grid_search_results.pkl', 'w') as f:
        f.write(dict_)


if __name__ == "__main__":
    main()
