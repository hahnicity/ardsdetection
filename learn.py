"""
learn
~~~~~

Performs learning for our classifier
"""
from argparse import ArgumentParser
import csv
import operator
from random import randint, sample
import time

from numpy import append, inf, nan
from numpy.random import permutation
import pandas as pd
from sklearn.cross_validation import KFold, train_test_split
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve

from collate import (
    broad_feature_set,
    collate_all_to_non_rolling_frame,
    collate_all_to_rolling_average,
    collate_all_to_rolling_frame,
    flow_time_feature_set,
)


PATHO = {0: 'ctrl', 1: 'ards', 2: 'copd'}

def get_cross_patient_train_test_idx(df, split_ratio):
    unique_patients = df['filename'].unique()
    mapping = {'control': [], 'ards': [], 'copd': []}
    for f in list(unique_patients):
        type_ = f.split('/')[0].replace('cohort', '')
        mapping[type_].append(f)

    patients_to_use = []
    total_test_patients = round(len(unique_patients) * split_ratio)
    for k, v in mapping.items():
        num_patients = len(v)
        proportion = float(num_patients) / len(unique_patients)
        num_to_use = int(round(total_test_patients * proportion))
        if num_to_use < 1:
            raise Exception("You do not have enough patients for {} cohort".format(k))
        patients = sample(v, num_to_use)
        print("number {} patients in training: {}".format(k, num_patients - len(patients)))
        print("number {} patients in test: {}".format(k, len(patients)))
        patients_to_use.extend(patients)

    train_patient_data = df.query('filename not in {}'.format(patients_to_use))
    test_patient_data = df.query('filename in {}'.format(patients_to_use))
    return train_patient_data.index, test_patient_data.index


def get_cross_patient_kfold_idxs(df, folds):
    idxs = []
    unique_patients = df['filename'].unique()
    mapping = {'control': [], 'ards': [], 'copd': []}
    for f in list(unique_patients):
        type_ = f.split('/')[0].replace('cohort', '')
        mapping[type_].append(f)
    total_test_patients = round(len(unique_patients) / folds)
    for i in range(folds):
        patients_to_use = []
        for k, v in mapping.items():
            lower_bound = int(round(i * len(v) / folds))
            upper_bound = int(round((i + 1) * len(v) / folds))
            if upper_bound < 1:
                raise Exception("You do not have enough patients for {} cohort".format(k))
            patients = v[lower_bound:upper_bound]
            patients_to_use.extend(patients)
        train_patient_data = df.query('filename not in {}'.format(patients_to_use))
        test_patient_data = df.query('filename in {}'.format(patients_to_use))
        idxs.append((train_patient_data.index, test_patient_data.index))

    return idxs


def train_test_split_by_patient(x, y, train_idx, test_idx):
    x_train = x.loc[train_idx].dropna()
    x_test = x.loc[test_idx].dropna()
    y_train = y.loc[train_idx].dropna()
    y_test = y.loc[test_idx].dropna()
    return x_train, x_test, y_train, y_test


def preprocess_and_split_x_y(
    df,
    split_ratio,
    is_cross_patient_split,
    samples,
    breaths_to_stack,
    is_cross_patient_kfold,
    folds,
    is_simple_split
):
    def finalize_data(x_train, x_test):
        x_train, scaling_factors = perform_initial_scaling(x_train, breaths_to_stack)
        x_test = perform_subsequent_scaling(x_test, scaling_factors)
        return x_train, x_test

    if is_cross_patient_split:
        train_idx, test_idx = get_cross_patient_train_test_idx(df, split_ratio)
    elif is_cross_patient_kfold:
        idxs = get_cross_patient_kfold_idxs(df, folds)

    y = df['y']
    suppl_data = pd.DataFrame({'ventBN': df.ventBN.values, 'patient': df.filename.values}, index=df.index)
    suppl_data['patient'] = suppl_data.patient.str.extract('(\d{4}RPI\d{10})')
    suppl_data['patho'] = 'ctrl'
    for i in range(3):
        patho = PATHO[i]
        suppl_data.loc[y[y == i].index, 'patho'] = patho
    df = df.drop(['y', 'filename', 'ventBN'], axis=1)

    x = pd.DataFrame(df)
    y = pd.Series(y)

    if is_cross_patient_split:
        x_train, x_test, y_train, y_test = train_test_split_by_patient(
            x, y, train_idx, test_idx
        )
        x_train, x_test = finalize_data(x_train, x_test)
        yield (x_train, x_test, y_train, y_test, suppl_data)
    elif is_simple_split:
        if samples:
            x = x.sample(n=samples)
            y = y.loc[x.index]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=split_ratio, random_state=randint(0, 100)
        )
        x_train, x_test = finalize_data(x_train, x_test)
        yield (x_train, x_test, y_train, y_test, suppl_data)
    elif is_cross_patient_kfold:
        for train_idx, test_idx in idxs:
            x_train, x_test, y_train, y_test = train_test_split_by_patient(
                x, y, train_idx, test_idx
            )
            x_train, x_test = finalize_data(x_train, x_test)
            yield (x_train, x_test, y_train, y_test, suppl_data)


def perform_initial_scaling(df, stacked_breaths):
    """
    Since breaths are stacked we look at max and mins across multiple
    stacked rows
    """
    max_mins = {}
    for col in range(df.shape[1]):
        modulo_idx = col % stacked_breaths
        max_mins.setdefault(modulo_idx, {'max': 0, 'min': 0})
        min = df.iloc[:, col].min()
        if min < max_mins[modulo_idx]['max']:
            max_mins[modulo_idx]['min'] = min
        max = df.iloc[:, col].max()
        if max > max_mins[modulo_idx]['min']:
            max_mins[modulo_idx]['max'] = max
    perform_subsequent_scaling(df, max_mins)
    return df, max_mins


def perform_subsequent_scaling(df, max_mins):
    for col in range(df.shape[1]):
        breaths_to_stack = len(max_mins)
        modulo_idx = col % breaths_to_stack
        val = max_mins[modulo_idx]
        df.iloc[:, col] = (df.iloc[:, col] - val['min']) / (val['max'] - val['min'])
    return df


def get_fns_idx(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return pos_loc[pos_loc != label].index


def get_fns(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return len(pos_loc[pos_loc != label])


def get_tns(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return len(neg_loc[neg_loc != label])


def get_fps_idx(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return neg_loc[neg_loc == label].index


def get_fps_full_rows(actual, predictions, label, filename):
    idx = get_fps_idx(actual, predictions, label)
    full_df = read_pickle(filename)
    return full_df.loc[idx]


def get_fps(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return len(neg_loc[neg_loc == label])


def get_tps(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return len(pos_loc[pos_loc == label])


def false_positive_rate(actual, predictions, label):
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 0
    else:
        return round(float(fp) / (fp + tn), 4)


def specificity(actual, predictions, label):
    """
    Also known as the true negative rate
    """
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 1
    else:
        return round(float(tn) / (tn + fp), 4)


def sensitivity(actual, predictions, label):
    """
    Also known as recall
    """
    tps = get_tps(actual, predictions, label)
    fns = get_fns(actual, predictions, label)
    if tps == 0 and fns == 0:
        return nan
    return tps / (tps+fns)


def aggregate_statistics(actual, predictions, results):
    cols = []
    tmp = []
    for i in range(0, 3):
        patho = {0: 'ctrl', 1: 'ards', 2: 'copd'}[i]
        tmp.extend([
            get_tps(actual, predictions, i),
            get_fps(actual, predictions, i),
            get_tns(actual, predictions, i),
            get_fns(actual, predictions, i),
        ])
        cols.extend([
            '{}_tps'.format(patho),
            '{}_fps'.format(patho),
            '{}_tns'.format(patho),
            '{}_fns'.format(patho),
        ])
    if isinstance(results, type(None)):
        results = pd.DataFrame([tmp], columns=cols)
    else:
        results = results.append(pd.Series(tmp, index=cols), ignore_index=True)
    return results


def make_predictions(clf, x_test, y_test, suppl_data):
    predictions = clf.predict(x_test)
    print("Accuracy: " + str(round(accuracy_score(y_test, predictions), 4)))
    print("Precision: " + str(precision_score(y_test, predictions, average='macro')))
    print("Recall: " + str(recall_score(y_test, predictions, average='macro')))
    predictions = pd.Series(predictions, index=x_test.index)
    print("Control recall: ", sensitivity(y_test, predictions, 0))
    print("ARDS recall: ", sensitivity(y_test, predictions, 1))
    print("COPD recall: ", sensitivity(y_test, predictions, 2))
    print("Control specificity", specificity(y_test, predictions, 0))
    print("ARDS specificity", specificity(y_test, predictions, 1))
    print("COPD specificity", specificity(y_test, predictions, 2))
    error = abs(y_test - predictions)
    failure_idx = error[error != 0]
    with open("failure.test", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "ventBN", "real val"])
        for idx in failure_idx.index:
            pt_data = suppl_data.loc[idx].tolist()
            actual = y_test.loc[idx]
            writer.writerow(pt_data + [actual])
    return predictions


def perform_cross_validation(x_train, x_test, y_train, y_test, suppl_data):
    params = {
        "n_estimators": list(range(50, 90, 5)),
        "max_features": list(range(23, 26)),
        "criterion": ["entropy"],
    }
    clf = GridSearchCV(RandomForestClassifier(), params)
    clf.fit(x_train, y_train.values)
    print("Params: ", clf.best_params_)
    print("Best CV score: ", clf.best_score_)
    return make_predictions(clf, x_test, y_test, suppl_data)


def perform_learning(x_train, x_test, y_train, y_test, suppl_data):
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train.values)
    predictions = make_predictions(clf, x_test, y_test, suppl_data)
    return predictions


def perform_voting(predictions, suppl_data, no_plot):
    """
    Perform majority rules voting on what disease subtype that a patient has
    """
    voting = {}
    ards_pred, ards_real = set(), set()
    copd_pred, copd_real = set(), set()
    control_pred, control_real = set(), set()
    pred = {}
    patients = {}
    patient_idxs = {}
    for idx in predictions.index:
        cohort = suppl_data.loc[idx].patho
        patient = suppl_data.loc[idx].patient
        if patient not in patient_idxs:
            patient_idxs[patient] = []
        patients[patient] = {"ctrl": 0, "ards": 1, "copd": 2}[cohort]
        voting.setdefault(patient, {0: 0, 1: 0, 2: 0})
        voting[patient][predictions.loc[idx]] += 1
        patient_idxs[patient].append(idx)

    actual = pd.Series(list(patients.values()), index=list(patients.keys()))

    for pt, vals in voting.items():
        maxed = max(vals.items(), key=operator.itemgetter(1))[0]
        pred[pt] = maxed
        pt_predictions = pd.Series(list(pred.values()), index=list(pred.keys()))

    print("")
    for label in [0, 1, 2]:
        sen = sensitivity(actual, pt_predictions, label)
        spec = specificity(actual, pt_predictions, label)
        # sensitivity is recall
        print("Patient sensitivity for label {}: {}".format(label, sen))
        print("Patient pecificity for label {}: {}".format(label, spec))

    diff = (pt_predictions - actual)
    incorrect_pts = diff[diff != 0].index
    for i in incorrect_pts:
        print("Patient {}: Prediction: {}, Actual: {}. Voting {}".format(
            i, pt_predictions.loc[i], actual.loc[i], voting[i]
        ))

    # XXX temporarily out of commission
#    if not no_plot:
#        for patient in incorrect_pts:
#            obs = predictions.loc[patient_idxs[patient]]
#            obs['idx'] = range(0, len(obs))
#            obs.plot.scatter('idx', 'y', title=patient)
#            plt.show()


def calc_label(y_train, y_test, label):
    train_samples = len(y_train[y_train == label])
    test_samples = len(y_test[y_test == label])
    print("{} train samples for label {}".format(train_samples, label))
    print("{} test samples for label {}".format(test_samples, label))


def create_df(args):
    feature_set = {"flow_time": flow_time_feature_set, "broad": broad_feature_set}[args.feature_set]
    if args.from_pickle:
        return pd.read_pickle(args.from_pickle)

    if args.metadata_processing_type == "non_rolling":
        df = collate_all_to_non_rolling_frame(args.samples, feature_set)
        args.stacks = df.shape[1]
    elif args.metadata_processing_type == "rolling_average":
        df = collate_all_to_rolling_average(args.stacks, args.samples, feature_set)
    elif args.metadata_processing_type == "rolling_frame":
        df = collate_all_to_rolling_frame(args.stacks, args.samples, feature_set)

    if args.to_pickle:
        df.to_pickle(args.to_pickle)
    return df


def print_results(results):
    for i in range(0, 3):
        patho = {0: 'ctrl', 1: 'ards', 2: 'copd'}[i]
        # sensitivity (recall) = tps / (tps+fns)
        # specificity =  tns / (tns+fps)
        tps = results['{}_tps'.format(patho)].sum()
        fns = results['{}_fns'.format(patho)].sum()
        tns = results['{}_tns'.format(patho)].sum()
        fps = results['{}_fps'.format(patho)].sum()
        acc = round((tps+tns) / (tps+tns+fps+fns), 4)
        sen = round(tps / (tps+fns), 4)
        spec = round(tns / (tns+fps), 4)
        prec = round(tps / (tps+fps), 4)
        print("{} total accuracy: {}".format(patho, acc))
        print("{} total sensitivity: {}".format(patho, sen))
        print("{} total specificity: {}".format(patho, spec))
        print("{} total precision: {}".format(patho, prec))


def main():
    parser = ArgumentParser()
    parser.add_argument("--samples", type=int, default=None, help="used only with simple split")
    parser.add_argument(
        "--test-size", type=float, default=0.20, help="percentage of patients to use in test"
    )
    parser.add_argument("--feature-set", default="flow_time", choices=["flow_time", "broad"])
    parser.add_argument("--pca", type=int, help="perform PCA analysis/transform on data")
    parser.add_argument("--metadata-processing-type", default="rolling_average", choices=["rolling_average", "non_rolling", "rolling_frame"])
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--simple-split", action="store_true")
    parser.add_argument("--cross-patient-split", action="store_true")
    parser.add_argument("--cross-patient-kfold", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stacks", default=20, type=int)
    parser.add_argument("--to-pickle", help="name of file the data frame will be pickled in")
    parser.add_argument("-p", "--from-pickle", help="name of file to retrieve pickled data from")
    parser.add_argument("--no-plot", action="store_true", help="Do not perform any plotting actions at the end of fold execution")
    args = parser.parse_args()
    df = create_df(args)
    results = None
    for x_train, x_test, y_train, y_test, suppl_data in preprocess_and_split_x_y(
        df, args.test_size, args.cross_patient_split, args.samples, args.stacks, args.cross_patient_kfold, args.folds, args.simple_split
    ):
        for i in range(0, 3):
            patho = PATHO[i]
            test_pts = suppl_data.loc[y_test[y_test == i].index].patient.unique()
            print("{} cohort patients: {}".format(patho, ", ".join(test_pts)))

        calc_label(y_train, y_test, 0)
        calc_label(y_train, y_test, 1)
        calc_label(y_train, y_test, 2)
        print("")

        if args.pca:
            pca = PCA(n_components=args.pca)
            pca.fit(x_train, y_train)
            x_train = pd.DataFrame(pca.transform(x_train), index=x_train.index)
            x_test = pd.DataFrame(pca.transform(x_test), index=x_test.index)

        if args.cross_validate:
            predictions = perform_cross_validation(x_train, x_test, y_train, y_test, suppl_data)
        else:
            predictions = perform_learning(x_train, x_test, y_train, y_test, suppl_data)
        results = aggregate_statistics(y_test, predictions, results)
        perform_voting(predictions, suppl_data, args.no_plot)
        print("-------------------")

    print_results(results)

if __name__ == "__main__":
    main()
