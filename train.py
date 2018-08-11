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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

from collate import Dataset
from metrics import *


PATHO = {0: 'ctrl', 1: 'ards', 2: 'copd'}

def get_cross_patient_train_test_idx(df, split_ratio):
    unique_patients = df['patient'].unique()
    mapping = {'ctrl': [], 'ards': [], 'copd': []}
    for patient in list(unique_patients):
        patient_rows = df[df.patient == patient]
        type_ = PATHO[df.y.unique()[0]]
        mapping[type_].append(patient)

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

    train_patient_data = df.query('patient not in {}'.format(patients_to_use))
    test_patient_data = df.query('patient in {}'.format(patients_to_use))
    return train_patient_data.index, test_patient_data.index


def get_cross_patient_kfold_idxs(df, folds):
    idxs = []
    unique_patients = df['patient'].unique()
    mapping = {'ctrl': [], 'ards': [], 'copd': []}
    for patient in list(unique_patients):
        patient_rows = df[df.patient == patient]
        type_ = PATHO[df.y.unique()[0]]
        mapping[type_].append(patient)
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
        train_patient_data = df.query('patient not in {}'.format(patients_to_use))
        test_patient_data = df.query('patient in {}'.format(patients_to_use))
        idxs.append((train_patient_data.index, test_patient_data.index))

    return idxs


def train_test_split_by_patient(x, y, train_idx, test_idx):
    x_train = x.loc[train_idx].dropna()
    x_test = x.loc[test_idx].dropna()
    y_train = y.loc[train_idx].dropna()
    y_test = y.loc[test_idx].dropna()
    return x_train, x_test, y_train, y_test


def preprocess_and_split_x_y(df, is_cross_patient_split, breaths_to_stack, is_cross_patient_kfold, folds):
    def finalize_data(x_train, x_test):
        x_train, scaling_factors = perform_initial_scaling(x_train, breaths_to_stack)
        x_test = perform_subsequent_scaling(x_test, scaling_factors)
        return x_train, x_test

    if is_cross_patient_split:
        train_idx, test_idx = get_cross_patient_train_test_idx(df, split_ratio)
    elif is_cross_patient_kfold:
        idxs = get_cross_patient_kfold_idxs(df, folds)

    y = df['y']
    suppl_data = pd.DataFrame({'ventBN': df.ventBN.values, 'patient': df.patient.values}, index=df.index)
    suppl_data['patient'] = suppl_data.patient.str.extract('(\d{4}RPI\d{10})')
    suppl_data['patho'] = 'ctrl'
    for i in range(3):
        patho = PATHO[i]
        suppl_data.loc[y[y == i].index, 'patho'] = patho
    df = df.drop(['y', 'patient', 'ventBN'], axis=1)

    x = pd.DataFrame(df)
    y = pd.Series(y)

    if is_cross_patient_split:
        x_train, x_test, y_train, y_test = train_test_split_by_patient(
            x, y, train_idx, test_idx
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


def perform_voting(patient_results, predictions, suppl_data):
    """
    Perform majority rules voting on what disease subtype that a patient has
    """
    patient_results.setdefault("predicted", {})
    patient_results.setdefault('actual', {})
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
    incorrect_pts = diff[diff != 0].index.tolist()
    correct_pts = diff[diff == 0].index.tolist()
    for k, v in dict(actual).items():
        patient_results['actual'][k] = v
    for k, v in dict(pt_predictions).items():
        patient_results['predicted'][k] = v

    for i in incorrect_pts:
        print("Patient {}: Prediction: {}, Actual: {}. Voting {}".format(
            i, pt_predictions.loc[i], actual.loc[i], voting[i]
        ))
    return patient_results


def calc_label(y_train, y_test, label):
    train_samples = len(y_train[y_train == label])
    test_samples = len(y_test[y_test == label])
    print("{} train samples for label {}".format(train_samples, label))
    print("{} test samples for label {}".format(test_samples, label))


def create_df(args):
    """
    Create dataframe for use in model

    :param args: Arguments from CLI parser
    """
    if args.from_pickle:
        return pd.read_pickle(args.from_pickle)

    df = Dataset(args.cohort_description, args.feature_set, args.stacks, args.load_intermediates).get()
    if args.to_pickle:
        df.to_pickle(args.to_pickle)
    return df


def print_results(results, patient_results, fold_copd):
    actual_arr = []
    pred_arr = []
    for k, v in patient_results['actual'].items():
        actual_arr.append(v)
        pred_arr.append(patient_results['predicted'][k])

    # sigh... this analysis is hack upon hack
    if fold_copd:
        for i, v in enumerate(actual_arr):
            if v == 2:
                actual_arr[i] = 0
            if pred_arr[i] == 2:
                pred_arr[i] = 0
        print("AUC {}".format(roc_auc_score(actual_arr, pred_arr)))

        for k, v in patient_results['actual'].items():
            if v == 2:
                patient_results['actual'][k] = 0
        for k, v in patient_results['predicted'].items():
            if v == 2:
                patient_results['predicted'][k] = 0
        max_range = 2
    else:
        max_range = 3

    for i in range(0, max_range):
        actual_pts = []
        predicted_pts = []
        for k, v in patient_results['actual'].items():
            if v == i:
                actual_pts.append(k)
        for k, v in patient_results['predicted'].items():
            if v == i:
                predicted_pts.append(k)

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
        pt_tps = 0
        pt_fps = 0
        pt_tns = 0
        pt_fns = 0
        for patient in actual_pts:
            if patient in predicted_pts:
                pt_tps += 1
            else:
                pt_fns += 1
        for patient in predicted_pts:
            if patient not in actual_pts:
                pt_fps += 1
        pt_tns = len(patient_results['actual']) - pt_tps - pt_fns - pt_fps
        print("{} patient accuracy: {}".format(patho, (pt_tps+pt_tns) / (pt_tps+pt_tns+pt_fps+pt_fns)))
        print("{} patient sensitivity: {}".format(patho, pt_tps / (pt_tps+pt_fns)))
        print("{} patient specificity: {}".format(patho, pt_tns / (pt_tns+pt_fps)))
        print("{} patient precision: {}".format(patho, pt_tps / (pt_tps+pt_fps)))
        print("")


def main():
    parser = ArgumentParser()
    parser.add_argument('--cohort-description', default='cohort-description.csv', help='path to cohort description file')
    parser.add_argument("--feature-set", default="flow_time", choices=["flow_time", "broad"])
    parser.add_argument('--load-intermediates', action='store_true', help='do best to load from intermediate data')
    parser.add_argument("--pca", type=int, help="perform PCA analysis/transform on data")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--cross-patient-split", action="store_true")
    parser.add_argument("--cross-patient-kfold", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stacks", default=20, type=int)
    parser.add_argument("--to-pickle", help="name of file the data frame will be pickled in")
    parser.add_argument("-p", "--from-pickle", help="name of file to retrieve pickled data from")
    parser.add_argument("--fold-copd", action="store_true")
    args = parser.parse_args()

    df = create_df(args)
    results = None
    patient_results = {}
    for x_train, x_test, y_train, y_test, suppl_data in preprocess_and_split_x_y(df, args.cross_patient_split, args.stacks, args.cross_patient_kfold, args.folds):
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
        patient_results = perform_voting(patient_results, predictions, suppl_data)
        print("-------------------")

    print_results(results, patient_results, args.fold_copd)

if __name__ == "__main__":
    main()
