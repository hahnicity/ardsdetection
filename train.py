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

import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold, train_test_split
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

from collate import Dataset
from metrics import *


PATHO = {0: 'ctrl', 1: 'ards', 2: 'copd'}


class ARDSDetectionModel(object):
    def __init__(self, args, data):
        """
        :param args: CLI args
        :param data: DataFrame containing train+test data
        """
        self.args = args
        self.data = data
        self.models = []
        results_cols = ["patient", "patho"]
        # XXX in future update so that we can merge COPD patients into OTHERS
        # if desired
        for n, patho in PATHO.iteritems():
            results_cols.extend([
                "{}_tps".format(patho), "{}_fps".format(patho),
                "{}_tns".format(patho), "{}_fns".format(patho),
                "{}_votes".format(patho),
            ])
        results_cols += ["model_idx", "prediction"]
        self.results = pd.DataFrame([], columns=results_cols)

    def get_cross_patient_train_test_idx(self):
        unique_patients = self.data['patient'].unique()
        mapping = {'ctrl': [], 'ards': [], 'copd': []}
        for patient in list(unique_patients):
            patient_rows = self.data[self.data.patient == patient]
            type_ = PATHO[patient_rows.y.unique()[0]]
            mapping[type_].append(patient)

        patients_to_use = []
        total_test_patients = round(len(unique_patients) * self.args.split_ratio)
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

        train_patient_data = self.data.query('patient not in {}'.format(patients_to_use))
        test_patient_data = self.data.query('patient in {}'.format(patients_to_use))
        return train_patient_data.index, test_patient_data.index

    def get_cross_patient_kfold_idxs(self):
        idxs = []
        unique_patients = self.data['patient'].unique()
        mapping = {'ctrl': [], 'ards': [], 'copd': []}
        for patient in unique_patients:
            patient_rows = self.data[self.data.patient == patient]
            type_ = PATHO[patient_rows.y.unique()[0]]
            mapping[type_].append(patient)
        total_test_patients = round(len(unique_patients) / self.args.folds)
        for i in range(self.args.folds):
            patients_to_use = []
            for k, v in mapping.items():
                lower_bound = int(round(i * len(v) / self.args.folds))
                upper_bound = int(round((i + 1) * len(v) / self.args.folds))
                if upper_bound < 1:
                    raise Exception("You do not have enough patients for {} cohort".format(k))
                patients = v[lower_bound:upper_bound]
                patients_to_use.extend(patients)
            train_patient_data = self.data.query('patient not in {}'.format(patients_to_use))
            test_patient_data = self.data.query('patient in {}'.format(patients_to_use))
            idxs.append((train_patient_data.index, test_patient_data.index))

        return idxs

    def perform_data_splits(self):
        if self.args.cross_patient_split:
            idxs = self.get_cross_patient_train_test_idx()
        elif self.args.cross_patient_kfold:
            idxs = self.get_cross_patient_kfold_idxs()

        y = self.data.y
        x = self.data.drop(['y', 'patient', 'ventBN'], axis=1)

        for train_idx, test_idx in idxs:
            x_train = x.loc[train_idx].dropna()
            x_test = x.loc[test_idx].dropna()
            y_train = y.loc[train_idx].dropna()
            y_test = y.loc[test_idx].dropna()

            scaler = MinMaxScaler()
            x_train = pd.DataFrame(scaler.fit_transform(x_train), index=y_train.index)
            x_test = pd.DataFrame(scaler.transform(x_test), index=y_test.index)
            yield (x_train, x_test, y_train, y_test)

    def train(self, x_train, y_train):
        clf = RandomForestClassifier(random_state=1)
        clf.fit(x_train, y_train)
        self.models.append(clf)

    def train_and_test(self):
        """
        Train models and then run testing afterwards.
        """
        for model_idx, (x_train, x_test, y_train, y_test) in enumerate(self.perform_data_splits()):
            if self.args.pca:
                pca = PCA(n_components=self.args.pca)
                pca.fit(x_train, y_train)
                x_train = pd.DataFrame(pca.transform(x_train), index=x_train.index)
                x_test = pd.DataFrame(pca.transform(x_test), index=x_test.index)

            if self.args.grid_search:
                self.perform_grid_search(x_train, y_train)
            else:
                self.train(x_train, y_train)

            predictions = pd.Series(self.models[-1].predict(x_test), index=y_test.index)
            results = self.aggregate_statistics(y_test, predictions, model_idx)
            self.print_model_stats(y_test, predictions, model_idx)
            print("-------------------")

        self.print_aggregate_results()

    def perform_grid_search(self, x_train, y_train):
        params = {
            "n_estimators": list(range(50, 90, 5)),
            "max_features": list(range(23, 26)),
            "criterion": ["entropy"],
        }
        clf = GridSearchCV(RandomForestClassifier(random_state=1), params)
        clf.fit(x_train, y_train.values)
        print("Params: ", clf.best_params_)
        print("Best CV score: ", clf.best_score_)
        self.models.append(clf)

    def aggregate_statistics(self, y_test, predictions, model_idx):
        x_test_expanded = self.data.loc[y_test.index]
        for pt in x_test_expanded.patient.unique():
            i = len(self.results)
            pt_rows = x_test_expanded[x_test_expanded.patient == pt]
            patho_n = pt_rows.y.unique()[0]
            pt_actual = y_test.loc[pt_rows.index]
            pt_pred = predictions.loc[pt_rows.index]
            pt_results = [pt, patho_n]
            for n, patho in PATHO.iteritems():
                pt_results.extend([
                    get_tps(pt_actual, pt_pred, n), get_fps(pt_actual, pt_pred, n),
                    get_tns(pt_actual, pt_pred, n), get_fns(pt_actual, pt_pred, n),
                    len(pt_pred[pt_pred == n]),
                ])
            pt_results.extend([model_idx, np.argmax([pt_results[6], pt_results[11], pt_results[16]])])
            self.results.loc[i] = pt_results

    def print_model_stats(self, y_test, predictions, model_idx):
        """
        Perform majority rules voting on what disease subtype that a patient has
        """
        model_results = self.results[self.results.model_idx == model_idx]
        incorrect_pts = model_results[model_results.patho != model_results.prediction]

        print("Model accuracy: {}".format(accuracy_score(y_test, predictions)))
        for n, patho in PATHO.iteritems():
            print("{} recall: {}".format(patho, recall_score(y_test, predictions, labels=[n], average='macro')))
            print("{} precision: {}".format(patho, precision_score(y_test, predictions, labels=[n], average='macro')))

        for idx, row in incorrect_pts.iterrows():
            print("Patient {}: Prediction: {}, Actual: {}. Voting:\n{}".format(
                row.patient, row.prediction, row.patho, row[['ctrl_votes', 'ards_votes', 'copd_votes']]
            ))

    def print_aggregate_results(self):
        # XXX for now just stick to analyzing aggregate over patients

        for n, patho in PATHO.iteritems():
            tps = float(len(self.results[(self.results.patho == n) & (self.results.prediction == n)]))
            tns = float(len(self.results[(self.results.patho != n) & (self.results.prediction != n)]))
            fps = float(len(self.results[(self.results.patho != n) & (self.results.prediction == n)]))
            fns = float(len(self.results[(self.results.patho == n) & (self.results.prediction != n)]))

            print("{} patient accuracy: {}".format(patho, (tps+tns) / (tps+tns+fps+fns)))
            print("{} patient sensitivity: {}".format(patho, tps / (tps+fns)))
            print("{} patient specificity: {}".format(patho, tns / (tns+fps)))
            print("{} patient precision: {}".format(patho, tps / (tps+fps)))
            print("")


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


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--cohort-description', default='cohort-description.csv', help='path to cohort description file')
    parser.add_argument("--feature-set", default="flow_time", choices=["flow_time", "broad"])
    parser.add_argument('--load-intermediates', action='store_true', help='do best to load from intermediate data')
    parser.add_argument('--split-ratio', type=float, default=.2)
    parser.add_argument("--pca", type=int, help="perform PCA analysis/transform on data")
    parser.add_argument("--grid-search", action="store_true", help='perform grid search for model hyperparameters')
    parser.add_argument("--cross-patient-split", action="store_true")
    parser.add_argument("--cross-patient-kfold", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stacks", default=20, type=int)
    parser.add_argument("--to-pickle", help="name of file the data frame will be pickled in")
    parser.add_argument("-p", "--from-pickle", help="name of file to retrieve pickled data from")
    parser.add_argument("--fold-copd", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    df = create_df(args)
    model = ARDSDetectionModel(args, df)
    model.train_and_test()


if __name__ == "__main__":
    main()
