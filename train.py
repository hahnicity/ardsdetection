"""
train
~~~~~

Performs learning for our classifier
"""
from argparse import ArgumentParser
from collections import Counter
import csv
from math import sqrt, ceil
import multiprocessing
import operator
from random import randint, sample
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from prettytable import PrettyTable
import seaborn as sns
from scipy import interp
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning
from sklearn.feature_selection import chi2, mutual_info_classif, RFE, SelectFromModel, SelectKBest
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from collate import Dataset
import dtw_lib
from metrics import *
from results import ModelCollection

sns.set()
sns.set_style('ticks')
sns.set_context('paper')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class NoFeaturesSelectedError(Exception):
    pass


class NoIndicesError(Exception):
    pass


class ARDSDetectionModel(object):

    def __init__(self, args, data):
        """
        :param args: CLI args
        :param data: DataFrame containing train+test data
        """
        self.args = args
        self.data = data
        self.pathos = {0: 'OTHER', 1: 'ARDS', 2: 'COPD'}
        self.feature_ranks = {}
        if not self.args.no_copd_to_ctrl:
            self.data.loc[self.data[self.data.y == 2].index, 'y'] = 0
            del self.pathos[2]

        self.models = []
        if self.args.load_model:
            self.models.append(pd.read_pickle(self.args.load_model))

        self.results = ModelCollection()
        # XXX this var is unused, and anything relying on it will break
        self.patient_predictions = {}

    def get_bootstrap_idxs(self):
        unique_patients = self.data['patient'].unique()
        idxs = []
        for _ in range(self.args.n_bootstraps):
            pts = list(np.random.choice(unique_patients, size=self.args.bootstrap_n_pts, replace=self.args.no_bootstrap_replace))
            train_patient_data = self.data.query('patient in {}'.format(pts))
            test_patient_data = self.data.query('patient not in {}'.format(pts))
            idxs.append((train_patient_data.index, test_patient_data.index))
        return idxs

    def get_holdout_random_idxs(self):
        """
        Split patients according to some kind of random split with proportions defined
        by CLI args.
        """
        unique_patients = sorted(self.data['patient'].unique())
        mapping = {patho: [] for n, patho in self.pathos.items()}
        for patient in list(unique_patients):
            patient_rows = self.data[self.data.patient == patient]
            type_ = self.pathos[patient_rows.y.unique()[0]]
            mapping[type_].append(patient)

        patients_to_use = []
        total_test_patients = round(len(unique_patients) * self.args.split_ratio)
        for k, v in mapping.items():
            num_patients = len(v)
            proportion = float(num_patients) / len(unique_patients)
            num_to_use = int(round(total_test_patients * proportion))
            if num_to_use < 1:
                raise Exception("You do not have enough patients for {} cohort".format(k))
            # Randomly choose <num_to_use> patients from the array.
            patients = sample(v, num_to_use)
            if not self.args.no_print_results:
                print("number {} patients in training: {}".format(k, num_patients - len(patients)))
                print("number {} patients in test: {}".format(k, len(patients)))
            patients_to_use.extend(patients)

        train_patient_data = self.data.query('patient not in {}'.format(patients_to_use))
        test_patient_data = self.data.query('patient in {}'.format(patients_to_use))
        return [(train_patient_data.index, test_patient_data.index)]

    def get_holdout_idxs(self):
        train_patient_data = self.data[self.data.set_type == 'train']
        test_patient_data = self.data[self.data.set_type == 'test']
        return [(train_patient_data.index, test_patient_data.index)]

    def _get_kfolds_when_train_test_equal(self, x, y, folds, train_cohort, test_cohort, is_random):
        """
        Get indices for train and test kfold data assuming that number of patients in the training
        cohort is equivalent to number of patients in testing. For cases in normal kfold where we
        are just iterating over the entire dataset this will be true.

        Gathers a list of patients and splits them by pathophysiology. Then just takes a continuous
        kfold split of them.
        """
        idxs = []
        mapping = {patho: [] for n, patho in self.pathos.items()}

        if is_random:
            unique_patients = x.patient.unique()
            np.random.shuffle(unique_patients)
        else:
            unique_patients = sorted(x.patient.unique())

        for patient in unique_patients:
            patient_rows = x[x.patient == patient]
            type_ = self.pathos[y.loc[patient_rows.index].unique()[0]]
            mapping[type_].append(patient)

        for i in range(folds):
            patients_to_use = []
            for k, v in mapping.items():
                lower_bound = int(round(i * len(v) / float(folds)))
                upper_bound = int(round((i + 1) * len(v) / float(folds)))
                if upper_bound < 1:
                    raise Exception("You do not have enough patients for {} cohort".format(k))
                patients = v[lower_bound:upper_bound]
                patients_to_use.extend(patients)

            train_pt_data = x[(x.set_type == train_cohort) & (~x.patient.isin(patients_to_use))]
            test_pt_data = x[(x.set_type == test_cohort) & (x.patient.isin(patients_to_use))]
            idxs.append((train_pt_data.index, test_pt_data.index))
        return idxs

    def _get_kfolds_when_train_test_unequal(self, x, y, folds, is_random):
        # This function assumes that train is larger than test and that there are no
        # patients in the test set outside the train set
        idxs = []
        if is_random:
            test_pts = x[x.set_type == 'test'].patient.unique()
            train_pts = x[x.set_type == 'train'].patient.unique()
            np.random.shuffle(train_pts)
            np.random.shuffle(test_pts)
        else:
            test_pts = sorted(x[x.set_type == 'test'].patient.unique())
            train_pts = sorted(x[x.set_type == 'train'].patient.unique())

        tmp_split_classes = []
        for i, pt_set in enumerate([test_pts, set(train_pts).difference(set(test_pts))]):
            for pt in pt_set:
                patho = x[x.patient == pt].y.iloc[0]
                # i*2+patho is important because it signifies which bin our patient is in.
                # There are 4 choices:
                # * in test and train sets and OTHER
                # * in test and train sets and ARDS
                # * in train only and OTHER
                # * in train only and ARDS
                tmp_split_classes.append([pt, i*2+patho])
        tmp_split_classes = np.array(tmp_split_classes)
        stratified = StratifiedKFold(n_splits=folds, shuffle=False)
        for i, j in stratified.split(tmp_split_classes[:, 0], tmp_split_classes[:, 1]):
            train_pts = tmp_split_classes[:, 0][i]
            test_pts = tmp_split_classes[:, 0][j]
            train_pt_data = x[(x.set_type == 'train') & (x.patient.isin(train_pts))]
            test_pt_data = x[(x.set_type == 'test') & (x.patient.isin(test_pts))]
            idxs.append((train_pt_data.index, test_pt_data.index))
        return idxs

    def get_cross_patient_kfold_idxs(self, x, y, folds, is_random):
        """
        Get indexes to split dataset
        """
        if 'set_type' not in x.columns:
            x['set_type'] = 'train_test'
            train_cohort = 'train_test'
            test_cohort = 'train_test'
        elif len(x['set_type'].unique()) == 1:
            train_cohort = x['set_type'].unique()[0]
            test_cohort = x['set_type'].unique()[0]
        elif len(x['set_type'].unique()) == 2:
            train_cohort = 'train'
            test_cohort = 'test'

        if test_cohort != train_cohort:
            total_test_patients = len(x[x.set_type == test_cohort].patient.unique())
            total_train_patients = len(x[x.set_type == train_cohort].patient.unique())
        else:
            total_test_patients = total_train_patients = len(x.patient.unique())

        if total_test_patients != total_train_patients:
            return self._get_kfolds_when_train_test_unequal(x, y, folds, is_random)
        else:
            return self._get_kfolds_when_train_test_equal(x, y, folds, train_cohort, test_cohort, is_random)

    def get_and_fit_scaler(self, x_train):
        """
        Get the scaler we wish to use and fit it to train data.

        :param x_train: Training set we wish to fit to
        """
        if not self.args.load_scaler:
            scaler = MinMaxScaler()
            scaler.fit(x_train)
        else:
            scaler = pd.read_pickle(self.args.load_scaler)

        if self.args.save_model_to and 'kfold' in self.args.split_type:
            raise Exception('Saving a model/scaler while in kfold is not supported!')
        elif self.args.save_model_to:
            pd.to_pickle(scaler, "scaler-" + self.args.save_model_to)

        return scaler

    def perform_data_splits(self):
        y = self.data.y
        x = self.data

        if self.args.split_type == "holdout_random":
            idxs = self.get_holdout_random_idxs()
        elif self.args.split_type == 'holdout':
            idxs = self.get_holdout_idxs()
        elif self.args.split_type == 'kfold':
            idxs = self.get_cross_patient_kfold_idxs(x, y, self.args.folds, False)
        elif self.args.split_type == 'kfold_random':
            idxs = self.get_cross_patient_kfold_idxs(x, y, self.args.folds, True)
        elif self.args.split_type == 'train_all':
            idxs = [(x.index, [])]
        elif self.args.split_type == 'test_all':
            idxs = [([], x.index)]
        elif self.args.split_type == 'bootstrap':
            idxs = self.get_bootstrap_idxs()

        if len(idxs) == 0:
            raise NoIndicesError('No indices were found for split. Did you enter your arguments correctly?')

        # try dropping cols
        for col in ['hour', 'row_time', 'y', 'patient', 'ventBN', 'set_type']:
            try:
                x = x.drop(col, axis=1)
            except:
                pass

        colnames = x.columns
        for train_idx, test_idx in idxs:
            x_train = x.loc[train_idx].dropna()
            x_test = x.loc[test_idx].dropna()
            y_train = y.loc[x_train.index]
            y_test = y.loc[x_test.index]
            scaler = self.get_and_fit_scaler(x_train)
            if len(x_train) != 0:
                x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=colnames)
            if len(x_test) != 0:
                x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=colnames)
            yield (x_train, x_test, y_train, y_test)

    def _get_hyperparameterized_model(self):
        hyperparams = self._get_hyperparameters()
        if self.args.algo == 'RF':
            clf = RandomForestClassifier(**hyperparams)
        elif self.args.algo == 'MLP':
            clf = MLPClassifier(**hyperparams)
        elif self.args.algo == 'SVM':
            clf = SVC(**hyperparams)
        elif self.args.algo == 'LOG_REG':
            clf = LogisticRegression(**hyperparams)
        elif self.args.algo == 'ADA':
            clf = AdaBoostClassifier(**hyperparams)
        elif self.args.algo == 'NB':
            clf = GaussianNB(**hyperparams)
        elif self.args.algo == 'GBC':
            clf = GradientBoostingClassifier(**hyperparams)
        elif self.args.algo == 'ATS_MODEL':
            clf = RandomForestClassifier(**hyperparams)

        return clf

    def train(self, x_train, y_train):
        clf = self._get_hyperparameterized_model()
        clf.fit(x_train, y_train)
        if self.args.algo in ['RF']:
            header = '--- Feature OOB scores ---'
            oob_scores = clf.feature_importances_
            scores = sorted(oob_scores, key=lambda x: -x)
            self.feature_score_rounding = lambda x: round(x, 4)
            self.rank_order = -1
        else:
            header = '--- Feature chi2 p-values ---'
            scores, pvals = chi2(x_train, y_train)
            scores = sorted(pvals)
            self.feature_score_rounding = lambda x: round(x, 100)
            self.rank_order = 1

        table = PrettyTable()
        table.field_names = ['rank', 'feature', 'score']
        for rank, feature, score in zip(range(1, len(scores)+1), x_train.columns, scores):
            if feature not in self.feature_ranks:
                self.feature_ranks[feature] = [(rank, score)]
            else:
                self.feature_ranks[feature].append((rank, score))

            table.add_row([rank, feature, self.feature_score_rounding(score)])

        if self.args.print_feature_selection:
            print(header)
            print(table)
        self.models.append(clf)

    def _get_hyperparameters(self, algo=None):
        """
        Get hyperparameters for model according to what we sent in as arguments.

        :param algo: Override algo setting to get params for specific algo.
        """
        algo = algo if algo else self.args.algo
        split_type = self.args.split_type if self.args.split_type in ['kfold', 'holdout'] else 'kfold'
        frame_size = self.args.frame_size if self.args.frame_size in [20, 100, 400] else 20
        hyperparameter_type = self.args.hyperparameter_type if self.args.split_type != 'holdout' else 'majority'
        status_post = self.args.post_hour
        params = {
            "RF": {
                24: {
                    'kfold': {
                        20: {
                            "average": {
                                "max_depth": 5,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 33,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                            "majority": {
                                "max_depth": 5,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 15,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                        },
                        100: {
                            "average": {
                                "max_depth": 2,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 33,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                            "majority": {
                                "max_depth": 1,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 5,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                    'holdout': {
                        100: {
                            'majority': {
                                "max_depth": 6,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 5,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                            'average': {
                                "max_depth": 16,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 29,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                },
                6: {
                    'kfold': {
                        100: {
                            "average": {
                                "max_depth": 2,
                                "max_features": 'log2',
                                'criterion': 'entropy',
                                'n_estimators': 16,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                            "majority": {
                                "max_depth": 1,
                                "max_features": 'log2',
                                'criterion': 'entropy',
                                'n_estimators': 5,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                    'holdout': {
                        100: {
                            'majority': {
                                "max_depth": 3,
                                "max_features": 'log2',
                                'criterion': 'gini',
                                'n_estimators': 10,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                            'average': {
                                "max_depth": 2,
                                "max_features": 'log2',
                                'criterion': 'gini',
                                'n_estimators': 40,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                },
                12: {
                    'kfold': {
                        100: {
                            "average": {
                                "max_depth": 2,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 24,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                            "majority": {
                                "max_depth": 2,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 5,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                    # XXX these are params from 24 hr model. should be changed
                    'holdout': {
                        100: {
                            'majority': {
                                "max_depth": 2,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 15,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                            'average': {
                                "max_depth": 2,
                                "max_features": 'auto',
                                'criterion': 'gini',
                                'n_estimators': 13,
                                'oob_score': True,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                },
            },
            'ADA': {
                24: {
                    'kfold': {
                        20: {
                            "average": {
                                'n_estimators': 80,
                                'learning_rate': 0.23125,
                                'algorithm': 'SAMME.R',
                                'random_state': np.random.RandomState(),
                            },
                            "majority": {
                                'n_estimators': 120,
                                'learning_rate': 0.03125,
                                'algorithm': 'SAMME.R',
                                'random_state': np.random.RandomState(),
                            },
                        },
                        100: {
                            "average": {
                                'n_estimators': 105,
                                'learning_rate': 0.11337890625,
                                'algorithm': 'SAMME.R',
                                'random_state': np.random.RandomState(),
                            },
                            "majority": {
                                # no majority found so take average of all estimators
                                'n_estimators': 15,
                                'learning_rate': 0.03125,
                                'algorithm': 'SAMME.R',
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                    'holdout': {
                        100: {
                            'majority': {
                                # no majority found so take average of all estimators
                                'n_estimators': 15,
                                'learning_rate': 0.25,
                                'algorithm': 'SAMME',
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                },
            },
            'LOG_REG': {
                24: {
                    'kfold': {
                        20: {
                            'average': {
                                'penalty': 'l1',
                                'C': 0.0875,
                                'max_iter': 100,
                                'tol': 0.000420004,
                                'solver': 'liblinear',
                                'random_state': np.random.RandomState(),
                            },
                            'majority': {
                                'penalty': 'l1',
                                'C': 0.0625,
                                'max_iter': 100,
                                'tol': 1e-8,
                                'solver': 'liblinear',
                                'random_state': np.random.RandomState(),
                            },
                        },
                        100: {
                            'average': {
                                'penalty': 'l1',
                                'C': 0.0675,
                                'max_iter': 230,
                                'tol': 0.00053800003,
                                'solver': 'liblinear',
                                'random_state': np.random.RandomState(),
                            },
                            'majority': {
                                'penalty': 'l1',
                                'C': 0.0625,
                                'max_iter': 100,
                                'tol': 1e-3,
                                'solver': 'liblinear',
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                    'holdout': {
                        100: {
                            'majority': {
                                'penalty': 'l2',
                                'C': 0.5,
                                'max_iter': 100,
                                'tol': .001,
                                'solver': 'sag',
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                },
            },
            'SVM': {
                24: {
                    'kfold': {
                        20: {
                            'average': {
                                'C': 7,
                                'kernel': 'sigmoid',
                                'cache_size': 512,
                                'random_state': np.random.RandomState(),
                            },
                            'majority': {
                                'C': 8,
                                'kernel': 'sigmoid',
                                'cache_size': 512,
                                'random_state': np.random.RandomState(),
                            },
                        },
                        100: {
                            'average': {
                                'C': 2.2130859375,
                                'kernel': 'poly',
                                'cache_size': 512,
                                'degree': 2,
                                'gamma': 'scale',
                                'tol': 0.004006,
                                'random_state': np.random.RandomState(),
                            },
                            'majority': {
                                'C': 0.03125,
                                'kernel': 'poly',
                                'cache_size': 512,
                                'degree': 2,
                                'gamma': 'scale',
                                'tol': 1e-5,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                    'holdout': {
                        100: {
                            'majority': {
                                'C': 0.03125,
                                'kernel': 'rbf',
                                'cache_size': 512,
                                'gamma': 'scale',
                                'tol': 1e-5,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                },
            },
            'MLP': {
                24: {
                    'kfold': {
                        20: {
                            'average': {
                                "hidden_layer_sizes": (38, 16),
                                "solver": 'adam',
                                'activation': 'identity',
                                'learning_rate_init': .07525,
                                'random_state': np.random.RandomState(),
                            },
                            'majority': {
                                'hidden_layer_sizes': (32, 16),
                                "solver": 'adam',
                                'activation': 'identity',
                                'learning_rate_init': .1,
                                'random_state': np.random.RandomState(),
                            },
                        },
                        100: {
                            'average': {
                                "hidden_layer_sizes": (32, 32),
                                "solver": 'sgd',
                                'activation': 'identity',
                                'learning_rate_init': .025058,
                                'random_state': np.random.RandomState(),
                            },
                            'majority': {
                                'hidden_layer_sizes': (32, 32),
                                "solver": 'sgd',
                                'activation': 'identity',
                                'learning_rate_init': .0005,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                    'holdout': {
                        100: {
                            'majority': {
                                "hidden_layer_sizes": (32, 64),
                                "solver": 'sgd',
                                'activation': 'relu',
                                'learning_rate_init': .0005,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                },
            },
            'GBC': {
                24: {
                    'kfold': {
                        20: {
                            'average': {
                                'n_estimators': 180,
                                'criterion': 'mae',
                                'loss': 'exponential',
                                'max_features': 'log2',
                                'n_iter_no_change': 100,
                                'random_state': np.random.RandomState(),
                            },
                            'majority': {
                                'n_estimators': 50,
                                'criterion': 'mae',
                                'loss': 'exponential',
                                'max_features': 'log2',
                                'n_iter_no_change': 100,
                                'random_state': np.random.RandomState(),
                            },
                        },
                        100: {
                            'average': {
                                'n_estimators': 180,
                                'criterion': 'mae',
                                'loss': 'exponential',
                                'max_features': 'log2',
                                'n_iter_no_change': 100,
                                'random_state': np.random.RandomState(),
                            },
                            'majority': {
                                'n_estimators': 50,
                                'criterion': 'mae',
                                'loss': 'exponential',
                                'max_features': 'log2',
                                'n_iter_no_change': 100,
                                'random_state': np.random.RandomState(),
                            },
                        },
                    },
                },
            },
            'NB': {
                24: {
                    'kfold': {
                        20: {
                            'average': {
                                'var_smoothing': .244,
                            },
                            'majority': {
                                'var_smoothing': 0.01,
                            },
                        },
                        100: {
                            'average': {
                                'var_smoothing': .244,
                            },
                            'majority': {
                                'var_smoothing': 0.01,
                            },
                        },
                    },
                    'holdout': {
                        100: {
                            'majority': {
                                'var_smoothing': 0.1,
                            },
                        },
                    },
                },
            },
            'ATS_MODEL': {
                24: {
                    'holdout': {
                        100: {
                            'majority': {
                                'oob_score': True,
                            },
                        },
                    },
                    'kfold': {
                        20: {
                            'average': {
                                'oob_score': True,
                            },
                        },
                        100: {
                            'average': {
                                'oob_score': True,
                            },
                        },
                        400: {
                            'average': {
                                'oob_score': True,
                            },
                        },
                    },
                },
            },
        }
        try:
            return params[algo][status_post][split_type][frame_size][hyperparameter_type]
        except KeyError:
            raise ValueError('We were unable to find hyperparams for choice algo: {}, status post: {}, split: {}, fs: {}, hyperparam type: {}. Check args or add new hyperparams'.format(algo, status_post, split_type, frame_size, hyperparameter_type))

    def train_and_test(self):
        """
        Train models and then run testing afterwards.
        """
        for model_idx, (x_train, x_test, y_train, y_test) in enumerate(self.perform_data_splits()):
            if not self.args.no_print_results and 'kfold' in self.args.split_type:
                print("----Run fold {}----".format(model_idx+1))

            for iter_n in range(self.args.n_runs):
                if self.args.grid_search:
                    self.perform_grid_search(x_train, y_train)
                    prediction_set = x_test
                elif self.args.feature_selection_method:
                    # sometimes x_test is modified by this func
                    prediction_set = self.perform_feature_selection(x_train, y_train, x_test)
                elif not self.args.load_model:
                    self.train(x_train, y_train)
                    prediction_set = x_test

                if self.args.save_model_to and 'kfold' in self.args.split_type:
                    raise Exception('Saving a model/scaler while in kfold is not supported!')
                elif self.args.save_model_to:
                    pd.to_pickle(self.models[-1], "model-" + self.args.save_model_to)

                predictions = pd.Series(self.models[-1].predict(prediction_set), index=y_test.index)
                self.results.add_model(y_test, predictions, self.data.loc[y_test.index], model_idx)

            self.results.calc_fold_stats(.5, model_idx, print_results=not self.args.no_print_results)
            if not self.args.no_print_results:
                print("-------------------")

        self.results.calc_aggregate_stats(.5, print_results=not self.args.no_print_results)

        if self.args.print_thresh_table:
            self.results.print_thresh_table(self.args.thresh_interval)

        if not self.args.no_print_results:
            self.results.get_youdens_results()

        if self.args.plot_predictions or self.args.plot_disease_evolution or self.args.plot_dtw_with_disease:
            self.plot_predictions()

        if self.args.plot_pairwise_features:
            self.plot_pairwise_feature_visualizations()

        if self.args.grid_search:
            self.aggregate_grid_search_results()

        if self.args.plot_roc_all_folds:
            self.results.plot_roc_all_folds()

        if self.args.plot_sen_spec_vs_thresh:
            self.results.plot_sen_spec_vs_thresh(self.args.thresh_interval)

        if self.args.print_feature_selection:
            self.print_aggregate_feature_results()

    def aggregate_grid_search_results(self):
        print("---- Grid Search Final Results ----")
        print("----")
        grid_results = {}
        for model in self.models:
            for param in model.best_params_:
                if param not in grid_results:
                    grid_results[param] = []
                grid_results[param].append(model.best_params_[param])
        pprint(grid_results)
        print('---- Grid Search Averages ----')
        for param in grid_results:
            first_inst = grid_results[param][0]
            if isinstance(first_inst, int) or isinstance(first_inst, float):
                print("param: {}. average: {}".format(param, sum(grid_results[param]) / len(grid_results[param])))
        print('---- Grid Search Majority ----')
        for param in grid_results:
            counts = Counter(grid_results[param])
            param_val, _ = max(counts.items(), key=lambda x: x[1])
            print('param: {}. majority: {}'.format(param, param_val))

    def convert_loc_to_iloc(self, df, loc_indices):
        copied = df.copy()
        copied['idx'] = range(len(df))
        return [(copied.loc[train_loc, 'idx'].values, copied.loc[test_loc, 'idx'])
                for train_loc, test_loc in loc_indices]

    def perform_grid_search(self, x_train, y_train):
        if self.args.algo == 'RF':
            self._perform_rf_grid_search(x_train, y_train)
        elif self.args.algo == 'MLP':
            self._perform_mlp_grid_search(x_train, y_train)
        elif self.args.algo == 'SVM':
            self._perform_svm_grid_search(x_train, y_train)
        elif self.args.algo == 'LOG_REG':
            self._perform_log_reg_grid_search(x_train, y_train)
        elif self.args.algo == 'ADA':
            self._perform_adaboost_grid_search(x_train, y_train)
        elif self.args.algo == 'NB':
            self._perform_nb_grid_search(x_train, y_train)
        elif self.args.algo == 'GBC':
            self._perform_gbc_grid_search(x_train, y_train)

    def _perform_rf_grid_search(self, x_train, y_train):
        params = {
            "n_estimators": range(5, 140, 5),
            "max_features": ['auto', 'log2', None],
            "criterion": ["entropy", 'gini'],
            "max_depth": range(1, 30, 1) + [None],
            #"criterion": ["entropy"],
        }
        self._perform_grid_search(RandomForestClassifier(), params, x_train, y_train)

    def _perform_mlp_grid_search(self, x_train, y_train):
        hiddens = [
            (16,), (32,), (64,),
            (16, 16), (16, 32), (16, 64),
            (32, 16), (32, 32), (32, 64),
            (64, 16), (64, 32), (64, 64),
        ]
        activations = ['relu', 'tanh', 'logistic']
        params = [{
            'activation': activations,
            'solver': ['lbfgs'],
            'hidden_layer_sizes': hiddens,
        }, {
            'activation': activations,
            'solver': ['sgd', 'adam'],
            'learning_rate_init': [.0001, .0005, .001, .005, .01],
            'hidden_layer_sizes': hiddens,
            # cut down on # params to search otherwise it will take forever
            #'alpha': [0.00001, .0001, .001, .01, .1],
            #'batch_size': [8, 16, 32, 64, 128, 256],
        }]
        self._perform_grid_search(MLPClassifier(), params, x_train, y_train)

    def _perform_svm_grid_search(self, x_train, y_train):
        C = [2**i for i in range(-10, 1)] + [i for i in range(2, 17)]
        tol = [10**i for i in range(-5, -1)]
        params = [{
            'C': C,
            'kernel': ['rbf', 'sigmoid'],
            'gamma': ['auto', 'scale'],
            'tol': tol,
        }, {
            'C': C,
            'kernel': ['linear'],
            'tol': tol,
        }, {
            'C': C,
            'kernel': ['poly'],
            'gamma': ['auto', 'scale'],
            'degree': range(1, 10),
            'tol': tol,
        }]
        self._perform_grid_search(SVC(cache_size=512), params, x_train, y_train)

    def _perform_adaboost_grid_search(self, x_train, y_train):
        params = {
            'learning_rate': [2**i for i in range(-15, 3)],
            'n_estimators': [5*i for i in range(1, 60)],
            'algorithm': ['SAMME', 'SAMME.R'],
        }
        self._perform_grid_search(AdaBoostClassifier(), params, x_train, y_train)

    def _perform_gbc_grid_search(self, x_train, y_train):
        # Another case where we cannot go too crazy on tuning everything
        params = {
            'loss': ['deviance', 'exponential'],
            # I need to reduce the parameters because grid search is running to long.
            #'learning_rate': [2**i for i in range(-5, 1)],
            'n_estimators': [50*i for i in range(1, 8)],
            'criterion': ['friedman_mse', 'mse', 'mae'],
            'max_features': [None, 'log2', 'auto'],
        }
        # implement early stopping because grid search is taking too long
        self._perform_grid_search(GradientBoostingClassifier(n_iter_no_change=100), params, x_train, y_train)

    def _perform_nb_grid_search(self, x_train, y_train):
        params = {
            'var_smoothing': [10**i for i in range(-14, 5)],
        }
        self._perform_grid_search(GaussianNB(), params, x_train, y_train)

    def _perform_log_reg_grid_search(self, x_train, y_train):
        params = [{
            'penalty': ['l2'],
            'solver': ['newton-cg', 'lbfgs', 'sag'],
            'C': [2**i for i in range(-5, 5)],
            'tol': [10**i for i in range(-10, -2)],
            'max_iter': [100, 200, 300, 400],
        }, {
            'penalty': ['l1'],
            'C': [2**i for i in range(-5, 5)],
            'tol': [10**i for i in range(-10, -2)],
            'solver': ['liblinear', 'saga'],
        }]
        self._perform_grid_search(LogisticRegression(), params, x_train, y_train)

    def _perform_grid_search(self, cls, params, x_train, y_train):
        x_train_expanded = self.data.loc[x_train.index]
        cv = self.get_cross_patient_kfold_idxs(x_train_expanded, y_train, self.args.grid_search_kfolds, False)
        # sklearn does CV indexing with iloc and not loc. Annoying, but can be worked around
        cv = self.convert_loc_to_iloc(x_train, cv)
        # keep 1 core around to actually do other stuff
        clf = GridSearchCV(cls, params, cv=cv, n_jobs=self.args.grid_search_jobs)
        clf.fit(x_train, y_train)
        print("Params: ", clf.best_params_)
        print("Best CV score: ", clf.best_score_)
        self.models.append(clf)

    def perform_feature_selection(self, x_train, y_train, x_test):
        clf = self._get_hyperparameterized_model()
        orig_train_idx = x_train.index
        orig_test_idx = x_test.index

        if self.args.feature_selection_method == 'RFE':
            selector = RFE(clf, self.args.n_new_features, step=1)
            selector.fit(x_train, y_train)
            self.selected_features = list(x_train.columns[selector.support_])
            if not self.args.no_print_results:
                print('Selected features: {}'.format(self.selected_features))
            self.models.append(selector)
        elif self.args.feature_selection_method == 'chi2':
            selector = SelectKBest(chi2, k=self.args.n_new_features)
            x_train = selector.fit_transform(x_train, y_train)
            self.selected_features = list(x_test.columns[selector.get_support()])
            x_train = pd.DataFrame(x_train, columns=self.selected_features, index=orig_train_idx)
            if not self.args.no_print_results:
                print('Selected features: {}'.format(self.selected_features))
            x_test = pd.DataFrame(selector.transform(x_test), columns=self.selected_features, index=orig_test_idx)
            clf.fit(x_train, y_train)
            self.models.append(clf)
        elif self.args.feature_selection_method == 'mutual_info':
            func = lambda x, y: mutual_info_classif(x, y, discrete_features=False)
            selector = SelectKBest(func, k=self.args.n_new_features)
            x_train = selector.fit_transform(x_train, y_train)
            self.selected_features = list(x_test.columns[selector.get_support()])
            if not self.args.no_print_results:
                print('Selected features: {}'.format(self.selected_features))
            x_test = pd.DataFrame(selector.transform(x_test), columns=self.selected_features)
            clf.fit(x_train, y_train)
            self.models.append(clf)
        elif self.args.feature_selection_method == 'gini':
            rf = RandomForestClassifier(n_estimators=25)
            selector = SelectFromModel(rf, threshold=self.args.select_from_model_thresh)
            selector.fit(x_train, y_train)
            self.selected_features = list(x_test.columns[selector.get_support()])
            if len(self.selected_features) == 0:
                raise NoFeaturesSelectedError('No features selected via gini. Maybe lower --select-from-model-thresh param')
            if not self.args.no_print_results:
                print('Selected features: {}'.format(self.selected_features))
            x_train = selector.transform(x_train)
            x_train = pd.DataFrame(x_train, columns=self.selected_features, index=orig_train_idx)
            x_test = pd.DataFrame(selector.transform(x_test), columns=self.selected_features)
            clf.fit(x_train, y_train)
            self.models.append(clf)
        elif self.args.feature_selection_method == 'lasso':
            lasso = LassoCV(cv=5)
            selector = SelectFromModel(lasso, threshold=self.args.select_from_model_thresh)
            selector.fit(x_train, y_train)
            self.selected_features = list(x_test.columns[selector.get_support()])
            if len(self.selected_features) == 0:
                raise NoFeaturesSelectedError('No features selected via lasso. Maybe lower --select-from-model-thresh param')
            if not self.args.no_print_results:
                print('Selected features: {}'.format(self.selected_features))
            x_train = selector.transform(x_train)
            x_test = pd.DataFrame(selector.transform(x_test), columns=self.selected_features)
            clf.fit(x_train, y_train)
            self.models.append(clf)
        elif self.args.feature_selection_method == 'PCA':
            pca = PCA(n_components=self.args.n_new_features)
            pca.fit(x_train, y_train)
            x_train = pd.DataFrame(pca.transform(x_train), index=x_train.index)
            x_test = pd.DataFrame(pca.transform(x_test), index=x_test.index)
            clf.fit(x_train, y_train)
            self.models.append(clf)

        return x_test

    def plot_predictions(self):
        """
        Plot votes on specific class was predicted for each patient.
        """
        colors = ['sky blue', 'deep red', 'eggplant']
        #fontname = 'Osaka'
        #cmap = sns.color_palette(sns.xkcd_palette(colors))
        cmap = ['#6c89b7', '#ff919c']
        plt.rcParams['font.family'] = 'Osaka'
        plt.rcParams['legend.loc'] = 'upper right'
        if self.args.plot_predictions:
            # plot fraction of votes for all patients
            step_size = 10
            for i in range(0, len(self.patient_results), step_size):
                slice = self.patient_results.loc[i:i+step_size-1]
                slice = slice.sort_values(by='patient')
                ind = np.arange(len(slice))
                bottom = np.zeros(len(ind))
                plots = []
                totals = [0] * len(ind)
                vote_cols = ['{}_votes'.format(patho) for _, patho in self.pathos.items()]
                # get sum of votes across all patients
                total_votes = slice[vote_cols].sum(axis=1).values
                for n, patho in self.pathos.items():
                    plots.append(plt.bar(ind, slice['{}_votes'.format(patho)].values / total_votes, bottom=bottom, color=cmap[n]))
                    bottom = bottom + (slice['{}_votes'.format(patho)].values / total_votes)
                plt.xticks(ind, slice.patient.str[:4].values, rotation=45)
                plt.ylabel('Fraction Votes')
                plt.xlabel('Patient Token ID')
                plt.legend([p[0] for p in plots], [patho for _, patho in self.pathos.items()], fontsize=11)
                plt.yticks(np.arange(0, 1.01, .1))
                plt.subplots_adjust(bottom=0.15)
                plt.show()

        if self.args.plot_dtw_with_disease:
            hourly_predictions = self.results.get_all_hourly_preds()
            for _, pt_rows in hourly_predictions.groupby('patient_id'):
                pt = pt_rows.iloc[0].patient_id
                dtw_lib.analyze_patient(pt, self.args.data_path, self.args.cohort_description, self.args.dtw_cache_dir)

            if not self.args.tiled_disease_evol:
                # ensure all results are processed and cached first

                for _, pt_rows in hourly_predictions.groupby('patient_id'):
                    self.plot_disease_evolution(pt_rows, cmap)
                    self.plot_dtw_patient_data(pt_rows, True, 1, True)
                    plt.show()
            else:
                self.plot_tiled_disease_evol(hourly_predictions, cmap, True)

        if self.args.plot_disease_evolution:
            # Plot fraction of votes for a single patient over 24 hrs.
            hourly_predictions = self.results.get_all_hourly_preds()
            if not self.args.tiled_disease_evol:
                for _, pt_rows in hourly_predictions.groupby('patient_id'):
                    self.plot_disease_evolution(pt_rows, cmap)
                    plt.show()
            else:
                self.plot_tiled_disease_evol(hourly_predictions, cmap, False)

    def plot_dtw_patient_data(self, pt_rows, set_label, lw, xy_visible):
        """
        Plot DTW for an individual patient

        :param pt_rows: Rows grouped by patient from the dataframe received from
                        self.results.get_all_hourly_preds
        """
        pt = pt_rows.iloc[0].patient_id
        dtw = dtw_lib.analyze_patient(pt, self.args.data_path, self.args.cohort_description, self.args.dtw_cache_dir)
        ax2 = plt.gca().twinx()
        ax2.plot(dtw[:, 0], dtw[:, 1], lw=lw, label='DTW', color='#663a3e')
        if set_label:
            ax2.set_ylabel('DTW Score')
        if not xy_visible:
            ax2.set_yticks([])
            ax2.set_xticks([])

    def plot_tiled_disease_evol(self, hourly_predictions, cmap, plot_with_dtw):
        """
        just focus on ARDS for now.

        want to have true ARDS, false pos ARDS, false neg ARDS, and true neg ARDS
        """
        pts = self.results.get_all_patient_results_dataframe()
        tps, tns, fps, fns = [], [], [], []
        for i, rows in pts.groupby('patient_id'):
            pt = rows.iloc[0].patient_id
            total_votes = rows[['other_votes', 'ards_votes']].sum().sum()
            ards_votes = rows['ards_votes'].sum()
            ground_truth = rows.iloc[0].ground_truth
            if ards_votes / float(total_votes) >= .5:
                pred = 1
            else:
                pred = 0

            if pred == 1 and ground_truth == 1:
                tps.append(pt)
            elif pred == 0 and ground_truth == 0:
                tns.append(pt)
            elif pred == 1 and ground_truth == 0:
                fps.append(pt)
            elif pred != 1 and ground_truth == 1:
                fns.append(pt)

        for arr, title in [
            (tps, 'ARDS True Pos'),
            (tns, 'ARDS True Neg'),
            (fps, 'ARDS False Pos'),
            (fns, 'ARDS False Neg'),
        ]:
            for idx, pt in enumerate(arr):
                layout = int(ceil(sqrt(len(arr))))
                plt.suptitle(title)
                pt_rows = hourly_predictions[hourly_predictions.patient_id == pt]
                plt.subplot(layout, layout, idx+1)
                self.plot_disease_evolution(pt_rows, cmap, legend=False, fontsize=6, xylabel=False, xy_visible=False)
                if plot_with_dtw:
                    self.plot_dtw_patient_data(pt_rows, False, .08, False)
            plt.show()

    def plot_disease_evolution(self, pt_rows, cmap, legend=True, fontsize=11, xylabel=True, xy_visible=True):
        """
        :param pt_rows: Result from self.results.get_all_hourly_preds but grouped by patient
        :param cmap: color map palette.
        """
        pt = pt_rows.iloc[0].patient_id
        bar_data = [[0] * len(self.pathos) for _ in range(24)]
        for hour in range(0, 24):
            ards_colname = 'hour_{}_ards_votes'.format(hour)
            other_colname = 'hour_{}_other_votes'.format(hour)
            colnames = [other_colname, ards_colname]
            pt_rows[colnames]
            all_votes = pt_rows[colnames].sum().sum()
            bar_data[hour] = [pt_rows[other_colname].sum() / all_votes, pt_rows[ards_colname].sum() / all_votes]

        plots = []
        bottom = np.zeros(24)
        for n in self.pathos:
            bar_fracs = np.array([bar_data[hour][n] for hour in range(0, 24)])
            plots.append(plt.bar(range(0, 24), bar_fracs, bottom=bottom, color=cmap[n]))
            bottom = bottom + bar_fracs

        plt.title("Patient {}".format(pt[:4]), fontsize=fontsize, pad=1)
        if xylabel:
            plt.ylabel('Fraction Predicted', fontsize=fontsize)
            plt.xlabel('Hour', fontsize=fontsize)
        plt.xlim(-.8, 23.8)
        if legend:
            votes_cols = list(set(pt_rows.columns).difference(['patient_id']))
            all_votes = pt_rows[votes_cols].sum().sum()
            ards_vote_cols = ['hour_{}_ards_votes'.format(hour) for hour in range(24)]
            other_vote_cols = ['hour_{}_other_votes'.format(hour) for hour in range(24)]
            mapping = {
                'Non-ARDS_percent': round(pt_rows[other_vote_cols].sum().sum() / all_votes, 3) * 100,
                'ARDS_percent': round(pt_rows[ards_vote_cols].sum().sum() / all_votes, 3) * 100,
            }

            plt.legend([
                "{}: {}%".format(patho, mapping['{}_percent'.format(patho)]) for patho in ['Non-ARDS', 'ARDS']
            ], fontsize=fontsize)
        if not xy_visible:
            plt.yticks([])
            plt.xticks([])
        else:
            plt.yticks(np.arange(0, 1.01, .1))
            plt.xticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])

    def plot_pairwise_feature_visualizations(self):
        """
        Visualize predictions versus feature relationships in the data.
        """
        max_features_per_plot = 5
        all_rows = None
        all_preds = None
        new_cols = None
        for pt, (rows, preds) in self.patient_predictions.items():
            rows = rows.drop(['ventBN', 'hour', 'set_type'], axis=1)
            new_cols = rows.columns
            if all_rows is None:
                all_rows = rows.values
                all_preds = preds.values
            else:
                all_rows = np.append(all_rows, rows.values, axis=0)
                all_preds = np.append(all_preds, preds.values, axis=0)

        all_rows = pd.DataFrame(all_rows, columns=new_cols)
        all_rows['preds'] = all_preds
        to_plot = sorted(list(set(all_rows.columns).difference(set(['y', 'preds', 'patient', 'row_time']))))
        for i in range(0, len(to_plot), max_features_per_plot):
            sns.pairplot(all_rows, vars=to_plot[i:i+max_features_per_plot], hue="preds")
            plt.show()

    def print_aggregate_feature_results(self):
        feature_avg_scores = []
        feature_all_ranks = {}
        for feature in self.feature_ranks:
            feature_avg_scores.append((feature, np.mean([x[1] for x in self.feature_ranks[feature]])))
            feature_all_ranks[feature] = [str(x[0]) for x in self.feature_ranks[feature]]
        table = PrettyTable()
        table.field_names = ['feature', 'avg_score', 'avg_rank']
        feature_avg_scores = sorted(feature_avg_scores, key=lambda x: self.rank_order * x[1])
        for feature, score in feature_avg_scores:
            table.add_row([feature, self.feature_score_rounding(score), np.mean(np.array(feature_all_ranks[feature]).astype(int))])
        print(table)


def create_df(args):
    """
    Create dataframe for use in model

    :param args: Arguments from CLI parser
    """
    if args.from_pickle:
        return pd.read_pickle(args.from_pickle)
    data_cls = Dataset(
        args.data_path,
        args.cohort_description,
        args.feature_set,
        args.frame_size,
        args.no_load_intermediates,
        args.experiment,
        args.post_hour,
        args.start_hour_delta,
        args.frame_func,
        args.split_type,
        args.test_frame_size,
        args.test_post_hour,
        args.test_start_hour_delta,
        use_ehr_features=args.use_ehr_features,
        use_demographic_features=args.use_demographic_features,
    )
    if args.load_from_unframed:
        unframed = pd.read_pickle(args.load_from_unframed)
        df = data_cls.get_framed_from_unframed_dataset(unframed)
    else:
        df = data_cls.get()
    # Perform evaluation on number of frames dropped if we want
    if args.print_dropped_frame_eval:
        table = PrettyTable()
        table.field_names = ['patient', 'Current Frames', 'Frames Dropped', '% Dropped']
        for patient, frames_dropped in data_cls.frames_dropped.items():
            n_frames_cur = len(df[df.patient == patient])
            table.add_row([patient, n_frames_cur, frames_dropped, round(100 * float(frames_dropped) / (frames_dropped+n_frames_cur), 3)])
        print(table)

    if args.to_pickle:
        df.to_pickle(args.to_pickle)

    return df


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection')
    parser.add_argument('--cohort-description', default='cohort-description.csv', help='path to cohort description file')
    parser.add_argument("--feature-set", default="flow_time", choices=Dataset.vent_feature_sets)
    parser.add_argument('--no-load-intermediates', action='store_false', help='do not load from intermediate data')
    parser.add_argument('-sr', '--split-ratio', type=float, default=.2)
    parser.add_argument("--grid-search", action="store_true", help='perform a grid search  for model hyperparameters')
    parser.add_argument("--grid-search-kfolds", type=int, default=4, help='number of validation kfolds to use in the grid search')
    parser.add_argument('--split-type', choices=['holdout', 'holdout_random', 'kfold', 'kfold_random', 'train_all', 'test_all', 'bootstrap'], help='All splits are performed so there is no test/train patient overlap', default='kfold')
    parser.add_argument('--save-model-to', help='save model+scaler to a pickle file')
    parser.add_argument('--load-model')
    parser.add_argument('--load-scaler')
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument('-fs', "--frame-size", type=int, default=100)
    parser.add_argument('-ff', '--frame-func', choices=['median', 'mean', 'var', 'std', 'mean+var', 'mean+std', 'median+var', 'median+std'], default='median')
    parser.add_argument('-sd', '--start-hour-delta', default=0, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data')
    parser.add_argument('-sp', '--post-hour', type=int, default=24)
    parser.add_argument('-tfs', "--test-frame-size", default=None, type=int)
    parser.add_argument('-tsd', '--test-start-hour-delta', default=None, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data. Only for usage in testing set')
    parser.add_argument('-tsp', '--test-post-hour', default=None, type=int)
    parser.add_argument("--to-pickle", help="name of file the data frame will be pickled in")
    parser.add_argument("-p", "--from-pickle", help="name of file to retrieve pickled data from")
    parser.add_argument('-e', '--experiment', help='Experiment number we wish to run. If you wish to mix patients from different experiments you can do <num>+<num>+... eg. 1+3  OR 1+2+3', default='1')
    parser.add_argument("--no-copd-to-ctrl", action="store_true", help='Dont convert copd annotations to ctrl annotations')
    parser.add_argument('--no-print-results', action='store_true', help='Dont print results of our model')
    parser.add_argument('--plot-disease-evolution', action='store_true', help='Plot evolution of disease over time')
    parser.add_argument('--plot-predictions', action='store_true', help='Plot prediction bars')
    parser.add_argument('--tiled-disease-evol', action='store_true', help='Plot disease evolution in tiled manner')
    parser.add_argument('--plot-pairwise-features', action='store_true', help='Plot pairwise relationships between features to better visualize their relationships and predictions')
    parser.add_argument('--algo', help='The type of algorithm you want to do ML with', choices=['RF', 'MLP', 'SVM', 'LOG_REG', 'GBC', 'NB', 'ADA', 'ATS_MODEL'], default='RF')
    parser.add_argument('-gsj', '--grid-search-jobs', type=int, default=multiprocessing.cpu_count(), help='run grid search with this many cores')
    parser.add_argument('-ehr', '--use-ehr-features', action='store_true', help='use EHR data in learning')
    parser.add_argument('-demo', '--use-demographic-features', action='store_true', help='use demographic data in learning')
    parser.add_argument('-ht', '--hyperparameter-type', choices=['average', 'majority'], default='average')
    parser.add_argument('-pdfe', '--print-dropped-frame-eval', action='store_true', help='Print evaluation of all the frames we drop')
    parser.add_argument('--load-from-unframed', help='create new framed dataset from an existing unframed one')
    parser.add_argument('-fsm', '--feature-selection-method', choices=['RFE', 'chi2', 'mutual_info', 'gini', 'lasso', 'PCA'], help='Feature selection method')
    parser.add_argument('--n-new-features', type=int, help='number of features to select using feature selection', default=1)
    parser.add_argument('--select-from-model-thresh', type=float, default=.2, help='Threshold to use for feature importances when using lasso and gini selection')
    parser.add_argument('--plot-sen-spec-vs-thresh', action='store_true', help='Plot the sensitivity and specificity values versus the ARDS threshold used')
    parser.add_argument('--plot-roc-all-folds', action='store_true', help='Plot ROC curve but with individual roc curves and then an average.')
    parser.add_argument('--thresh-interval', type=int, default=25)
    parser.add_argument('--print-thresh-table', action='store_true')
    parser.add_argument('--n-runs', type=int, help='number of times to run the model', default=10)
    parser.add_argument('--print-feature-selection', action='store_true')
    parser.add_argument('--bootstrap-n-pts', type=int, default=80, help='number of patients to sample on a single bootstrap')
    parser.add_argument('--no-bootstrap-replace', action='store_false', help='Dont use replacement when sampling patients with bootstrap')
    parser.add_argument('--n-bootstraps', type=int, default=10, help='number of bootstrapped patient samplees to take')
    parser.add_argument('--plot-dtw-with-disease', action='store_true', help='Plot DTW observations by hour versus patient disease evolution')
    parser.add_argument('--dtw-cache-dir', default='dtw_cache')
    return parser


def main():
    args = build_parser().parse_args()
    df = create_df(args)
    model = ARDSDetectionModel(args, df)
    model.train_and_test()


if __name__ == "__main__":
    main()
