"""
learn
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from prettytable import PrettyTable
import seaborn as sns
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif, RFE, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from collate import Dataset
from metrics import *

sns.set()
sns.set_style('ticks')
sns.set_context('paper')


class ARDSDetectionModel(object):

    def __init__(self, args, data):
        """
        :param args: CLI args
        :param data: DataFrame containing train+test data
        """
        self.args = args
        self.data = data
        self.pathos = {0: 'OTHER', 1: 'ARDS', 2: 'COPD'}
        if not self.args.no_copd_to_ctrl:
            self.data.loc[self.data[self.data.y == 2].index, 'y'] = 0
            del self.pathos[2]

        self.models = []
        if self.args.load_model:
            self.models.append(pd.read_pickle(self.args.load_model))

        results_cols = ["patient", "patho"]
        for n, patho in self.pathos.items():
            results_cols.extend([
                "{}_tps".format(patho), "{}_fps".format(patho),
                "{}_tns".format(patho), "{}_fns".format(patho),
                "{}_votes".format(patho),
            ])
        results_cols += ["model_idx", "prediction"]
        # self.results is meant to be a high level dataframe of aggregated statistics
        # from our model.
        #
        # self.patient_results is redundant with self.results. We should probably find
        # a way to consolidate the two
        self.results = pd.DataFrame([], columns=results_cols)
        self.patient_results = pd.DataFrame(
            [], columns=['patient'] + ["{}_votes".format(patho) for _, patho in self.pathos.items()] + ['actual']
        )
        # self.patient_predictions is a dictionary of low level data for each patient that
        # consists of x, y, and prediction frames for each patient. The format is like so:
        #
        # {
        #   patient: (
        #       x+y,
        #       predictions
        #   ),
        #   patient2: (
        #       ...
        #   )
        #   ...
        # }
        self.patient_predictions = {}

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

    def get_cross_patient_kfold_idxs(self, x, y, folds):
        """
        Get indexes to split dataset
        """
        idxs = []
        unique_patients = sorted(x.patient.unique())
        mapping = {patho: [] for n, patho in self.pathos.items()}

        for patient in unique_patients:
            patient_rows = x[x.patient == patient]
            type_ = self.pathos[y.loc[patient_rows.index].unique()[0]]
            mapping[type_].append(patient)

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

        total_test_patients = round(len(unique_patients) / float(folds))
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

        if self.args.save_model_to and self.args.split_type == 'kfold':
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
            idxs = self.get_cross_patient_kfold_idxs(x, y, self.args.folds)
        elif self.args.split_type == 'train_all':
            idxs = [(x.index, [])]
        elif self.args.split_type == 'test_all':
            idxs = [([], x.index)]

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
        return clf

    def train(self, x_train, y_train):
        clf = self._get_hyperparameterized_model()
        clf.fit(x_train, y_train)
        if self.args.algo == 'RF' and not self.args.no_print_results:
            print('--- OOB scores ---')
            oob_table = PrettyTable()
            oob_table.field_names = ['feature', 'score']
            oob_scores = zip(x_train.columns, clf.feature_importances_)
            for feature, score in oob_scores:
                oob_table.add_row([feature, round(score, 4)])
            print(oob_table)
        self.models.append(clf)

    def _get_hyperparameters(self):
        params = {
            "RF": {
                20: {
                    "average": {
                        "random_state": 1,
                        "max_depth": 5,
                        "max_features": 'auto',
                        'criterion': 'gini',
                        'n_estimators': 33,
                        'oob_score': True,
                    },
                    "majority": {
                        "random_state": 1,
                        "max_depth": 5,
                        "max_features": 'auto',
                        'criterion': 'gini',
                        'n_estimators': 15,
                        'oob_score': True,
                    },
                },
                100: {
                    "average": {
                        "random_state": 1,
                        "max_depth": 5,
                        "max_features": 'auto',
                        'criterion': 'entropy',
                        'n_estimators': 84,
                        'oob_score': True,
                    },
                    "majority": {
                        "random_state": 1,
                        "max_depth": 5,
                        "max_features": 'auto',
                        'criterion': 'entropy',
                        'n_estimators': 95,
                        'oob_score': True,
                    },
                },
            },
            'ADA': {
                20: {
                    "average": {
                        'random_state': 1,
                        'n_estimators': 80,
                        'learning_rate': 0.23125,
                        'algorithm': 'SAMME.R',
                    },
                    "majority": {
                        'random_state': 1,
                        'n_estimators': 120,
                        'learning_rate': 0.03125,
                        'algorithm': 'SAMME.R',
                    },
                },
                100: {
                    "average": {
                        'random_state': 1,
                        'n_estimators': 164,
                        'learning_rate': 0.3625,
                        'algorithm': 'SAMME.R',
                    },
                    "majority": {
                        'random_state': 1,
                        # no majority found so take average of all estimators
                        'n_estimators': 164,
                        'learning_rate': 0.03125,
                        'algorithm': 'SAMME.R',
                    },
                },
            },
            'LOG_REG': {
                20: {
                    'average': {
                        'random_state': 1,
                        'penalty': 'l1',
                        'C': 0.0875,
                        'max_iter': 100,
                        'tol': 0.000420004,
                        'solver': 'liblinear',
                    },
                    'majority': {
                        'random_state': 1,
                        'penalty': 'l1',
                        'C': 0.0625,
                        'max_iter': 100,
                        'tol': 1e-8,
                        'solver': 'liblinear',
                    },
                },
                100: {
                    'average': {
                        'random_state': 1,
                        'penalty': 'l1',
                        'C': 0.05625,
                        'max_iter': 100,
                        'tol': 0.000202006,
                        'solver': 'liblinear',
                    },
                    'majority': {
                        'random_state': 1,
                        'penalty': 'l1',
                        'C': 0.03125,
                        'max_iter': 100,
                        'tol': 1e-08,
                        'solver': 'liblinear',
                    },
                },
            },
            'SVM': {
                20: {
                    'average': {
                        'C': 7,
                        'kernel': 'sigmoid',
                        'cache_size': 512,
                        'random_state': 1,
                    },
                    'majority': {
                        'C': 8,
                        'kernel': 'sigmoid',
                        'cache_size': 512,
                        'random_state': 1,
                    },
                },
                100: {
                    'average': {
                        'C': 9,
                        'kernel': 'poly',
                        'cache_size': 512,
                        'random_state': 1,
                        'degree': 2,
                    },
                    'majority': {
                        'C': 16,
                        'kernel': 'poly',
                        'cache_size': 512,
                        'random_state': 1,
                        'degree': 2,
                    },
                },
            },
            'MLP': {
                20: {
                    'average': {
                        "hidden_layer_sizes": (38, 16),
                        "solver": 'adam',
                        'activation': 'identity',
                        'learning_rate_init': .07525,
                    },
                    'majority': {
                        'hidden_layer_sizes': (32, 16),
                        "solver": 'adam',
                        'activation': 'identity',
                        'learning_rate_init': .1,
                    },
                },
                100: {
                    'average': {
                        "hidden_layer_sizes": (70, 27),
                        "solver": 'sgd',
                        'activation': 'identity',
                        'learning_rate_init': .0244,
                    },
                    'majority': {
                        'hidden_layer_sizes': (32, 32),
                        "solver": 'sgd',
                        # its even between relu and iden. choodse relu because of runs from frame size 20
                        'activation': 'identity',
                        'learning_rate_init': .005,  # its even between 0.01 and .001
                    },
                },
            },
            'GBC': {
                20: {
                    'average': {
                        'random_state': 1,
                        'n_estimators': 180,
                        'criterion': 'mae',
                        'loss': 'exponential',
                        'max_features': 'log2',
                        'n_iter_no_change': 100,
                    },
                    'majority': {
                        'random_state': 1,
                        'n_estimators': 50,
                        'criterion': 'mae',
                        'loss': 'exponential',
                        'max_features': 'log2',
                        'n_iter_no_change': 100,
                    },
                },
                100: {
                    # technically we haven't done the runs for this and we aren't
                    # going to do the runs for this, so just copy params from 20
                    # to make sure the code doesn't break
                    'average': {
                        'random_state': 1,
                        'n_estimators': 180,
                        'criterion': 'mae',
                        'loss': 'exponential',
                        'max_features': 'log2',
                        'n_iter_no_change': 100,
                    },
                    'majority': {
                        'random_state': 1,
                        'n_estimators': 50,
                        'criterion': 'mae',
                        'loss': 'exponential',
                        'max_features': 'log2',
                        'n_iter_no_change': 100,
                    },
                },
            },
            'NB': {
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
                        'var_smoothing': 0.1,
                    },
                },
            },
        }
        return params[self.args.algo][self.args.frame_size][self.args.hyperparameter_type]

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
            elif self.args.feature_selection_method:
                # sometimes x_test is modified by this func
                x_test = self.perform_feature_selection(x_train, y_train, x_test)
            elif not self.args.load_model:
                self.train(x_train, y_train)

            if self.args.save_model_to and self.args.split_type == 'kfold':
                raise Exception('Saving a model/scaler while in kfold is not supported!')
            elif self.args.save_model_to:
                pd.to_pickle(self.models[-1], "model-" + self.args.save_model_to)

            predictions = pd.Series(self.models[-1].predict(x_test), index=y_test.index)
            results = self.aggregate_statistics(y_test, predictions, model_idx)
            if not self.args.no_print_results:
                self.print_model_stats(y_test, predictions, model_idx)
                print("-------------------")

        self.aggregate_results()
        if not self.args.no_print_results:
            self.print_aggregate_results()
        if self.args.plot_predictions or self.args.plot_disease_evolution:
            self.plot_predictions()
        if self.args.plot_pairwise_features:
            self.plot_pairwise_feature_visualizations()

        if self.args.grid_search:
            self.aggregate_grid_search_results()

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
        self._perform_grid_search(RandomForestClassifier(random_state=1), params, x_train, y_train)

    def _perform_mlp_grid_search(self, x_train, y_train):
        hiddens = [
            (16,), (32,), (64,), (128,),
            (16, 16), (16, 32), (16, 64), (16, 128),
            (32, 16), (32, 32), (32, 64), (32, 128),
            (64, 16), (64, 32), (64, 64), (64, 128),
            (128, 16), (128, 32), (128, 64), (128, 128),
        ]
        activations = ['relu', 'tanh', 'logistic', 'identity']
        params = [{
            'activation': activations,
            'solver': ['lbfgs'],
            'hidden_layer_sizes': hiddens,
        }, {
            'activation': activations,
            'solver': ['sgd', 'adam'],
            'learning_rate_init': [.0001, .0005, .001, .005, .01, .05, .1],
            'hidden_layer_sizes': hiddens,
            # cut down on # params to search otherwise it will take forever
            #'alpha': [0.00001, .0001, .001, .01, .1],
            #'batch_size': [8, 16, 32, 64, 128, 256],
        }]
        self._perform_grid_search(MLPClassifier(random_state=1), params, x_train, y_train)

    def _perform_svm_grid_search(self, x_train, y_train):
        params = [{
            'C': [i for i in range(-16, 17)],
            'kernel': ['rbf', 'linear', 'sigmoid'],
        }, {
            'C': [i for i in range(-16, 17)],
            'kernel': ['poly'],
            'degree': range(2, 8),
        }]
        self._perform_grid_search(SVC(random_state=1, cache_size=512), params, x_train, y_train)

    def _perform_adaboost_grid_search(self, x_train, y_train):
        params = {
            'learning_rate': [2**i for i in range(-15, 3)],
            'n_estimators': [5*i for i in range(1, 60)],
            'algorithm': ['SAMME', 'SAMME.R'],
        }
        self._perform_grid_search(AdaBoostClassifier(random_state=1), params, x_train, y_train)

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
        self._perform_grid_search(GradientBoostingClassifier(random_state=1, n_iter_no_change=100), params, x_train, y_train)

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
        self._perform_grid_search(LogisticRegression(random_state=1), params, x_train, y_train)

    def _perform_grid_search(self, cls, params, x_train, y_train):
        x_train_expanded = self.data.loc[x_train.index]
        cv = self.get_cross_patient_kfold_idxs(x_train_expanded, y_train, self.args.grid_search_kfolds)
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
        if self.args.feature_selection_method == 'RFE':
            selector = RFE(clf, self.args.n_new_features, step=1)
            selector.fit(x_train, y_train)
            print('Selected features: {}'.format(list(x_train.columns[selector.support_])))
            self.models.append(selector)
        elif self.args.feature_selection_method == 'chi2':
            selector = SelectKBest(chi2, k=self.args.n_new_features)
            x_train = selector.fit_transform(x_train, y_train)
            cols = list(x_test.columns[selector.get_support()])
            print('Selected features: {}'.format(cols))
            x_test = pd.DataFrame(selector.transform(x_test), columns=cols)
            clf.fit(x_train, y_train)
            self.models.append(clf)
        elif self.args.feature_selection_method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=self.args.n_new_features)
            x_train = selector.fit_transform(x_train, y_train)
            cols = list(x_test.columns[selector.get_support()])
            print('Selected features: {}'.format(cols))
            x_test = pd.DataFrame(selector.transform(x_test), columns=cols)
            clf.fit(x_train, y_train)
            self.models.append(clf)
        elif self.args.feature_selection_method == 'gini':
            # XXX
            raise NotImplementedError()

        return x_test

    def aggregate_statistics(self, y_test, predictions, model_idx):
        """
        After a group of patients is run through the model, record all necessary stats
        such as true positives, false positives, etc.
        """
        x_test = self.data.loc[y_test.index]
        for pt in x_test.patient.unique():
            pt_idx = len(self.patient_results)
            pt_rows = x_test[x_test.patient == pt]
            patho_n = pt_rows.y.unique()[0]
            pt_actual = y_test.loc[pt_rows.index]
            pt_pred = predictions.loc[pt_rows.index]
            self.patient_predictions[pt] = (pt_rows, pt_pred)

            self.patient_results.loc[pt_idx] = [pt] + [len(pt_pred[pt_pred == n]) for n in self.pathos] + [pt_actual.unique()[0]]

            i = len(self.results)
            pt_results = [pt, patho_n]
            for n, patho in self.pathos.items():
                pt_results.extend([
                    get_tps(pt_actual, pt_pred, n), get_fps(pt_actual, pt_pred, n),
                    get_tns(pt_actual, pt_pred, n), get_fns(pt_actual, pt_pred, n),
                    len(pt_pred[pt_pred == n]),
                ])

            patho_pred = np.argmax([pt_results[6 + 5*k] for k in range(len(self.pathos))])
            pt_results.extend([model_idx, patho_pred])
            self.results.loc[i] = pt_results

    def print_model_stats(self, y_test, predictions, model_idx):
        """
        Perform majority rules voting on what disease subtype that a patient has
        """
        model_results = self.results[self.results.model_idx == model_idx]
        incorrect_pts = model_results[model_results.patho != model_results.prediction]

        print("Model accuracy: {}".format(accuracy_score(y_test, predictions)))
        for n, patho in self.pathos.items():
            print("{} recall: {}".format(patho, recall_score(y_test, predictions, labels=[n], average='macro')))
            print("{} precision: {}".format(patho, precision_score(y_test, predictions, labels=[n], average='macro')))

        patho_votes = ["{}_votes".format(k) for k in self.pathos.values()]
        for idx, row in incorrect_pts.iterrows():
            print("Patient {}: Prediction: {}, Actual: {}. Voting:\n{}".format(
                row.patient, row.prediction, row.patho, row[patho_votes]
            ))

    def plot_predictions(self):
        """
        Plot votes on specific class was predicted for each patient.
        """
        colors = ['viridian', 'pumpkin orange', 'eggplant']
        #fontname = 'Osaka'
        cmap = sns.color_palette(sns.xkcd_palette(colors))
        plt.rcParams['font.family'] = 'Osaka'
        plt.rcParams['legend.loc'] = 'upper right'
        if self.args.plot_predictions:
            # plot fraction of votes for all patients
            step_size = 10
            for i in range(0, len(self.patient_results), step_size):
                slice = self.patient_results.loc[i:i+step_size-1]
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

        if self.args.plot_disease_evolution:
            # Plot fraction of votes for a single patient over 24 hrs.
            if not self.args.tiled_disease_evol:
                for pt, (pt_rows, pt_preds) in self.patient_predictions.items():
                    self.plot_disease_evolution(pt, pt_rows, pt_preds, cmap)
                    plt.show()
            else:
                # just focus on ARDS for now.
                #
                # want to have true ARDS, false pos ARDS, false neg ARDS, and true neg ARDS
                true_pos = self.results[(self.results.patho == 1) & (self.results.prediction == 1)].patient
                true_neg = self.results[(self.results.patho != 1) & (self.results.prediction != 1)].patient
                false_pos = self.results[(self.results.patho != 1) & (self.results.prediction == 1)].patient
                false_neg = self.results[(self.results.patho == 1) & (self.results.prediction != 1)].patient
                for arr, title in [
                    (true_pos, 'ARDS True Pos'),
                    (true_neg, 'ARDS True Neg'),
                    (false_pos, 'ARDS False Pos'),
                    (false_neg, 'ARDS False Neg'),
                ]:
                    for idx, pt in enumerate(arr):
                        layout = int(ceil(sqrt(len(arr))))
                        plt.suptitle(title)
                        pt_rows, pt_preds = self.patient_predictions[pt]
                        plt.subplot(layout, layout, idx+1)
                        self.plot_disease_evolution(pt, pt_rows, pt_preds, cmap, legend=False, fontsize=6, xylabel=False, xy_visible=False)
                    plt.show()

    def plot_disease_evolution(self, pt, pt_rows, pt_preds, cmap, legend=True, fontsize=11, xylabel=True, xy_visible=True):
        pt_rows['pred'] = pt_preds
        hour_preds = pt_rows[['hour', 'pred']]
        bar_data = [[0] * len(self.pathos) for _ in range(24)]
        for hour in range(0, 24):
            hour_frame = hour_preds[hour_preds.hour == hour]
            counts = hour_frame.pred.value_counts()
            for n in self.pathos:
                try:
                    bar_data[hour][n] = counts[n] / float(counts.sum())
                except (IndexError, KeyError):
                    continue
        plots = []
        bottom = np.zeros(24)
        for n in self.pathos:
            bar_fracs = np.array([bar_data[hour][n] for hour in range(0, 24)])
            plots.append(plt.bar(range(0, 24), bar_fracs, bottom=bottom, color=cmap[n]))
            bottom = bottom + bar_fracs

        plt.title(pt, fontsize=fontsize, pad=1)
        if xylabel:
            plt.ylabel('Fraction Predicted', fontsize=fontsize)
            plt.xlabel('Hour', fontsize=fontsize)
        plt.xlim(-.8, 23.8)
        if legend:
            plt.legend([
                "{}: {}%".format(patho, round(len(pt_preds[pt_preds == n]) / float(len(pt_preds)), 3)*100)
                for n, patho in self.pathos.items()
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

    def aggregate_results(self):
        """
        Aggregate final results for all patients into a friendly data frame
        """
        aggregate_results = []
        for n, patho in self.pathos.items():
            tps = float(len(self.results[(self.results.patho == n) & (self.results.prediction == n)]))
            tns = float(len(self.results[(self.results.patho != n) & (self.results.prediction != n)]))
            fps = float(len(self.results[(self.results.patho != n) & (self.results.prediction == n)]))
            fns = float(len(self.results[(self.results.patho == n) & (self.results.prediction != n)]))
            accuracy = round((tps+tns) / (tps+tns+fps+fns), 4)
            try:
                sensitivity = round(tps / (tps+fns), 4)
            except ZeroDivisionError:
                sensitivity = 0
            try:
                specificity = round(tns / (tns+fps), 4)
            except ZeroDivisionError:
                specificity = 0
            try:
                precision = round(tps / (tps+fps), 4)
            except ZeroDivisionError:  # Can happen when no predictions for cls are made
                precision = 0
            if len(self.pathos) > 2:
                auc = np.nan
            elif len(self.pathos) == 2:
                auc = round(roc_auc_score(self.results.patho.tolist(), self.results.prediction.tolist()), 4)
            aggregate_results.append([patho, tps, tns, fps, fns, accuracy, sensitivity, specificity, precision, auc])

        self.aggregate_results = pd.DataFrame(
            aggregate_results,
            columns=['patho', 'tps', 'tns', 'fps', 'fns', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc']
        )

    def print_aggregate_results(self):
        for n, patho in self.pathos.items():
            row = self.aggregate_results[self.aggregate_results.patho == patho].iloc[0]
            print("{} patient accuracy: {}".format(patho, row.accuracy))
            print("{} patient sensitivity: {}".format(patho, row.sensitivity))
            print("{} patient specificity: {}".format(patho, row.specificity))
            print("{} patient precision: {}".format(patho, row.precision))
            print("")

        print("Model AUC: {}".format(row.auc))


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
    parser.add_argument("--feature-set", default="flow_time", choices=["flow_time", "flow_time_opt", "flow_time_orig", "broad", "broad_opt"])
    parser.add_argument('--no-load-intermediates', action='store_false', help='do not load from intermediate data')
    parser.add_argument('-sr', '--split-ratio', type=float, default=.2)
    parser.add_argument("--pca", type=int, help="perform PCA analysis/transform on data")
    parser.add_argument("--grid-search", action="store_true", help='perform a grid search  for model hyperparameters')
    parser.add_argument("--grid-search-kfolds", type=int, default=3, help='number of validation kfolds to use in the grid search')
    parser.add_argument('--split-type', choices=['holdout', 'holdout_random', 'kfold', 'train_all', 'test_all'], help='All splits are performed so there is no test/train patient overlap', default='kfold')
    parser.add_argument('--save-model-to', help='save model+scaler to a pickle file')
    parser.add_argument('--load-model')
    parser.add_argument('--load-scaler')
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument('-fs', "--frame-size", default=20, type=int)
    parser.add_argument('-ff', '--frame-func', choices=['median', 'mean', 'var', 'std', 'mean+var', 'mean+std', 'median+var', 'median+std'], default='median')
    parser.add_argument('-sd', '--start-hour-delta', default=0, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data')
    parser.add_argument('-sp', '--post-hour', default=24, type=int)
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
    parser.add_argument('--algo', help='The type of algorithm you want to do ML with', choices=['RF', 'MLP', 'SVM', 'LOG_REG', 'GBC', 'NB', 'ADA'], default='RF')
    parser.add_argument('-gsj', '--grid-search-jobs', type=int, default=multiprocessing.cpu_count(), help='run grid search with this many cores')
    parser.add_argument('-ehr', '--use-ehr-features', action='store_true', help='use EHR data in learning')
    parser.add_argument('-demo', '--use-demographic-features', action='store_true', help='use demographic data in learning')
    parser.add_argument('-ht', '--hyperparameter-type', choices=['average', 'majority'], default='average')
    parser.add_argument('-pdfe', '--print-dropped-frame-eval', action='store_true', help='Print evaluation of all the frames we drop')
    parser.add_argument('--load-from-unframed', action='store_true')
    parser.add_argument('-fsm', '--feature-selection-method', choices=['RFE', 'chi2', 'mutual_info', 'gini'], help='Feature selection method')
    parser.add_argument('--n-new-features', type=int, help='number of features to select using feature selection', default=1)
    return parser


def main():
    args = build_parser().parse_args()
    df = create_df(args)
    model = ARDSDetectionModel(args, df)
    model.train_and_test()


if __name__ == "__main__":
    main()
