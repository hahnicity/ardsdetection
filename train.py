"""
learn
~~~~~

Performs learning for our classifier
"""
from argparse import ArgumentParser
import csv
from math import sqrt, ceil
import operator
from random import randint, sample
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import KFold, train_test_split
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

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
            # XXX For now just stick to 50/50 split. But later this code should be
            # removed
            patients = []
            for i in [0, 1]:
                patients.extend(list(self.data[self.data.y == i].patient.unique()[:50]))
            self.data = self.data[self.data.patient.isin(patients)]

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
        self.results = pd.DataFrame([], columns=results_cols)
        self.patient_results = pd.DataFrame(
            [], columns=['patient'] + ["{}_votes".format(patho) for _, patho in self.pathos.items()] + ['actual']
        )
        self.patient_predictions = {}

    def get_cross_patient_train_test_idx(self):
        """
        """
        unique_patients = self.data['patient'].unique()
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
            patients = sample(v, num_to_use)
            print("number {} patients in training: {}".format(k, num_patients - len(patients)))
            print("number {} patients in test: {}".format(k, len(patients)))
            patients_to_use.extend(patients)

        train_patient_data = self.data.query('patient not in {}'.format(patients_to_use))
        test_patient_data = self.data.query('patient in {}'.format(patients_to_use))
        return [(train_patient_data.index, test_patient_data.index)]

    def get_cross_patient_kfold_idxs(self, x, y, folds):
        """
        Get indexes to split dataset
        """
        idxs = []
        unique_patients = x.patient.unique()
        mapping = {patho: [] for n, patho in self.pathos.items()}

        for patient in unique_patients:
            patient_rows = x[x.patient == patient]
            # XXX I can simpify this line, especially because y is still
            # attached to x at this point
            type_ = self.pathos[y.loc[patient_rows.index].unique()[0]]
            mapping[type_].append(patient)

        if 'set_type' not in x.columns:
            x['set_type'] = 'train_test'
            train_cohort = 'train_test'
            test_cohort = 'train_test'
        elif len(x['set_type'].unique()) == 1:
            train_cohort = 'train_test'
            test_cohort = 'train_test'
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

        # XXX this split function will be broken with addition of new test params
        if self.args.split_type == "simple":
            idxs = self.get_cross_patient_train_test_idx()
        elif self.args.split_type == 'kfold':
            idxs = self.get_cross_patient_kfold_idxs(x, y, self.args.folds)
        # XXX these two if/elseif will be broken with addition of new test params
        elif self.args.split_type == 'train_all':
            idxs = [(x.index, [])]
        elif self.args.split_type == 'test_all':
            idxs = [([], x.index)]

        try:
            x = x.drop(['hour'], axis=1)
        except:
            pass
        try:
            x = x.drop(['y', 'patient', 'ventBN', 'set_type'], axis=1)
        except:  # maybe we didnt define ventBN, its not that important anyhow.
            x = x.drop(['y', 'patient', 'set_type'], axis=1)

        for train_idx, test_idx in idxs:
            x_train = x.loc[train_idx].dropna()
            x_test = x.loc[test_idx].dropna()
            y_train = y.loc[train_idx].dropna()
            y_test = y.loc[test_idx].dropna()

            scaler = self.get_and_fit_scaler(x_train)
            if len(x_train) != 0:
                x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index)
            if len(x_test) != 0:
                x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index)
            yield (x_train, x_test, y_train, y_test)

    def train(self, x_train, y_train):
        clf = RandomForestClassifier(random_state=1, oob_score=True)
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

    def convert_loc_to_iloc(self, df, loc_indices):
        copied = df.copy()
        copied['idx'] = range(len(df))
        return [(copied.loc[train_loc, 'idx'].values, copied.loc[test_loc, 'idx'])
                for train_loc, test_loc in loc_indices]

    def perform_grid_search(self, x_train, y_train):
        params = {
            "n_estimators": range(10, 50, 5),
            "max_features": ['auto', 'log2', None],
            "criterion": ["entropy", 'gini'],
            "max_depth": range(5, 30, 5) + [None],
            "oob_score": [True, False],
            "warm_start": [True, False],
            "min_samples_split": [2, 5, 10, 15],
        }
        x_train_expanded = self.data.loc[x_train.index]
        cv = self.get_cross_patient_kfold_idxs(x_train_expanded, y_train, 10)
        # sklearn does CV indexing with iloc and not loc. Annoying, but can be worked around
        cv = self.convert_loc_to_iloc(x_train, cv)
        clf = GridSearchCV(RandomForestClassifier(random_state=1), params, cv=cv)
        clf.fit(x_train, y_train.values)
        print("Params: ", clf.best_params_)
        print("Best CV score: ", clf.best_score_)
        self.models.append(clf)

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
                tp_layout = int(ceil(sqrt(len(true_pos))))
                for idx, pt in enumerate(true_pos):
                    pt_rows, pt_preds = self.patient_predictions[pt]
                    plt.subplot(tp_layout, tp_layout, idx+1)
                    self.plot_disease_evolution(pt, pt_rows, pt_preds, cmap, legend=False, fontsize=3)
                plt.show()

    def plot_disease_evolution(self, pt, pt_rows, pt_preds, cmap, legend=True, fontsize=11):
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

        plt.title(pt, fontsize=fontsize)
        plt.ylabel('Fraction Predicted', fontsize=fontsize)
        plt.xlabel('Hour', fontsize=fontsize)
        plt.xlim(-.8, 23.8)
        if legend:
            plt.legend([
                "{}: {}%".format(patho, round(len(pt_preds[pt_preds == n]) / float(len(pt_preds)), 3)*100)
                for n, patho in self.pathos.items()
            ], fontsize=fontsize)
        plt.yticks(np.arange(0, 1.01, .1))
        plt.xticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])

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
            sensitivity = round(tps / (tps+fns), 4)
            specificity = round(tns / (tns+fps), 4)
            precision = round(tps / (tps+fps), 4)
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
    df = Dataset(
        args.cohort_description,
        args.feature_set,
        args.frame_size,
        args.no_load_intermediates,
        args.experiment,
        args.post_hour,
        args.start_hour_delta,
        args.frame_func,
        args.test_frame_size,
        args.test_post_hour,
        args.test_start_hour_delta,
    ).get()
    if args.to_pickle:
        df.to_pickle(args.to_pickle)

    return df


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--cohort-description', default='cohort-description.csv', help='path to cohort description file')
    parser.add_argument("--feature-set", default="flow_time", choices=["flow_time", "flow_time_opt", "flow_time_orig", "broad", "broad_opt"])
    parser.add_argument('--no-load-intermediates', action='store_false', help='do not load from intermediate data')
    parser.add_argument('--split-ratio', type=float, default=.2)
    parser.add_argument("--pca", type=int, help="perform PCA analysis/transform on data")
    parser.add_argument("--grid-search", action="store_true", help='perform grid search for model hyperparameters')
    parser.add_argument('--split-type', choices=['simple', 'kfold', 'train_all', 'test_all'], help='All splits are performed so there is no test/train patient overlap', default='kfold')
    parser.add_argument('--save-model-to', help='save model+scaler to a pickle file')
    parser.add_argument('--load-model')
    parser.add_argument('--load-scaler')
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument('-fs', "--frame-size", default=20, type=int)
    parser.add_argument('-ff', '--frame-func', choices=['median', 'mean', 'var'], default='median')
    parser.add_argument('-sd', '--start-hour-delta', default=0, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data')
    parser.add_argument('-sp', '--post-hour', default=24, type=int)
    parser.add_argument('-tfs', "--test-frame-size", default=None, type=int)
    parser.add_argument('-tsd', '--test-start-hour-delta', default=None, type=int, help='time delta post ARDS detection time or vent start to begin analyzing data. Only for usage in testing set')
    parser.add_argument('-tsp', '--test-post-hour', default=None, type=int)
    parser.add_argument("--to-pickle", help="name of file the data frame will be pickled in")
    parser.add_argument("-p", "--from-pickle", help="name of file to retrieve pickled data from")
    parser.add_argument('-e', '--experiment', help='Experiment number we wish to run. If you wish to mix patients from different experiments you can do <num>+<num>+... eg. 1+3  OR 1+2+3', default='1+4')
    parser.add_argument("--no-copd-to-ctrl", action="store_true", help='Dont convert copd annotations to ctrl annotations')
    parser.add_argument('--no-print-results', action='store_true', help='Dont print results of our model')
    parser.add_argument('--plot-disease-evolution', action='store_true', help='Plot evolution of disease over time')
    parser.add_argument('--plot-predictions', action='store_true', help='Plot prediction bars')
    parser.add_argument('--tiled-disease-evol', action='store_true', help='Plot disease evolution in tiled manner')
    return parser


def main():
    args = build_parser().parse_args()
    df = create_df(args)
    model = ARDSDetectionModel(args, df)
    model.train_and_test()


if __name__ == "__main__":
    main()
