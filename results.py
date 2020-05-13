import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy import interp
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from metrics import janky_roc


class PatientResults(object):
    def __init__(self, patient_id, ground_truth, fold_idx, model_idx):
        self.patient_id = patient_id
        self.other_votes = 0
        self.ards_votes = 0
        self.ground_truth = ground_truth
        self.majority_prediction = np.nan
        self.fold_idx = fold_idx
        self.model_idx = model_idx
        # the intention is that this will map hour to the number of predictions made
        # for non-ARDS and ARDS
        self.hourly_preds = {i: [np.nan, np.nan] for i in range(24)}

    def set_results(self, predictions, x_test_pt):
        for i in predictions.values:
            if i == 0:
                self.other_votes += 1
            elif i == 1:
                self.ards_votes += 1

        # at least in python 2.7 int() essentially acts as math.floor
        ards_percentage = int(100 * (self.ards_votes / float(len(predictions))))
        self.majority_prediction = 1 if self.ards_votes >= self.other_votes else 0
        x_test_pt['pred'] = predictions
        grouping = x_test_pt[['hour', 'pred']].groupby('hour')
        for i, rows in grouping:
            ards_count = rows.pred.sum()
            self.hourly_preds[rows.iloc[0].hour] = [len(rows) - ards_count, ards_count]

    def to_list(self):
        return [
            self.patient_id,
            self.other_votes,
            self.ards_votes,
            self.ards_votes / (float(self.other_votes) + self.ards_votes),
            self.majority_prediction,
            self.fold_idx,
            self.model_idx,
            self.ground_truth,
        ], ['patient_id', 'other_votes', 'ards_votes', 'frac_votes', 'majority_prediction', 'fold_idx', 'model_idx', 'ground_truth']

    def get_hourly_preds(self):
        results = [self.patient_id]
        columns = ['patient_id']
        for hour, preds in self.hourly_preds.items():
            results.extend(preds)
            columns.extend(['hour_{}_other_votes'.format(hour), 'hour_{}_ards_votes'.format(hour)])
        return results, columns


class ModelResults(object):
    def __init__(self, fold_idx, model_idx):
        self.fold_idx = fold_idx
        self.all_patient_results = []
        self.model_idx = model_idx

    def set_results(self, y_test, predictions, x_test):
        """
        """
        for pt in x_test.patient.unique():
            pt_rows = x_test[x_test.patient == pt]
            pt_gt = y_test.loc[pt_rows.index]
            pt_predictions = predictions.loc[pt_rows.index]
            ground_truth_label = pt_gt.iloc[0]
            results = PatientResults(pt, ground_truth_label, self.fold_idx, self.model_idx)
            results.set_results(pt_predictions, pt_rows)
            self.all_patient_results.append(results)

    def get_patient_results_dataframe(self):
        tmp = []
        for result in self.all_patient_results:
            lst, cols = result.to_list()
            tmp.append(lst)
        return pd.DataFrame(tmp, columns=cols)

    def get_patient_results(self):
        tmp = []
        for result in self.all_patient_results:
            lst, cols = result.to_list()
            tmp.append(lst)
        return pd.DataFrame(tmp, columns=cols)

    def get_patient_hourly_preds(self):
        tmp = []
        for result in self.all_patient_results:
            lst, cols = result.get_hourly_preds()
            tmp.append(lst)
        return pd.DataFrame(tmp, columns=cols)

    def count_predictions(self, threshold):
        """
        """
        assert 0 <= threshold <= 1
        stat_cols = []
        for patho in ['other', 'ards']:
            stat_cols.extend([
                '{}_tps_{}'.format(patho, threshold),
                '{}_tns_{}'.format(patho, threshold),
                '{}_fps_{}'.format(patho, threshold),
                '{}_fns_{}'.format(patho, threshold)
            ])
        stat_cols += ['fold_idx']

        patient_results = self.get_patient_results()
        stat_results = []
        for patho in [0, 1]:
            # The 2 idx is the prediction fraction from the patient results class
            #
            # In this if statement we are differentiating between predictions made
            # for ARDS and predictions made otherwise. the eq_mask signifies
            # predictions made for the pathophysiology. For instance if our pathophys
            # is 0 then we want the fraction votes for ARDS to be < prediction threshold.
            if patho == 0:
                eq_mask = patient_results.frac_votes < threshold
                neq_mask = patient_results.frac_votes >= threshold
            else:
                eq_mask = patient_results.frac_votes >= threshold
                neq_mask = patient_results.frac_votes < threshold

            stat_results.extend([
                len(patient_results[eq_mask][patient_results.loc[eq_mask, 'ground_truth'] == patho]),
                len(patient_results[neq_mask][patient_results.loc[neq_mask, 'ground_truth'] != patho]),
                len(patient_results[eq_mask][patient_results.loc[eq_mask, 'ground_truth'] != patho]),
                len(patient_results[neq_mask][patient_results.loc[neq_mask, 'ground_truth'] == patho]),
            ])
        return stat_results + [self.fold_idx], stat_cols


class ModelCollection(object):
    def __init__(self, args):
        self.models = []
        self.model_results = {
            'folds': {},
            'aggregate': None,
        }
        self.model_idx = 0
        self.args = args
        self.experiment_name = self.args.experiment_name

    def add_model(self, y_test, predictions, x_test, fold_idx):
        model = ModelResults(fold_idx, self.model_idx)
        model.set_results(y_test, predictions, x_test)
        self.models.append(model)
        self.model_idx += 1

    def get_aggregate_predictions_dataframe(self, threshold):
        """
        Get aggregated results of all the dataframes
        """
        tmp = []
        for model in self.models:
            results, cols = model.count_predictions(threshold)
            tmp.append(results)
        return pd.DataFrame(tmp, columns=cols)

    def get_all_hourly_preds(self):
        tmp = [model.get_patient_hourly_preds() for model in self.models]
        return pd.concat(tmp, ignore_index=True)

    def get_all_patient_results_dataframe(self):
        tmp = [model.get_patient_results_dataframe() for model in self.models]
        return pd.concat(tmp, axis=0, ignore_index=True)

    def get_all_patient_results_in_fold_dataframe(self, fold_idx):
        # if you don't want to reconstitute this all the time you
        # can probably keep a boolean variable that tells you when you need to remake
        # and then can store as a global var
        tmp = [model.get_patient_results_dataframe() for model in self.models if model.fold_idx == fold_idx]
        return pd.concat(tmp, axis=0, ignore_index=True)

    def calc_fold_stats(self, threshold, fold_idx, print_results=True):
        if threshold > 1:  # threshold is a percentage
            threshold = threshold / 100.0
        df = self.get_aggregate_predictions_dataframe(threshold)
        fold_results = df[df.fold_idx == fold_idx]
        patient_results = self.get_all_patient_results_in_fold_dataframe(fold_idx)
        results_df = self.calc_results(fold_results, threshold, patient_results)
        self.model_results['folds'][fold_idx] = results_df
        if print_results:
            self.print_results_table(results_df)

    def calc_aggregate_stats(self, threshold, print_results=True):
        if threshold > 1:  # threshold is a percentage
            threshold = threshold / 100.0
        df = self.get_aggregate_predictions_dataframe(threshold)
        patient_results = self.get_all_patient_results_dataframe()
        results_df = self.calc_results(df, threshold, patient_results)
        self.save_to_pickle()
        self.model_results['aggregate'] = results_df
        if print_results:
            print('---Aggregate Results---')
            self.print_results_table(results_df)

    def calc_results(self, dataframe, threshold, patient_results):
        columns = ['patho', 'acc', 'recall', 'spec', 'prec', 'npv', 'auc', 'acc_ci', 'recall_ci', 'spec_ci', 'prec_ci', 'npv_ci', 'auc_ci']
        stats_tmp = []
        aucs = self.get_auc_results(patient_results)
        uniq_pts = len(patient_results.patient_id.unique())
        mean_auc = aucs.mean().round(3)
        auc_ci = (1.96 * np.sqrt(mean_auc * (1-mean_auc) / uniq_pts)).round(3)
        for patho in ['other', 'ards']:
            stats = self.get_summary_statistics_from_frame(dataframe, patho, threshold)
            means = stats.mean().round(3)
            cis = (1.96 * np.sqrt(means*(1-means)/uniq_pts)).round(3)
            stats_tmp.append([
                patho,
                means[0],
                means[1],
                means[2],
                means[3],
                means[4],
                aucs.mean().round(2),
                cis[0],
                cis[1],
                cis[2],
                cis[3],
                cis[4],
                auc_ci,
            ])
        return pd.DataFrame(stats_tmp, columns=columns)

    def print_results_table(self, results_df):
        table = PrettyTable()
        table.field_names = ['patho', 'sensitivity', 'specificity', 'precision', 'npv', 'auc']
        for i, row in results_df.iterrows():
            results_row = [
                row.patho,
                u"{}\u00B1{}".format(row.recall, row.recall_ci),
                u"{}\u00B1{}".format(row.spec, row.spec_ci),
                u"{}\u00B1{}".format(row.prec, row.prec_ci),
                u"{}\u00B1{}".format(row.npv, row.npv_ci),
                u"{}\u00B1{}".format(row.auc, row.auc_ci),
            ]
            table.add_row(results_row)
        print(table)

    def plot_roc_all_folds(self, savefig=False, dpi=1200, fmt_out='png'):
        # I might be able to find confidence std using p(1-p). Nah. we actually cant do
        # this because polling is using the identity of std from a binomial distribution. So
        # in order to have conf interval we need some kind of observable std.
        tprs = []
        aucs = []
        threshes = set()
        mean_fpr = np.linspace(0, 1, 100)
        results = self.get_all_patient_results_dataframe()
        uniq_pts = len(results.patient_id.unique())

        color_map = {
            0: 'green',
            1: 'orange',
            2: 'purple',
            3: 'saddlebrown',
            4: 'fuchsia',
            'main': 'royalblue'
        }
        for fold_idx in results.fold_idx.unique():
            fold_preds = results[results.fold_idx == fold_idx]
            model_aucs = self.get_auc_results(fold_preds)
            fpr, tpr, thresh = roc_curve(fold_preds.ground_truth, fold_preds.frac_votes)
            threshes.update(thresh)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1.5, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (fold_idx+1, roc_auc),
                     color=color_map[fold_idx])

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = round(auc(mean_fpr, mean_tpr), 2)
        std_auc = np.std(aucs)

        model_aucs = self.get_auc_results(results)
        auc_ci = (1.96 * np.sqrt(mean_auc * (1-mean_auc) / uniq_pts)).round(3)
        plt.plot(mean_fpr, mean_tpr, color=color_map['main'],
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, auc_ci),
                 lw=2.5, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        if savefig:
            plt.savefig('roc-all-folds.{}'.format(fmt_out), dpi=dpi)
        plt.show()

    def plot_sen_spec_vs_thresh(self, thresh_interval, savefig=False, dpi=1200, fmt_out='png'):
        y1 = []
        y2 = []
        pred_threshes = range(0, 100+thresh_interval, thresh_interval)
        for i in pred_threshes:
            thresh = i / 100.0
            df = self.get_aggregate_predictions_dataframe(thresh)
            stats = self.get_summary_statistics_from_frame(df, 'ards', thresh)
            means = stats.mean()
            y1.append(means[1])
            y2.append(means[2])
        patho = 'ARDS'
        plt.plot(pred_threshes, y1, label='{} sensitivity'.format(patho), lw=2)
        plt.plot(pred_threshes, y2, label='{} specificity'.format(patho), lw=2)
        plt.legend(loc='lower right')
        plt.title('Sensitivity v Specificity analysis')
        plt.ylabel('Score')
        plt.xlabel('Percentage ARDS votes')
        plt.ylim(0.0, 1.01)
        plt.yticks(np.arange(0, 1.01, .1))
        plt.xticks(np.arange(0, 101, 10))
        plt.grid()
        if savefig:
            plt.savefig('sen-spec-vs-thresh.{}'.format(fmt_out), dpi=dpi)
        plt.show()

    def get_youdens_results(self):
        """
        Get Youden results for all models derived
        """
        results = self.get_all_patient_results_dataframe()
        uniq_pts = len(results.patient_id.unique())
        # -1 stands for the ground truth idx, and 2 stands for prediction frac idx
        all_tpr, all_fpr, threshs = janky_roc(results.ground_truth, results.frac_votes)
        j_scores = np.array(all_tpr) - np.array(all_fpr)
        tmp = zip(j_scores, threshs)
        ordered_j_scores = []
        for score, thresh in tmp:
            if thresh in np.arange(0, 101, 1) / 100.0:
                ordered_j_scores.append((score, thresh))
        ordered_j_scores = sorted(ordered_j_scores, key=lambda x: (x[0], -x[1]))
        optimal_pred_frac = ordered_j_scores[-1][1]
        data_at_frac = self.get_aggregate_predictions_dataframe(optimal_pred_frac)
        # get closest prediction thresh
        optimal_table = PrettyTable()
        optimal_table.field_names = ['patho', '% votes', 'acc', 'sen', 'spec', 'prec', 'npv']
        for patho in ['other', 'ards']:
            stats = self.get_summary_statistics_from_frame(data_at_frac, patho, optimal_pred_frac)
            means = stats.mean().round(2)
            cis = (1.96 * np.sqrt(means*(1-means)/uniq_pts)).round(3)
            optimal_table.add_row([
                patho,
                optimal_pred_frac,
                u"{}\u00B1{}".format(means[0], cis[0]),
                u"{}\u00B1{}".format(means[1], cis[1]),
                u"{}\u00B1{}".format(means[2], cis[2]),
                u"{}\u00B1{}".format(means[3], cis[3]),
                u"{}\u00B1{}".format(means[4], cis[4]),
            ])

        print('---Youden Results---')
        print(optimal_table)

    def get_summary_statistics_from_frame(self, dataframe, patho, threshold):
        """
        Get summary statistics about all models in question given a pathophysiology and
        threshold to evaluate at.
        """
        tps = "{}_tps_{}".format(patho, threshold)
        tns = "{}_tns_{}".format(patho, threshold)
        fps = "{}_fps_{}".format(patho, threshold)
        fns = "{}_fns_{}".format(patho, threshold)
        sens = dataframe[tps] / (dataframe[tps] + dataframe[fns])
        specs = dataframe[tns] / (dataframe[tns] + dataframe[fps])
        precs = dataframe[tps] / (dataframe[fps] + dataframe[tps])
        npvs = dataframe[tns] / (dataframe[tns] + dataframe[fns])
        accs = (dataframe[tns] + dataframe[tps]) / (dataframe[tns] + dataframe[tps] + dataframe[fns] + dataframe[fps])
        stats = pd.concat([accs, sens, specs, precs, npvs], axis=1)
        return stats

    def get_auc_results(self, patient_results):
        group = patient_results.groupby('model_idx')
        aucs = []
        for i, model_pts in group:
            fpr, tpr, thresholds = roc_curve(model_pts.ground_truth, model_pts.frac_votes, pos_label=1)
            aucs.append(auc(fpr, tpr))
        return np.array(aucs)

    def print_thresh_table(self, thresh_interval):
        assert 1 <= thresh_interval <= 100
        table = PrettyTable()
        table.field_names = ['patho', 'vote %', 'acc', 'sen', 'spec', 'prec', 'npv']
        pred_threshes = range(0, 100+thresh_interval, thresh_interval)
        patient_results = self.get_all_patient_results_dataframe()
        uniq_pts = len(patient_results.patient_id.unique())
        for i in pred_threshes:
            thresh = i / 100.0
            df = self.get_aggregate_predictions_dataframe(thresh)
            stats = self.get_summary_statistics_from_frame(df, 'ards', thresh)
            means = stats.mean().round(2)
            cis = (1.96 * np.sqrt(means*(1-means)/uniq_pts)).round(3)
            row = [
                'ards',
                i,
                u"{}\u00B1{}".format(means[0], cis[0]),
                u"{}\u00B1{}".format(means[1], cis[1]),
                u"{}\u00B1{}".format(means[2], cis[2]),
                u"{}\u00B1{}".format(means[3], cis[3]),
                u"{}\u00B1{}".format(means[4], cis[5]),
            ]
            table.add_row(row)
        print(table)

    def save_to_pickle(self):
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        model_time = time.time()
        if self.experiment_name is not None:
            base_filename = 'model_collection_results_{}_{}.pkl'.format(self.experiment_name, int(model_time))
        else:
            base_filename = 'model_collection_results_{}.pkl'.format(int(model_time))
        pd.to_pickle(self, os.path.join(results_dir, base_filename))
