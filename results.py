import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy import interp
from sklearn.metrics import auc, roc_curve

from metrics import janky_roc


class PatientResults(object):
    def __init__(self, patient_id, ground_truth, fold_idx):
        self.patient_id = patient_id
        self.other_votes = 0
        self.ards_votes = 0
        self.ground_truth = ground_truth
        self.majority_prediction = np.nan
        self.fold_idx = fold_idx

    def set_results(self, predictions):
        for i in predictions:
            if i == 0:
                self.other_votes += 1
            elif i == 1:
                self.ards_votes += 1

        # at least in python 2.7 int() essentially acts as math.floor
        ards_percentage = int(100 * (self.ards_votes / float(len(predictions))))
        self.majority_prediction = 1 if self.ards_votes >= self.other_votes else 0

    def to_list(self):
        return [
            self.other_votes,
            self.ards_votes,
            self.ards_votes / (float(self.other_votes) + self.ards_votes),
            self.majority_prediction,
            self.fold_idx,
            self.ground_truth,
        ], ['other_votes', 'ards_votes', 'frac_votes', 'prediction', 'fold_idx', 'ground_truth']

    def get_patient_id(self):
        return self.patient_id


class ModelResults(object):
    def __init__(self, fold_idx):
        self.fold_idx = fold_idx
        self.all_patient_results = []

    def set_results(self, y_test, predictions, x_test):
        """
        """
        for pt in x_test.patient.unique():
            pt_rows = x_test[x_test.patient == pt]
            pt_gt = y_test.loc[pt_rows.index]
            pt_predictions = predictions.loc[pt_rows.index]
            ground_truth_label = pt_gt.iloc[0]
            results = PatientResults(pt, ground_truth_label, self.fold_idx)
            results.set_results(pt_predictions.values)
            self.all_patient_results.append(results)

    def get_patient_results_dataframe(self):
        tmp = []
        for result in self.all_patient_results:
            lst, cols = result.to_list()
            tmp.append(lst)
        return pd.DataFrame(tmp, columns=cols)

    def get_patient_results_np_array(self):
        tmp = []
        for result in self.all_patient_results:
            lst, cols = result.to_list()
            tmp.append(lst)
        return np.array(tmp)

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

        patient_results = self.get_patient_results_np_array()
        stat_results = []
        for patho in [0, 1]:
            # The 2 idx is the prediction fraction from the patient results class
            #
            # In this if statement we are differentiating between predictions made
            # for ARDS and predictions made otherwise. the eq_mask signifies
            # predictions made for the pathophysiology. For instance if our pathophys
            # is 0 then we want the fraction votes for ARDS to be < prediction threshold.
            if patho == 0:
                eq_mask = patient_results[:, 2] < threshold
                neq_mask = patient_results[:, 2] >= threshold
            else:
                eq_mask = patient_results[:, 2] >= threshold
                neq_mask = patient_results[:, 2] < threshold

            stat_results.extend([
                len(patient_results[eq_mask][patient_results[eq_mask, -1] == patho]),
                len(patient_results[neq_mask][patient_results[neq_mask, -1] != patho]),
                len(patient_results[eq_mask][patient_results[eq_mask, -1] != patho]),
                len(patient_results[neq_mask][patient_results[neq_mask, -1] == patho]),
            ])
        return stat_results + [self.fold_idx], stat_cols


class ModelCollection(object):
    def __init__(self):
        self.models = []
        self.model_results = {
            'folds': {},
            'aggregate': None,
        }

    def add_model(self, y_test, predictions, x_test, fold_idx):
        model = ModelResults(fold_idx)
        model.set_results(y_test, predictions, x_test)
        self.models.append(model)

    def get_aggregate_predictions_dataframe(self, threshold):
        """
        Get aggregated results of all the dataframes
        """
        tmp = []
        for model in self.models:
            results, cols = model.count_predictions(threshold)
            tmp.append(results)
        return pd.DataFrame(tmp, columns=cols)

    def get_all_patient_results_np_array(self):
        tmp = [model.get_patient_results_np_array() for model in self.models]
        return np.concatenate(tmp, axis=0)

    def get_all_patient_results_dataframe(self):
        tmp = [model.get_patient_results_dataframe() for model in self.models]
        return pd.concat(tmp, axis=0)

    def get_all_patient_results_in_fold(self, fold_idx):
        # if you don't want to reconstitute this all the time you
        # can probably keep a boolean variable that tells you when you need to remake
        # and then can store as a global var
        tmp = [model.get_patient_results_np_array() for model in self.models if model.fold_idx == fold_idx]
        return np.concatenate(tmp, axis=0)

    def calc_fold_stats(self, threshold, fold_idx, print_results=True):
        if threshold > 1:  # threshold is a percentage
            threshold = threshold / 100.0
        df = self.get_aggregate_predictions_dataframe(threshold)
        fold_results = df[df.fold_idx == fold_idx]
        patient_results = self.get_all_patient_results_in_fold(fold_idx)
        results_df = self.calc_results(fold_results, threshold)
        self.model_results['folds'][fold_idx] = results_df
        if print_results:
            self.print_results_table(results_df)

    def calc_aggregate_stats(self, threshold, print_results=True):
        if threshold > 1:  # threshold is a percentage
            threshold = threshold / 100.0
        df = self.get_aggregate_predictions_dataframe(threshold)
        patient_results = self.get_all_patient_results_np_array()
        results_df = self.calc_results(df, threshold)
        self.model_results['aggregate'] = results_df
        if print_results:
            print('---Aggregate Results---')
            self.print_results_table(results_df)

    def calc_results(self, dataframe, threshold):
        columns = ['patho', 'recall', 'spec', 'prec', 'recall_ci', 'spec_ci', 'prec_ci']
        stats_tmp = []
        for patho in ['other', 'ards']:
            stats = self.get_summary_statistics_from_frame(dataframe, patho, threshold)
            cis = (1.96 * stats.std() / np.sqrt(len(stats))).round(3)
            means = stats.mean().round(2)
            # XXX need to add auc somehow
            #auc = round(roc_auc_score(results.patho.tolist(), results.pred_frac.tolist()), 4)
            stats_tmp.append([
                patho,
                means[0],
                means[1],
                means[2],
                cis[0],
                cis[1],
                cis[2]
            ])
        return pd.DataFrame(stats_tmp, columns=columns)

    def print_results_table(self, results_df):
        table = PrettyTable()
        table.field_names = ['patho', 'sensitivity', 'specificity', 'precision']
        for i, row in results_df.iterrows():
            results_row = [
                row.patho,
                u"{}\u00B1{}".format(row.recall, row.recall_ci),
                u"{}\u00B1{}".format(row.spec, row.spec_ci),
                u"{}\u00B1{}".format(row.prec, row.prec_ci),
            ]
            table.add_row(results_row)
        print(table)

    def plot_roc_all_folds(self):
        # I might be able to find confidence std using p(1-p). Nah. we actually cant do
        # this because polling is using the identity of std from a binomial distribution. So
        # in order to have conf interval we need some kind of observable std.
        tprs = []
        aucs = []
        threshes = set()
        mean_fpr = np.linspace(0, 1, 100)
        results = self.get_all_patient_results_dataframe()

        for fold_idx in results.fold_idx.unique():
            fold_preds = results[results.fold_idx == fold_idx]
            fpr, tpr, thresh = roc_curve(fold_preds.ground_truth, fold_preds.frac_votes)
            threshes.update(thresh)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (fold_idx+1, roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # 1.96 is a constant for normal distribution and 95% CI
        #
        # XXX I am doing this inconsistently with where I compute aucs elsewhere.
        ci = 1.96 * (std_auc / np.sqrt(len(aucs)))
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, ci),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    def plot_sen_spec_vs_thresh(self):
        # XXX need to fix this one up after I finish the ROC work
        patho = 'ARDS'
        row = self.thresh_eval[(self.thresh_eval.patho == patho) & (self.thresh_eval.fold_idx == -1)].iloc[0]
        y1 = [row['sen@{}'.format(i)] for i in self.pred_threshes]
        y2 = [row['spec@{}'.format(i)] for i in self.pred_threshes]
        plt.plot(self.pred_threshes, y1, label='{} sensitivity'.format(patho))
        plt.plot(self.pred_threshes, y2, label='{} specificity'.format(patho))
        plt.legend(loc='lower right')
        plt.title('Sensitivity v Specificity analysis')
        plt.ylabel('Score')
        plt.xlabel('Percentage ARDS votes')
        plt.ylim(0.0, 1.01)
        plt.yticks(np.arange(0, 1.01, .1))
        plt.grid()
        plt.show()

    def get_youdens_results(self):
        """
        Get Youden results for all models derived
        """
        results = self.get_all_patient_results_np_array()
        # -1 stands for the ground truth idx, and 2 stands for prediction frac idx
        all_tpr, all_fpr, threshs = janky_roc(results[:, -1], results[:, 2])
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
        optimal_table.field_names = ['patho', '% votes', 'sen', 'spec', 'prec']
        for patho in ['other', 'ards']:
            stats = self.get_summary_statistics_from_frame(data_at_frac, patho, optimal_pred_frac)
            means = stats.mean().round(2)
            optimal_table.add_row([patho, optimal_pred_frac, means[0], means[1], means[2]])

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
        stats = pd.concat([sens, specs, precs], axis=1)
        return stats
