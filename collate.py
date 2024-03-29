"""
collate
~~~~~~~

Create dataset of items we wish to use for the training/testing of our
model.
"""
from copy import copy
import csv
from datetime import datetime
from glob import glob
from io import open
import logging
from operator import xor
from pathlib import Path
import os
import re
import sys
import traceback

import coloredlogs
import numpy as np
import pandas as pd
from parliament.analyze import FileCalculations
from parliament.polynomial_model import perform_polynomial_model
from parliament.other_calcs import calc_volumes
from scipy.signal import butter, sosfilt
from ventmap.breath_meta import get_experimental_breath_meta, get_file_experimental_breath_meta
from ventmap.constants import EXPERIMENTAL_META_HEADER
from ventmap.raw_utils import extract_raw

try:
    from algorithms.tor5 import perform_tor_with_bs_be
    from ventmode import datasets
    from ventmode.main import merge_periods_with_low_time_thresh, run_dataset_with_classifier_and_lookahead
    tor_possible = True
except ImportError:
    tor_possible = False

coloredlogs.install()
DEMOGRAPHIC_DATA_PATH = 'demographic/cohort_demographics.csv'
EHR_DATA_PATH = 'ehr/pva_study_20181127_temperature_and_lab_results_no_phi.csv'


class Dataset(object):
    # Feature sets are mapped by (feature_name, breath_meta_feature_index)
    necessities = ['ventBN']
    vent_feature_sets = {
        'flow_time': necessities + [
            'mean_flow_from_pef',
            'inst_RR',
            'minF_to_zero',
            'pef_+0.16_to_zero',
            'iTime',
            'eTime',
            'I:E ratio',
            'dyn_compliance',
            'tve:tvi ratio',
            'stat_compliance',
            'resist',
        ],
        'flow_time_biased': necessities + [
            'mean_flow_from_pef',
            'inst_RR',
            'minF_to_zero',
            'pef_+0.16_to_zero',
            'iTime',
            'eTime',
            'I:E ratio',
            'dyn_compliance',
            'tve:tvi ratio',
            'tvi',
            'tve',
            'stat_compliance',
            'resist',
        ],
        'flow_time_orig': necessities + [
            'mean_flow_from_pef',
            'inst_RR',
            'minF_to_zero',
            'pef_+0.16_to_zero',
            'iTime',
            'eTime',
            'I:E ratio',
            'dyn_compliance',
            'tve:tvi ratio',
        ],
        'flow_time_opt': necessities + [
            'dyn_compliance',
            'tve:tvi ratio',
            'mean_flow_from_pef',
            'eTime',
            'I:E ratio',
        ],
        # this is pertaining to a cluster of experiments performed 4/24/19 where
        # we ran an exhaustive search on train24+test24 for the most optimal possible
        # feature set that we could find.
        "holdout_exhaustive_rf_search": necessities + [
            'pef_+0.16_to_zero',
        ],
        # This pertains to the optimal feature set we found for our 2019 ATS submission
        'ats_optimal': necessities + [
            'dyn_compliance',
            'tve:tvi ratio',
            'mean_flow_from_pef',
            'eTime',
            'I:E ratio',
        ],
        'stat_compliance': necessities + ['stat_compliance'],
        'resist': necessities + ['resist'],
    }
    vent_feature_sets.update({
        'broad': vent_feature_sets['flow_time'] + [
            'tvi',
            'tve',
            'Maw',
            'ipAUC',
            'PIP',
            'PEEP',
            'epAUC',
        ],
        'broad_opt': necessities + [
            'PEEP',
            'I:E Ratio',
            'inst_RR',
            'TVi',
            'PIP',
            'iTime',
        ],
    })
    ehr_features = [
        "TEMPERATURE_F",
        "WBC",
        # For now just focus on arterial vars, we can figure out what to do
        # with venous information later
        #
        # also: pco2 doesn't correlate well with abg and vbg, which may diminish utility
        "ABG_P_F_RATIO",
        "ABG_PH_ARTERIAL",
        "PCO2_ARTERIAL",
        # XXX Vent ratio?? = (min vent * pco2) / (pbw * 100 * 37.5)
        # supposed to correspond well with dead space
        # https://www.atsjournals.org/doi/pdf/10.1164/rccm.201804-0692OC
        #
        # Let's leave this for later. For now just keep going with basics
    ]
    demographic_features = [
        "AGE",
        "SEX",
        "HEIGHT_CM",
        "WEIGHT_KG"
    ]

    def __init__(self,
                 data_dir_path,
                 cohort_description,
                 feature_set,
                 frame_size,
                 load_intermediates,
                 experiment_num,
                 post_hour,
                 start_hour_delta,
                 frame_func,
                 split_type,
                 test_frame_size=None,
                 test_post_hour=None,
                 test_start_hour_delta=None,
                 custom_vent_features=None,
                 use_ehr_features=True,
                 use_demographic_features=True,
                 vent_bn_frac_missing=.5,
                 use_ventmode=False,
                 ventmode_model_path='',
                 ventmode_scaler_path='',
                 use_tor=False,
                 fft_filtering_low=None,
                 fft_filtering_high=None,
                 butter_low=None,
                 butter_high=None,):
        """
        Define a dataset for use in training an ARDS detection algorithm. If we desire we can
        have separate parameterization for train and test sets. This causes a completely new
        testing set to be created after the training set with differing parameterization for
        all patients

        :param data_dir_path: path to directory where breath data is located
        :param cohort_description: path to cohort description file
        :param feature_set: flow_time/flow_time_opt/flow_time_orig/broad/broad_opt/custom
        :param frame_size: stack N breaths in the data
        :param load_intermediates: Will do best to load intermediate preprocessed data from file
        :param experiment_num: The experiment we wish to run
        :param post_hour: The number of hours post ARDS diagnosis we wish to examine
        :param start_hour_delta: The hour delta that we want to start looking at data for
        :param frame_func: Function to apply on breath frames. choices: median, mean, var, mean+var, median+var, mean+std, median+std
        :param split_type: Type of split to perform on dataset. choices: holdout, kfold, holdout_random, train_all, test_all
        :param test_frame_size: frame size to set only for testing set
        :param test_post_hour: post_hour to set only for testing set
        :param test_start_hour_delta: start delta to set only for testing set
        :param custom_vent_features: If you set features manually you must specify which to use in format (feature name, index)
        :param use_ehr_features: Should we use EHR derived features?
        :param use_demographic_features: Should we use demographic features?
        :param vent_bn_frac_missing: Define amount of sequential BNs we will allow missing from a frame
        :param use_ventmode: bool for whether or not we should use ventmode in our frame
        :param ventmode_model_path:
        :param ventmode_scaler_path:
        :param use_tor: bool for whether or not we should use tor
        :param fft_filtering_low: lower bound hz. perform fft filtering
        :param fft_filtering_high: upper bound hz. perform fft filtering.
        :param butter_low: lower bound hz for butterworth filtering
        :param butter_high: upper bound hz for butterworth filtering
        """
        self.data_dir_path = data_dir_path
        self.split_type = split_type
        self.experiment_num = experiment_num
        self.desc = pd.read_csv(cohort_description)
        self.compliance_upper_lim = 500
        self.compliance_lower_lim = 0
        self.compliance_algo = 'polynomial'

        if feature_set != 'custom':
            self.vent_features = self.vent_feature_sets[feature_set]
        elif feature_set == 'custom':
            self.vent_features = custom_vent_features

        frame_funcs = frame_func.split('+')
        self.frame_funcs = []
        for func in frame_funcs:
            if func == 'median':
                self.frame_funcs.append(np.nanmedian)
            elif func == 'mean':
                self.frame_funcs.append(np.nanmean)
            elif func == 'var':
                self.frame_funcs.append(np.nanvar)
            elif func == 'std':
                self.frame_funcs.append(np.nanstd)
            else:
                raise Exception('Chosen frame function: {} is not currently supported!'.format(frame_func))

        self.frame_size = frame_size
        self.load_intermediates = load_intermediates
        self.post_hour = post_hour
        self.start_hour_delta = start_hour_delta
        self.vent_bn_frac_missing = vent_bn_frac_missing
        # keep track of the number of frames dropped per patient
        self.dropped_data = {}
        if test_frame_size or test_post_hour or test_start_hour_delta:
            self.test_frame_size = test_frame_size if isinstance(test_frame_size, int) else frame_size
            self.test_post_hour = test_post_hour if isinstance(test_post_hour, int) else post_hour
            self.test_start_hour_delta = test_start_hour_delta if isinstance(test_start_hour_delta, int) else start_hour_delta
        else:
            self.test_frame_size = test_frame_size
            self.test_post_hour = test_post_hour
            self.test_start_hour_delta = test_start_hour_delta
        self.use_ehr_features = use_ehr_features
        if use_ehr_features:
            self.ehr_data = pd.read_csv(os.path.join(data_dir_path, EHR_DATA_PATH))
            self.ehr_data['DATA_TIME'] = pd.to_datetime(self.ehr_data.DATA_TIME, format="%m/%d/%y %H:%M")
            self.ehr_data = self.ehr_data.sort_values(by=['PATIENT_ID', 'DATA_TIME'])
            self.ehr_data.index = range(len(self.ehr_data))
        self.use_demographic_features = use_demographic_features
        if use_demographic_features:
            self.demographic_data = pd.read_csv(os.path.join(data_dir_path, DEMOGRAPHIC_DATA_PATH))
            self.demographic_data.loc[self.demographic_data.SEX == 'M', 'SEX'] = 0
            self.demographic_data.loc[self.demographic_data.SEX == 'F', 'SEX'] = 1
        self.use_ventmode = use_ventmode
        self.use_tor = use_tor
        self.ventmode_model_path = ventmode_model_path
        self.ventmode_scaler_path = ventmode_scaler_path
        self.fft_filtering_low = fft_filtering_low
        self.fft_filtering_high = fft_filtering_high
        self.butter_low = butter_low
        self.butter_high = butter_high

    def _get_patient_file_map(self, cohort_dir):
        raw_dirs = []
        file_map = {}
        for i in self.experiment_num.split('+'):
            raw_dirs.append(os.path.join(self.data_dir_path, 'experiment{num}/{cohort_dir}/raw'.format(num=i, cohort_dir=cohort_dir)))
        for dir_ in raw_dirs:
            for patient in os.listdir(dir_):
                files = glob(os.path.join(dir_, patient, "*.csv"))
                # Don't include patients who have no data
                if len(files) == 0:
                    continue
                # Ensure there are only duplicate files for same patient. This is
                # bascially a sanity check.
                if patient in file_map:
                    prev_fs = [os.path.basename(f) for f in file_map[patient]]
                    cur_fs = [os.path.basename(f) for f in files]
                    assert sorted(prev_fs) == sorted(cur_fs), patient
                file_map[patient] = files
        return file_map

    def get(self):
        """
        Get dataset with framed data that we will use for our
        generic learning algorithms
        """
        return self._get_dataset('framed')

    def get_unframed_dataset(self):
        """
        Get dataset with unframed data. This is normally used for debugging
        purposes but can also be used to preprocess features and load them into
        a dataframe for future purposes
        """
        return self._get_dataset('unframed')

    def get_framed_from_unframed_dataset(self, unframed):
        """
        Get a framed dataset using a previously processed unframed dataset. This can
        be helpful in cases where we are re-computing frequently on the same dataset
        like in the case of feature grid search

        :param unframed: The pd.DataFrame instance of the unframed dataset
        """
        all_pts = None
        cohorts = self._get_data_split_params()
        cols = self._get_dataframe_colnames()

        for cohort, params in cohorts.items():
            start_hour_delta = params['sd']
            post_hour = params['sp']
            frame_size = params['frame_size']
            cohort_unframed = unframed[unframed.set_type == cohort]

            for patient_id in cohort_unframed.patient.unique():
                patient_rows = unframed[unframed.patient == patient_id]
                patient_rows = patient_rows.replace([np.inf, -np.inf], np.nan)
                desc_pt_row = self.desc[self.desc['Patient Unique Identifier'].astype(str) == patient_id].iloc[0]
                patho = patient_rows.iloc[0].y
                if patho != 1:
                    pt_start_time = desc_pt_row['vent_start_time']
                else:
                    pt_start_time = desc_pt_row['Date when Berlin criteria first met (m/dd/yyy)']
                pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%m/%d/%y %H:%M"))
                meta = patient_rows[self.vent_features].dropna().values
                meta, stack_times = self.create_breath_frames(meta, frame_size, patient_rows.abs_time_at_BS.values, patient_id)
                if len(meta) == 0:
                    logging.warn('Filtered all data for patient: {} start time: {}'.format(patient_id, start_time))
                    continue
                tmp = pd.DataFrame(meta, columns=cols)
                tmp['patient'] = patient_id
                tmp['row_time'] = stack_times
                tmp['set_type'] = cohort
                tmp['y'] = patho
                hour_row = np.zeros((len(tmp), 1))
                for hour in range(0, 24):
                    mask = np.logical_and(
                        (pt_start_time + np.timedelta64(hour, 'h')) <= stack_times,
                        (pt_start_time + np.timedelta64(hour+1, 'h')) > stack_times
                    )
                    hour_row[mask] = hour
                tmp['hour'] = hour_row
                try:
                    tmp = tmp.drop(['dropme'], axis=1)
                except (KeyError, ValueError):  # its possible we only have 1 feature type to use
                    pass
                if all_pts is None:
                    all_pts = tmp
                else:
                    all_pts = all_pts.append(tmp)

        # reindex and return
        all_pts.index = range(len(all_pts))
        return all_pts

    def _get_dataset(self, type_):
        df = None
        cohorts = self._get_data_split_params()

        for cohort, params in cohorts.items():
            start_hour_delta = params['sd']
            post_hour = params['sp']
            frame_size = params['frame_size']
            file_map = self._get_patient_file_map(params['cohort_dir'])

            for patient in file_map:
                pt_row = self.desc[self.desc['Patient Unique Identifier'].astype(str) == patient]
                if len(pt_row) == 0:
                    raise Exception('Found no information in patient mapping for patient: {}'.format(patient))
                pt_row = pt_row.iloc[0]

                if not self._is_patient_available_in_frame(pt_row, patient, start_hour_delta, post_hour):
                    continue

                patho = pt_row['Pathophysiology'].strip()
                files = file_map[patient]

                if int(patient[:4]) <= 50:
                    date_fmt = r'(\d{4}-\d{2}-\d{2}__\d{2}:\d{2})'
                    strp_fmt = '%Y-%m-%d__%H:%M'
                else:
                    date_fmt = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})'
                    strp_fmt = '%Y-%m-%d-%H-%M'

                if 'ARDS' not in patho:
                    pt_start_time = pt_row['vent_start_time']
                    if pt_start_time is np.nan:
                        first_file = sorted(files)[0]
                        date_str = re.search(date_fmt, first_file).groups()[0]
                        pt_start_time = np.datetime64(datetime.strptime(date_str, strp_fmt)) + np.timedelta64(start_hour_delta, 'h')
                    else:
                        try:
                            pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%m/%d/%y %H:%M")) + np.timedelta64(start_hour_delta, 'h')
                        except ValueError:  # anon
                            pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%Y-%m-%d %H:%M:%S")) + np.timedelta64(start_hour_delta, 'h')

                # Handle COPD+ARDS as just ARDS wrt to the model for now. We can be
                # more granular later
                if 'ARDS' in patho:
                    gt_label = 1
                    pt_start_time = pt_row['Date when Berlin criteria first met (m/dd/yyy)']
                    try:
                        pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%m/%d/%y %H:%M")) + np.timedelta64(start_hour_delta, 'h')
                    except ValueError:  # anon
                        pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%Y-%m-%d %H:%M:%S")) + np.timedelta64(start_hour_delta, 'h')
                # For now we only get first day of recorded data. Maybe in future we will want
                # first day of vent data.
                elif 'COPD' in patho or 'ASTHMA' in patho:
                    gt_label = 2
                else:
                    gt_label = 0

                if type_ == 'unframed':
                    tmp = self.process_unframed_patient_data(patient, files, pt_start_time, post_hour)
                elif type_ == 'framed':
                    tmp = self.process_framed_patient_data(patient, files, pt_start_time, post_hour, frame_size)

                # If we don't have any data don't bother with further steps
                if len(tmp) == 0:
                    continue

                tmp['y'] = gt_label
                tmp['set_type'] = cohort

                # Add relative hour timing to here based on the row_time
                hour_row = np.zeros((len(tmp), 1))
                row_time = tmp.row_time.values
                for hour in range(0, 24):
                    mask = np.logical_and(
                        (pt_start_time + np.timedelta64(hour, 'h')) <= row_time,
                        (pt_start_time + np.timedelta64(hour+1, 'h')) > row_time
                    )
                    hour_row[mask] = hour
                tmp['hour'] = hour_row

                # append patient data frames together
                if df is None:
                    df = tmp
                else:
                    df = df.append(tmp)

        df.index = range(len(df))
        return df

    def fft_filter_waveform(self, waveform):
        freqs = np.fft.fftshift(np.fft.fftfreq(len(waveform), d=0.02))
        freq_mask = np.logical_and(np.abs(freqs) >= self.fft_filtering_low, np.abs(freqs) <= self.fft_filtering_high)  # mask outside frequency bands
        filtered = np.fft.fftshift(np.fft.fft(waveform, axis=-1))
        filtered[~freq_mask] = 0
        filt = list(np.fft.ifft(np.fft.ifftshift(filtered), axis=-1).real)
        return filt

    def butter_filter_waveform(self, waveform):
        if self.butter_low == 0:
            sos = butter(10, self.butter_high, fs=50, output='sos', btype='lowpass')
        elif self.butter_high == 25:
            sos = butter(10, self.butter_low, fs=50, output='sos', btype='highpass')
        else:
            wn = (self.butter_low, self.butter_high)
            sos = butter(10, wn, fs=50, output='sos', btype='bandpass')
        return list(sosfilt(sos, waveform, axis=-1))

    def load_compliance_file(self, patient_id, filename, meta):
        base_filename = os.path.basename(filename)
        intermediate_fname = "compliance_{}".format(base_filename)
        compliance_dir = os.path.dirname(filename).replace('raw', 'meta')
        compliance_path = os.path.join(compliance_dir, intermediate_fname)
        load_from_raw = True

        if self.load_intermediates:
            try:
                return pd.read_csv(compliance_path)
            except IOError:
                pass

        meta = pd.DataFrame(meta, columns=EXPERIMENTAL_META_HEADER)
        c_res = []
        for i, breath in enumerate(extract_raw(open(filename, errors='ignore', encoding='ascii'), False)):
            r = meta.iloc[i]
            flow = np.array(breath['flow']) / 60
            vols = calc_volumes(flow, breath['dt'])
            compliance, resist, _, _ = perform_polynomial_model(
                flow, vols, np.array(breath['pressure']), int(r.x0_index), float(r.PEEP), float(r.tvi)/1000
            )
            # compliance is mult by 1000 because we divide tvi by 1000.
            compliance = compliance * 1000
            c_res.append([breath['rel_bn'], breath['vent_bn'], compliance, resist])
        df = pd.DataFrame(c_res, columns=['rel_bn', 'vent_bn', 'stat_compliance', 'resist'])
        mask = (
            (df['stat_compliance'] > self.compliance_upper_lim) |
            (df['stat_compliance'] < self.compliance_lower_lim)
        )
        df.loc[mask, 'stat_compliance'] = np.nan
        df.loc[df['resist'] < 0, 'resist'] = np.nan
        df.to_csv(compliance_path, index=False)
        return df

    def load_breath_meta_file(self, filename):
        """
        Load breath metadata from a file. If we want to load intermediate
        products then do that if we can. Otherwise load from raw data.
        Save all breath metadata to an intermediate directory
        """
        # We can change this to load from numpy objects, but this method is
        # not where most time is being spent if we have an intermediate. This
        # only takes up about 8% of the time in that case. It's the finding
        # of median that takes up all the time.
        base_filename = os.path.basename(filename)
        intermediate_fname = "breath_meta_{}".format(base_filename)
        meta_dir = os.path.dirname(filename).replace('raw', 'meta')
        metadata_path = os.path.join(meta_dir, intermediate_fname)
        load_from_raw = True

        if self.load_intermediates:
            try:
                with open(metadata_path) as f:
                    meta = []
                    reader = csv.reader(f)
                    for l in reader:
                        meta.append(l)
                load_from_raw = False
            except IOError:
                pass

        if load_from_raw:
            try:
                if self.fft_filtering_low is not None and self.fft_filtering_high is not None:
                    meta = []
                    for breath in extract_raw(open(filename, errors='ignore', encoding='ascii'), False):
                        if len(breath['flow']) == 0 or len(breath['pressure']) == 0:
                            continue
                        breath['flow'] = self.fft_filter_waveform(breath['flow'])
                        breath['pressure'] = self.fft_filter_waveform(breath['pressure'])
                        bm = get_experimental_breath_meta(breath)
                        meta.append(bm)
                elif self.butter_low is not None and self.butter_high is not None:
                    meta = []
                    for breath in extract_raw(open(filename, errors='ignore', encoding='ascii'), False):
                        if len(breath['flow']) == 0 or len(breath['pressure']) == 0:
                            continue
                        breath['flow'] = self.butter_filter_waveform(breath['flow'])
                        breath['pressure'] = self.butter_filter_waveform(breath['pressure'])
                        bm = get_experimental_breath_meta(breath)
                        meta.append(bm)
                else:
                    meta = get_file_experimental_breath_meta(filename, ignore_missing_bes=False)[1:]
            except Exception as err:
                logging.error('Unable to load data from file: {}'.format(filename))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print('-------- BEGIN TRACEBACK -----------')
                traceback.print_tb(exc_traceback)
                print('-------- END TRACEBACK ------------')
                raise err

            Path(meta_dir).mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(meta)

        return meta

    def process_ventmode_tor(self, patient_id, pt_files, meta):
        if xor(self.use_ventmode, self.use_tor):
            # this is really just because I don't have to do ventmode and tor
            # together in any other case other than to just make nice with
            # a reviewer.
            #
            # but in general this whole process takes a butt-ton of time so
            # it shouldn't be done unless we absolutely need it
            raise Exception('do ventmode and tor together otherwise this line of code will continue to be angry')

        # XXX this is kinda bad. you should have ventmode preds in same dir as breath metadata
        mode_preds_filepath = Path('/tmp/').joinpath('{}-ventmode.pkl'.format(patient_id))
        if mode_preds_filepath.exists() and self.load_intermediates:
            ventmode_preds = pd.read_pickle(str(mode_preds_filepath))
        else:
            fileset = {'x': [(patient_id, f) for f in pt_files]}
            # VFinal is pretty bloated and could be dramatically slimmed down if you
            # really need it for speed sake
            ventmode_dataset = datasets.VFinalFeatureSet(fileset, 10, 100)
            ventmode_df = ventmode_dataset.create_prediction_df()
            ventmode_model = pd.read_pickle(self.ventmode_model_path)
            ventmode_scaler = pd.read_pickle(self.ventmode_scaler_path)
            model_results = run_dataset_with_classifier_and_lookahead(
                ventmode_model, ventmode_scaler, ventmode_df, 'vfinal', 50, 0.6
            )
            model_results.predictions = merge_periods_with_low_time_thresh(model_results.predictions, model_results.patient, model_results.abs_bs, pd.Timedelta(minutes=3))
            ventmode_preds = model_results[['rel_bn', 'vent_bn', 'predictions']]
            ventmode_preds.to_pickle(str(mode_preds_filepath))

        # convert to dataframe because pandas has a handy merge function
        meta = pd.DataFrame(meta)
        meta = meta.rename(columns={0: 'rel_bn', 1: 'vent_bn'})
        meta['rel_bn'] = meta.rel_bn.astype(int)
        meta['vent_bn'] = meta.vent_bn.astype(int)
        meta = meta.merge(ventmode_preds, on=['rel_bn', 'vent_bn'], how='inner')
        if len(meta) == 0:
            raise Exception('ventmode merge was not completed successfully for pt: {}'.format(patient_id))
        solo3 = []
        for f in pt_files:
            solo_file = os.path.splitext(Path(f).name)[0] + '_v5_1_0__solo3.csv'
            solo_filepath = Path('/tmp').joinpath(solo_file)
            if solo_filepath.exists() and self.load_intermediates:
                solo3.append(pd.read_csv(str(solo_filepath)))
            else:
                solo3.append(perform_tor_with_bs_be(f, [], '/tmp', 64, 'F', None)[0])

        solo3 = pd.concat(solo3)
        solo3 = solo3.rename(columns={'BN': 'rel_bn', 'ventBN': 'vent_bn', 'dbl.4': 'dta', 'bs.1or2': 'bsa'})[['rel_bn', 'vent_bn', 'dta', 'bsa']]
        meta = meta.merge(solo3, on=['rel_bn', 'vent_bn'], how='inner')
        if len(meta) == 0:
            raise Exception('solo merge was not completed successfully for pt: {}'.format(patient_id))
        return meta.values

    def process_framed_patient_data(self, patient_id, pt_files, start_time, post_hour, frame_size):
        """
        Process all patient framed data for use in our learning algorithms. In this
        function we go from loading all raw breath metadata to compiling it into
        a near usable frame.

        :param patient_id: patient pseudo-id
        :param pt_files: abspath to all patient vent files
        :param start_time: numpy datetime that we want to start analysis on
        :param post_hour: numpy datetime that we want to end analysis
        :param frame_size: size of frames to use
        """
        # PROCESS ALL VENTILATOR DATA
        pt_files = sorted(pt_files)
        meta = self.load_breath_meta_file(pt_files[0])
        compliance = [self.load_compliance_file(patient_id, pt_files[0], meta)]
        vm_tor_cond = (self.use_ventmode or self.use_tor) and tor_possible

        for f in pt_files[1:]:
            # This takes up about 21% of the time in this function if ehr and demo
            # data is not processed
            tmp = self.load_breath_meta_file(f)
            meta.extend(tmp)
            compliance.append(self.load_compliance_file(patient_id, f, tmp))
        compliance = pd.concat(compliance)

        if vm_tor_cond:
            meta = self.process_ventmode_tor(patient_id, pt_files, meta)

        if len(meta) != 0:
            # This takes up 54% of the time in this function if ehr and demo data
            # is not processed
            meta = np.array(meta)
            proper_len = 50 if not vm_tor_cond else 53
            len_mask = [len(r) == proper_len for r in meta]
            meta = np.array(list(meta[len_mask]))
            c_vals = compliance[['stat_compliance', 'resist']].values
            if len(meta) == len(c_vals):
                meta = np.append(meta, c_vals, axis=1)
            else:  # corner case
                compliance = compliance[['rel_bn', 'vent_bn', 'stat_compliance', 'resist']]
                meta = pd.DataFrame(meta)
                meta = meta.rename(columns={0: 'rel_bn', 1: 'vent_bn'})
                meta[['rel_bn', 'vent_bn']] = meta[['rel_bn', 'vent_bn']].astype(int)

                meta = meta.merge(compliance, on=['rel_bn', 'vent_bn'])
                meta = meta.values
            meta, bs_times = self.process_breath_features(meta, start_time, post_hour, patient_id)
            # This takes up 15% of computation time if ehr and demo data is not
            # processed
            meta, stack_times = self.create_breath_frames(meta, frame_size, bs_times, patient_id)
            if len(meta) == 0:
                logging.warn('Filtered all data for patient: {} start time: {}'.format(patient_id, start_time))

        # If all data was filtered by our starting time criteria
        if len(meta) == 0:
            meta = []
        cols = self._get_dataframe_colnames()

        # PROCESS EHR DATA
        if self.use_ehr_features and len(meta) > 0:
            pt_data = self.ehr_data[self.ehr_data.PATIENT_ID == patient_id]
            if len(pt_data) == 0:
                stripped_id = patient_id[:4]
                patients_in_data = self.ehr_data.PATIENT_ID.str[:4].unique()
                if stripped_id in patients_in_data:
                    logging.error('unable to find ehr data for {}. The linkage is bad'.format(patient_id))
                else:
                    logging.error('unable to find ehr data for {}. We were unable to find the patient'.format(patient_id))
                ehr_obs = np.empty((len(meta), len(self.ehr_features)))
                ehr_obs[:] = np.nan
            else:
                # this func now takes only 17% of the time for this entire func
                ehr_obs = self.link_breath_and_ehr_features(pt_data, stack_times, patient_id)

            meta = np.append(meta, ehr_obs, axis=1)
            cols = cols + self.ehr_features
        elif self.use_ehr_features:
            cols = cols + self.ehr_features

        # PROCESS DEMOGRAPHIC DATA
        if self.use_demographic_features and len(meta) > 0:
            pt_data = self.demographic_data[self.demographic_data.PATIENT_ID == patient_id]
            if len(pt_data) == 0:
                logging.error('unable to find demographic data for {}.'.format(patient_id))
                demo_obs = np.empty((len(meta), len(self.demographic_features)))
                demo_obs[:] = np.nan
            elif len(pt_data) > 1:
                raise Exception('Found more than one row of demographic data for {}'.format(patient_id))
            else:
                row = pt_data.iloc[0][self.demographic_features].values
                demo_obs = np.repeat([row], len(meta), axis=0).astype(np.float32)
            meta = np.append(meta, demo_obs, axis=1)
            cols = cols + self.demographic_features
        elif self.use_demographic_features:
            cols = cols + self.demographic_features

        df = pd.DataFrame(meta, columns=cols)
        df['row_time'] = stack_times
        try:
            df = df.drop(['dropme'], axis=1)
        except (KeyError, ValueError):  # its possible we only have 1 feature type to use
            pass
        df['patient'] = patient_id
        return df

    def process_unframed_patient_data(self, patient_id, pt_files, start_time, post_hour):
        """
        Process all patient data so that we can gather a comprehensive inventory of all
        patient data occurring at a given time. This is useful for categorizing how
        many breaths we have from a patient, when the breaths occurred, and how
        many patients we have in aggregate
        """
        pt_files = sorted(pt_files)
        meta = self.load_breath_meta_file(pt_files[0])
        for f in pt_files[1:]:
            meta.extend(self.load_breath_meta_file(f))

        if (self.use_ventmode or self.use_tor) and tor_possible:
            meta = self.process_ventmode_tor(patient_id, pt_files, meta)

        if len(meta) != 0:
            meta = np.array(meta)
            if isinstance(meta[0], list):
                raise Exception('Rows inside metadata are a list for patient: {}. Something went wrong. Try deleting metadata files and re-run'.format(patient_id))
            try:
                bs_times = pd.to_datetime(meta[:, 29], format="%Y-%m-%d %H-%M-%S.%f").values
            except ValueError:
                bs_times = pd.to_datetime(meta[:, 29], format="%Y-%m-%d %H:%M:%S.%f").values
            # Currently we aren't filtering data before start time in unframed data frames
            mask = np.logical_and(start_time <= bs_times, bs_times <= (start_time + np.timedelta64(post_hour, 'h')))
            meta = meta[mask]

        # If all data was filtered by our starting time criteria
        if len(meta) == 0:
            meta = []
        cols = copy(EXPERIMENTAL_META_HEADER)
        if self.use_ventmode and self.use_tor:
            cols += ['ventmode', 'dta', 'bsa']

        df = pd.DataFrame(meta, columns=cols)
        # setup standard datatypes, remove things that are not helpful
        to_drop = [' ', 'BS.1', 'x01', 'tvi1', 'tve1', 'x02', 'tvi2', 'tve2']
        dtypes = {name: "float32" for name in cols if name not in to_drop}
        # exemptions
        exemptions = {
            "BN": 'int16', 'ventBN': 'int16', 'BS': 'float16','x0_index': 'int16',
            'abs_time_at_BS': 'object', 'abs_time_at_x0': 'object',
            'abs_time_at_BE': 'object'
        }
        dtypes.update(exemptions)
        df = df.astype(dtype=dtypes)
        try:
            df['abs_time_at_BS'] = pd.to_datetime(df['abs_time_at_BS'], format="%Y-%m-%d %H-%M-%S.%f")
        except ValueError:
            df['abs_time_at_BS'] = pd.to_datetime(df['abs_time_at_BS'], format="%Y-%m-%d %H:%M:%S.%f")
        df['row_time'] = df['abs_time_at_BS']
        df = df.drop(to_drop, axis=1)
        df['patient'] = patient_id
        return df

    def process_breath_features(self, mat, start_time, post_hour, patient_id):
        """
        Preprocess all breath_meta information. This mainly involves cutting the
        data off when it doesn't correspond to the 24 hr window we're interested in
        examining. Also involves cutting off ventmodes and asynchronies we dont want
        if we are performing this analysis as well.

        :param mat: matrix of data to process
        :param start_time: time we wish to start using data from patient
        :param post_hour: time to stop analyzing data relative to start of ventilation of berlin match
        :param patient_id: patient identifier
        """
        # index of abs bs is 29. So sort by BS time.
        abs_bs_idx = EXPERIMENTAL_META_HEADER.index('abs_time_at_BS')
        try:
            bs_times = pd.to_datetime(mat[:, abs_bs_idx], format="%Y-%m-%d %H-%M-%S.%f").values
        except ValueError:
            bs_times = pd.to_datetime(mat[:, abs_bs_idx], format="%Y-%m-%d %H:%M:%S.%f").values
        # perform filtering via start and end times
        mask = np.logical_and(start_time <= bs_times, bs_times <= (start_time + np.timedelta64(post_hour, 'h')))
        mat = mat[mask]
        bs_times = bs_times[mask]

        row_idxs = [
            EXPERIMENTAL_META_HEADER.index(feature) for feature in self.vent_features
            if feature in EXPERIMENTAL_META_HEADER
        ] + (lambda x: [] if 'stat_compliance' not in x else [-2])(self.vent_features) + (lambda x: [] if 'resist' not in x else [-1])(self.vent_features)
        # drop the nan condition because it preserves data and there is higher
        # likelihood of static compliance calcs being nan
        mask = np.isinf(mat[:, row_idxs].astype(np.float32))
        if (self.use_ventmode and self.use_tor) and tor_possible :
            # filter out undesirable ventmodes (basically anything that isnt VC/PC)
            mask[mat[:, -5].astype(int) > 1] = True
            # filter out DTA/BSA
            mask[np.any(mat[:, [-4, -3]].astype(int)>0, axis=1)] = True

        mat = mat[:, row_idxs]
        mat = mat.astype(np.float32)

        # this block just analyzes where we dropped data.
        #
        # XXX ventmode and tor break this logic tho! At least for now its not
        # a priority to fix this.
        self.dropped_data[patient_id] = {
            'too_many_discontinuous_bns': {'vent_bns': [], 'count': 0},
            'nan_inf_dropping': {
                'drop_vent_bns': None,
                'out_of_n': len(mat),
                'cols': {feature: 0 for feature in self.vent_features}
            },
        }
        if mask.any():
            vent_bn_idx = self.vent_features.index('ventBN')
            vent_bns = list(mat[np.any(mask, axis=1), vent_bn_idx].ravel())
            self.dropped_data[patient_id]['nan_inf_dropping']['drop_vent_bns'] = vent_bns
            cols_dropped = np.where(mask)[1]
            for k, v in pd.value_counts(cols_dropped).items():
                self.dropped_data[patient_id]['nan_inf_dropping']['cols'][self.vent_features[k]] = v

        mask = np.any(mask, axis=1)
        return mat[~mask], bs_times[~mask]

    def create_breath_frames(self, mat, frame_size, bs_times, patient_id):
        """
        Calculate our desired statistics on stacks of breaths.

        :param mat: Matrix to perform rolling average on.
        :param frame_size: number of breaths in stack
        :param bs_times: breath start times for each breath in the matrix
        :param patient_id: patient identifier
        """
        stacks = []
        stack_times = []
        # make sure we capture the last frame even if it's not as complete as we
        # might like it to be
        for low_idx in range(0, len(mat), frame_size):
            row = None
            stack = mat[low_idx:low_idx+frame_size]
            # compare vent bn diffs on stacked breaths.
            vent_bn_idx = self.vent_features.index('ventBN')
            diffs = stack[:-1, vent_bn_idx] + 1 - stack[1:, vent_bn_idx]
            # do not include the stack if it is discontiguous to too large a degree
            bns_missing = sum(abs(diffs))
            missing_thresh = int(frame_size * self.vent_bn_frac_missing)
            if bns_missing > missing_thresh:
                # last vent BN possible is 65536 (2^16) I'd like to recognize if this is occurring
                if not abs(bns_missing - (2 ** 16)) <= missing_thresh:
                    self.dropped_data[patient_id]['too_many_discontinuous_bns']['vent_bns'].append(list(stack[:, vent_bn_idx].ravel()))
                    self.dropped_data[patient_id]['too_many_discontinuous_bns']['count'] += 1
                    continue
            stack_times.append(bs_times[low_idx:low_idx+frame_size][0])
            # We still have ventBN in the matrix, and this essentially gives func(BN)
            #
            # axis=0 takes function across a column
            for func in self.frame_funcs:
                if row is None:
                    row = func(stack, axis=0)
                else:
                    row = np.append(row, func(stack, axis=0))
            stacks.append(row)
        return np.array(stacks), np.array(stack_times)

    def link_breath_and_ehr_features(self, ehr_data, stack_times, patient_id):
        """
        Link breath data to EHR data.

        :param ehr_data: EHR data for a specific patient we want to link
        :param stack_times: Times our breath frames are occurring
        """
        ehr_obs = []
        # add feedforward and feedbackward data
        ehr_data = ehr_data.fillna(method='ffill').fillna(method='bfill')

        for iloc, (loc, row) in enumerate(ehr_data.iterrows()):
            linked_vals = []
            next_loc = loc + 1 if iloc < len(ehr_data) - 1 else None
            # case in which there is only one recording for ehr data
            if iloc == 0 and next_loc is None:
                n_stacks = len(stack_times)
            elif iloc == 0 and next_loc is not None:
                n_stacks = len(stack_times[stack_times < ehr_data.loc[next_loc].DATA_TIME.asm8])
            elif iloc > 0 and next_loc is not None:
                n_stacks = len(stack_times[np.logical_and(stack_times >= row.DATA_TIME.asm8, stack_times < ehr_data.loc[next_loc].DATA_TIME.asm8)])
            elif iloc > 0 and next_loc is None:
                n_stacks = len(stack_times[stack_times >= row.DATA_TIME.asm8])

            for feature in self.ehr_features:
                val = row[feature]
                if not isinstance(val, str) and np.isnan(val) and len(ehr_obs) == 0 and n_stacks > 0:
                    logging.warn('EHR feature {} for patient {} is nan. This may cause patient data to be dropped'.format(feature, patient_id))
                if isinstance(val, str) and '<0.2' in val:
                    val = 0
                elif isinstance(val, str) and '<6.87' in val:
                    val = 6
                linked_vals.append(val)
            ehr_obs.extend([linked_vals] * n_stacks)

        return np.array(ehr_obs).astype(np.float32)

    def _get_dataframe_colnames(self):
        cols = []
        for idx, func in enumerate(self.frame_funcs):
            cols.extend(["{}_{}".format(func.__name__, feature) for feature in self.vent_features])
            # perform a bit of cleanup on hour and ventBN cols
            if idx == 0:
                ventbn_colname = 'ventBN'
            elif idx > 0:
                ventbn_colname = 'dropme'
            cols[cols.index('{}_ventBN'.format(func.__name__))] = ventbn_colname
        return cols

    def _get_data_split_params(self):
        # In case we are using separate parameterization for both train and test
        if (self.test_post_hour or self.test_start_hour_delta or self.test_frame_size) and self.split_type in ['kfold']:
            cohorts = {
                'train': {
                    'sd': self.start_hour_delta,
                    'sp': self.post_hour,
                    'frame_size': self.frame_size,
                    'cohort_dir': 'all_data',
                },
                'test': {
                    'sd': self.test_start_hour_delta,
                    'sp': self.test_post_hour,
                    'frame_size': self.test_frame_size,
                    'cohort_dir': 'all_data',
                },
            }
        elif self.split_type in ['kfold', 'holdout_random', 'train_all', 'test_all']:
            cohorts = {
                'train_test': {
                    'sd': self.start_hour_delta,
                    'sp': self.post_hour,
                    'frame_size': self.frame_size,
                    'cohort_dir': 'all_data',
                }
            }
        elif (self.test_post_hour or self.test_start_hour_delta or self.test_frame_size) and self.split_type in ['holdout', 'holdout_random', 'train_all', 'test_all']:
            raise NotImplementedError('Havent implemented {} split with varying test params'.format(self.split_type))
        elif self.split_type == 'holdout':
            cohorts = {
                'train': {
                    'sd': self.start_hour_delta,
                    'sp': self.post_hour,
                    'frame_size': self.frame_size,
                    'cohort_dir': 'training',
                },
                'test': {
                    'sd': self.start_hour_delta,
                    'sp': self.post_hour,
                    'frame_size': self.frame_size,
                    'cohort_dir': 'testing',
                },
            }
        return cohorts

    def _is_patient_available_in_frame(self, pt_row, patient, start_hour_delta, post_hour):
        if start_hour_delta != 0 or post_hour != 24:
            # evalulate whether the patient should be included in dataset
            pt_avail_rowname = 'available_for_{}-{}_analytics'.format(start_hour_delta, post_hour)
            try:
                pt_row[pt_avail_rowname]
            except KeyError:
                raise Exception('No indicator row {} exists for patient {}! You must add it to your cohort description!'.format(pt_avail_rowname, patient))

            if pt_row[pt_avail_rowname] != 1:
                return False
        return True
