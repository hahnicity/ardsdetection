"""
collate
~~~~~~~

Create dataset of items we wish to use for the training/testing of our
model.
"""
from collections import OrderedDict
import csv
from datetime import datetime
from glob import glob
import os
import re
from warnings import warn

import numpy as np
import pandas as pd

from algorithms.breath_meta import get_file_experimental_breath_meta
from algorithms.constants import EXPERIMENTAL_META_HEADER


class Dataset(object):
    # Feature sets are mapped by (feature_name, breath_meta_feature_index)
    necessities = [('ventBN', 2), ('hour', -1)]
    flow_time_feature_set = necessities + [
        # minF_to_zero is just pef_to_zero
        ('mean_flow_from_pef', 38),
        ('inst_RR', 8),
        ('minF_to_zero', 36),
        ('pef_+0.16_to_zero', 37),
        ('iTime', 6),
        ('eTime', 7),
        ('I:E ratio', 5),
        ('dyn_compliance', 39),
        ('TVratio', 11),
    ]
    flow_time_original = necessities + [
        ('mean_flow_from_pef', 38),
        ('inst_RR', 8),
        ('minF_to_zero', 36),
        ('pef_+0.16_to_zero', 37),
        ('iTime', 6),
        ('eTime', 7),
        ('I:E ratio', 5),
        ('dyn_compliance', 39),
    ]
    flow_time_optimal = necessities + [
        ('dyn_compliance', 39),
        ('TVratio', 11),
        ('mean_flow_from_pef', 38),
        ('eTime', 7),
        ('I:E ratio', 5),
    ]
    broad_feature_set = flow_time_feature_set + [
        ('TVi', 9),
        ('TVe', 10),
        ('Maw', 16),
        ('ipAUC', 18),
        ('PIP', 15),
        ('PEEP', 17),
        ('epAUC', 19),
    ]
    broad_optimal = necessities + [
        ('PEEP', 17),
        ('I:E Ratio', 5),
        ('inst_RR', 8),
        ('TVi', 9),
        ('PIP', 15),
        ('iTime', 6),
    ]

    def __init__(self,
                 cohort_description,
                 feature_set,
                 frame_size,
                 load_intermediates,
                 experiment_num,
                 post_hour,
                 start_hour_delta,
                 frame_func,
                 test_frame_size=None,
                 test_post_hour=None,
                 test_start_hour_delta=None,
                 custom_features=None):
        """
        Define a dataset for use in training an ARDS detection algorithm. If we desire we can
        have separate parameterization for train and test sets. This causes a completely new
        testing set to be created after the training set with differing parameterization for
        all patients

        :param cohort_description: path to cohort description file
        :param feature_set: flow_time/flow_time_opt/flow_time_orig/broad/broad_opt/custom
        :param frame_size: stack N breaths in the data
        :param load_intermediates: Will do best to load intermediate preprocessed data from file
        :param experiment_num: The experiment we wish to run
        :param post_hour: The number of hours post ARDS diagnosis we wish to examine
        :param start_hour_delta: The hour delta that we want to start looking at data for
        :param frame_func: Function to apply on breath frames. choices: median, mean, var, mean+var, median+var, mean+std, median+std
        :param test_frame_size: frame size to set only for testing set
        :param test_post_hour: post_hour to set only for testing set
        :param test_start_hour_delta: start delta to set only for testing set
        :param custom_features: If you set features manually you must specify which to use in format (feature name, index)
        """
        raw_dirs = []
        for i in experiment_num.split('+'):
            raw_dirs.append('data/experiment{num}/training/raw'.format(num=i))
        self.desc = pd.read_csv(cohort_description)
        self.file_map = {}
        self.experiment_num = experiment_num
        for dir_ in raw_dirs:
            for patient in os.listdir(dir_):
                files = glob(os.path.join(dir_, patient, "*.csv"))
                # Don't include patients who have no data
                if len(files) == 0:
                    continue
                # Ensure there are only duplicate files for same patient. This is
                # bascially a sanity check.
                if patient in self.file_map:
                    prev_fs = [os.path.basename(f) for f in self.file_map[patient]]
                    cur_fs = [os.path.basename(f) for f in files]
                    assert sorted(prev_fs) == sorted(cur_fs), patient
                self.file_map[patient] = files

        if feature_set == 'flow_time':
            self.features = OrderedDict(self.flow_time_feature_set)
        if feature_set == 'flow_time_orig':
            self.features = OrderedDict(self.flow_time_original)
        elif feature_set == 'flow_time_opt':
            self.features = OrderedDict(self.flow_time_optimal)
        elif feature_set == 'broad':
            self.features = OrderedDict(self.broad_feature_set)
        elif feature_set == 'broad_opt':
            self.features = OrderedDict(self.broad_optimal)
        elif feature_set == 'custom':
            self.features = OrderedDict(custom_features)

        frame_funcs = frame_func.split('+')
        self.frame_funcs = []
        for func in frame_funcs:
            if func == 'median':
                self.frame_funcs.append(np.median)
            elif func == 'mean':
                self.frame_funcs.append(np.mean)
            elif func == 'var':
                self.frame_funcs.append(np.var)
            elif func == 'std':
                self.frame_funcs.append(np.std)
            else:
                raise Exception('Chosen frame function: {} is not currently supported!'.format(frame_func))

        self.frame_size = frame_size
        self.load_intermediates = load_intermediates
        self.post_hour = post_hour
        self.start_hour_delta = start_hour_delta
        self.test_frame_size = test_frame_size if isinstance(test_frame_size, int) else frame_size
        self.test_post_hour = test_post_hour if isinstance(test_post_hour, int) else post_hour
        self.test_start_hour_delta = test_start_hour_delta if isinstance(test_start_hour_delta, int) else start_hour_delta

    def get(self):
        """
        Get dataset with framed data that we will use for our
        generic learning algorithms
        """
        return self._get_dataset('framed')

    def get_unframed_dataset(self):
        """
        Get dataset with unframed data. This is normally used for debugging
        purposes
        """
        return self._get_dataset('unframed')

    def _get_dataset(self, type_):
        df = None
        # We are using separate parameterization for both train and test
        if self.test_post_hour or self.test_start_hour_delta or self.test_frame_size:
            cohorts = {
                'train': {
                    'sd': self.start_hour_delta,
                    'sp': self.post_hour,
                    'frame_size': self.frame_size
                },
                'test': {
                    'sd': self.test_start_hour_delta,
                    'sp': self.test_post_hour,
                    'frame_size': self.test_frame_size
                },
            }
        else:
            cohorts = {
                'train_test': {
                    'sd': self.start_hour_delta,
                    'sp': self.post_hour,
                    'frame_size': self.frame_size
                }
            }

        for cohort, params in cohorts.items():
            start_hour_delta = params['sd']
            post_hour = params['sp']
            frame_size = params['frame_size']

            for patient in self.file_map:
                pt_row = self.desc[self.desc['Patient Unique Identifier'] == patient]
                if len(pt_row) == 0:
                    raise Exception('Found no information in patient mapping for patient: {}'.format(patient))
                pt_row = pt_row[(pt_row.experiment_group.isin([int(i) for i in self.experiment_num.split('+')])) & (pt_row['Potential Enrollment'] == 'Y')]
                if len(pt_row) == 0:
                    raise Exception("patient {} is not supposed to be in the cohort!".format(patient))
                else:
                    pt_row = pt_row.iloc[0]

                patho = pt_row['Pathophysiology'].strip()
                # Sanity check
                files = self.file_map[patient]

                if int(patient[:4]) <= 50:
                    date_fmt = r'(\d{4}-\d{2}-\d{2}__\d{2}:\d{2})'
                    strp_fmt = '%Y-%m-%d__%H:%M'
                else:
                    date_fmt = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})'
                    strp_fmt = '%Y-%m-%d-%H-%M'

                if 'ARDS' not in patho:
                    pt_start_time = pt_row['vent_start_time']
                    # XXX in future change this behavior if we don't want to use patients
                    # without a vent start time
                    if pt_start_time is np.nan:
                        first_file = sorted(files)[0]
                        date_str = re.search(date_fmt, first_file).groups()[0]
                        pt_start_time = np.datetime64(datetime.strptime(date_str, strp_fmt)) + np.timedelta64(start_hour_delta, 'h')
                    else:
                        pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%m/%d/%y %H:%M")) + np.timedelta64(start_hour_delta, 'h')

                # Handle COPD+ARDS as just ARDS wrt to the model for now. We can be
                # more granular later
                if 'ARDS' in patho:
                    gt_label = 1
                    pt_start_time = pt_row['Date when Berlin criteria first met (m/dd/yyy)']
                    pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%m/%d/%y %H:%M")) + np.timedelta64(start_hour_delta, 'h')
                # For now we only get first day of recorded data. Maybe in future we will want
                # first day of vent data.
                elif 'COPD' in patho or 'ASTHMA' in patho:
                    gt_label = 2
                else:
                    gt_label = 0

                if type_ == 'unframed':
                    tmp = self.process_unframed_patient_data(patient, files, pt_start_time, post_hour)
                elif type_ == 'framed':
                    tmp = self.process_patient_data(patient, files, pt_start_time, post_hour, frame_size)
                tmp['y'] = gt_label
                tmp['set_type'] = cohort

                if df is None:
                    df = tmp
                else:
                    df = df.append(tmp)

        df.index = range(len(df))
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
            meta = get_file_experimental_breath_meta(filename, ignore_missing_bes=False)[1:]
            try:
                os.mkdir(meta_dir)
            except OSError:  # dir likely exists
                pass

            with open(metadata_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(meta)

        return meta

    def process_patient_data(self, patient_id, pt_files, start_time, post_hour, frame_size):
        """
        Process all patient data for use in our learning algorithms

        :param patient_id: patient pseudo-id
        :param pt_files: abspath to all patient vent files
        :param start_time: numpy datetime that we want to start analysis on
        :param post_hour: numpy datetime that we want to end analysis
        :param frame_size: size of frames to use
        """
        # Cut off the header with [1:]
        meta = self.load_breath_meta_file(pt_files[0])
        for f in pt_files[1:]:
            meta.extend(self.load_breath_meta_file(f))

        if len(meta) != 0:
            meta = self.process_features(np.array(meta), start_time, post_hour)
            meta = self.create_frames(meta, frame_size)
            if len(meta) == 0:
                warn('Filtered all data for patient: {} start time: {}'.format(patient_id, start_time))

        # If all data was filtered by our starting time criteria
        if len(meta) == 0:
            meta = []

        cols = []
        for idx, func in enumerate(self.frame_funcs):
            cols.extend(["{}_{}".format(func.__name__, feature) for feature in self.features.keys()])
            # perform a bit of cleanup on hour and ventBN cols
            if idx == 0:
                hour_colname = 'hour'
                ventbn_colname = 'ventBN'
            elif idx > 0:
                hour_colname = 'dropme'
                ventbn_colname = 'dropme'
            cols[cols.index('{}_hour'.format(func.__name__))] = hour_colname
            cols[cols.index('{}_ventBN'.format(func.__name__))] = ventbn_colname

        df = pd.DataFrame(meta, columns=cols)
        df = df.drop(['dropme'], axis=1)
        df['patient'] = patient_id
        return df

    def process_unframed_patient_data(self, patient_id, pt_files, start_time, post_hour):
        """
        Process all patient data so that we can gather a comprehensive inventory of all their
        data occurring at their given times
        """
        meta = self.load_breath_meta_file(pt_files[0])
        for f in pt_files[1:]:
            meta.extend(self.load_breath_meta_file(f))

        if len(meta) != 0:
            meta = np.array(meta)
            if isinstance(meta[0], list):
                raise Exception('Rows inside metadata are a list for patient: {}. Something went wrong. Try deleting metadata files and re-run'.format(patient_id))
            meta = meta[meta[:, 29].argsort()]
            if start_time is not None:
                try:
                    dt = pd.to_datetime(meta[:, 29], format="%Y-%m-%d %H-%M-%S.%f").values
                except ValueError:
                    dt = pd.to_datetime(meta[:, 29], format="%Y-%m-%d %H:%M:%S.%f").values
                mask = dt <= (start_time + np.timedelta64(post_hour, 'h'))
                meta = meta[mask]

        # If all data was filtered by our starting time criteria
        if len(meta) == 0:
            meta = []
        df = pd.DataFrame(meta, columns=EXPERIMENTAL_META_HEADER)
        # setup standard datatypes, remove things that are not helpful
        to_drop = [' ', 'BS.1', 'x01', 'tvi1', 'tve1', 'x02', 'tvi2', 'tve2']
        dtypes = {name: "float32" for name in EXPERIMENTAL_META_HEADER if name not in to_drop}
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
        df = df.drop(to_drop, axis=1)
        df['patient'] = patient_id
        return df

    def process_features(self, mat, start_time, post_hour):
        """
        Preprocess all breath_meta information

        :param mat: matrix of data to process
        :param start_time: time we wish to start using data from patient
        :param post_hour: time to stop analyzing data relative to start of ventilation of berlin match
        """
        # index of abs bs is 29. So sort by BS time.
        mat = mat[mat[:, 29].argsort()]
        if start_time is not None:
            try:
                dt = pd.to_datetime(mat[:, 29], format="%Y-%m-%d %H-%M-%S.%f").values
            except ValueError:
                dt = pd.to_datetime(mat[:, 29], format="%Y-%m-%d %H:%M:%S.%f").values
            mask = dt <= (start_time + np.timedelta64(post_hour, 'h'))
            mat = mat[mask]

        hour_row = np.zeros((len(mat), 1))
        try:
            dt = pd.to_datetime(mat[:, 29], format="%Y-%m-%d %H-%M-%S.%f").values
        except ValueError:
            dt = pd.to_datetime(mat[:, 29], format="%Y-%m-%d %H:%M:%S.%f").values
        for hour in range(0, 24):
            mask = np.logical_and(
                (start_time + np.timedelta64(hour, 'h')) <= dt,
                (start_time + np.timedelta64(hour+1, 'h')) > dt
            )
            hour_row[mask] = hour
        row_idxs = list(self.features.values())
        mat = np.append(mat, hour_row, axis=1)
        mat = mat[:, row_idxs]
        mat = mat.astype(np.float32)
        mask = np.any(np.isnan(mat) | np.isinf(mat), axis=1)
        return mat[~mask]

    def create_frames(self, mat, frame_size):
        """
        Find median of stacks of breaths on a matrix

        :param mat: Matrix to perform rolling average on.
        :param frame_size: number of breaths in stack
        """
        stacks = []
        # make sure we capture the last frame even if it's not as complete as we
        # might like it to be
        for low_idx in range(0, len(mat), frame_size):
            row = None
            stack = mat[low_idx:low_idx+frame_size]
            # We still have ventBN in the matrix, and this essentially gives average BN
            #
            # axis=0 takes function across a column
            for func in self.frame_funcs:
                if row is None:
                    row = func(stack, axis=0)
                else:
                    row = np.append(row, func(stack, axis=0))
            stacks.append(row)
        return np.array(stacks)
