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


class Dataset(object):
    # Feature sets are mapped by (feature_name, breath_meta_feature_index)
    necessities = [('ventBN', 2)]
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
        ('pef_+0.16_to_zero', 37),
        ('TVratio', 11),
        ('eTime', 7)
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

    def __init__(self, cohort_description, feature_set, frame_size, load_intermediates, experiment_num, post_hour, start_hour_delta, frame_func, custom_features=None):
        """
        :param cohort_description: path to cohort description file
        :param feature_set: flow_time/flow_time_opt/flow_time_orig/broad/broad_opt/custom
        :param frame_size: stack N breaths in the data
        :param load_intermediates: Will do best to load intermediate preprocessed data from file
        :param experiment_num: The experiment we wish to run
        :param post_hour: The number of hours post ARDS diagnosis we wish to examine
        :param start_hour_delta: The hour delta that we want to start looking at data for
        :param frame_func: Function to apply on breath frames. choices: median, mean, var
        :param custom_features: If you set features manually you must specify which to use in format (feature name, index)
        """
        raw_dirs = []
        for i in experiment_num.split('+'):
            raw_dirs.append('data/experiment{num}/training/raw'.format(num=i))
        self.desc = pd.read_csv(cohort_description)
        self.file_map = {}
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

        if frame_func == 'median':
            self.frame_func = np.median
        elif frame_func == 'mean':
            self.frame_func = np.mean
        elif frame_func == 'var':
            self.frame_func = np.var
        else:
            raise Exception('Chosen frame function: {} is not currently supported!'.format(frame_func))

        self.frame_size = frame_size
        self.load_intermediates = load_intermediates
        self.post_hour = post_hour
        self.start_hour_delta = start_hour_delta

    def get(self):
        # So what we do for this is go through patient by patient and extract
        # their metadata in a way the rpi will enjoy
        df = None
        for patient in self.file_map:
            pt_row = self.desc[self.desc['Patient Unique Identifier'] == patient]
            # Trust the cohort descriptor file that all patients are respresented
            # equally in different experiments. This is my current philosophy. But will
            # this idea change? In which case I will need to segment by experiment number
            # and then do a check for patients with similar and dissimilar experimental
            # setups. I can't imagine that this will actually happen in the project and
            # it would be a major design decision that would affect the paper.
            #
            # XXX If you ever change this policy be sure to change code
            if len(pt_row) == 0:
                raise Exception('Found more than no rows for patient: {}'.format(patient))
            pt_row = pt_row.iloc[0]
            patho = pt_row['Pathophysiology'].strip()
            files = self.file_map[patient]

            if int(patient[:4]) <= 50:
                date_fmt = r'(\d{4}-\d{2}-\d{2}__\d{2}:\d{2})'
                strp_fmt = '%Y-%m-%d__%H:%M'
            else:
                date_fmt = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})'
                strp_fmt = '%Y-%m-%d-%H-%M'

            if 'ARDS' not in patho:
                first_file = sorted(files)[0]
                date_str = re.search(date_fmt, first_file).groups()[0]
                pt_start_time = np.datetime64(datetime.strptime(date_str, strp_fmt)) + np.timedelta64(self.start_hour_delta, 'h')

            # Handle COPD+ARDS as just ARDS wrt to the model for now. We can be
            # more granular later
            if 'ARDS' in patho:
                gt_label = 1
                pt_start_time = pt_row['Date when Berlin criteria first met (m/dd/yyy)']
                pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%m/%d/%y %H:%M")) + np.timedelta64(self.start_hour_delta, 'h')
            # For now we only get first day of recorded data. Maybe in future we will want
            # first day of vent data.
            elif 'COPD' in patho:
                gt_label = 2
            else:
                gt_label = 0

            # XXX If you want to make this faster you can always run this operation in
            # parallel and just concat whatever you get from the pooling.
            if df is None:
                df = self.process_patient_data(patient, files, pt_start_time)
                df['y'] = gt_label
            else:
                tmp = self.process_patient_data(patient, files, pt_start_time)
                tmp['y'] = gt_label
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

    def process_patient_data(self, patient_id, pt_files, start_time):
        # Cut off the header with [1:]
        meta = self.load_breath_meta_file(pt_files[0])
        for f in pt_files[1:]:
            meta.extend(self.load_breath_meta_file(f))

        if len(meta) != 0:
            meta = self.process_features(np.array(meta), start_time)
            meta = self.create_frames(meta)
            if len(meta) == 0:
                warn('Filtered all data for patient: {} start time: {}'.format(patient_id, start_time))

        # If all data was filtered by our starting time criteria
        if len(meta) == 0:
            meta = []
        cols = list(self.features.keys())
        df = pd.DataFrame(meta, columns=cols)
        df['patient'] = patient_id
        return df

    def process_features(self, mat, start_time):
        """
        Preprocess all breath_meta information

        :param mat: matrix of data to process
        :param start_time: time we wish to start using data from patient
        """
        # index of abs bs is 29. So sort by BS time.
        mat = mat[mat[:, 29].argsort()]
        if start_time is not None:
            try:
                dt = pd.to_datetime(mat[:, 29], format="%Y-%m-%d %H-%M-%S.%f").values
            except ValueError:
                dt = pd.to_datetime(mat[:, 29], format="%Y-%m-%d %H:%M:%S.%f").values
            mask = dt <= (start_time + np.timedelta64(self.post_hour, 'h'))
            mat = mat[mask]
        row_idxs = list(self.features.values())
        mat = mat[:, row_idxs]
        mat = mat.astype(np.float32)
        mask = np.any(np.isnan(mat) | np.isinf(mat), axis=1)
        return mat[~mask]

    def create_frames(self, mat):
        """
        Find median of stacks of breaths on a matrix

        :param mat: Matrix to perform rolling average on.
        """
        stacks = []
        for low_idx in range(0, len(mat)-self.frame_size, self.frame_size):
            stack = mat[low_idx:low_idx+self.frame_size]
            # We still have ventBN in the matrix, and this essentially gives average BN
            stacks.append(self.frame_func(stack, axis=0))
        return np.array(stacks)
