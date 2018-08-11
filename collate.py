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

import numpy as np
import pandas as pd

from algorithms.breath_meta import get_file_experimental_breath_meta


class Dataset(object):
    # Feature sets are mapped by (feature_name, breath_meta_feature_index)
    neccessities = [('ventBN', 2)]
    flow_time_feature_set = neccessities + [
        # minF_to_zero is just pef_to_zero
        ('mean_flow_from_pef', 37), ('inst_RR', 8), ('minF_to_zero', 36),
        ('pef_+0.16_to_zero', 37), ('iTime', 6), ('eTime', 7), ('I:E ratio', 5),
        ('dyn_compliance', 38),
    ]
    broad_feature_set = flow_time_feature_set + [
        ('TVi', 9), ('TVe', 10), ('Maw', 16), ('ipAUC', 18), ('PIP', 15), ('PEEP', 17),
        ('epAUC', 19),
    ]

    def __init__(self, cohort_description, feature_set, breaths_to_stack, load_intermediates):
        """
        :param cohort_description: path to cohort description file
        :param feature_set: flow_time or broad
        :param breaths_to_stack: stack N breaths in the data
        :param load_intermediates: Will do best to load intermediate preprocessed data from file
        """
        # XXX Currently just analyze experiment 1. In the future this will
        # be configurable tho.
        self.raw_dir = 'data/experiment1/training/raw'
        self.desc = pd.read_csv(cohort_description)
        self.file_map = {}
        for patient in os.listdir(self.raw_dir):
            files = glob(os.path.join(self.raw_dir, patient, "*.csv"))
            # Don't include patients who have no data
            if len(files) > 0:
                self.file_map[patient] = files

        if feature_set == 'flow_time':
            self.features = OrderedDict(self.flow_time_feature_set)
        elif feature_set == 'broad':
            self.features = OrderedDict(self.broad_feature_set)

        self.breaths_to_stack = breaths_to_stack
        self.load_intermediates = load_intermediates

    def get(self):
        # So what we do for this is go through patient by patient and extract
        # their metadata in a way the rpi will enjoy
        df = None
        for patient in self.file_map:
            pt_row = self.desc[self.desc['Patient Unique Identifier'] == patient]
            if len(pt_row) > 1:
                raise Exception('Found more than 1 row for patient: {}'.format(patient))
            elif len(pt_row) == 0:
                raise Exception('Found more than no rows for patient: {}'.format(patient))
            pt_row = pt_row.iloc[0]
            patho = pt_row['Pathophysiology'].strip()
            files = self.file_map[patient]

            # Handle COPD+ARDS as just ARDS wrt to the model for now. We can be
            # more granular later
            if 'ARDS' in patho:
                gt_label = 1
                pt_start_time = pt_row['Date when Berlin criteria first met (m/dd/yyy)']
                pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%m/%d/%y %H:%M"))
            elif 'COPD' in patho:
                gt_label = 2
                pt_start_time = None
            else:
                gt_label = 0
                pt_start_time = None

            if df is None:
                df = self.process_patient_data(patient, files, pt_start_time)
                df['y'] = gt_label
            else:
                tmp = self.process_patient_data(patient, files, pt_start_time)
                tmp['y'] = gt_label
                df = df.append(tmp)
        return df

    def load_breath_meta_file(self, filename):
        """
        Load breath metadata from a file. If we want to load intermediate
        products then do that if we can. Otherwise load from raw data.
        Save all breath metadata to an intermediate directory
        """
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
            meta = get_file_experimental_breath_meta(filename)[1:]
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
        for f in pt_files:
            meta.extend(self.load_breath_meta_file(f))
            # XXX in future add filename as well. Especially if debugging necessary

        if len(meta) != 0:
            meta = self.process_features(np.array(meta), start_time)
            meta = self.to_stacked_median(meta)
        cols = list(self.features.keys())
        try:
            df = pd.DataFrame(meta, columns=cols)
        except:
            raise Exception('Failed to capture any data for patient: {}'.format(patient_id))
        df['patient'] = patient_id
        return df

    def process_features(self, mat, start_time):
        """
        Preprocess all breath_meta information

        :param mat: matrix of data to process
        :param start_time: time we wish to start using data from patient
        """
        mat = mat[mat[:, 29].argsort()]
        if start_time is not None:
            # index of abs bs is 29
            dt = pd.to_datetime(mat[:, 29], format="%Y-%m-%d %H-%M-%S.%f").values
            mask = dt >= start_time
            mat = mat[mask]
        row_idxs = list(self.features.values())
        mat = mat[:, row_idxs]
        mat = mat.astype(np.float32)
        mask = np.any(np.isnan(mat) | np.isinf(mat), axis=1)
        return mat[~mask]

    def to_stacked_median(self, mat):
        """
        Find median of stacks of breaths on a matrix

        :param mat: Matrix to perform rolling average on.
        """
        stacks = []
        for low_idx in range(0, len(mat)-self.breaths_to_stack, self.breaths_to_stack):
            stack = mat[low_idx:low_idx+self.breaths_to_stack]
            # We still have ventBN in the matrix, and this essentially gives average BN
            stacks.append(np.median(mat, axis=0))
        return np.array(stacks)
