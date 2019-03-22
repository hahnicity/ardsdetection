"""
patient_time_categorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Categorization of how much data that we have for each
"""
import argparse
from datetime import datetime
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collate import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection')
    parser.add_argument('-p', '--from-pickle')
    parser.add_argument('-t', '--to-pickle')
    parser.add_argument('-d', '--cohort-description', default='cohort-description.csv', help='Path to file describing the cohort')
    parser.add_argument('-e', '--experiment', default='1+4')
    args = parser.parse_args()

    if args.from_pickle:
        df = pd.read_pickle(args.from_pickle)
    else:
        cls = Dataset(args.data_path, args.cohort_description, 'flow_time', 20, True, args.experiment, 24, 0, 'mean')
        df = cls.get_unframed_dataset()

    if args.to_pickle:
        pd.to_pickle(df, args.to_pickle)

    desc = pd.read_csv(args.cohort_description)
    hour_bins = {}
    for patient in df.patient.unique():
        hour_bins[patient] = {i: 0 for i in range(24)}
        patient_df = df[df.patient == patient]

        pt_row = desc[desc['Patient Unique Identifier'] == patient]
        pt_row = pt_row.iloc[0]
        patho = pt_row['Pathophysiology'].strip()
        if int(patient[:4]) <= 50:
            date_fmt = r'(\d{4}-\d{2}-\d{2}__\d{2}:\d{2})'
            strp_fmt = '%Y-%m-%d__%H:%M'
        else:
            date_fmt = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})'
            strp_fmt = '%Y-%m-%d-%H-%M'

        if 'ARDS' not in patho:
            gt_label = 0
            pt_start_time = patient_df.iloc[0].abs_time_at_BS
            hour_bins[patient]['patho'] = 'other'

        if 'ARDS' in patho:
            gt_label = 1
            pt_start_time = pt_row['Date when Berlin criteria first met (m/dd/yyy)']
            pt_start_time = np.datetime64(datetime.strptime(pt_start_time, "%m/%d/%y %H:%M"))
            hour_bins[patient]['patho'] = 'ards'

        if 'COPD' in patho:
            gt_label = 2
            hour_bins[patient]['patho'] = 'copd'

        # count how much data patient has in each hour bin
        for hour in range(1, 25):
            mask = np.logical_and(
                pt_start_time + np.timedelta64(hour-1, 'h') <= patient_df.abs_time_at_BS,
                patient_df.abs_time_at_BS < pt_start_time + np.timedelta64(hour, 'h')
            )
            num_breaths = len(patient_df[mask])
            hour_bins[patient][hour-1] = num_breaths

    # get counts of patients who have data by the hour
    for patho in ['other', 'ards', 'copd']:
        patho_hours = [0] * 24
        for patient, bins in hour_bins.items():
            if bins['patho'] != patho:
                continue
            for hour in range(24):
                if bins[hour] > 0:
                    patho_hours[hour] += 1
        plt.title('{} patient count with data by hour'.format(patho.upper()))
        plt.ylabel('count')
        plt.xlabel('hour')
        plt.xlim((.5, 24.5))
        plt.bar(np.arange(1, 25), patho_hours)
        plt.show()


if __name__ == "__main__":
    main()
