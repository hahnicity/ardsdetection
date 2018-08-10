from argparse import ArgumentParser
from datetime import datetime, timedelta
import os
import re
import subprocess

import pandas as pd

SERVER_NAME = 'b2c-compute'
SERVER_DIRNAME = '/x1/data/results/backups'
# XXX change in future to be generalizable
DATA_PATH = 'data/experiment1/training/raw'


def get_first_days_data(patient_id, initial_dt, skip_existing_patients):
    out_dir = os.path.join(DATA_PATH, patient_id)
    try:
        os.mkdir(out_dir)
    except OSError:
        if skip_existing_patients and len(os.listdir(out_dir)) > 0:
            print('Skip patient {} because they have existing data'.format(patient_id))
            return

    print('Get data for patient: {}, at start time: {}'.format(patient_id, initial_dt.strftime('%Y-%m-%d-%H-%M')))
    hour_glob = "{{{}}}".format(",".join(["{:02d}".format(i) for i in range(initial_dt.hour - 2, 24)]))
    start_glob = "*{:04d}-{:02d}-{:02d}-{}-*".format(initial_dt.year, initial_dt.month, initial_dt.day, hour_glob)

    one_day_later = initial_dt + timedelta(days=1)
    hour_glob = "{{{}}}".format(",".join(["{:02d}".format(i) for i in range(0, one_day_later.hour+1)]))
    end_glob = "*{:04d}-{:02d}-{:02d}-{}-*".format(one_day_later.year, one_day_later.month, one_day_later.day, hour_glob)

    patient_start_path = os.path.join(SERVER_DIRNAME, patient_id, start_glob)
    proc = subprocess.Popen(
        ['scp', '{}:{}'.format(SERVER_NAME, patient_start_path), out_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    patient_start_path = os.path.join(SERVER_DIRNAME, patient_id, end_glob)
    proc = subprocess.Popen(
        ['scp', '{}:{}'.format(SERVER_NAME, patient_start_path), out_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()


def copy_ards_patient(row, skip_existing_patients):
    patient_id = row['Patient Unique Identifier']
    berlin_start = row['Date when Berlin criteria first met (m/dd/yyy)']
    try:
        dt = datetime.strptime(berlin_start, '%m/%d/%y %H:%M')
    except ValueError:
        print('Was unable to find Berlin start time for {}. skipping patient'.format(patient_id))
        return
    # should also be grabbing data from at least 2 hours prior because of the way our file rotation
    # is set up
    get_first_days_data(patient_id, dt, skip_existing_patients)


def copy_non_ards_patient(row, skip_existing_patients):
    # XXX Currently we do not know when patient was first intubated, so for
    # now it might be best to just gather first 24 hrs of data. But in future
    # we will need to gather first 24 exclusively
    patient_id = row['Patient Unique Identifier']
    first_file_cmd = "ls {} | head -n 1".format(os.path.join(SERVER_DIRNAME, patient_id))
    proc = subprocess.Popen(['ssh', SERVER_NAME, first_file_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stdout == '':
        print('No files found for patient: {}'.format(patient_id))
        return
    if isinstance(stdout, bytes):
        stdout = stdout.decode()
    date_str = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})', stdout).groups()[0]
    try:
        init_dt = datetime.strptime(date_str, '%Y-%m-%d-%H-%M')
    except ValueError:
        print('Was unable to get first file for patient: {}'.format(patient_id))
        return
    get_first_days_data(patient_id, init_dt, skip_existing_patients)


def main():
    parser = ArgumentParser()
    parser.add_argument('--cohort-description', default='cohort-description.csv', help='Path to file describing the cohort')
    parser.add_argument('--experiment', default=1, choices=[1, 2], type=int)
    parser.add_argument('--skip-existing-patients', action='store_true', help='Dont collect more data for patients who already have data')
    args = parser.parse_args()

    df = pd.read_csv(args.cohort_description)
    patients_to_get = df[df['Experiment Group 1 / 2'] == args.experiment]
    enrollment = patients_to_get[patients_to_get['Potential Enrollment'] == 'Y']

    for idx, row in enrollment.iterrows():
        patho = row['Pathophysiology']
        if patho == 'ARDS':
            copy_ards_patient(row, args.skip_existing_patients)
        else:
            copy_non_ards_patient(row, args.skip_existing_patients)


if __name__ == "__main__":
    main()
