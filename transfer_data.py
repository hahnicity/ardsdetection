from argparse import ArgumentParser
from datetime import datetime, timedelta
import os
import re
import subprocess
from warnings import warn

import pandas as pd

SERVER_NAME = 'b2c-compute'
SERVER_DIRNAME = '/x1/data/results/backups'
# XXX change in future to be generalizable to testing cohorts
DATA_PATH = 'data/experiment{num}/training/raw'


def get_first_days_data(patient_id, initial_dt, experiment_num):
    out_dir = os.path.join(DATA_PATH.format(num=experiment_num), patient_id)
    try:
        os.mkdir(out_dir)
    except OSError:
        pass

    # patients before patient 50 had a different file naming schema
    if int(patient_id[:4]) <= 50:
        file_glob = "*{:04d}-{:02d}-{:02d}__{}:*"
    else:
        file_glob = "*{:04d}-{:02d}-{:02d}-{}-*"

    print('Get data for patient: {}, at start time: {}'.format(patient_id, initial_dt.strftime('%Y-%m-%d-%H-%M')))
    hour_glob = "{{{}}}".format(",".join(["{:02d}".format(i) for i in range(initial_dt.hour - 2, 24)]))
    start_glob = file_glob.format(initial_dt.year, initial_dt.month, initial_dt.day, hour_glob)

    one_day_later = initial_dt + timedelta(days=1)
    hour_glob = "{{{}}}".format(",".join(["{:02d}".format(i) for i in range(0, one_day_later.hour+1)]))
    end_glob = file_glob.format(one_day_later.year, one_day_later.month, one_day_later.day, hour_glob)

    patient_start_path = os.path.join(SERVER_DIRNAME, patient_id, start_glob)
    proc = subprocess.Popen(
        ['rsync', '{}:{}'.format(SERVER_NAME, patient_start_path), out_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    patient_start_path = os.path.join(SERVER_DIRNAME, patient_id, end_glob)
    proc = subprocess.Popen(
        ['rsync', '{}:{}'.format(SERVER_NAME, patient_start_path), out_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()


def copy_ards_patient(row, experiment_num):
    patient_id = row['Patient Unique Identifier']
    berlin_start = row['Date when Berlin criteria first met (m/dd/yyy)']
    try:
        dt = datetime.strptime(berlin_start, '%m/%d/%y %H:%M')
    except ValueError:
        print('Was unable to find Berlin start time for {}. skipping patient'.format(patient_id))
        return
    # should also be grabbing data from at least 2 hours prior because of the way our file rotation
    # is set up
    get_first_days_data(patient_id, dt, experiment_num)


def copy_non_ards_patient(row, experiment_num):
    patient_id = row['Patient Unique Identifier']
    try:
        dt = datetime.strptime(row['vent_start_time'], '%m/%d/%y %H:%M')
    except TypeError:
        warn('Unable to find a vent start time for patient: {}. Now looking for first file collected. However in future this may not be permissive.'.format(patient_id))
        first_file_cmd = "ls {} | head -n 1".format(os.path.join(SERVER_DIRNAME, patient_id, '*.csv'))
        proc = subprocess.Popen(['ssh', SERVER_NAME, first_file_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        stdout = stdout.strip()
        if isinstance(stdout, bytes):
            stdout = stdout.decode('utf-8')

        if stdout == '':
            print('No files found for patient: {}'.format(patient_id))
            return

        if int(patient_id[:4]) <= 50:
            date_fmt = r'(\d{4}-\d{2}-\d{2}__\d{2}:\d{2})'
            strp_fmt = '%Y-%m-%d__%H:%M'
        else:
            date_fmt = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})'
            strp_fmt = '%Y-%m-%d-%H-%M'

        date_str = re.search(date_fmt, stdout).groups()[0]
        try:
            dt = datetime.strptime(date_str, strp_fmt)
        except ValueError:
            print('Was unable to get first file for patient: {}'.format(patient_id))
            return
    get_first_days_data(patient_id, dt, experiment_num)


def main():
    parser = ArgumentParser()
    parser.add_argument('--cohort-description', default='cohort-description.csv', help='Path to file describing the cohort')
    parser.add_argument('-e', '--experiment', default=1, choices=[1, 2, 3, 4], type=int)
    parser.add_argument('-p', '--only-patient', help='Only gather data for specific patient id')
    args = parser.parse_args()

    df = pd.read_csv(args.cohort_description)
    patients_to_get = df[df.experiment_group == args.experiment]
    enrollment = patients_to_get[patients_to_get['Potential Enrollment'] == 'Y']
    if args.only_patient:
        enrollment = enrollment[enrollment['Patient Unique Identifier'] == args.only_patient]
        if len(enrollment) == 0:
            raise Exception('Could not find any rows in cohort for patient {}'.format(args.only_patient))

    for idx, row in enrollment.iterrows():
        patho = row['Pathophysiology']
        if 'ARDS' in patho:
            copy_ards_patient(row, args.experiment)
        else:
            copy_non_ards_patient(row, args.experiment)


if __name__ == "__main__":
    main()
