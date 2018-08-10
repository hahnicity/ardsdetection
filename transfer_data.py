from argparse import ArgumentParser
from datetime import datetime, timedelta
import subprocess

import pandas as pd

SERVER_NAME = 'b2c-compute'
SERVER_DIRNAME = '/x1/data/results/backups/'
# XXX change in future to be generalizable
DATA_PATH = 'data/experiment1/training/raw'


def get_first_days_data(patient_id, initial_dt):
    hour_glob = "{{{}}}".format(",".join(["{0:02d}".format(i) for i in range(initial_dt.hour - 2, 24)]))
    start_glob = "*{}-{}-{}-{}-*".format(initial_dt.year, initial_dt.month, initial_dt.day, hour_glob)

    one_day_later = initial_dt + timedelta(days=1)
    hour_glob = "{{{}}}".format(",".join(["{0:02d}".format(i) for i in range(0, one_day_later.hour+1)]))
    end_glob = "*{}-{}-{}-{}-*".format(one_day_later.year, one_day_later.month, one_day_later.day, hour_glob)

    patient_start_path = os.path.join(SERVER_DIRNAME, patient_id, start_glob)
    proc = subprocess.Popen(
        ['scp', '{}:{}'.format(SERVER_NAME, patient_start_path), DATA_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.communicate()
    patient_start_path = os.path.join(SERVER_DIRNAME, patient_id, end_glob)
    proc = subprocess.Popen(
        ['scp', '{}:{}'.format(SERVER_NAME, patient_start_path), DATA_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.communicate()


def copy_ards_patient(row):
    patient_id = row['Patient Unique Identifier']
    berlin_start = row['Date when Berlin criteria first met (m/dd/yyy)']
    try:
        dt = datetime.strptime(berlin_start, '%m/%d/%y %H:%M')
    except ValueError:
        print('Was unable to find Berlin start time for {}. skipping patient'.format(patient_id))
        return
    # should also be grabbing data from at least 2 hours prior because of the way our file rotation
    # is set up
    get_first_days_data(patient_id, dt)


def copy_non_ards_patient(row):
    # XXX Currently we do not know when patient was first intubated, so for
    # now it might be best to just gather first 24 hrs of data. But in future
    # we will need to gather first 24 exclusively
    patient_id = row['Patient Unique Identifier']
    first_file_cmd = "ls {} | head -n 1".format(os.path.join(SERVER_DIRNAME, patient_id))
    proc = subprocess.Popen(['ssh', SERVER_NAME, first_file_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    init_dt = datatime.strptime(stdout, '%Y-%m-%d-%H-%M-%S.%f')
    get_first_days_data(patient_id, init_dt)


def main():
    parser = ArgumentParser()
    parser.add_argument('--cohort-description', default='cohort-description.csv', help='Path to file describing the cohort')
    parser.add_argument('--experiment', default=1, choices=[1, 2])
    args = parser.parse_args()

    df = pd.read_csv(args.cohort_description)
    patients_to_get = df[df['Experiment Group 1 / 2'] == 'Group {}'.format(args.experiment)]
    enrollment = patients_to_get[patients_to_get['Potential Enrollment for Greg´s ARDS Project'] == 'Y']

    for idx, row in enrollment.iterrows():
        patho = row['ARDS / obstructive (COPD-asthma) / Others']
        if patho == 'ARDS':
            copy_ards_patient(row)
        else:
            copy_non_ards_patient(row)


if __name__ == "__main__":
    main()
