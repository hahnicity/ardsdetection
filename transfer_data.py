from argparse import ArgumentParser
from datetime import datetime, timedelta
import subprocess

import pandas as pd

SERVER_NAME = 'b2c-compute'
SERVER_DIRNAME = '/x1/data/results/backups/'
# XXX change in future to be generalizable
DATA_PATH = 'data/experiment1/training/raw'


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
    hour_glob = "{{{}}}".format(",".join(["{0:02d}".format(i) for i in range(dt.hour - 2, 24)]))
    start_glob = "*{}-{}-{}-{}-*".format(dt.year, dt.month, dt.day, hour_glob)

    one_day_later = dt + timedelta(days=1)
    hour_glob = "{{{}}}".format(",".join(["{0:02d}".format(i) for i in range(0, one_day_later.hour+1)]))
    end_glob = "*{}-{}-{}-{}-*".format(dt.year, dt.month, dt.day, hour_glob)

    patient_start_path = os.path.join(SERVER_DIRNAME, patient_id, start_glob)
    subprocess.Popen(
        ['scp', '{}:{}'.format(SERVER_NAME, patient_start_path), DATA_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def copy_non_ards_patient(row):
    # XXX Currently we do not know when patient was first intubated, so for
    # now it might be best to just gather first 24 hrs of data. But in future
    # we will need to gather first 24 exclusively
    patient_id = row['Patient Unique Identifier']
    pass


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
