"""
find where vwd collection started/stopped for patients.
"""
import argparse
from datetime import datetime
import re
from subprocess import PIPE, Popen

import pandas as pd

remote = 'b2c-compute'
user = 'grehm'
backups_dir = '/x1/data/results/backups'
old_date_pat = r'(\d{4}-\d{2}-\d{2}__\d{2}:\d{2})'
old_strp_fmt = '%Y-%m-%d__%H:%M'
cur_date_pat = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})'
cur_strp_fmt = '%Y-%m-%d-%H-%M'
final_fmt = '%Y-%m-%d %H:%M'


def main():
    # get a data frame of processed data, get patients, then go to remote
    # server and get start/end times. not too tough.
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    df = pd.read_pickle(args.dataset)
    patients = df.patient.unique()
    start_end_times = []
    for pt in patients:
        proc = Popen(['ssh', '{}@{}'.format(user, remote), 'ls {}/{}'.format(backups_dir, pt)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        if stderr:
            raise Exception(stderr)
        tmp = sorted(stdout.split())
        files = []
        for f in tmp:
            match = re.search(cur_date_pat, str(f))
            if match:
                files.append(str(f))
            elif re.search(old_date_pat, str(f)):
                files.append(str(f))
        # XXX will probably need to go thru final file to get most accurate end
        try:
            dt_start = re.search(cur_date_pat, str(files[0])).groups()[0]
            start = datetime.strptime(dt_start, cur_strp_fmt)
            dt_end = re.search(cur_date_pat, str(files[-1])).groups()[0]
            end = datetime.strptime(dt_end, cur_strp_fmt)
        except Exception as err:
            dt_start = re.search(old_date_pat, str(files[0])).groups()[0]
            start = datetime.strptime(dt_start, old_strp_fmt)
            dt_end = re.search(old_date_pat, str(files[-1])).groups()[0]
            end = datetime.strptime(dt_end, old_strp_fmt)
        row = [pt]
        for t in [start, end]:
            row.append(t.strftime(final_fmt))
        start_end_times.append(row)
    df = pd.DataFrame(start_end_times, columns=['patient', 'vwd_start_time', 'vwd_estimated_end'])
    df.to_csv('cohort-vwd-start-times.csv')


if __name__ == "__main__":
    main()
