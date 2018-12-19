import csv
import os

import numpy as np
import pandas as pd


def main():
    cohort = pd.read_csv("experiment5-cohort-description.csv")
    if not os.path.exists('experiment5-with-hour.pkl'):
        df = pd.read_pickle('experiment5.pkl')
        df = df[df.set_type == 'train']
        df['hour'] = np.nan
        for _, row in cohort.iterrows():
            patient = row['Patient Unique Identifier']
            ards_start = pd.to_datetime(row['Date when Berlin criteria first met (m/dd/yyy)'])
            pt = df[df.patient == patient]
            if len(pt) == 0:
                print('no data patient: {}'.format(patient))
                continue

            for hour in range(24):
                low_delta = pd.Timedelta(hours=hour)
                high_delta = pd.Timedelta(hours=hour+1)
                hour_rows = pt[(pt.abs_time_at_BS >= ards_start+low_delta) & (pt.abs_time_at_BS < ards_start+high_delta)]
                if len(hour_rows) == 0:
                    continue
                df.loc[hour_rows.index, 'hour'] = hour
        df.to_pickle('experiment5-with-hour.pkl')
    else:
        df = pd.read_pickle('experiment5-with-hour.pkl')

    analytics = [['patient', 'hour', 'n_breaths', 'mean_tvi']]
    for patient in sorted(df.patient.unique()):
        pt = df[df.patient == patient]
        pt = pt[~pt.hour.isna()]
        for hour in pt.hour.unique():
            hr = pt[pt.hour == hour]
            analytics.append([patient, hour, len(hr), hr.tvi.mean()])

    with_data = set(df.patient.unique())
    no_data = set(cohort['Patient Unique Identifier'].values).difference(with_data)
    for patient in sorted(no_data):
        analytics.append([patient, np.nan, 0, np.nan])

    with open('experiment5_analytics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(analytics)


if __name__ == "__main__":
    main()
