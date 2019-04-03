"""
non_phasic_coverage_plots
~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import argparse
from copy import copy
import math

import pandas as pd
import matplotlib.pyplot as plt

from cohort_tools.non_phasic_analysis import perform_patient_hour_mapping
from cohort_tools.quality_check import find_hourly_coverage


def plot_patient(idx, patient, coverage, patho, hours):
    max_square_plot = 16
    sqrt_max = math.sqrt(max_square_plot)
    plt.subplot(sqrt_max, sqrt_max, (idx % max_square_plot)+1)

    frac = coverage[patient]['frac_coverage']
    vals = frac.values()
    if len(frac.keys()) < hours:
        for _ in range(sorted(frac.keys())[-1] + 1, hours):
            vals.append(0)

    plt.bar(range(hours), vals)
    plt.title(patient, fontsize=8, pad=.5)
    plt.xticks([])
    plt.ylim((0, 1))
    plt.yticks([])
    plt.suptitle("{} {} Hour Coverage Reports".format(patho, hours))
    if ((idx + 1) % max_square_plot) == 0:
        plt.show()


def analyze_coverage(coverage, ards_patients, other_patients, hours):
    one_hr = 60 * 60
    ards_hours_covered = []
    for patient in ards_patients:
        # calculate total hours of coverage for a patient.
        seconds_covered = sum(coverage[patient]['seconds_covered'].values())
        ards_hours_covered.append(seconds_covered / float(one_hr))
    plt.hist(ards_hours_covered, bins=hours)
    plt.title('ARDS Coverage in {} Hours'.format(hours))
    plt.xticks(range(hours), range(hours))
    plt.xlabel('number hours data')
    plt.ylabel('# patients in bin')
    plt.show()

    other_hours_covered = []
    for patient in other_patients:
        # calculate total hours of coverage for a patient.
        seconds_covered = sum(coverage[patient]['seconds_covered'].values())
        other_hours_covered.append(seconds_covered / float(one_hr))
    plt.hist(other_hours_covered, bins=hours)
    plt.title('OTHER Coverage in {} Hours'.format(hours))
    plt.xticks(range(hours), range(hours))
    plt.xlabel('number hours data')
    plt.ylabel('# patients in bin')
    plt.show()

    # Generate Hourly Coverage Reports

    for idx, patient in enumerate(sorted(ards_patients)):
        plot_patient(idx, patient, coverage, 'ARDS', hours)
    plt.show()

    for idx, patient in enumerate(sorted(other_patients)):
        plot_patient(idx, patient, coverage, 'OTHER', hours)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-file', default='cohort-description.csv')
    parser.add_argument('data_frame')
    args = parser.parse_args()

    df = pd.read_pickle(args.data_frame)
    df = df.rename(columns={'abs_time_at_BS': 'abs_bs'})
    df['breath_time'] = df['iTime'] + df['eTime']
    cohort = pd.read_csv(args.cohort_file)
    cohort = cohort[((cohort.experiment_group == 1)) & (cohort['Potential Enrollment'] == 'Y')]
    # make phases file from cohort file
    ards_patients = df[df.y == 1].patient.unique()
    other_patients = df[(df.y == 0) | (df.y == 2)].patient.unique()

    ards_vent_starts = cohort.loc[cohort['Patient Unique Identifier'].isin(ards_patients), ['Patient Unique Identifier', 'Date when Berlin criteria first met (m/dd/yyy)']]
    ards_vent_starts = ards_vent_starts.rename(columns={'Date when Berlin criteria first met (m/dd/yyy)': 'vent_start_time'})
    other_vent_starts = cohort.loc[cohort['Patient Unique Identifier'].isin(other_patients), ['Patient Unique Identifier', 'vent_start_time']]

    phases = pd.concat([ards_vent_starts, other_vent_starts])
    phases = phases.rename(columns={'vent_start_time': 'vent_start', 'Patient Unique Identifier': 'patient'})
    phases['vent_end'] = pd.to_datetime(phases.vent_start) + pd.Timedelta(hours=24)

    hour_idxs = perform_patient_hour_mapping(df, phases, 'no')
    coverage = find_hourly_coverage(df, hour_idxs)

    # analyze 24 hr coverage
    analyze_coverage(coverage, ards_patients, other_patients, 24)

    # analyze 6 hr coverage
    six_hr_coverage = copy(coverage)
    for patient in coverage:
        for key in coverage[patient]:
            for hour in range(6, 24):
                try:
                    del six_hr_coverage[patient][key][hour]
                except:
                    pass
    analyze_coverage(six_hr_coverage, ards_patients, other_patients, 6)



if __name__ == "__main__":
    main()
