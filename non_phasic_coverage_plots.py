"""
non_phasic_coverage_plots
~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from cohort_tools.non_phasic_analysis import perform_patient_hour_mapping
from cohort_tools.quality_check import find_hourly_coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase_file')
    parser.add_argument('data_frame')
    args = parser.parse_args()

    df = pd.read_pickle(args.data_frame)
    df = df.rename(columns={'abs_time_at_BS': 'abs_bs'})
    df['breath_time'] = df['iTime'] + df['eTime']
    phases = pd.read_csv(args.phase_file)

    hour_idxs = perform_patient_hour_mapping(df, phases, 'no')
    coverage = find_hourly_coverage(df, hour_idxs)

    max_square_plot = 25
    for idx, patient in enumerate(coverage.keys()):
        plt.suptitle('Coverage Reports')
        plt.subplot(5, 5, (idx % max_square_plot)+1)

        frac = coverage[patient]['frac_coverage']
        vals = frac.values()
        if len(frac.keys()) < 24:
            for _ in range(sorted(frac.keys())[-1] + 1, 24):
                vals.append(0)

        plt.bar(range(24), vals)
        plt.title(patient, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        if ((idx + 1) % max_square_plot) == 0:
            plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    main()
