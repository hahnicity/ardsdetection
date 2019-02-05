import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ehr_data', help='path to ehr data')
    parser.add_argument('codekey', help='path to codekey')
    parser.add_argument('--output-path', default='data/demographic/cohort_demographics.csv')
    args = parser.parse_args()

    ehr = pd.read_csv(args.ehr_data)
    codekey = pd.read_csv(args.codekey)
    # map of EHR:codekey patient id differences
    patient_id_diffs = {
        '0127RPI0120160121': '0127RPI0120160124',
        '0166RPI2120160227': '0166RPI2220160227'
    }
    data = []
    for pt in ehr.PATIENT_ID.unique():
        if pt in patient_id_diffs:
            pt = patient_id_diffs[pt]
        key_rows = codekey[codekey['Patient Unique Identifier'] == pt]
        if len(key_rows) > 1:
            raise Exception('Found more than one row for pt {}'.format(pt))
        elif len(key_rows) == 0:
            raise Exception('Did not find a corresponding row for pt {}'.format(pt))
        key_rows = key_rows.iloc[0]
        data.append([pt, key_rows['Weight (kg)'], key_rows['Sex'], key_rows['Height (cm)'], key_rows['Age']])

    demographics = pd.DataFrame(data, columns=['PATIENT_ID', 'WEIGHT_KG', 'SEX', 'HEIGHT_CM', 'AGE'])
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    demographics.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
