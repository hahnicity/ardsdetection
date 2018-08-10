import csv
from glob import glob
import os

from numpy import append, array, empty, inf, nan
from pandas import DataFrame, read_csv, Series

necessities = ['ventBN']
flow_time_feature_set = necessities + [
    'mean_flow_from_pef', 'inst_RR', 'minF_to_zero', 'pef_+0.16_to_zero',
    'iTime', 'eTime', 'I:E ratio', 'dyn_compliance',
]
broad_feature_set = flow_time_feature_set + [
    'TVi', 'TVe', 'Maw', 'ipAUC', 'PIP', 'PEEP', 'epAUC',
]


def process_features(df, feature_set):
    df['min_vent'] = df['TVi'] * df['inst_RR']
    df = df[feature_set]
    df = df.replace([inf, -inf], nan).dropna()
    return df


def get_cohort_files(cohort):
    cohorts = ["ardscohort", "controlcohort", "copdcohort", "ardscopdcohort"]
    if cohort not in cohorts:
        raise Exception("Input must either be one of {}".format(cohorts))
    dirs = os.listdir(cohort)
    cohort_files = []
    for dir in dirs:
        files = glob("{}/{}/0*_breath_meta*.csv".format(cohort, dir))
        for f in files:
            cohort_files.append(f)

    return cohort_files


def collate_from_breath_meta_to_list(cohort):
    """
    Gets all breath_meta.csv files in our specific cohort and then gets all
    the data from these files and stores them in a list.
    """
    cohort_files = get_cohort_files(cohort)
    data = []
    for f in cohort_files:
        with open(f) as meta:
            reader = csv.reader(meta)
            for line in reader:
                data.append(line)
    return data


def df_to_rolling_average(df, breaths_to_stack):
    low = 0
    stacks = []
    for i in range(breaths_to_stack, len(df), breaths_to_stack):
        stats = []
        stack = df[low:i]
        low = i
        vent_bn = stack.iloc[0]['ventBN']
        stats.append(vent_bn)
        del stack['ventBN']
        for col_idx in stack.columns:
            stats.append(stack[col_idx].median())
            #stats.append(stack[col_idx].var())
        stacks.append(stats)
    df = DataFrame(stacks, index=list(range(0, len(stacks))))
    df = df.rename(columns={0: "ventBN"})
    return df


def collate_non_rolling(cohort, _, samples, feature_set):
    cohort_files = get_cohort_files(cohort)
    df = process_features(read_csv(cohort_files[0]), feature_set)
    file_array = [cohort_files[0]] * len(df)
    for f in cohort_files[1:]:
        if samples and len(df) >= samples:
            break
        new = process_features(read_csv(f), feature_set)
        df = df.append(new)
        file_array.extend([f] * len(new))
    df['filename'] = file_array
    return df


def collate_rolling_average(cohort, breaths_to_stack, samples, feature_set):
    cohort_files = get_cohort_files(cohort)
    df = df_to_rolling_average(process_features(read_csv(cohort_files[0]), feature_set), breaths_to_stack)
    file_array = [cohort_files[0]] * len(df)
    for f in cohort_files[1:]:
        if samples and len(df) >= samples:
            break
        new = df_to_rolling_average(process_features(read_csv(f), feature_set), breaths_to_stack)
        df = df.append(new)
        file_array.extend([f] * len(new))
    df['filename'] = file_array
    return df


def collate_from_breath_meta_to_data_frame(cohort, breaths_to_stack, samples, feature_set):
    cohort_files = get_cohort_files(cohort)
    df = process_features(read_csv(cohort_files[0]), feature_set)
    initial_features = list(df.columns.values)
    rolling = create_rolling_frame(df, breaths_to_stack, feature_set)
    file_array = [cohort_files[0]] * len(rolling)
    for f in cohort_files[1:]:
        if samples and len(rolling) >= samples:
            break
        new = process_features(read_csv(f), feature_set)
        if len(new.index) == 0:
            continue
        new = create_rolling_frame(new, breaths_to_stack, feature_set)
        rolling = append(rolling, new, axis=0)
        file_array.extend([f] * len(new))
    df = DataFrame(rolling)
    df['filename'] = file_array
    df = df.rename(columns={(len(initial_features) - 1) * breaths_to_stack: 'ventBN'})
    return df


def create_rolling_frame(df, breaths_in_frame, feature_set):
    """
    What we do for rolling frame is that we take each breath, and associate it with
    a certain window. The window length is defined by the breaths_in_frame param.
    each window contains each variable of every single breath in sequential order.
    For example if my window size was 2 and my vars for each breath were TVi and TVe
    then my rolling frame would look like

    row1: TVi_1, TVe_1, TVi_2, TVe_2
    row2: TVi_3, TVe_3, TVi_4, TVe_4
    ...

    A rolling frame is a weird construct: I really am dubious about whether
    or not it was a good idea but that is besides the point.
    """
    matrix = df.as_matrix()
    # The +1 is for the start bn
    rolling = empty((0, ((len(matrix[0]) - 1) * breaths_in_frame) + 1), float)
    # The [1:] means cut off the vent bn
    row = matrix[0][1:]
    start_bn = 0
    for i, _ in enumerate(df.index[:-1]):
        # Ensure we can attach initial vent bn without interfering with our model
        if start_bn == 0:
            start_bn = int(matrix[i][0])
        if (i + 1) % breaths_in_frame == 0:
            row = append(row, [start_bn])
            rolling = append(rolling, [row], axis=0)
            row = array([])
            start_bn = 0
        # The [1:] means cut off the vent bn
        row = append(row, matrix[i + 1][1:])
    return rolling



def collate_all(breaths_to_stack, samples, func, feature_set):
    copd = func("copdcohort", breaths_to_stack, samples, feature_set)
    copd['y'] = Series(2, index=copd.index)
    ards = func("ardscohort", breaths_to_stack, samples, feature_set)
    ards['y'] = Series(1, index=ards.index)
    control = func("controlcohort", breaths_to_stack, samples, feature_set)
    control['y'] = Series(0, index=control.index)
    return ards.append([control, copd], ignore_index=True)



def collate_all_to_rolling_frame(breaths_to_stack, samples, feature_set):
    return collate_all(
        breaths_to_stack, samples, collate_from_breath_meta_to_data_frame, feature_set
    )


def collate_all_to_non_rolling_frame(samples, feature_set):
    return collate_all(None, samples, collate_non_rolling, feature_set)


def collate_all_to_rolling_average(breaths_to_stack, samples, feature_set):
    return collate_all(breaths_to_stack, samples, collate_rolling_average, feature_set)
