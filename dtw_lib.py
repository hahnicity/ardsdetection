from datetime import datetime
from glob import glob
import os

from dtwco.warping.core import dtw
from matplotlib.dates import date2num
import numpy as np
import pandas as pd
from ventmap.constants import OUT_DATETIME_FORMAT
from ventmap.raw_utils import read_processed_file


def _find_per_breath_dtw_score(prev_pressure_flow_waves, breath, use_pressure):
    # compare the last n_breaths to current breath to compute DTW score
    score = 0
    for pressure, flow in prev_pressure_flow_waves:
        score += dtw(flow, breath['flow'])
        if use_pressure:
            score += dtw(pressure, breath['pressure'])
    return score / len(prev_pressure_flow_waves)


def dtw_analyze(file_gen_list, n_breaths, rolling_av_len, use_pressure):
    """
    :param file_gen_list: list of generators for specific patient's vent files
    :param n_breaths: number of breaths we want to look back in the window
    :param rolling_av_len: An additional rolling average to compute on top of the stats. Can be 1 if you don't want a rolling average
    :param use_pressure: Use pressure waveform along with flow to calc DTW
    """
    pressure_flow_waves = []
    dtw_scores = [np.nan] * n_breaths
    rel_bns = []
    timestamps = []

    for generator in file_gen_list:
        for breath in generator:
            # XXX I'd like to setup system that drops breaths if vent bn is too far away.

            timestamps.append(datetime.strptime(breath['abs_bs'], OUT_DATETIME_FORMAT))
            rel_bns.append(breath['rel_bn'])
            if len(pressure_flow_waves) == (n_breaths+1):
                pressure_flow_waves.pop(0)

            if len(pressure_flow_waves) < (n_breaths):
                pressure_flow_waves.append((breath['pressure'], breath['flow']))
                continue

            dtw_scores.append(_find_per_breath_dtw_score(pressure_flow_waves, breath, use_pressure))
            pressure_flow_waves.append((breath['pressure'], breath['flow']))
            # XXX I'd like to setup system that drops breaths if vent bn is too far away.
            prev_vent_bn = breath['vent_bn']

    rolling_av = np.convolve(dtw_scores, np.ones((rolling_av_len,))/rolling_av_len, mode='valid')
    return np.append([np.nan]*(rolling_av_len-1), rolling_av), rel_bns, timestamps


def analyze_patient(patient_id, dataset_path, cohort_file, cache_dir, use_pressure):
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if not os.path.exists(os.path.join(cache_dir, patient_id)):
        os.mkdir(os.path.join(cache_dir, patient_id))

    n_breaths = 4
    rolling_len = 5
    cache_file = "{}_n{}_rolling{}_pressure{}.npy".format(patient_id, n_breaths, rolling_len, use_pressure)
    cache_file_path = os.path.join(cache_dir, patient_id, cache_file)
    if os.path.exists(cache_file_path):
        return np.load(cache_file_path)

    files = glob(os.path.join(dataset_path, 'experiment1/all_data/raw', patient_id, '*.raw.npy'))
    desc = pd.read_csv(cohort_file).drop_duplicates(subset=['Patient Unique Identifier'])

    gen_list = []
    for f in files:
        proc_file = f.replace('.raw.npy', '.processed.npy')
        gen_list.append(read_processed_file(f, proc_file))

    dtw_scores, rel_bns, timestamps = dtw_analyze(gen_list, 4, 5, use_pressure)
    patient_row = desc[desc['Patient Unique Identifier'] == patient_id].iloc[0]

    if patient_row.Pathophysiology != 'ARDS':
        start_time = patient_row.vent_start_time
    else:
        start_time = patient_row['Date when Berlin criteria first met (m/dd/yyy)']
    start_time = pd.to_datetime(start_time)

    # Translate timestamps to hour
    hrs = [(ts - start_time).total_seconds() / (60 * 60) for ts in timestamps]
    arr = np.array([hrs, dtw_scores]).T
    arr = arr[np.argsort(arr[:, 0])]
    np.save(cache_file_path, arr)
    # return an array of factional hours to their corresponding DTW scoring
    return arr
