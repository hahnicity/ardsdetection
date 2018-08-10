"""
metrics
~~~~~~~
"""
import pandas as pd


def get_fns_idx(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return pos_loc[pos_loc != label].index


def get_fns(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return len(pos_loc[pos_loc != label])


def get_tns(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return len(neg_loc[neg_loc != label])


def get_fps_idx(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return neg_loc[neg_loc == label].index


def get_fps_full_rows(actual, predictions, label, filename):
    idx = get_fps_idx(actual, predictions, label)
    full_df = read_pickle(filename)
    return full_df.loc[idx]


def get_fps(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return len(neg_loc[neg_loc == label])


def get_tps(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return len(pos_loc[pos_loc == label])


def false_positive_rate(actual, predictions, label):
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 0
    else:
        return round(float(fp) / (fp + tn), 4)


def specificity(actual, predictions, label):
    """
    Also known as the true negative rate
    """
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 1
    else:
        return round(float(tn) / (tn + fp), 4)


def sensitivity(actual, predictions, label):
    """
    Also known as recall
    """
    tps = get_tps(actual, predictions, label)
    fns = get_fns(actual, predictions, label)
    if tps == 0 and fns == 0:
        return nan
    return tps / (tps+fns)


def aggregate_statistics(actual, predictions, results):
    cols = []
    tmp = []
    for i in range(0, 3):
        patho = {0: 'ctrl', 1: 'ards', 2: 'copd'}[i]
        tmp.extend([
            get_tps(actual, predictions, i),
            get_fps(actual, predictions, i),
            get_tns(actual, predictions, i),
            get_fns(actual, predictions, i),
        ])
        cols.extend([
            '{}_tps'.format(patho),
            '{}_fps'.format(patho),
            '{}_tns'.format(patho),
            '{}_fns'.format(patho),
        ])
    if isinstance(results, type(None)):
        results = pd.DataFrame([tmp], columns=cols)
    else:
        results = results.append(pd.Series(tmp, index=cols), ignore_index=True)
    return results
