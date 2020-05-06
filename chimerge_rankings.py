from glob import glob
import os

import pandas as pd


def parse_file(filename):
    txt = open(filename).read()
    res = []
    for l in txt.split('\n'):
        if not l or l.startswith('+') or l.split('|')[1].strip() == 'feature':
            continue
        feature = l.split('|')[1].strip()
        imp = float(l.split('|')[2].strip())
        res.append([feature, imp])
    return res


files = glob('chi2/results/*.txt')
results = {}
for file in files:
    basename = os.path.basename(file)
    spl = basename.split('-')
    strat = spl[0]
    bins = spl[1]
    fold = spl[2]
    res = parse_file(file)
    results[file] = {
        "strat": strat,
        "bins": int(bins),
        "fold": int(fold),
        "results": res,
    }

rankings = []
for file, v in results.items():
    feat_order = [f[0] for f in v['results']]
    for i, f in enumerate(feat_order):
        row = [f, i+1, v['bins'], v['strat'], v['fold']]
        rankings.append(row)

rankings = pd.DataFrame(rankings, columns=['feature', 'f_rank', 'bins', 'strat', 'fold'])
for i, group in rankings.groupby(by=['fold', 'f_rank']):
    print('fold {}, rank {} top 2:'.format(i[0], i[1]))
    print(group.feature.value_counts().head(2))
    print('')
import IPython; IPython.embed()
