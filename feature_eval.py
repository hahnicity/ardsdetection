pat = re.compile('fs(?P<frame_size>\d+)_ff(?P<func>[a-z\+]+)_.*spNone_(?P<algo>[A-Z]+(:?_REG)?)')
feature_results = {}
for f in fs:
    specifics = pat.search(f).groupdict()
    if specifics['func'] in ['std', 'var']:
        continue
    if int(specifics['frame_size']) != 100:
        continue
    if specifics['algo'] != 'LOG_REG':
        continue
    tmp = pd.read_pickle(f)
    for row in tmp:
        if "-".join(row['features']) not in feature_results:
            feature_results["-".join(row['features'])] = {'aucs': [row['auc']]}
        else:
            feature_results["-".join(row['features'])]['aucs'].append(row['auc'])

for key in feature_results:
    feature_results[key]['mean_auc'] = np.mean(feature_results[key]['aucs'])
    feature_results[key]['median_auc'] = np.median(feature_results[key]['aucs'])
    feature_results[key]['auc_std'] = np.std(feature_results[key]['aucs'])

list_res = [(k, feature_results[k]['mean_auc'], feature_results[k]['median_auc'], feature_results[k]['auc_std']) for k in feature_results]
list_res_by_mean_median = sorted(list_res, key=lambda x: (x[1], x[2]))
print(list_res_by_mean_median[-5:])
