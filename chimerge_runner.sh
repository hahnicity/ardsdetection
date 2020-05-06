#!/bin/bash
bins=(10 20 30 40 50 60 70 80 90 100)
strats=("quantile" "uniform" "kmeans")
folds=(0 1 2 3 4)
for bin in ${bins[@]}
do
    for strat in ${strats[@]}
    do
        for fold in ${folds[@]}
        do
            filename="${strat}-${bin}-${fold}-results.txt"
            python chi2_ks_test.py -cb $bin -s $strat chi2/chi2_fold_${fold}.csv chimerge >> chi2/results/$filename
        done
    done
done
