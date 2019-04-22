#!/bin/bash

for algo in 'RF' 'MLP' 'SVM' 'NB' 'ADA' 'LOG_REG'
do
    for fsm in 'RFE' 'chi2' 'mutual_info' 'gini' 'lasso' 'PCA'
    do
        ts python feature_selection.py --split-type kfold -p 50-50-cohort-flow-time-fix-features2-fs100.pkl --algo ${algo} --savefig /Users/greg/${algo}-${fsm}-kfold.png -fsm ${fsm}
    done
done
