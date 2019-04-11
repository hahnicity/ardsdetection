#!/bin/bash
for i in 'RF' 'NB'
do
    for j in 100 200
    do
        for k in 'mean' 'var' 'std' 'median+var' 'median+std' 'mean+var' 'mean+std'
        do
            ts python feature_grid_search.py --load-from-unframed unframed-fix-features2.pkl --algo $i -fs $j -ff $k --threads 8
        done
    done
done
