# ARDS Early Recognition Project

#Running
## Locally
To run the analysis perform the installation steps for local installs then run in
python. The following command will perform a cross patient split on a certain proportion
of patients as defined by the `--test-size` flag. By default the test size is 20% of
a particular pathophysiology. So if ARDS has 10 patients then a 20% split would imply
we test on 2 patients and train on 8.

    python train.py --cross-patient-split

## Split types
There are multiple ways to split your dataset for evaluation.

 * --cross-patient-split : Evaluates the first kfold split in your data in accordance to how many folds you tell the analysis to use via the `--folds` flag
 * --cross-patient-kfold : Runs through all kfolds and aggregates results. Each fold tries to have equivalent amounts of ARDS, COPD, and control patients as the other.

## With Pickle Files
Storing data in pickle files will prevent the need to completely reprocess the data
set at the beginning of analysis. You can store the current dataset into a pickle
file using the `--to-pickle` argument

    python train.py --to-pickle dataset.pickle

Then after the data has been saved you can retrieve that processed data using the
`-p` flag.

    python train.py -p dataset.pickle

This will dramatically increase execution time of your analysis since initial calculation
of all breath metadata will not occur.

## Preprocessing
### PCA
We can run a PCA analysis/transform on our data

    python train.py --pca 4 ...

Will perform a PCA transform using 4 principal components for our data.
