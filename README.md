# ARDS Early Recognition Project

#Installation
## Mac OSX
Install python

    brew install python

Install virtualenv

    sudo pip install virtualenv

Create a new virtualenvironment and activate

    virtualenv venv --python python3
    source venv/bin/activate

Install requirements

    pip install -r requirements.txt

#Running
## Locally
To run the analysis perform the installation steps for local installs then run in
python. The following command will perform a cross patient split on a certain proportion
of patients as defined by the `--test-size` flag. By default the test size is 20% of
a particular pathophysiology. So if ARDS has 10 patients then a 20% split would imply
we test on 2 patients and train on 8.

    python learn.py --metadata-processing-type rolling_average --cross-patient-split

The `--metadata-processing-type rolling_average` argument means we will calculate the breath metadata for
every single breath and then average them for a given window of time. The size of
this window is set in number of breaths by the `--stacks` argument.

There are 3 possible arguments that can be given to the `--metadata-processing-type` flag.

 * rolling_average - breaths are stacked in a window of size according to the `--stacks` flag. Each feature is then averaged with the same feature from other breaths in the window.
 * non_rolling - all breaths are placed in sequential order
 * rolling_frame - really weird, don't use this. I'm pretty sure this is experimental

## Split types
There are multiple ways to split your dataset for evaluation.

 * --simple-split : use this if you want something like an 80/20 split on your data
 * --cross-patient-split : Evaluates the first kfold split in your data in accordance to how many folds you tell the analysis to use via the `--folds` flag
 * --cross-patient-kfold : Runs through all kfolds and aggregates results. Each fold tries to have equivalent amounts of ARDS, COPD, and control patients as the other.

## With Pickle Files
Storing data in pickle files will prevent the need to completely reprocess the data
set at the beginning of analysis. You can store the current dataset into a pickle
file using the `--to-pickle` argument

    python learn.py --to-pickle dataset.pickle --metadata-processing-type rolling_average --stacks 30 --cross-patient-split

Then after the data has been saved you can retrieve that processed data using the
`-p` flag.

    python learn.py -p dataset.pickle --cross-patient-split

This will dramatically increase execution time of your analysis since initial calculation
of all breath metadata will not occur.

## Preprocessing
### PCA
We can run a PCA analysis/transform on our data

    python learn.py --pca 4 ...

Will perform a PCA transform using 4 principal components for our data.
