# ARDS Early Recognition Project

## Overview

This gives a general overview of the way the ARDS detection works.

![](img/ARDS-paper-Figure1.png)

**A.** Ventilator waveform data (VWD) from each subject were divided into consecutive 100-breath observation windows. Physiologic features were calculated for each breath in a window, and median values were used to represent the entire window. Each window was labeled as ARDS or non-ARDS and tagged with a subject identifier.

**B.** ARDS screening ML models were developed using a two-step process. First, we trained the model to classify all individual 100-breath windows from the training set as either ARDS or non-ARDS.

**C.** We determined patient-level model performance by attributing all breath window predictions from the validation sets to each subject and assigning the subject class as ARDS or non-ARDS using a specific threshold for the percentage of individual windows classified as ARDS in any given time bin

## Install
Use anaconda for setting up your environment

    conda env create -f environment.yml
    conda activate ards

    # install ventparliament
    cd ..
    git clone https://github.com/hahnicity/ventparliament.git
    cd ventparliament
    conda env update --file environment.yml --name ards
    pip install -e .

    # go back to ardsdetection code
    cd ../ardsdetection

## Dataset

For access to the dataset please contact Jason Adams `jyadams@ucdavis.edu` for dataset
access. After this you will be able to use the code described in this repo.

## Reproduction

### Process Your Dataset

First you need to process the dataset into a pickled format. If you are using
our dataset from Jason Adams then you need to specify pathing to the dataset and
pathing to the cohort description.

Steps:

 1. Unzip the dataset. Remember the final path of the dataset.
```
7z x ardsdetection_dataset_anonymized.7z -o/dataset/base/path/
```
 2. Move to the code path for this repository
```
cd /path/to/ardsdetection
```
 3. Run dataset processing
```
python train.py -dp /dataset/base/path/ardsdetection_anon --cohort-description /dataset/base/path/ardsdetection_anon/anon-desc.csv --to-pickle processed_dataset.pkl
```

### Running

#### Standard Trials

A basic run will look something like

    python train.py -p processed_dataset.pkl

There are a number of options for running the algorithms with a number of different options. For example, you can change the number of features and the feature selection method

    python train.py -p processed_dataset.pkl -fsm chi2 --n-new-features 7

You can also change the algorithm

    python train.py -p processed_dataset.pkl --algo MLP

Another option is to redo the dataset with different feature sets. For instance, if you want to just have static compliance as your only feature:

```
python train.py -dp /dataset/base/path/ardsdetection_anon --cohort-description /dataset/base/path/ardsdetection_anon/anon-desc.csv --to-pickle processed_dataset.pkl --feature-set stat_compliance
```

Another option is to change the split type of your data to a random bootstrap

    python train.py -p processed_dataset.pkl --split-type bootstrap

#### ATS Conference Results

Checkout the `ats-abstract` branch and run. This code only runs with Python2.7.

	git checkout ats-abstract
	python train.py -p flow-time-opt-ats.pkl --folds 5

#### CCX Results

Checkout the `ccx` tag

    git checkout ccx
    python train.py --from-pickle ccx_dataset.pkl --algo RF -fsm chi2 --split-type kfold --n-new-features 8

## Citing
If you used our dataset or the work herein please cite us :bowtie:

```
@article{rehm2021use,
  title={Use of Machine Learning to Screen for Acute Respiratory Distress Syndrome Using Raw Ventilator Waveform Data},
  author={Rehm, Gregory B and Cort{\'e}s-Puch, Irene and Kuhn, Brooks T and Nguyen, Jimmy and Fazio, Sarina A and Johnson, Michael A and Anderson, Nicholas R and Chuah, Chen-Nee and Adams, Jason Y},
  journal={Critical care explorations},
  volume={3},
  number={1},
  year={2021},
  publisher={Wolters Kluwer Health}
}
```
