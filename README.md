# ARDS Early Recognition Project

## Overview

This gives a general overview of the way the ARDS detection works.

![](img/ARDS-paper-Figure1.png)

**A.** Ventilator waveform data (VWD) from each subject were divided into consecutive 100-breath observation windows. Physiologic features were calculated for each breath in a window, and median values were used to represent the entire window. Each window was labeled as ARDS or non-ARDS and tagged with a subject identifier.

**B.** ARDS screening ML models were developed using a two-step process. First, we trained the model to classify all individual 100-breath windows from the training set as either ARDS or non-ARDS.

**C.** We determined patient-level model performance by attributing all breath window predictions from the validation sets to each subject and assigning the subject class as ARDS or non-ARDS using a specific threshold for the percentage of individual windows classified as ARDS in any given time bin

## Install
Use anaconda for setting up your environment

	conda create -n ards python=2.7
    conda activate ards
	conda install matplotlib seaborn
	pip install -r requirements.txt

## Dataset

For access to the dataset please contact Jason Adams `jyadams@ucdavis.edu` for dataset
access. After this you will be able to use the code described in this repo.

## Reproduction

### ATS Conference Results

Checkout the `ats-abstract` branch and run.

	git checkout ats-abstract
	python train.py -p flow-time-opt-ats.pkl --folds 5

### CCX Results

Checkout the `ccx` tag

    git checkout ccx
    python train.py --algo RF -fsm chi2 --split-type kfold --n-new-features 8

## Citing
If you used my dataset or the work herein please cite us :bowtie:

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
