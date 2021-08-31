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

## Reproduction

### ATS Conference Results

Checkout the `ats-abstract` branch and run.

	git checkout ats-abstract
	python train.py -p flow-time-opt-ats.pkl --folds 5

### CCX Results

Checkout the `ccx` tag

    git checkout ccx
    python train.py --algo RF -fsm chi2 --split-type kfold --n-new-features 8
