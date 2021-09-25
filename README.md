# ARDS Early Recognition Project

## Install
Use anaconda for setting up your environment

	conda create -n ards python=2.7
	conda install matplotlib seaborn
	pip install -r requirements.txt

## Reproduction

### ATS Conference Results

Checkout the `ats-abstract` branch and run.

	git checkout ats-abstract
	python train.py -p flow-time-opt-ats.pkl --folds 5

### CCX Results

    git checkout ccx
    python train.py -p ccx_dataset.pkl --n-new-features 8 -fsm chi2 --algo RF
