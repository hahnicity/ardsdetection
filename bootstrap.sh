#!/bin/bash
# bootstrap the installation of a spark master on rhel/fedora

sudo yum install -y gcc-c++ gcc python python-pip python-virtualenv python-devel atlas-devel lapack-devel blas-devel libgfortran libyaml-devel

virtualenv venv
source venv/bin/activate
pip install numpy
pip install -r requirements.txt
