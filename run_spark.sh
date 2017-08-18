#!/bin/bash
export SPARK_HOME=/home/ec2-user/spark
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
export PYSPARK_PYTHON=/home/ec2-user/ecs251-final-project/venv/bin/python

connect_str="spark://`lsof | grep TCP | grep 7077 | head -n1 | awk '{print $9}'`"
source venv/bin/activate
python learn.py --with-spark --connect-str $connect_str
