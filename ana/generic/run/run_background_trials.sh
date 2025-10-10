#!/bin/bash

cd /cfs/klemming/home/j/jabei
source miniconda.init.sh
conda activate conda-dirs/envs/my-asteria/

cd /cfs/klemming/home/j/jabei/Private/project/ASTERIA/ASTERIA/docs/gen2-ana/seperate

IDIST=$1
SAMPLES=100000000

python run_background_trials.py $IDIST $SAMPLES