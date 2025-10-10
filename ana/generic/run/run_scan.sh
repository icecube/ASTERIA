#!/bin/bash

cd /cfs/klemming/home/j/jabei
source conda.init.sh
conda activate conda-dirs/envs/my-conda-dardel/

cd /cfs/klemming/home/j/jabei/Private/project/ASTERIA/ASTERIA/docs/gen2-ana/seperate

IAMPL=$1
SAMPLES=10000

python run_scan.py $IAMPL $SAMPLES