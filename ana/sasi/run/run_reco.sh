#!/bin/bash

cd /cfs/klemming/home/j/jabei
source conda.init.sh
conda activate conda-dirs/envs/my-conda-dardel/

cd /cfs/klemming/home/j/jabei/Private/project/ASTERIA/ASTERIA/docs/gen2-ana/seperate

ARRAY_ID=$1
IAMPL=$((ARRAY_ID / 60))
IDIST=$((ARRAY_ID % 60))
SAMPLES=100000

python run_reco.py $IAMPL $IDIST $SAMPLES