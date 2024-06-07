#!/bin/bash

OUTPUTVAL=$1

SAMPLES=100000

for DIST in $(seq 1 0.2 2); do

    if [[ $(echo $DIST | awk -F. '{print $2}') != 0 ]]; then
        python run_background_trials.py $DIST $SAMPLES $OUTPUTVAL
    fi
done