#!/bin/bash

OUTPUTVAL=$1

SAMPLES=100000

for DIST in {1..2}
do
    python run_background_trials.py $DIST $SAMPLES $OUTPUTVAL
done
