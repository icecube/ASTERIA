#!/bin/bash

OUTPUTVAL=$1

SAMPLES=100000000

for ((IDIST=0; i<296; i++))
    python run_background_trials.py $IDIST $SAMPLES 0
done