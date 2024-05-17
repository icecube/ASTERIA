#!/bin/bash

samples=100000

for dist in {1..2}
do
    python run_background_trials.py $dist $samples
done
