#!/bin/bash

samples=10000

for dist in {1..60}
do
    python run_background_trials.py $dist $samples
done
