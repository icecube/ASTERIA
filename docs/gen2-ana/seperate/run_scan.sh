#!/bin/bash

trials=10000

for ((i=10; i<=100; i+=10))
do
    python run_scan.py $i $trials
done
