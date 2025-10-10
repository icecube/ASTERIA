#!/bin/bash -l
#SBATCH --output=/cfs/klemming/home/j/jabei/Private/project/ASTERIA/ASTERIA/docs/gen2-ana/seperate/outputs/slurm_%A_%a.out
#SBATCH -A naiss2024-22-718
#SBATCH -J BKG_MSWI
#SBATCH -p shared
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -a 0-299

mkdir -p /cfs/klemming/home/j/jabei/Private/project/ASTERIA/ASTERIA/docs/gen2-ana/seperate/outputs/

srun -n 1 run_background_trials.sh $SLURM_ARRAY_TASK_ID