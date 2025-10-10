#!/bin/bash -l
#SBATCH --output=/cfs/klemming/home/j/jabei/Private/project/ASTERIA/ASTERIA/docs/gen2-ana/seperate/outputs/slurm_%A_%a.out
#SBATCH -A naiss2024-22-718
#SBATCH -J RECO_DIST
#SBATCH -p shared
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -a 0-719
#SBATCH --mem-per-cpu=4GB

mkdir -p /cfs/klemming/home/j/jabei/Private/project/ASTERIA/ASTERIA/docs/gen2-ana/seperate/outputs/

srun -n 1 run_reco.sh $SLURM_ARRAY_TASK_ID