#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=30:00
#SBATCH --output=./hpc/slurm/features/v2/cblas/%j.out

srun ./bin/v2_temp datasets/features.csv 10
srun ./bin/v2_temp datasets/features.csv 40
srun ./bin/v2_temp datasets/features.csv 70
srun ./bin/v2_temp datasets/features.csv 100
