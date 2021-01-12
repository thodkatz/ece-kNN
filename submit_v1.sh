#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=5:00
#SBATCH --output=./hpc/slurm/tv/timesnow/v1/%j.out

srun ./bin/v1 datasets/tv/TIMESNOW.txt 10
srun ./bin/v1 datasets/tv/TIMESNOW.txt 40
srun ./bin/v1 datasets/tv/TIMESNOW.txt 70
srun ./bin/v1 datasets/tv/TIMESNOW.txt 100
