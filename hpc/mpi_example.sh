#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks=8
#SBATCH --time=3:00
#SBATCH --output=./hpc/slurm/corel/histo/v2/%j.out

srun ./bin/v2 datasets/corel/ColorHistogram.asc 10
srun ./bin/v2 datasets/corel/ColorHistogram.asc 40
srun ./bin/v2 datasets/corel/ColorHistogram.asc 70
srun ./bin/v2 datasets/corel/ColorHistogram.asc 100
