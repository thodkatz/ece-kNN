#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks=8
#SBATCH --time=1:00
#SBATCH --output=./hpc/slurm/%j.out

srun ./bin/v1 datasets/corel/ColorHistogram.asc
