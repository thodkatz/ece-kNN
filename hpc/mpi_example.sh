#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=20
#SBATCH --nodes=1
#SBATCH --time=1:00:00

module load gcc openmpi

mpicc example.c -o example

srun ./example
