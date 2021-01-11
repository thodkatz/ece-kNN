#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --time=1:00

mpic++ example.c -o example
srun ./example
