#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=20:00
#SBATCH --output=./hpc/slurm/tv/timesnow/%j.out

./bin/v0 datasets/tv/TIMESNOW.txt 10
./bin/v0 datasets/tv/TIMESNOW.txt 40
./bin/v0 datasets/tv/TIMESNOW.txt 70
./bin/v0 datasets/tv/TIMESNOW.txt 100
