#!/bin/bash

echo "Hello, the scheduling started!"

sbatch --ntasks=1 submit_openmp.sh
for ((i=2; i<=20; i+=2))
do
sbatch --ntasks=$i submit_openmp.sh
done
