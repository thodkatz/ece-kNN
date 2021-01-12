#!/bin/bash

echo "Hello, the scheduling started!"

sbatch --ntasks=1 submit_v0.sh

for ((i=4; i<=20; i+=4))
do
sbatch --ntasks=$i submit_v1.sh
done

for ((i=4; i<=20; i+=4))
do
sbatch --ntasks=$i submit_v2.sh
done
