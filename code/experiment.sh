#!/bin/bash

module purge
eval "$(conda shell.bash hook)"
conda activate default

export OPENBLAS_NUM_THREADS=1

python experiment.py $1 $2 $3
