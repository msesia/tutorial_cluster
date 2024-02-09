#!/bin/bash

N_TRAIN_LIST=(100 200 500 1000 2000 5000 10000)
BATCH_LIST=$(seq 1 5)

for BATCH in $BATCH_LIST; do
  for N_TRAIN in "${N_TRAIN_LIST[@]}"; do

    ./experiment.sh $N_TRAIN $BATCH
done
done
