#!/bin/bash

N_TRAIN_LIST=(100 200 500 1000 2000 5000 10000)
BATCH_LIST=$(seq 1 5)
MODEL_LIST=("rf" "dnn")

# Slurm parameters
MEMO=1G                             # Memory required (5 GB)
TIME=00-00:05:00                    # Time required (20 m)

ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

mkdir -p "results/"
mkdir -p "logs/"

for BATCH in $BATCH_LIST; do
  for N_TRAIN in "${N_TRAIN_LIST[@]}"; do
    for MODEL in "${MODEL_LIST[@]}"; do
      
      JOBN="n"$N_TRAIN"_batch"$BATCH"_"$MODEL
      OUTF="logs/"$JOBN".out"
      ERRF="logs/"$JOBN".err"

      OUT_FILE="results/"$JOBN".txt"
      COMPLETE=0
      #ls $OUT_FILE
      if [[ -f $OUT_FILE ]]; then
        COMPLETE=1
      fi
      
      if [[ $COMPLETE -eq 0 ]]; then
        SCRIPT="experiment.sh $N_TRAIN $BATCH $MODEL"
        ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
        $ORD
      fi

    done
  done
done
