#!/usr/bin/env bash

# Define datasets as an array (index 0-based)
DATASETS=("Magic" "Adult" "Mushroom" "Shuttle" "Wine" "Heart" "Iris")  # Add more as needed

# Get the current array index
INDEX=$PBS_ARRAY_INDEX

# Logging
LOGFILE="sh_log/run_${INDEX}_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -ia "$LOGFILE")
exec 2> >(tee -ia "$LOGFILE" >&2)

source ~/.bashrc
conda activate ~/bartpg
# Run for this dataset
python compare.py "${DATASETS[$INDEX]}"
conda deactivate
