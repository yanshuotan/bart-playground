#!/bin/bash

#SBATCH --job-name=bart_schedule_linear_additive_10
#SBATCH --partition=low
#SBATCH --array=1-100
#SBATCH --output=slurm_logs/schedule_runs/linear_additive_10/%A_%a.out
#SBATCH --error=slurm_logs/schedule_runs/linear_additive_10/%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# --- Configuration ---
EXPERIMENT_SCRIPT="experiments/schedule.py"
DGP_CONFIG="dgp=linear_additive dgp_params.n_features=10"
LOG_DIR="slurm_logs/schedule_runs/linear_additive_10"

# --- Setup ---
mkdir -p "${LOG_DIR}"

echo "Starting Slurm task ${SLURM_ARRAY_TASK_ID} for job ${SLURM_JOB_ID}"
SEED=${SLURM_ARRAY_TASK_ID}

# --- Execution ---
python ${EXPERIMENT_SCRIPT} ${DGP_CONFIG} experiment_params.main_seed=${SEED} experiment_params.plot_results=False
echo "Slurm task ${SLURM_ARRAY_TASK_ID} finished." 