#!/usr/bin/env bash
set -euo pipefail

cd /root/bart-playground/diagnosis/fixed100_testpoints

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bartts

export PYTHONPATH=/root/bart-playground:${PYTHONPATH:-}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

STORE_DIR="store_fixed100_short10k_tmax100_geom10_3runs_CLEAN_SEED3024"

for ds in abalone concrete friedman; do
  echo "========== START ${ds} =========="
  date

  python -u run_fixed100_tmax100_search_v3.py \
    --datasets "${ds}" \
    --n-runs 3 \
    --n-chains 4 \
    --n-jobs 2 \
    --short-ndpost 10000 \
    --short-nskip 0 \
    --skip-long \
    --store-dir "${STORE_DIR}" \
    --base-chain-seed 3024 \
    --ladder-tmax 100 \
    --ladder-init-size 10 \
    --ladder-max-temperatures 0 \
    --ladder-search-points 1000 \
    --ladder-max-rounds 10 \
    --ladder-repeats 3 \
    --ladder-ndpost 500 \
    --ladder-nskip 500 \
    2>&1 | tee "logs/short10k_geom10_${ds}_3runs_njobs2_seed3024_clean_$(date +%Y%m%d_%H%M).log"

  echo "========== DONE ${ds} =========="
  date
done
