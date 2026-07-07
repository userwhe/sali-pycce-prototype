#!/bin/bash
#SBATCH --job-name=sali-shard-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/gscratch/scrubbed/whe3/sali/repo
#SBATCH --output=/gscratch/scrubbed/whe3/sali/logs/%x-%j.out
#SBATCH --error=/gscratch/scrubbed/whe3/sali/logs/%x-%j.err

set -euo pipefail

WORK_ROOT=/gscratch/scrubbed/whe3/sali
CHECKOUT_DIR="${WORK_ROOT}/repo"
PROJECT_DIR="${CHECKOUT_DIR}/sali"
DATASET_DIR="${WORK_ROOT}/datasets/paper-low-full"
CONDA_ENV="${WORK_ROOT}/conda-env"

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  PROJECT_DIR="${CHECKOUT_DIR}"
fi

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/datasets"
cd "${PROJECT_DIR}"

module load conda

conda run --no-capture-output --prefix "${CONDA_ENV}" python scripts/shard_jobs.py \
  --preset paper \
  --field low \
  --dataset-dir "${DATASET_DIR}" \
  --shard-size 10000 \
  --normalization-samples 20000 \
  --shard-dtype float32 \
  --raw-signal-dtype float32 \
  prepare
