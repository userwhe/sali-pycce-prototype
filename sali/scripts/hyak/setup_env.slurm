#!/bin/bash
#SBATCH --job-name=sali-setup-env
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
REPO_DIR="${WORK_ROOT}/repo"
VENV="${WORK_ROOT}/venv"

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/datasets"
cd "${REPO_DIR}"

python3 -m venv "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "${REPO_DIR}[dev]"
python scripts/shard_jobs.py --help >/dev/null
