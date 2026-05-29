#!/bin/bash
#SBATCH --job-name=sali-smoke
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sali-smoke-%j.out
#SBATCH --error=logs/sali-smoke-%j.err

set -euo pipefail

REPO=${REPO:-/mmfs1/gscratch/scrubbed/whe3/projects/sali-pycce-prototype}
ENV_PATH=${ENV_PATH:-/mmfs1/gscratch/scrubbed/whe3/conda/envs/sali-pycce}

mkdir -p "$REPO/logs" "$REPO/checkpoints" "$REPO/runs"
cd "$REPO"

module load conda
conda activate "$ENV_PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

hostname
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY

pytest -q

python -m sali_pycce.train \
  --train-samples 256 \
  --val-samples 64 \
  --epochs 1 \
  --batch-size 16 \
  --signal-points 128 \
  --max-spins 3 \
  --device cuda \
  --out checkpoints/hyak_smoke.pt \
  --threads ${SLURM_CPUS_PER_TASK:-4}

python -m sali_pycce.evaluate \
  --checkpoint checkpoints/hyak_smoke.pt \
  --samples 32 \
  --batch-size 16 \
  --signal-points 128 \
  --max-spins 3 \
  --device cuda \
  --threads ${SLURM_CPUS_PER_TASK:-4}
