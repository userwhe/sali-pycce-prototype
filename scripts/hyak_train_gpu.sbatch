#!/bin/bash
#SBATCH --job-name=sali-train
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sali-train-%j.out
#SBATCH --error=logs/sali-train-%j.err

set -euo pipefail

REPO=${REPO:-/mmfs1/gscratch/scrubbed/whe3/projects/sali-pycce-prototype}
ENV_PATH=${ENV_PATH:-/mmfs1/gscratch/scrubbed/whe3/conda/envs/sali-pycce}

mkdir -p "$REPO/logs" "$REPO/checkpoints" "$REPO/runs"
cd "$REPO"

module load conda
conda activate "$ENV_PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

hostname
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY

python -m sali_pycce.train \
  --train-samples 20000 \
  --val-samples 2000 \
  --epochs 20 \
  --batch-size 128 \
  --signal-points 512 \
  --max-spins 10 \
  --device cuda \
  --out checkpoints/hyak_medium.pt \
  --threads ${SLURM_CPUS_PER_TASK:-8}

python -m sali_pycce.evaluate \
  --checkpoint checkpoints/hyak_medium.pt \
  --samples 500 \
  --batch-size 128 \
  --signal-points 512 \
  --max-spins 10 \
  --threshold 0.25 \
  --device cuda \
  --threads ${SLURM_CPUS_PER_TASK:-8}

python examples/predict_one.py \
  --checkpoint checkpoints/hyak_medium.pt \
  --signal-points 512 \
  --max-spins 10 \
  --device cuda \
  --out runs/hyak_prediction_demo.png \
  --threads ${SLURM_CPUS_PER_TASK:-8}
