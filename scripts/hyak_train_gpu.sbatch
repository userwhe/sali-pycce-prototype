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
  --preset colab-medium \
  --epochs 20 \
  --batch-size 128 \
  --device cuda \
  --out checkpoints/hyak_medium.pt \
  --history-out runs/hyak_medium_history.csv \
  --threads ${SLURM_CPUS_PER_TASK:-8}

python -m sali_pycce.evaluate \
  --checkpoint checkpoints/hyak_medium.pt \
  --samples 500 \
  --batch-size 128 \
  --threshold 0.25 \
  --device cuda \
  --metrics-out runs/hyak_medium_metrics.json \
  --threads ${SLURM_CPUS_PER_TASK:-8}

python examples/predict_one.py \
  --checkpoint checkpoints/hyak_medium.pt \
  --device cuda \
  --out runs/hyak_prediction_demo.png \
  --heatmap-comparison-out runs/hyak_heatmap_comparison.png \
  --threads ${SLURM_CPUS_PER_TASK:-8}
