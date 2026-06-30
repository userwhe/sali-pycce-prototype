# Running SALI-PyCCE on UW Hyak

This guide assumes your UW NetID is `whe3` and you are working on Hyak Klone.

## 0. What you must fill in

You must fill in your valid Slurm account/allocation.

Run on Hyak:

```bash
hyakalloc
```

Pick an account/partition combination that can run GPU jobs. Do **not** assume your NetID `whe3` is the Slurm account; an invalid account/partition combination will fail at submission time.

The scripts in this repo do **not** hard-code `#SBATCH --account=...`. Instead, submit with:

```bash
sbatch -A <YOUR_ACCOUNT> scripts/hyak_smoke_gpu.sbatch
sbatch -A <YOUR_ACCOUNT> scripts/hyak_train_gpu.sbatch
```

If you have a specific GPU partition, you may also override partition/GPU flags on the command line.

## 1. Login

```bash
ssh whe3@klone.hyak.uw.edu
```

Do not run training on the login node. Use it only for cloning, editing, installing lightweight packages, and submitting Slurm jobs.

## 2. Prepare workspace

```bash
mkdir -p /mmfs1/gscratch/scrubbed/whe3/projects
cd /mmfs1/gscratch/scrubbed/whe3/projects

git clone https://github.com/userwhe/sali-pycce-prototype.git
cd sali-pycce-prototype
```

If the repo is private, use SSH or a GitHub token.

## 3. Create conda environment

Recommended location:

```bash
mkdir -p /mmfs1/gscratch/scrubbed/whe3/conda/envs
module load conda
conda create -p /mmfs1/gscratch/scrubbed/whe3/conda/envs/sali-pycce python=3.11 -y
conda activate /mmfs1/gscratch/scrubbed/whe3/conda/envs/sali-pycce
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For the current prototype, PyCCE is optional because the default training backend uses the analytic simulator. To try PyCCE:

```bash
pip install -e ".[pycce,dev]"
```

## 4. Quick CPU sanity check on login node

This should be light and short:

```bash
python - <<'PY'
import torch
import sali_pycce
print('torch', torch.__version__)
print('cuda visible now?', torch.cuda.is_available())
print('package import ok')
PY

pytest -q
```

Do not run real training here.

## 5. Check available GPUs

```bash
sinfo -s
sinfo -p ckpt-all -O nodehost,cpusstate,freemem,gres,gresused -S nodehost | grep -v null
```

## 6. Optional interactive GPU smoke test

Replace `<YOUR_ACCOUNT>`:

```bash
salloc -A <YOUR_ACCOUNT> -p ckpt-all --gpus-per-node=2080ti:1 --mem=16G --time=00:30:00
cd /mmfs1/gscratch/scrubbed/whe3/projects/sali-pycce-prototype
module load conda
conda activate /mmfs1/gscratch/scrubbed/whe3/conda/envs/sali-pycce
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m sali_pycce.train --preset smoke --epochs 1 --batch-size 16 --device cuda --out checkpoints/interactive_smoke.pt --history-out runs/interactive_smoke_history.csv
exit
```

## 7. Submit GPU smoke job

```bash
mkdir -p logs checkpoints runs
sbatch -A <YOUR_ACCOUNT> scripts/hyak_smoke_gpu.sbatch
squeue -u whe3
```

Check logs:

```bash
ls -lh logs/
tail -f logs/sali-smoke-<JOBID>.out
```

## 8. Submit medium training job

```bash
sbatch -A <YOUR_ACCOUNT> scripts/hyak_train_gpu.sbatch
squeue -u whe3
```

Outputs go to:

```text
checkpoints/hyak_medium.pt
logs/sali-train-<JOBID>.out
logs/sali-train-<JOBID>.err
```

## 9. Evaluate after training

```bash
module load conda
conda activate /mmfs1/gscratch/scrubbed/whe3/conda/envs/sali-pycce

python -m sali_pycce.evaluate \
  --checkpoint checkpoints/hyak_medium.pt \
  --samples 500 \
  --threshold 0.25 \
  --device cuda \
  --metrics-out runs/hyak_medium_metrics.json
```

Run evaluation inside a GPU allocation or submit a separate Slurm job if it is not small.

## 10. What still needs physics-specific filling

The current code is runnable, but serious research use still needs these choices:

1. Real PyCCE bath construction in `src/sali_pycce/pycce_backend.py`.
2. Exact central-spin basis and NV electronic states.
3. Magnetic-field value and direction.
4. Hyperfine tensor convention and units.
5. Pulse sequence convention: CPMG timing, whether `tau` is half-spacing, pulse phases, and finite pulse effects.
6. Noise/readout model: shot noise, contrast, decoherence, pulse errors.
7. Dataset size and parameter range matching your experiment.

Start with the analytic backend first to validate the ML pipeline, then replace the simulator with PyCCE-generated traces.
