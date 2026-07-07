# Hyak SALI Shard Generation

Working root:

```text
/gscratch/scrubbed/whe3/sali
```

Dataset target:

```text
/gscratch/scrubbed/whe3/sali/datasets/paper-low-full
```

The repo-standard paper split is 2,520,000 train, 540,000 validation, and 540,000 test samples. With `shard_size=10000`, the plan contains 360 shards.

## First Login Checks

```bash
ssh whe3@klone.hyak.uw.edu
hyakalloc
hyakstorage --home
hyakstorage /gscratch/scrubbed/whe3
df -h /gscratch/scrubbed/whe3
```

Choose the account and partition from `hyakalloc`. Set these variables before submitting jobs:

```bash
: "${SALI_HYAK_ACCOUNT:?Set SALI_HYAK_ACCOUNT to the exact account shown by hyakalloc}"
: "${SALI_HYAK_PARTITION:?Set SALI_HYAK_PARTITION to the exact partition shown by hyakalloc}"
```

The Slurm scripts intentionally do not hard-code account or partition.

## Environment Setup

From the local repo root, upload the code without large local run outputs:

```bash
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'runs/' \
  ./ whe3@klone.hyak.uw.edu:/gscratch/scrubbed/whe3/sali/repo/
```

On Hyak:

```bash
cd /gscratch/scrubbed/whe3/sali
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'repo[dev]'
mkdir -p logs datasets
```

If you are using the normal Hyak password + Duo login flow, you can avoid
multiple interactive SSH commands by running the bootstrap helper from your Mac:

```bash
ssh whe3@klone.hyak.uw.edu 'SALI_SUBMIT_MODE=setup bash -s' < scripts/hyak/bootstrap_from_github.sh
```

That command clones the pushed `codex/sali-reproduction` branch from GitHub,
uses the `sali/` project subdirectory in the checkout when present, prints
storage status, and prints `hyakalloc`. After choosing the account and partition
from `hyakalloc`, submit the compute-node environment setup, prepare job, and
four-shard pilot with:

```bash
ssh whe3@klone.hyak.uw.edu \
  'SALI_HYAK_ACCOUNT=<account> SALI_HYAK_PARTITION=<partition> SALI_SUBMIT_MODE=pilot bash -s' \
  < scripts/hyak/bootstrap_from_github.sh
```

The helper supports `SALI_SUBMIT_MODE=setup`, `env`, `prepare`, `pilot`, and
`full`. Use `pilot` first; `full` submits environment setup, prepare, all 360
shard-array tasks, and finalize with Slurm dependencies.

## Local Hyak Smoke Test

Use an interactive or short batch job to run:

```bash
cd /gscratch/scrubbed/whe3/sali/repo
source /gscratch/scrubbed/whe3/sali/venv/bin/activate
python scripts/shard_jobs.py --preset practical --field low \
  --dataset-dir /gscratch/scrubbed/whe3/sali/datasets/tiny-smoke \
  --train-samples 8 --val-samples 4 --test-samples 4 \
  --shard-size 4 --normalization-samples 4 prepare
for id in 0 1 2 3; do
  python scripts/shard_jobs.py --preset practical --field low \
    --dataset-dir /gscratch/scrubbed/whe3/sali/datasets/tiny-smoke \
    --train-samples 8 --val-samples 4 --test-samples 4 \
    --shard-size 4 --normalization-samples 4 write --shard-id "$id"
done
python scripts/shard_jobs.py --preset practical --field low \
  --dataset-dir /gscratch/scrubbed/whe3/sali/datasets/tiny-smoke \
  --train-samples 8 --val-samples 4 --test-samples 4 \
  --shard-size 4 --normalization-samples 4 finalize
```

## Pilot Array

Submit the full prepare job first:

```bash
: "${SALI_HYAK_ACCOUNT:?Set SALI_HYAK_ACCOUNT first}"
: "${SALI_HYAK_PARTITION:?Set SALI_HYAK_PARTITION first}"
sbatch -A "$SALI_HYAK_ACCOUNT" -p "$SALI_HYAK_PARTITION" scripts/hyak/prepare_shards.slurm
```

After it succeeds, run a four-shard pilot by overriding the array:

```bash
: "${SALI_HYAK_ACCOUNT:?Set SALI_HYAK_ACCOUNT first}"
: "${SALI_HYAK_PARTITION:?Set SALI_HYAK_PARTITION first}"
sbatch -A "$SALI_HYAK_ACCOUNT" -p "$SALI_HYAK_PARTITION" --array=0-3%2 scripts/hyak/generate_shards_array.slurm
```

Inspect logs and generated file sizes:

```bash
ls -lh /gscratch/scrubbed/whe3/sali/datasets/paper-low-full | head
du -h --max-depth 1 /gscratch/scrubbed/whe3/sali/datasets/paper-low-full
```

## Full Array

After the pilot succeeds:

```bash
: "${SALI_HYAK_ACCOUNT:?Set SALI_HYAK_ACCOUNT first}"
: "${SALI_HYAK_PARTITION:?Set SALI_HYAK_PARTITION first}"
sbatch -A "$SALI_HYAK_ACCOUNT" -p "$SALI_HYAK_PARTITION" scripts/hyak/generate_shards_array.slurm
```

When all 360 array tasks complete:

```bash
: "${SALI_HYAK_ACCOUNT:?Set SALI_HYAK_ACCOUNT first}"
: "${SALI_HYAK_PARTITION:?Set SALI_HYAK_PARTITION first}"
sbatch -A "$SALI_HYAK_ACCOUNT" -p "$SALI_HYAK_PARTITION" scripts/hyak/finalize_shards.slurm
```

## Transfer

After `manifest.json` exists and validates, transfer the dataset off Hyak. Prefer a method that avoids storing Google Drive credentials on Hyak. One option is to sync to a local machine that has Google Drive mounted:

```bash
rsync -az --partial --progress \
  whe3@klone.hyak.uw.edu:/gscratch/scrubbed/whe3/sali/datasets/paper-low-full/ \
  "/Users/weitao/Library/CloudStorage/GoogleDrive-heweutao@gmail.com/My Drive/02_Research/DQP/sali/datasets/paper-low-full/"
```
