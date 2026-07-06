# Hyak Parallel Shard Generation Design

## Goal

Generate the repo-standard SALI paper dataset shards on Hyak, validate the workflow first on this local machine, validate the full generated dataset on Hyak scratch storage, and then transfer the validated dataset to Google Drive for durable storage.

The target dataset is the existing paper split:

- train: 2,520,000 samples
- validation: 540,000 samples
- test: 540,000 samples
- total: 3,600,000 samples

With the current default `shard_size=10000`, this produces 360 shard files:

- 252 train shards
- 54 validation shards
- 54 test shards

The Hyak working root is:

```text
/gscratch/scrubbed/whe3/sali
```

This location is active scratch storage, not long-term storage. The workflow must copy validated outputs to Google Drive after generation.

## Constraints

Hyak compute must run through Slurm jobs, not on login nodes. The workflow should use a Slurm array so shard generation can run in parallel across compute nodes.

Hyak `$HOME` is too small for this workflow. Code, virtual environment, shard outputs, logs, and transfer bundles should live under `/gscratch/scrubbed/whe3/sali`.

The current repo generator, `generate_shards()`, is deterministic and compatible with parallelization at the shard level, but it is not itself safe to call concurrently because each process writes the same `manifest.json`. The parallel workflow must avoid concurrent writes to shared manifest files.

Google Drive should not be written by every Slurm array task. Parallel generation should write only to Hyak scratch. A separate transfer step should run after validation succeeds.

## Architecture

The workflow has five stages.

1. Local smoke validation

Run a tiny parallel-generation test locally before touching Hyak. This verifies shard plan creation, per-shard writing, manifest finalization, and compatibility with existing sharded training loaders.

2. Hyak environment setup

Create a working directory under `/gscratch/scrubbed/whe3/sali`, upload or clone the repo, create a Python environment there, and install the repo dependencies. Keep generated data, logs, and job scripts under this root.

3. Hyak setup job

Create a single source of truth for the full paper split:

- config snapshot
- config hash
- normalization statistics
- shard plan
- expected shard paths, split names, start indices, stop indices, and sample counts

This setup job writes only small metadata files. It does not generate the 3.6M samples.

4. Slurm array shard generation

Submit one array job over the shard plan. Each task receives one shard id, reads the shared setup metadata, generates exactly one shard, and writes it to a unique path.

Each task must write to a temporary file first, then atomically rename it into the final `.npz` path only after the shard is complete. This prevents partial files from being mistaken for valid shards.

The array job must not write `manifest.json`.

5. Finalization and transfer

After the array completes, run a finalizer job that validates every shard against the plan, checks array shapes and indices, and writes the final `manifest.json` in the format expected by the existing sharded training path.

Only after validation succeeds should the dataset be transferred to Google Drive. The preferred transfer shape is either resumable sync of the dataset directory or tar bundles grouped by split/chunk. The transfer should be a separate step from generation.

## Components

### Shard Plan

The shard plan is a metadata file listing all shards to generate. Each row contains:

- shard id
- split
- shard index within split
- start sample index
- stop sample index
- sample count
- output filename

The plan is deterministic for a given config and shard size.

### Per-Shard Generator

The per-shard generator reuses the existing deterministic sample generation logic. It should generate sample indices `[start, stop)` for a single split and write arrays compatible with the current `.npz` shard format:

- `signals`
- `raw_signals`
- `heatmaps`
- `nuclei_count`
- `nuclei_az_khz`
- `nuclei_aperp_khz`
- `indices`

It should fail fast if the output shard already exists but is invalid. It may skip a shard only when validation confirms the existing file is complete and matches the expected shape and indices.

### Final Manifest Writer

The finalizer builds the single authoritative `manifest.json`. It must include the same compatibility fields used by current `load_shard_manifest()`, including:

- schema version
- data mode
- config hash
- config snapshot
- split sizes
- shard size
- dtypes
- normalization stats
- shard list
- generation timestamps

Existing training code should then load the dataset through `load_shard_manifest()` without special Hyak-specific behavior.

### Slurm Scripts

The Slurm scripts are thin wrappers. They should set paths, activate the Python environment, and call repo Python entrypoints. Resource settings should be easy to edit for the available Hyak account and partition.

The first production array should be conservative: run a small subset of shard ids to measure runtime, memory, and file size before launching all 360 shards.

## Data Flow

```text
repo config
  -> setup metadata and shard plan
  -> Slurm array per-shard `.npz` files
  -> final validation
  -> manifest.json
  -> optional local Hyak training check
  -> Google Drive transfer
```

The dataset itself remains deterministic. A regenerated shard for the same config, split, and index range should contain the same examples.

## Validation

Local validation should cover:

- tiny split plan generation
- one or more per-shard writes
- final manifest creation
- `load_shard_manifest()` compatibility
- sample loading from generated shards
- a tiny sharded training smoke test

Hyak validation should cover:

- setup metadata exists and matches the intended paper split
- every planned shard file exists
- every shard has the expected arrays, shapes, dtypes, and `indices`
- final `manifest.json` loads through existing repo code
- total split counts match `2_520_000 / 540_000 / 540_000`
- optional short training read test against the generated dataset

## Error Handling And Resume

The workflow should be resumable at shard granularity.

If an array task fails, rerunning the same shard id should either skip a valid existing shard or replace an invalid/incomplete temporary output. Failed tasks should not require regenerating completed shards.

The finalizer should refuse to write `manifest.json` if any shard is missing, malformed, or generated from a mismatched config.

Temporary files should be clearly named, for example:

```text
train-000123.npz.tmp-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
```

Only final `.npz` files should be considered part of the dataset.

## Transfer Plan

After final validation, transfer the dataset from `/gscratch/scrubbed/whe3/sali` to Google Drive.

The transfer step should be separate from shard generation. Acceptable transfer methods include:

- `rsync` from Hyak to a local machine with Google Drive mounted
- `rclone` from Hyak to Google Drive if authentication is configured safely on Hyak
- tar bundles copied off Hyak, then uploaded to Drive

The first implementation should prefer the method that requires the least credential exposure on Hyak. If direct Drive credentials would need to live on Hyak, prefer transferring through the local machine instead.

## Open Operational Decisions

Before full production launch, choose:

- Hyak account and partition names from `hyakalloc`
- CPU and memory request per shard task
- array concurrency limit
- whether to store shards as `float32` or use `float16` for storage reduction
- Google Drive transfer method

The default data format should remain repo-compatible `float32` unless storage measurements from a pilot run make that impractical.

## Success Criteria

- A tiny local parallel-generation smoke test passes.
- A small Hyak pilot array generates and validates a subset of shards.
- The full Hyak array generates all 360 paper-split shard files under `/gscratch/scrubbed/whe3/sali`.
- The final manifest loads with existing `load_shard_manifest()` code.
- The generated dataset can be used by existing sharded training code.
- The validated dataset is transferred to Google Drive.
