# SALI Sharded Dataset Training Design

## Purpose

Move paper-scale SALI training from repeated on-the-fly sample generation to a two-phase pipeline:

1. generate the dataset once into durable compressed shards
2. train from those shards without generating signals or heatmaps in the hot path

The current streamed run confirmed that the A100 is underutilized because CPU DataLoader workers spend most of their time generating synthetic signals and rendering target heatmaps. This design preserves the current dense heatmap training target and heatmap loss while shifting that work ahead of training.

## Scope

This change adds a sharded data mode beside the existing materialized and streamed modes.

In scope:

- deterministic train/validation/test shard generation
- compressed shard files containing signals, target heatmaps, and nuclei metadata
- a manifest that records config, split sizes, shard layout, dtypes, normalization stats, and compatibility metadata
- a PyTorch dataset that reads the generated shards for training and validation
- a training path that preserves checkpoint/resume, JSON/CSV metric logging, periodic threshold calibration, and sample plots
- CLI commands for staged shard generation and sharded training

Out of scope for the first revision:

- changing the model target representation
- changing the heatmap loss semantics
- distributed multi-GPU training
- full dataset upload/download to the local Mac
- replacing diagnostics and post-processing APIs

## Dataset Format

The first implementation will use compressed NumPy `.npz` shard files. This avoids adding a new storage dependency and keeps the format easy to inspect from Colab and local tests.

Each shard stores a contiguous range of deterministic sample indices for one split:

- `signals`: normalized signal tensor with shape `(N, 2, signal_points)`
- `raw_signals`: raw signal tensor with shape `(N, 2, signal_points)` for diagnostics and reconstruction plots
- `heatmaps`: dense target heatmaps with shape `(N, 1, output_height, output_width)`
- `nuclei_count`: number of nuclei per sample, shape `(N,)`
- `nuclei_az_khz`: padded nucleus Az values, shape `(N, max_nuclei)`
- `nuclei_aperp_khz`: padded nucleus Aperp values, shape `(N, max_nuclei)`
- `indices`: absolute split-local sample indices, shape `(N,)`

Default dtypes:

- `signals`: `float16`
- `raw_signals`: `float16`
- `heatmaps`: `float16`
- `nuclei_count`: `uint8`
- `nuclei_az_khz`: `float32`
- `nuclei_aperp_khz`: `float32`
- `indices`: `int64`

Training will cast `signals` and `heatmaps` to `float32` tensors before model and loss computation. This preserves the current loss implementation while reducing shard storage and I/O.

## Manifest

Shard generation writes `manifest.json` under the dataset directory.

The manifest includes:

- schema version
- data mode: `sharded`
- field and full run config snapshot
- config hash for compatibility checks
- split sizes
- shard size
- shard list with split, path, start index, stop index, and sample count
- stored dtypes
- normalization stats
- generation start and end timestamps

Sharded training must load the manifest before training. It must reject a manifest whose shape-relevant config fields differ from the active run config. This prevents accidentally training on shards generated with a different field, output grid, signal length, split seed, or coupling ranges.

## Generation Flow

Shard generation runs independently from model training.

For each split:

1. estimate or load training normalization stats
2. generate deterministic samples by split and index
3. render the current dense target heatmap once
4. normalize signals
5. write one compressed shard per configured shard size
6. update the manifest only after a shard file is successfully written

The generation command must be resumable. If a shard file already exists and matches the expected sample count and index range, generation skips it. This lets long Colab generation jobs resume after runtime loss without discarding completed shards.

For paper scale, shards should be generated into Google Drive for durability, then copied in chunks to Colab local SSD for training when possible. Training directly from Drive is supported but not preferred because Drive random reads and decompression can become a new bottleneck.

## Training Flow

The sharded training path keeps the current batch contract:

`(signal32, signal256, target_heatmap)`

The sharded dataset reads shard files and yields tensors without calling:

- `generate_sample_signals()`
- `render_heatmap()`
- `generate_indexed_sample()`

Training behavior remains aligned with `train_streamed_model()`:

- checkpoint `checkpoints/latest.pt` every epoch
- optional archived epoch checkpoints every `N` epochs
- `best_model.pt` as a plain state dict
- `history.json`
- `history.csv`
- `run_state.json`
- resume from `latest` or a specific checkpoint
- optional periodic threshold calibration on a bounded validation sample view
- optional periodic sample plots
- capped final test evaluation unless `--full-test-eval` is passed

Per-epoch shuffling will shuffle shard order and sample order within shards using a deterministic epoch seed. Validation and test iteration remain stable.

## Avoiding Unnecessary Training Work

During model training, the hot path should only:

- load compressed arrays from shard files
- cast signal and heatmap arrays to torch tensors
- move tensors to the selected device
- run forward/backward/loss

The hot path should not:

- generate couplings
- simulate signals
- normalize by recomputing statistics
- render heatmaps
- materialize diagnostic samples unless a periodic diagnostic hook is due
- run full test evaluation during training

Diagnostic and plotting helpers may reconstruct `Sample` objects from shard contents, but only for bounded sample counts.

## CLI

`scripts/train_colab.py` will gain:

- `--data-mode sharded`
- `--dataset-dir PATH`
- `--shard-size N`
- `--generate-shards`
- `--shard-dtype float16|float32`
- `--raw-signal-dtype float16|float32`
- `--cache-dataset-dir PATH`
- `--skip-existing-shards`

Typical staged workflow:

1. generate 100k shards
2. train 100k from shards and compare GPU utilization
3. generate 500k shards
4. train 500k from shards and compare epoch time/metrics
5. generate full paper-scale shards
6. launch full training from shards

Example Colab paths:

- durable dataset root: `/content/drive/MyDrive/02_Research/DQP/sali/datasets`
- durable run root: `/content/drive/MyDrive/02_Research/DQP/sali/runs`
- optional local cache: `/content/sali-dataset-cache`

## Expected Storage

For the full `3.6M` sample paper split:

- heatmaps as `float16`: roughly `153 GB` before compression
- raw signals as `float16`: roughly `14.4 GB`
- normalized signals as `float16`: roughly `14.4 GB`
- raw plus normalized signals as `float16`: roughly `28.8 GB`
- nuclei metadata: comparatively small

Because dense heatmaps are mostly zeros, compressed `.npz` shards should reduce storage substantially. If storing both `signals` and `raw_signals` proves too large, the first reduction is to store `raw_signals` only for validation/test or for a bounded diagnostics subset. The default first revision will store both so existing diagnostics remain straightforward and comparable.

## Error Handling

Generation must fail clearly when:

- split sizes are invalid
- shard size is invalid
- the dataset directory is not writable
- Drive is requested but not mounted
- normalization statistics are non-finite
- an existing shard has the wrong shape, dtype, or index range

Training must fail clearly when:

- `--data-mode sharded` is used without `--dataset-dir`
- `manifest.json` is missing
- the manifest does not match the active run config
- required shard files are missing
- a shard contains non-finite signals or heatmaps
- a resume checkpoint is incompatible with the active manifest/config

## Tests

Tests will cover:

- tiny shard generation writes expected files and manifest
- generated shard tensors have expected shapes and dtypes
- sharded samples match deterministic streamed samples for the same split/index
- sharded training does not call `generate_indexed_sample()` or `render_heatmap()`
- sharded training writes `latest.pt`, archived checkpoint, `best_model.pt`, JSON history, and CSV history
- sharded resume continues at the next epoch
- CLI help exposes sharded dataset flags
- manifest compatibility checks reject shape-relevant config mismatches
- skip-existing shard generation does not overwrite valid completed shards

## Acceptance Criteria

The implementation is ready for a new Colab benchmark when:

- all local tests pass
- a tiny local generate-and-train sharded smoke run completes
- sharded training avoids data generation and heatmap rendering in the training loop
- the 100k sharded Colab run starts from pre-generated shards
- A100 GPU utilization and epoch time are compared against the stopped streamed run
- outputs remain durable under the user's Google Drive SALI folder
