# SALI Sharded Dataset Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-faithful sharded dataset mode that pre-generates SALI signals and dense heatmaps once, then trains from shards without signal simulation or heatmap rendering in the training hot path.

**Architecture:** Add a focused `sali.shards` module for shard manifests, generation, validation, sample reconstruction, and a shard-backed PyTorch dataset. Reuse the existing model, heatmap renderer, loss, postprocessing, diagnostics, checkpoint/resume, and Colab run script. `scripts/train_colab.py` becomes the orchestration layer for `materialized`, `stream`, and `sharded` modes.

**Tech Stack:** Python 3.10+, NumPy compressed `.npz`, PyTorch `IterableDataset`/`DataLoader`, existing SALI config/data/train/diagnostics modules, pytest, JSON manifests.

---

## Paper-Faithfulness Guardrails

Keep these unchanged from the paper and current reproduction:

- paper-scale split: `2_520_000` train, `540_000` validation, `540_000` test
- nuclei count: `1..20`
- `A^z`: `[-100, 100] kHz`
- positive `A^\perp`: `[2, 102] kHz`
- two inputs: `N=32` and `N=256` CPMG signals
- `1000` points per signal
- `T2 = 200 us`, `1000` simulated measurements, shot noise, and decoherence through the existing physics generator
- target heatmaps: dense `5 x 5` Gaussian-like regions rendered by `render_heatmap()`
- training target/loss contract: `(signal32, signal256, target_heatmap)` and MSE/weighted variants over all pixels
- postprocessing/evaluation: existing erosion, dilation, thresholding, connected components, IoU, precision/recall, coupling MAE, and signal-reconstruction MAE

Default shard dtypes must be `float32` for `signals`, `raw_signals`, and `heatmaps`. `float16` can exist as an explicit option, but not as the paper-faithful default.

## File Structure

- Create `src/sali/shards.py`: manifest dataclasses/helpers, config hashing, shard generation, shard validation, shard-backed iterable dataset, bounded sample reconstruction.
- Modify `src/sali/train.py`: add `train_sharded_model()` and reuse existing checkpoint/history machinery.
- Modify `scripts/train_colab.py`: add sharded CLI flags, shard generation command path, sharded training/evaluation path, and diagnostics/sample-plot helpers using shard sample views.
- Create `tests/test_shards.py`: shard generation, manifest, dtype/shape, determinism, paper-faithful defaults, no-generation training hot-path guard.
- Modify `tests/test_train.py`: sharded checkpoint/resume tests.
- Modify `tests/test_train_colab_script.py`: CLI flag and default-mode tests.

---

### Task 1: Add Shard Generation Tests

**Files:**
- Create: `tests/test_shards.py`
- Implementation target: `src/sali/shards.py`

- [ ] **Step 1: Write failing tests for shard generation, manifest, dtypes, and deterministic equality**

Add this file:

```python
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sali.data import estimate_normalization_stats, generate_indexed_sample
from sali.shards import (
    DEFAULT_SHARD_DTYPE,
    ShardedSaliDataset,
    generate_shards,
    load_shard_manifest,
    materialize_sharded_samples,
)


def test_generate_shards_writes_manifest_and_float32_arrays(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"

    manifest = generate_shards(
        tiny_config,
        dataset_dir,
        shard_size=3,
        normalization_samples=4,
    )

    manifest_path = dataset_dir / "manifest.json"
    assert manifest_path.exists()
    assert manifest.schema_version == 1
    assert manifest.data_mode == "sharded"
    assert manifest.dtypes["signals"] == "float32"
    assert manifest.dtypes["raw_signals"] == "float32"
    assert manifest.dtypes["heatmaps"] == "float32"
    assert DEFAULT_SHARD_DTYPE == np.float32
    assert len(manifest.shards) == 7

    first_train = next(shard for shard in manifest.shards if shard.split == "train")
    arrays = np.load(dataset_dir / first_train.path)
    assert arrays["signals"].dtype == np.float32
    assert arrays["raw_signals"].dtype == np.float32
    assert arrays["heatmaps"].dtype == np.float32
    assert arrays["signals"].shape == (3, 2, tiny_config.physics.signal_points)
    assert arrays["heatmaps"].shape == (
        3,
        1,
        tiny_config.model.output_height,
        tiny_config.model.output_width,
    )
    assert arrays["nuclei_count"].dtype == np.uint8
    assert arrays["indices"].tolist() == [0, 1, 2]

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw_manifest["normalization_stats"]["epsilon"] == pytest.approx(tiny_config.data.norm_epsilon)


def test_sharded_samples_match_streamed_indexed_samples(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest = generate_shards(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    stats = manifest.normalization_stats

    samples = materialize_sharded_samples(tiny_config, dataset_dir, "train", max_samples=3)

    for index, sample in enumerate(samples):
        expected = generate_indexed_sample(tiny_config, "train", index, stats)
        np.testing.assert_array_equal(sample.signals, expected.signals)
        np.testing.assert_array_equal(sample.raw_signals, expected.raw_signals)
        np.testing.assert_array_equal(sample.heatmap, expected.heatmap)
        np.testing.assert_array_equal(sample.nuclei.az_khz, expected.nuclei.az_khz)
        np.testing.assert_array_equal(sample.nuclei.aperp_khz, expected.nuclei.aperp_khz)


def test_sharded_dataset_returns_torch_ready_tensors(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    generate_shards(tiny_config, dataset_dir, shard_size=5, normalization_samples=4)

    dataset = ShardedSaliDataset(tiny_config, dataset_dir, "train")
    signal32, signal256, heatmap = next(iter(dataset))

    assert len(dataset) == tiny_config.data.train_samples
    assert tuple(signal32.shape) == (1, tiny_config.physics.signal_points)
    assert tuple(signal256.shape) == (1, tiny_config.physics.signal_points)
    assert tuple(heatmap.shape) == (1, tiny_config.model.output_height, tiny_config.model.output_width)
    assert signal32.dtype == torch.float32
    assert signal256.dtype == torch.float32
    assert heatmap.dtype == torch.float32


def test_generate_shards_skips_existing_valid_shards(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = generate_shards(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    first_path = dataset_dir / first.shards[0].path
    first_mtime = first_path.stat().st_mtime_ns

    second = generate_shards(
        tiny_config,
        dataset_dir,
        shard_size=4,
        normalization_samples=4,
        skip_existing=True,
    )

    assert second.shards[0].path == first.shards[0].path
    assert first_path.stat().st_mtime_ns == first_mtime


def test_load_manifest_rejects_shape_relevant_config_mismatch(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    generate_shards(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)

    changed = tiny_config
    changed.model.output_height = 206

    with pytest.raises(ValueError, match="manifest does not match"):
        load_shard_manifest(dataset_dir, changed)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_shards.py -q`

Expected: import failure for `sali.shards`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_shards.py
git commit -m "test: cover SALI shard generation"
```

---

### Task 2: Implement Shard Manifest and Generation

**Files:**
- Create: `src/sali/shards.py`
- Test: `tests/test_shards.py`

- [ ] **Step 1: Implement manifest dataclasses, dtype parsing, config hashing, and JSON round trip**

Create `src/sali/shards.py` with these public interfaces:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from sali.config import RunConfig
from sali.data import NormalizationStats, Sample, _split_count, generate_indexed_sample
from sali.physics import Couplings


SCHEMA_VERSION = 1
DEFAULT_SHARD_DTYPE = np.float32
SUPPORTED_FLOAT_DTYPES = {"float32": np.float32, "float16": np.float16}


@dataclass(slots=True)
class ShardInfo:
    split: str
    path: str
    start: int
    stop: int
    count: int


@dataclass(slots=True)
class ShardManifest:
    schema_version: int
    data_mode: str
    config_hash: str
    config: dict[str, object]
    split_sizes: dict[str, int]
    shard_size: int
    dtypes: dict[str, str]
    normalization_stats: NormalizationStats
    shards: list[ShardInfo]
    generated_at: str
    completed_at: str | None


def dtype_from_name(name: str) -> np.dtype:
    if name not in SUPPORTED_FLOAT_DTYPES:
        raise ValueError("dtype must be one of: float32, float16")
    return np.dtype(SUPPORTED_FLOAT_DTYPES[name])


def _config_snapshot(cfg: RunConfig) -> dict[str, object]:
    snapshot = asdict(cfg)
    snapshot["output_dir"] = str(cfg.output_dir)
    return snapshot


def _shape_relevant_config(cfg: RunConfig) -> dict[str, object]:
    return {
        "physics": asdict(cfg.physics),
        "data": asdict(cfg.data),
        "model": asdict(cfg.model),
    }


def config_hash(cfg: RunConfig) -> str:
    payload = json.dumps(_shape_relevant_config(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_to_json(manifest: ShardManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload["normalization_stats"] = asdict(manifest.normalization_stats)
    payload["shards"] = [asdict(shard) for shard in manifest.shards]
    return payload


def _manifest_from_json(payload: dict[str, object]) -> ShardManifest:
    stats_payload = dict(payload["normalization_stats"])
    return ShardManifest(
        schema_version=int(payload["schema_version"]),
        data_mode=str(payload["data_mode"]),
        config_hash=str(payload["config_hash"]),
        config=dict(payload["config"]),
        split_sizes={str(key): int(value) for key, value in dict(payload["split_sizes"]).items()},
        shard_size=int(payload["shard_size"]),
        dtypes={str(key): str(value) for key, value in dict(payload["dtypes"]).items()},
        normalization_stats=NormalizationStats(
            mean=float(stats_payload["mean"]),
            var=float(stats_payload["var"]),
            epsilon=float(stats_payload["epsilon"]),
        ),
        shards=[ShardInfo(**dict(item)) for item in list(payload["shards"])],
        generated_at=str(payload["generated_at"]),
        completed_at=None if payload.get("completed_at") is None else str(payload["completed_at"]),
    )
```

- [ ] **Step 2: Implement shard array creation and validation**

Add:

```python
def _split_names() -> tuple[str, str, str]:
    return ("train", "val", "test")


def _split_sizes(cfg: RunConfig) -> dict[str, int]:
    return {split: _split_count(cfg, split) for split in _split_names()}


def _shard_filename(split: str, shard_index: int) -> str:
    return f"{split}-{shard_index:06d}.npz"


def _validate_shard_arrays(path: Path, expected_count: int, start: int, stop: int, cfg: RunConfig) -> bool:
    if not path.exists():
        return False
    try:
        arrays = np.load(path)
        return (
            arrays["signals"].shape == (expected_count, 2, cfg.physics.signal_points)
            and arrays["raw_signals"].shape == (expected_count, 2, cfg.physics.signal_points)
            and arrays["heatmaps"].shape == (expected_count, 1, cfg.model.output_height, cfg.model.output_width)
            and arrays["nuclei_count"].shape == (expected_count,)
            and arrays["nuclei_az_khz"].shape == (expected_count, cfg.data.max_nuclei)
            and arrays["nuclei_aperp_khz"].shape == (expected_count, cfg.data.max_nuclei)
            and arrays["indices"].tolist() == list(range(start, stop))
        )
    except Exception:
        return False


def _write_shard(
    cfg: RunConfig,
    split: str,
    start: int,
    stop: int,
    stats: NormalizationStats,
    path: Path,
    *,
    shard_dtype: np.dtype,
    raw_signal_dtype: np.dtype,
) -> None:
    count = stop - start
    signals = np.empty((count, 2, cfg.physics.signal_points), dtype=shard_dtype)
    raw_signals = np.empty((count, 2, cfg.physics.signal_points), dtype=raw_signal_dtype)
    heatmaps = np.empty((count, 1, cfg.model.output_height, cfg.model.output_width), dtype=shard_dtype)
    nuclei_count = np.empty((count,), dtype=np.uint8)
    nuclei_az = np.zeros((count, cfg.data.max_nuclei), dtype=np.float32)
    nuclei_aperp = np.zeros((count, cfg.data.max_nuclei), dtype=np.float32)
    indices = np.arange(start, stop, dtype=np.int64)

    for row, index in enumerate(range(start, stop)):
        sample = generate_indexed_sample(cfg, split, index, stats)
        count_i = sample.nuclei.count
        signals[row] = sample.signals.astype(shard_dtype, copy=False)
        raw_signals[row] = sample.raw_signals.astype(raw_signal_dtype, copy=False)
        heatmaps[row] = sample.heatmap.astype(shard_dtype, copy=False)
        nuclei_count[row] = count_i
        nuclei_az[row, :count_i] = sample.nuclei.az_khz
        nuclei_aperp[row, :count_i] = sample.nuclei.aperp_khz

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signals=signals,
        raw_signals=raw_signals,
        heatmaps=heatmaps,
        nuclei_count=nuclei_count,
        nuclei_az_khz=nuclei_az,
        nuclei_aperp_khz=nuclei_aperp,
        indices=indices,
    )
```

- [ ] **Step 3: Implement `generate_shards()` and `load_shard_manifest()`**

Add:

```python
def generate_shards(
    cfg: RunConfig,
    dataset_dir: Path,
    *,
    shard_size: int,
    normalization_samples: int,
    shard_dtype: str = "float32",
    raw_signal_dtype: str = "float32",
    skip_existing: bool = True,
) -> ShardManifest:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    signal_dtype = dtype_from_name(shard_dtype)
    raw_dtype = dtype_from_name(raw_signal_dtype)
    from sali.data import estimate_normalization_stats

    stats = estimate_normalization_stats(cfg, sample_limit=normalization_samples)
    if not np.isfinite(stats.mean) or not np.isfinite(stats.var):
        raise FloatingPointError("normalization statistics are non-finite")

    now = datetime.now(timezone.utc).isoformat()
    manifest = ShardManifest(
        schema_version=SCHEMA_VERSION,
        data_mode="sharded",
        config_hash=config_hash(cfg),
        config=_config_snapshot(cfg),
        split_sizes=_split_sizes(cfg),
        shard_size=int(shard_size),
        dtypes={
            "signals": str(signal_dtype.name),
            "raw_signals": str(raw_dtype.name),
            "heatmaps": str(signal_dtype.name),
            "nuclei_count": "uint8",
            "nuclei_az_khz": "float32",
            "nuclei_aperp_khz": "float32",
            "indices": "int64",
        },
        normalization_stats=stats,
        shards=[],
        generated_at=now,
        completed_at=None,
    )
    for split in _split_names():
        split_count = _split_count(cfg, split)
        for shard_index, start in enumerate(range(0, split_count, shard_size)):
            stop = min(start + shard_size, split_count)
            path = dataset_dir / _shard_filename(split, shard_index)
            count = stop - start
            if not (skip_existing and _validate_shard_arrays(path, count, start, stop, cfg)):
                _write_shard(
                    cfg,
                    split,
                    start,
                    stop,
                    stats,
                    path,
                    shard_dtype=signal_dtype,
                    raw_signal_dtype=raw_dtype,
                )
            manifest.shards.append(
                ShardInfo(split=split, path=path.name, start=start, stop=stop, count=count)
            )
            (dataset_dir / "manifest.json").write_text(
                json.dumps(_manifest_to_json(manifest), indent=2, default=str),
                encoding="utf-8",
            )
    manifest.completed_at = datetime.now(timezone.utc).isoformat()
    (dataset_dir / "manifest.json").write_text(
        json.dumps(_manifest_to_json(manifest), indent=2, default=str),
        encoding="utf-8",
    )
    return manifest


def load_shard_manifest(dataset_dir: Path, cfg: RunConfig | None = None) -> ShardManifest:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing shard manifest: {path}")
    manifest = _manifest_from_json(json.loads(path.read_text(encoding="utf-8")))
    if manifest.schema_version != SCHEMA_VERSION or manifest.data_mode != "sharded":
        raise ValueError("unsupported shard manifest")
    if cfg is not None and manifest.config_hash != config_hash(cfg):
        raise ValueError("manifest does not match active run config")
    for shard in manifest.shards:
        shard_path = dataset_dir / shard.path
        if cfg is None:
            if not shard_path.exists():
                raise FileNotFoundError(f"missing shard file: {shard_path}")
        elif not _validate_shard_arrays(shard_path, shard.count, shard.start, shard.stop, cfg):
            raise ValueError(f"invalid shard file: {shard_path}")
    return manifest
```

- [ ] **Step 4: Run shard tests**

Run: `.venv/bin/python -m pytest tests/test_shards.py -q`

Expected: shard generation tests pass or fail only because dataset iteration/sample reconstruction is still missing.

- [ ] **Step 5: Commit manifest/generation implementation**

```bash
git add src/sali/shards.py tests/test_shards.py
git commit -m "feat: generate SALI dataset shards"
```

---

### Task 3: Implement Shard Dataset and Sample Reconstruction

**Files:**
- Modify: `src/sali/shards.py`
- Test: `tests/test_shards.py`

- [ ] **Step 1: Add shard sample reconstruction helpers**

Implement in `src/sali/shards.py`:

```python
def _sample_from_arrays(arrays: np.lib.npyio.NpzFile, row: int, split: str, stats: NormalizationStats) -> Sample:
    count = int(arrays["nuclei_count"][row])
    nuclei = Couplings(
        az_khz=np.asarray(arrays["nuclei_az_khz"][row, :count], dtype=np.float32),
        aperp_khz=np.asarray(arrays["nuclei_aperp_khz"][row, :count], dtype=np.float32),
    )
    return Sample(
        signals=np.asarray(arrays["signals"][row], dtype=np.float32),
        raw_signals=np.asarray(arrays["raw_signals"][row], dtype=np.float32),
        heatmap=np.asarray(arrays["heatmaps"][row], dtype=np.float32),
        nuclei=nuclei,
        split=split,
    )


def iter_sharded_samples(
    cfg: RunConfig,
    dataset_dir: Path,
    split: str,
    *,
    max_samples: int | None = None,
) -> Iterator[Sample]:
    manifest = load_shard_manifest(dataset_dir, cfg)
    emitted = 0
    for shard in manifest.shards:
        if shard.split != split:
            continue
        arrays = np.load(dataset_dir / shard.path)
        for row in range(shard.count):
            if max_samples is not None and emitted >= max_samples:
                return
            yield _sample_from_arrays(arrays, row, split, manifest.normalization_stats)
            emitted += 1


def materialize_sharded_samples(
    cfg: RunConfig,
    dataset_dir: Path,
    split: str,
    *,
    max_samples: int,
) -> list[Sample]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    return list(iter_sharded_samples(cfg, dataset_dir, split, max_samples=max_samples))
```

- [ ] **Step 2: Implement `ShardedSaliDataset`**

Add:

```python
class ShardedSaliDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        cfg: RunConfig,
        dataset_dir: Path,
        split: str,
        *,
        epoch: int = 0,
        shuffle: bool = False,
    ) -> None:
        self.cfg = cfg
        self.dataset_dir = dataset_dir
        self.split = split
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)
        self.manifest = load_shard_manifest(dataset_dir, cfg)
        self.shards = [shard for shard in self.manifest.shards if shard.split == split]
        if not self.shards:
            raise ValueError(f"manifest contains no shards for split {split}")
        self.count = sum(shard.count for shard in self.shards)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        rng = np.random.default_rng([self.cfg.data.seed, self.epoch, {"train": 0, "val": 1, "test": 2}[self.split]])
        shard_order = np.arange(len(self.shards))
        if self.shuffle:
            rng.shuffle(shard_order)
        position = 0
        for shard_pos in shard_order:
            shard = self.shards[int(shard_pos)]
            arrays = np.load(self.dataset_dir / shard.path)
            row_order = np.arange(shard.count)
            if self.shuffle:
                rng.shuffle(row_order)
            for row in row_order:
                if position % worker_count == worker_id:
                    signals = np.asarray(arrays["signals"][int(row)], dtype=np.float32)
                    heatmap = np.asarray(arrays["heatmaps"][int(row)], dtype=np.float32)
                    yield (
                        torch.from_numpy(signals[0:1]),
                        torch.from_numpy(signals[1:2]),
                        torch.from_numpy(heatmap),
                    )
                position += 1
```

- [ ] **Step 3: Run shard tests**

Run: `.venv/bin/python -m pytest tests/test_shards.py -q`

Expected: all tests in `tests/test_shards.py` pass.

- [ ] **Step 4: Commit dataset implementation**

```bash
git add src/sali/shards.py tests/test_shards.py
git commit -m "feat: read SALI dataset shards"
```

---

### Task 4: Add Sharded Training Tests

**Files:**
- Modify: `tests/test_train.py`
- Implementation target: `src/sali/train.py`

- [ ] **Step 1: Write failing sharded checkpoint/resume test**

Append to `tests/test_train.py`:

```python
def test_train_sharded_model_writes_resume_checkpoints_and_csv(tiny_config, tmp_path) -> None:
    from sali.shards import generate_shards
    from sali.train import train_sharded_model

    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "sharded-run"
    generate_shards(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    tiny_config.output_dir = run_dir
    tiny_config.training.batch_size = 2
    tiny_config.training.max_epochs = 1

    result = train_sharded_model(tiny_config, dataset_dir, checkpoint_every_epochs=1)

    latest = run_dir / "checkpoints" / "latest.pt"
    epoch_checkpoint = run_dir / "checkpoints" / "epoch_0001.pt"
    history_json = run_dir / "history.json"
    history_csv = run_dir / "history.csv"
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    best_state = torch.load(result.best_checkpoint, map_location="cpu", weights_only=True)

    assert latest.exists()
    assert epoch_checkpoint.exists()
    assert result.best_checkpoint.exists()
    assert history_json.exists()
    assert history_csv.exists()
    assert checkpoint["epoch"] == 1
    assert checkpoint["normalization_stats"]["epsilon"] == pytest.approx(tiny_config.data.norm_epsilon)
    assert len(result.history["train_loss"]) == 1
    assert all(tensor.device.type == "cpu" for tensor in best_state.values())


def test_train_sharded_model_resume_continues_next_epoch(tiny_config, tmp_path) -> None:
    from sali.shards import generate_shards
    from sali.train import train_sharded_model

    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "sharded-resume"
    generate_shards(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    tiny_config.output_dir = run_dir
    tiny_config.training.batch_size = 2
    tiny_config.training.max_epochs = 1

    train_sharded_model(tiny_config, dataset_dir, checkpoint_every_epochs=1)

    resume_cfg = copy.deepcopy(tiny_config)
    resume_cfg.training.max_epochs = 2
    latest = run_dir / "checkpoints" / "latest.pt"
    result = train_sharded_model(
        resume_cfg,
        dataset_dir,
        checkpoint_every_epochs=1,
        resume_from=latest,
    )
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)

    assert checkpoint["epoch"] == 2
    assert len(result.history["train_loss"]) == 2
    assert len(result.history["val_loss"]) == 2
```

- [ ] **Step 2: Write failing no-generation hot-path test**

Append:

```python
def test_train_sharded_model_does_not_generate_samples_or_heatmaps(tiny_config, tmp_path, monkeypatch) -> None:
    from sali.shards import generate_shards
    from sali.train import train_sharded_model

    dataset_dir = tmp_path / "dataset"
    generate_shards(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    tiny_config.output_dir = tmp_path / "sharded-no-generation"
    tiny_config.training.batch_size = 2
    tiny_config.training.max_epochs = 1

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("training must not generate samples")

    def fail_heatmap(*_args, **_kwargs):
        raise AssertionError("training must not render heatmaps")

    monkeypatch.setattr("sali.data.generate_indexed_sample", fail_generate)
    monkeypatch.setattr("sali.targets.render_heatmap", fail_heatmap)

    train_sharded_model(tiny_config, dataset_dir, checkpoint_every_epochs=1)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train.py::test_train_sharded_model_writes_resume_checkpoints_and_csv tests/test_train.py::test_train_sharded_model_resume_continues_next_epoch tests/test_train.py::test_train_sharded_model_does_not_generate_samples_or_heatmaps -q`

Expected: import failure for `train_sharded_model`.

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/test_train.py
git commit -m "test: cover sharded SALI training"
```

---

### Task 5: Implement Sharded Training

**Files:**
- Modify: `src/sali/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Import shard helpers**

Modify imports in `src/sali/train.py`:

```python
from sali.shards import ShardedSaliDataset, load_shard_manifest
```

- [ ] **Step 2: Implement `train_sharded_model()`**

Add after `train_streamed_model()`:

```python
def train_sharded_model(
    cfg: RunConfig,
    dataset_dir: Path,
    *,
    checkpoint_every_epochs: int = 1,
    resume_from: Path | str | None = None,
    num_workers: int = 0,
    epoch_callback: Callable[[int, SaliNet, dict[str, list[float]]], None] | None = None,
) -> TrainResult:
    if checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    manifest = load_shard_manifest(dataset_dir, cfg)
    stats = manifest.normalization_stats
    drop_last = _validate_training_batches(cfg.data.train_samples, cfg.training.batch_size)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = cfg.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.training.device)
    _set_torch_seed(cfg.data.seed)
    model = SaliNet(cfg.model).to(device)
    criterion = make_heatmap_loss(cfg.training)
    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_reduction_factor,
        patience=cfg.training.lr_plateau_patience,
        min_lr=1e-8,
    )
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss = float("inf")
    early_stopping_loss = float("inf")
    best_checkpoint = cfg.output_dir / "best_model.pt"
    best_state = _cpu_state_dict(model)
    stale_epochs = 0
    start_epoch = 1
    if resume_from is not None:
        resume_path = _resolve_resume_checkpoint(cfg.output_dir, resume_from)
        checkpoint = _load_full_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        history = {
            key: [float(value) for value in values]
            for key, values in dict(checkpoint["history"]).items()
        }
        best_loss = float(checkpoint["best_loss"])
        early_stopping_loss = float(checkpoint["early_stopping_loss"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_checkpoint.exists():
            best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
        else:
            best_state = _cpu_state_dict(model)

    val_loader = DataLoader(
        ShardedSaliDataset(cfg, dataset_dir, "val"),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    latest_checkpoint = checkpoint_dir / "latest.pt"
    for epoch in range(start_epoch, cfg.training.max_epochs + 1):
        train_loader = DataLoader(
            ShardedSaliDataset(cfg, dataset_dir, "train", epoch=epoch, shuffle=True),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device, None)
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(lr)
        if val_loss < best_loss:
            best_loss = float(val_loss)
            best_state = _cpu_state_dict(model)
            torch.save(best_state, best_checkpoint)
        if val_loss < early_stopping_loss - cfg.training.min_delta:
            early_stopping_loss = float(val_loss)
            stale_epochs = 0
        else:
            stale_epochs += 1
        _save_history(cfg.output_dir / "history.json", history)
        _save_history_csv(cfg.output_dir / "history.csv", history)
        _save_full_checkpoint(
            latest_checkpoint,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_loss,
            early_stopping_loss=early_stopping_loss,
            stale_epochs=stale_epochs,
            history=history,
            cfg=cfg,
            stats=stats,
        )
        if epoch % checkpoint_every_epochs == 0:
            _save_full_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                early_stopping_loss=early_stopping_loss,
                stale_epochs=stale_epochs,
                history=history,
                cfg=cfg,
                stats=stats,
            )
        _write_run_state(
            cfg.output_dir,
            epoch=epoch,
            best_loss=best_loss,
            stale_epochs=stale_epochs,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=best_checkpoint,
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model, history)
        if stale_epochs >= cfg.training.early_stopping_patience:
            break

    _load_state_dict_to_device(model, best_state, device)
    if not best_checkpoint.exists():
        torch.save(best_state, best_checkpoint)
    return TrainResult(model=model, history=history, best_checkpoint=best_checkpoint)
```

- [ ] **Step 3: Run sharded training tests**

Run: `.venv/bin/python -m pytest tests/test_train.py::test_train_sharded_model_writes_resume_checkpoints_and_csv tests/test_train.py::test_train_sharded_model_resume_continues_next_epoch tests/test_train.py::test_train_sharded_model_does_not_generate_samples_or_heatmaps -q`

Expected: PASS.

- [ ] **Step 4: Commit sharded training implementation**

```bash
git add src/sali/train.py tests/test_train.py
git commit -m "feat: train SALI from dataset shards"
```

---

### Task 6: Add CLI Tests for Sharded Mode

**Files:**
- Modify: `tests/test_train_colab_script.py`
- Implementation target: `scripts/train_colab.py`

- [ ] **Step 1: Extend help/default tests**

Modify `test_train_colab_help_runs()`:

```python
    assert "--dataset-dir" in completed.stdout
    assert "--shard-size" in completed.stdout
    assert "--generate-shards" in completed.stdout
    assert "--shard-dtype" in completed.stdout
    assert "--raw-signal-dtype" in completed.stdout
    assert "--cache-dataset-dir" in completed.stdout
    assert "--skip-existing-shards" in completed.stdout
```

Add:

```python
def test_train_colab_defaults_paper_to_sharded_mode() -> None:
    from scripts.train_colab import default_data_mode

    assert default_data_mode("paper") == "sharded"
    assert default_data_mode("practical") == "materialized"
```

Remove or update the older assertion that paper defaults to `stream`.

- [ ] **Step 2: Run CLI tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_colab_script.py -q`

Expected: failures for missing sharded flags and old default.

- [ ] **Step 3: Commit failing CLI tests**

```bash
git add tests/test_train_colab_script.py
git commit -m "test: cover sharded SALI CLI mode"
```

---

### Task 7: Implement CLI Shard Generation and Training

**Files:**
- Modify: `scripts/train_colab.py`
- Test: `tests/test_train_colab_script.py`, `tests/test_shards.py`, `tests/test_train.py`

- [ ] **Step 1: Import shard APIs and sharded trainer**

Modify imports:

```python
from sali.shards import (
    generate_shards,
    iter_sharded_samples,
    materialize_sharded_samples,
)
from sali.train import (
    TrainResult,
    evaluate_model,
    evaluate_sample_iterable,
    train_model,
    train_sharded_model,
    train_streamed_model,
)
```

- [ ] **Step 2: Extend CLI arguments and default mode**

Modify `parse_args()`:

```python
parser.add_argument("--data-mode", choices=["materialized", "stream", "sharded"], default=None)
parser.add_argument("--dataset-dir", type=Path, default=None)
parser.add_argument("--shard-size", type=int, default=10_000)
parser.add_argument("--generate-shards", action="store_true")
parser.add_argument("--shard-dtype", choices=["float32", "float16"], default="float32")
parser.add_argument("--raw-signal-dtype", choices=["float32", "float16"], default="float32")
parser.add_argument("--cache-dataset-dir", type=Path, default=None)
parser.add_argument("--skip-existing-shards", action=argparse.BooleanOptionalAction, default=True)
```

Modify `default_data_mode()`:

```python
def default_data_mode(preset: str) -> str:
    if preset == "paper":
        return "sharded"
    if preset == "practical":
        return "materialized"
    raise ValueError("preset must be either 'practical' or 'paper'")
```

- [ ] **Step 3: Add dataset directory resolver**

Add:

```python
def resolve_dataset_dir(cfg: RunConfig, args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = Path(f"datasets/{args.preset}-{args.field}-{cfg.data.train_samples}-{cfg.data.val_samples}-{cfg.data.test_samples}")
    if args.run_root is not None and not dataset_dir.is_absolute():
        dataset_dir = args.run_root / dataset_dir
    if str(dataset_dir).startswith("/content/drive") and not DRIVE_MYDRIVE.exists():
        raise RuntimeError(
            "Google Drive is not mounted in this Colab runtime. "
            "Expected /content/drive/MyDrive before writing durable SALI dataset shards."
        )
    return dataset_dir
```

- [ ] **Step 4: Add sharded sample views and periodic artifact callback**

Add:

```python
def sharded_sample_views(
    cfg: RunConfig,
    dataset_dir: Path,
    max_samples: int,
) -> DataSplits:
    return DataSplits(
        train=materialize_sharded_samples(cfg, dataset_dir, "train", max_samples=max_samples),
        val=materialize_sharded_samples(cfg, dataset_dir, "val", max_samples=max_samples),
        test=materialize_sharded_samples(cfg, dataset_dir, "test", max_samples=max_samples),
    )


def maybe_run_periodic_sharded_artifacts(
    *,
    cfg: RunConfig,
    dataset_dir: Path,
    thresholds: list[float],
    epoch: int,
    model: torch.nn.Module,
    history: dict[str, list[float]],
    calibrate_every_epochs: int,
    sample_plots_every_epochs: int,
    diagnostic_samples: int,
    threshold_mode: str,
    disable_morphology: bool,
) -> None:
    if calibrate_every_epochs > 0 and epoch % calibrate_every_epochs == 0:
        val_samples = materialize_sharded_samples(cfg, dataset_dir, "val", max_samples=diagnostic_samples)
        diagnostics = diagnose_split(
            model,
            val_samples,
            cfg,
            thresholds=thresholds,
            disable_morphology=disable_morphology,
        )
        save_json(cfg.output_dir / f"diagnostics_epoch_{epoch:04d}.json", {"val": diagnostics})
        if threshold_mode == "calibrate":
            selected = select_threshold(diagnostics["threshold_sweep"])
            cfg.postprocess.threshold = float(selected["threshold"])
            save_json(cfg.output_dir / f"selected_threshold_epoch_{epoch:04d}.json", {"mode": "calibrate", **selected})
    if sample_plots_every_epochs > 0 and epoch % sample_plots_every_epochs == 0:
        sample = materialize_sharded_samples(cfg, dataset_dir, "test", max_samples=1)[0]
        make_example_plots_for_sample(cfg, sample, model, cfg.output_dir / "figures" / f"epoch_{epoch:04d}")
        plot_loss(history, cfg.output_dir / "figures" / f"loss_epoch_{epoch:04d}.png")
```

- [ ] **Step 5: Add `run_sharded()`**

Add:

```python
def run_sharded(cfg: RunConfig, args: argparse.Namespace, thresholds: list[float]) -> None:
    dataset_dir = resolve_dataset_dir(cfg, args)
    if args.generate_shards:
        manifest = generate_shards(
            cfg,
            dataset_dir,
            shard_size=args.shard_size,
            normalization_samples=args.normalization_samples,
            shard_dtype=args.shard_dtype,
            raw_signal_dtype=args.raw_signal_dtype,
            skip_existing=args.skip_existing_shards,
        )
        save_json(cfg.output_dir / "dataset_manifest.json", {
            "dataset_dir": str(dataset_dir),
            "config_hash": manifest.config_hash,
            "shard_count": len(manifest.shards),
        })

    def epoch_callback(epoch: int, model: torch.nn.Module, history: dict[str, list[float]]) -> None:
        maybe_run_periodic_sharded_artifacts(
            cfg=cfg,
            dataset_dir=dataset_dir,
            thresholds=thresholds,
            epoch=epoch,
            model=model,
            history=history,
            calibrate_every_epochs=args.calibrate_every_epochs,
            sample_plots_every_epochs=args.sample_plots_every_epochs,
            diagnostic_samples=args.diagnostic_samples,
            threshold_mode=args.threshold_mode,
            disable_morphology=args.disable_diagnostic_morphology,
        )

    result = train_sharded_model(
        cfg,
        dataset_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        resume_from=args.resume,
        num_workers=args.num_workers,
        epoch_callback=epoch_callback,
    )
    plot_loss(result.history, cfg.output_dir / "figures" / "loss.png")

    diagnostic_views = sharded_sample_views(cfg, dataset_dir, args.diagnostic_samples)
    diagnostics = diagnose_splits(
        result.model,
        diagnostic_views,
        cfg,
        thresholds=thresholds,
        max_samples=None,
        disable_morphology=args.disable_diagnostic_morphology,
    )
    save_json(cfg.output_dir / "diagnostics.json", diagnostics)
    selected_threshold = {"mode": args.threshold_mode, "threshold": cfg.postprocess.threshold}
    if args.threshold_mode == "calibrate":
        selected = select_threshold(diagnostics["val"]["threshold_sweep"])
        cfg.postprocess.threshold = float(selected["threshold"])
        selected_threshold = {"mode": "calibrate", **selected}
    save_json(cfg.output_dir / "selected_threshold.json", selected_threshold)

    if not args.no_final_eval:
        if args.full_test_eval:
            metrics = evaluate_sample_iterable(result.model, cfg, iter_sharded_samples(cfg, dataset_dir, "test"))
        else:
            test_samples = materialize_sharded_samples(cfg, dataset_dir, "test", max_samples=args.max_eval_samples)
            metrics = evaluate_model(result.model, cfg, test_samples)
        summary = aggregate_by_true_count(metrics)
        save_json(cfg.output_dir / "metrics_by_true_count.json", summary)
        plot_precision_recall(metrics, cfg.output_dir / "figures" / "precision_recall.png")
        plot_mae(metrics, cfg.output_dir / "figures" / "mae.png")
    sample = materialize_sharded_samples(cfg, dataset_dir, "test", max_samples=1)[0]
    make_example_plots_for_sample(cfg, sample, result.model, cfg.output_dir / "figures")
```

- [ ] **Step 6: Route main mode**

Modify `main()`:

```python
save_json(cfg.output_dir / "config.json", {"data_mode": data_mode, **asdict(cfg)})
if data_mode == "materialized":
    run_materialized(cfg, args, thresholds)
elif data_mode == "stream":
    run_streamed(cfg, args, thresholds)
else:
    run_sharded(cfg, args, thresholds)
```

- [ ] **Step 7: Run CLI tests**

Run: `.venv/bin/python -m pytest tests/test_train_colab_script.py -q`

Expected: PASS.

- [ ] **Step 8: Commit CLI implementation**

```bash
git add scripts/train_colab.py tests/test_train_colab_script.py
git commit -m "feat: add sharded SALI Colab workflow"
```

---

### Task 8: Local Smoke and Full Verification

**Files:**
- No required source edits unless verification reveals a bug.

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_shards.py tests/test_train.py tests/test_train_colab_script.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run tiny sharded smoke command**

Run:

```bash
.venv/bin/python scripts/train_colab.py \
  --preset practical \
  --data-mode sharded \
  --field low \
  --output-dir runs/local-sharded-smoke \
  --dataset-dir runs/local-sharded-smoke-dataset \
  --generate-shards \
  --device cpu \
  --train-samples 8 \
  --val-samples 4 \
  --test-samples 4 \
  --epochs 1 \
  --batch-size 2 \
  --normalization-samples 4 \
  --shard-size 4 \
  --diagnostic-samples 2 \
  --max-eval-samples 2 \
  --checkpoint-every-epochs 1 \
  --calibrate-every-epochs 1 \
  --sample-plots-every-epochs 1
```

Expected:

- command exits `0`
- `runs/local-sharded-smoke/checkpoints/latest.pt` exists
- `runs/local-sharded-smoke/history.json` exists
- `runs/local-sharded-smoke-dataset/manifest.json` exists
- `runs/local-sharded-smoke-dataset/train-000000.npz` exists

- [ ] **Step 4: Inspect smoke artifact sizes**

Run:

```bash
find runs/local-sharded-smoke runs/local-sharded-smoke-dataset -maxdepth 2 -type f -print
```

Expected: checkpoint, metrics, figures, manifest, and shard files are present.

- [ ] **Step 5: Commit any verification fixes**

Only if fixes were needed:

```bash
git add <fixed-files>
git commit -m "fix: stabilize sharded SALI smoke run"
```

---

### Task 9: Colab 100k Sharded Benchmark Launch

**Files:**
- Create as needed under `/private/tmp` for launch scripts; do not commit temporary launch files.

- [ ] **Step 1: Package updated code for Colab**

Run from `/Users/weitao/Code/python`:

```bash
tar -czf /private/tmp/sali-colab-sharded.tar.gz \
  --exclude=sali/.venv \
  --exclude=sali/.pytest_cache \
  --exclude=sali/runs \
  --exclude=sali/__pycache__ \
  --exclude=sali/src/sali/__pycache__ \
  --exclude=sali/tests/__pycache__ \
  sali
```

Expected: `/private/tmp/sali-colab-sharded.tar.gz` exists.

- [ ] **Step 2: Upload archive to A100 session**

Run:

```bash
colab upload -s sali-paper-a100 /private/tmp/sali-colab-sharded.tar.gz /content/sali-colab-sharded.tar.gz
```

Expected: upload succeeds.

- [ ] **Step 3: Generate and train 100k sharded benchmark**

Create `/private/tmp/run_sali_sharded_100k_a100.py` with a command equivalent to:

```bash
python /content/sali/scripts/train_colab.py \
  --preset paper \
  --data-mode sharded \
  --field low \
  --run-root /content/drive/MyDrive/02_Research/DQP/sali \
  --output-dir runs/sali-sharded-100k-a100 \
  --dataset-dir datasets/sali-sharded-100k-low \
  --generate-shards \
  --device auto \
  --train-samples 100000 \
  --val-samples 20000 \
  --test-samples 20000 \
  --epochs 5 \
  --batch-size 64 \
  --normalization-samples 10000 \
  --shard-size 5000 \
  --diagnostic-samples 1024 \
  --max-eval-samples 2048 \
  --checkpoint-every-epochs 1 \
  --calibrate-every-epochs 5 \
  --sample-plots-every-epochs 5 \
  --threshold-mode calibrate \
  --num-workers 4
```

Expected: first run may spend time generating shards, then training should show higher A100 utilization than the stopped streamed run.

- [ ] **Step 4: Monitor GPU and epoch time**

Use a read-only Colab status script that reports:

- training PID
- GPU utilization samples
- DataLoader worker CPU usage
- `history.json` epoch count
- `run_state.json`
- shard manifest summary

Expected: after shard generation finishes, training should not have 8 workers saturated on sample generation and should write epoch artifacts faster than the streamed run.
