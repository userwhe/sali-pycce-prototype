# Hyak Parallel Shard Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel-safe shard generation workflow for the repo-standard 3.6M-sample SALI paper split, validate it locally, then run it on Hyak under `/gscratch/scrubbed/whe3/sali`.

**Architecture:** Keep the existing `.npz` shard format and `load_shard_manifest()` training path unchanged. Add a focused `sali.shard_jobs` orchestration module that creates a deterministic shard plan, writes one shard per worker without touching the shared manifest, and finalizes a manifest only after every shard validates. Add a CLI and thin Hyak Slurm wrappers around that module.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, existing SALI modules, pytest, Slurm job arrays on Hyak.

---

## File Structure

- Create `src/sali/shard_jobs.py`: parallel-safe shard planning, single-shard writing, validation, and manifest finalization.
- Create `tests/test_shard_jobs.py`: unit and integration tests for plan creation, per-shard generation, resume/skip behavior, missing-shard failure, and final manifest compatibility.
- Create `scripts/shard_jobs.py`: command-line entrypoint with `prepare`, `write`, `finalize`, and `validate` subcommands.
- Create `tests/test_shard_jobs_script.py`: subprocess tests for the CLI on a tiny practical split.
- Create `scripts/hyak/prepare_shards.slurm`: Hyak setup job for metadata and normalization stats.
- Create `scripts/hyak/generate_shards_array.slurm`: Hyak Slurm array job, one task per shard id.
- Create `scripts/hyak/finalize_shards.slurm`: Hyak final validation and manifest writer.
- Create `scripts/hyak/README.md`: exact local-to-Hyak workflow, including pilot and full array commands.

Do not modify the existing sharded training loader except where tests reveal a compatibility bug. The new path should produce files that existing `sali.shards.load_shard_manifest()` and `sali.shards.ShardedSaliDataset` already understand.

---

### Task 1: Add Parallel Shard Job Tests

**Files:**
- Create: `tests/test_shard_jobs.py`
- No production code changes in this task

- [ ] **Step 1: Write failing tests**

Create `tests/test_shard_jobs.py` with this content:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sali.data import generate_indexed_sample
from sali.shards import load_shard_manifest, materialize_sharded_samples
from sali.shard_jobs import (
    build_shard_plan,
    finalize_shard_generation,
    load_generation_plan,
    prepare_shard_generation,
    write_planned_shard,
)


def test_build_shard_plan_covers_all_splits(tiny_config) -> None:
    plan = build_shard_plan(tiny_config, shard_size=3)

    assert [entry.shard_id for entry in plan] == list(range(7))
    assert [(entry.split, entry.start, entry.stop, entry.path) for entry in plan] == [
        ("train", 0, 3, "train-000000.npz"),
        ("train", 3, 6, "train-000001.npz"),
        ("train", 6, 8, "train-000002.npz"),
        ("val", 0, 3, "val-000000.npz"),
        ("val", 3, 4, "val-000001.npz"),
        ("test", 0, 3, "test-000000.npz"),
        ("test", 3, 4, "test-000001.npz"),
    ]
    assert sum(entry.count for entry in plan if entry.split == "train") == tiny_config.data.train_samples
    assert sum(entry.count for entry in plan if entry.split == "val") == tiny_config.data.val_samples
    assert sum(entry.count for entry in plan if entry.split == "test") == tiny_config.data.test_samples


def test_prepare_shard_generation_writes_plan_metadata(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"

    plan = prepare_shard_generation(
        tiny_config,
        dataset_dir,
        shard_size=4,
        normalization_samples=4,
        shard_dtype="float32",
        raw_signal_dtype="float32",
    )

    plan_path = dataset_dir / "generation_plan.json"
    raw = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan_path.exists()
    assert plan.config_hash == raw["config_hash"]
    assert raw["split_sizes"] == {"train": 8, "val": 4, "test": 4}
    assert raw["shard_size"] == 4
    assert raw["dtypes"]["signals"] == "float32"
    assert raw["normalization_stats"]["epsilon"] == pytest.approx(tiny_config.data.norm_epsilon)
    assert len(raw["shards"]) == 4


def test_write_planned_shard_matches_indexed_samples(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    plan = prepare_shard_generation(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)

    written = write_planned_shard(tiny_config, dataset_dir, shard_id=0)

    assert written.path == "train-000000.npz"
    arrays = np.load(dataset_dir / written.path)
    assert arrays["indices"].tolist() == [0, 1, 2, 3]
    expected = generate_indexed_sample(tiny_config, "train", 0, plan.normalization_stats)
    np.testing.assert_array_equal(arrays["signals"][0], expected.signals)
    np.testing.assert_array_equal(arrays["raw_signals"][0], expected.raw_signals)
    np.testing.assert_array_equal(arrays["heatmaps"][0], expected.heatmap)


def test_write_planned_shard_skips_existing_valid_file(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    prepare_shard_generation(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    first = write_planned_shard(tiny_config, dataset_dir, shard_id=0)
    first_path = dataset_dir / first.path
    first_mtime = first_path.stat().st_mtime_ns

    second = write_planned_shard(tiny_config, dataset_dir, shard_id=0, skip_existing=True)

    assert second.path == first.path
    assert first_path.stat().st_mtime_ns == first_mtime


def test_write_planned_shard_replaces_invalid_existing_file(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    prepare_shard_generation(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    bad_path = dataset_dir / "train-000000.npz"
    bad_path.write_text("not a valid npz", encoding="utf-8")

    written = write_planned_shard(tiny_config, dataset_dir, shard_id=0, skip_existing=True)

    arrays = np.load(dataset_dir / written.path)
    assert arrays["indices"].tolist() == [0, 1, 2, 3]


def test_write_planned_shard_rejects_config_mismatch(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    prepare_shard_generation(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    tiny_config.model.output_height = tiny_config.model.output_height + 2

    with pytest.raises(ValueError, match="generation plan does not match active config"):
        write_planned_shard(tiny_config, dataset_dir, shard_id=0)


def test_finalize_shard_generation_rejects_missing_shards(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    prepare_shard_generation(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    write_planned_shard(tiny_config, dataset_dir, shard_id=0)

    with pytest.raises(FileNotFoundError, match="missing shard file"):
        finalize_shard_generation(tiny_config, dataset_dir)


def test_finalize_shard_generation_writes_manifest_compatible_with_loader(tiny_config, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    plan = prepare_shard_generation(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)
    for entry in plan.shards:
        write_planned_shard(tiny_config, dataset_dir, shard_id=entry.shard_id)

    manifest = finalize_shard_generation(tiny_config, dataset_dir)
    loaded = load_shard_manifest(dataset_dir, tiny_config)
    samples = materialize_sharded_samples(tiny_config, dataset_dir, "train", max_samples=2)

    assert (dataset_dir / "manifest.json").exists()
    assert manifest.completed_at is not None
    assert loaded.config_hash == plan.config_hash
    assert len(loaded.shards) == len(plan.shards)
    assert len(samples) == 2
    assert samples[0].split == "train"


def test_load_generation_plan_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing generation plan"):
        load_generation_plan(tmp_path / "missing")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_shard_jobs.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'sali.shard_jobs'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add sali/tests/test_shard_jobs.py
git commit -m "test: cover parallel shard generation jobs"
```

---

### Task 2: Implement Parallel Shard Job Module

**Files:**
- Create: `src/sali/shard_jobs.py`
- Test: `tests/test_shard_jobs.py`

- [ ] **Step 1: Implement `sali.shard_jobs`**

Create `src/sali/shard_jobs.py` with this content:

```python
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sali.config import RunConfig
from sali.data import NormalizationStats, _split_count, estimate_normalization_stats
from sali.shards import (
    SCHEMA_VERSION,
    ShardInfo,
    ShardManifest,
    _manifest_to_json,
    _validate_shard_arrays,
    _write_shard,
    config_hash,
    dtype_from_name,
)


GENERATION_PLAN = "generation_plan.json"


@dataclass(slots=True)
class ShardPlanEntry:
    shard_id: int
    split: str
    shard_index: int
    start: int
    stop: int
    count: int
    path: str


@dataclass(slots=True)
class ShardGenerationPlan:
    schema_version: int
    data_mode: str
    config_hash: str
    config: dict[str, Any]
    split_sizes: dict[str, int]
    shard_size: int
    dtypes: dict[str, str]
    normalization_stats: NormalizationStats
    shards: list[ShardPlanEntry]
    generated_at: str


def _split_names() -> tuple[str, str, str]:
    return ("train", "val", "test")


def _shard_filename(split: str, shard_index: int) -> str:
    return f"{split}-{shard_index:06d}.npz"


def _config_snapshot(cfg: RunConfig) -> dict[str, Any]:
    snapshot = asdict(cfg)
    snapshot["output_dir"] = str(cfg.output_dir)
    return snapshot


def _split_sizes(cfg: RunConfig) -> dict[str, int]:
    return {split: _split_count(cfg, split) for split in _split_names()}


def build_shard_plan(cfg: RunConfig, shard_size: int) -> list[ShardPlanEntry]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    entries: list[ShardPlanEntry] = []
    for split in _split_names():
        split_count = _split_count(cfg, split)
        for shard_index, start in enumerate(range(0, split_count, shard_size)):
            stop = min(start + shard_size, split_count)
            entries.append(
                ShardPlanEntry(
                    shard_id=len(entries),
                    split=split,
                    shard_index=shard_index,
                    start=start,
                    stop=stop,
                    count=stop - start,
                    path=_shard_filename(split, shard_index),
                )
            )
    return entries


def _plan_to_json(plan: ShardGenerationPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["normalization_stats"] = asdict(plan.normalization_stats)
    payload["shards"] = [asdict(entry) for entry in plan.shards]
    return payload


def _plan_from_json(payload: dict[str, Any]) -> ShardGenerationPlan:
    stats_payload = dict(payload["normalization_stats"])
    return ShardGenerationPlan(
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
        shards=[
            ShardPlanEntry(
                shard_id=int(item["shard_id"]),
                split=str(item["split"]),
                shard_index=int(item["shard_index"]),
                start=int(item["start"]),
                stop=int(item["stop"]),
                count=int(item["count"]),
                path=str(item["path"]),
            )
            for item in list(payload["shards"])
        ],
        generated_at=str(payload["generated_at"]),
    )


def save_generation_plan(path: Path, plan: ShardGenerationPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plan_to_json(plan), indent=2, default=str), encoding="utf-8")


def load_generation_plan(dataset_dir: Path) -> ShardGenerationPlan:
    path = dataset_dir / GENERATION_PLAN
    if not path.exists():
        raise FileNotFoundError(f"missing generation plan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = _plan_from_json(payload)
    if plan.schema_version != SCHEMA_VERSION or plan.data_mode != "sharded":
        raise ValueError("unsupported generation plan")
    return plan


def prepare_shard_generation(
    cfg: RunConfig,
    dataset_dir: Path,
    *,
    shard_size: int,
    normalization_samples: int,
    shard_dtype: str = "float32",
    raw_signal_dtype: str = "float32",
) -> ShardGenerationPlan:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    signal_dtype = dtype_from_name(shard_dtype)
    raw_dtype = dtype_from_name(raw_signal_dtype)
    stats = estimate_normalization_stats(cfg, sample_limit=normalization_samples)
    if not np.isfinite(stats.mean) or not np.isfinite(stats.var):
        raise FloatingPointError("normalization statistics are non-finite")
    plan = ShardGenerationPlan(
        schema_version=SCHEMA_VERSION,
        data_mode="sharded",
        config_hash=config_hash(cfg),
        config=_config_snapshot(cfg),
        split_sizes=_split_sizes(cfg),
        shard_size=int(shard_size),
        dtypes={
            "signals": signal_dtype.name,
            "raw_signals": raw_dtype.name,
            "heatmaps": signal_dtype.name,
            "nuclei_count": "uint8",
            "nuclei_az_khz": "float32",
            "nuclei_aperp_khz": "float32",
            "indices": "int64",
        },
        normalization_stats=stats,
        shards=build_shard_plan(cfg, shard_size),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    save_generation_plan(dataset_dir / GENERATION_PLAN, plan)
    return plan


def _check_plan_matches_config(plan: ShardGenerationPlan, cfg: RunConfig) -> None:
    if plan.config_hash != config_hash(cfg):
        raise ValueError("generation plan does not match active config")


def _entry_by_id(plan: ShardGenerationPlan, shard_id: int) -> ShardPlanEntry:
    for entry in plan.shards:
        if entry.shard_id == shard_id:
            return entry
    raise IndexError(f"shard_id {shard_id} is out of range for {len(plan.shards)} planned shards")


def _temp_shard_path(dataset_dir: Path, entry: ShardPlanEntry) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", str(entry.shard_id))
    return dataset_dir / f"{entry.path}.tmp-{job_id}-{task_id}.npz"


def write_planned_shard(
    cfg: RunConfig,
    dataset_dir: Path,
    *,
    shard_id: int,
    skip_existing: bool = True,
) -> ShardPlanEntry:
    plan = load_generation_plan(dataset_dir)
    _check_plan_matches_config(plan, cfg)
    entry = _entry_by_id(plan, int(shard_id))
    target_path = dataset_dir / entry.path
    signal_dtype = dtype_from_name(plan.dtypes["signals"])
    raw_dtype = dtype_from_name(plan.dtypes["raw_signals"])

    if skip_existing and _validate_shard_arrays(target_path, entry.count, entry.start, entry.stop, cfg):
        return entry

    temp_path = _temp_shard_path(dataset_dir, entry)
    if temp_path.exists():
        temp_path.unlink()
    try:
        _write_shard(
            cfg,
            entry.split,
            entry.start,
            entry.stop,
            plan.normalization_stats,
            temp_path,
            shard_dtype=signal_dtype,
            raw_signal_dtype=raw_dtype,
        )
        if not _validate_shard_arrays(temp_path, entry.count, entry.start, entry.stop, cfg):
            raise ValueError(f"temporary shard failed validation: {temp_path}")
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return entry


def validate_planned_shards(cfg: RunConfig, dataset_dir: Path) -> list[ShardPlanEntry]:
    plan = load_generation_plan(dataset_dir)
    _check_plan_matches_config(plan, cfg)
    invalid: list[ShardPlanEntry] = []
    for entry in plan.shards:
        path = dataset_dir / entry.path
        if not path.exists():
            raise FileNotFoundError(f"missing shard file: {path}")
        if not _validate_shard_arrays(path, entry.count, entry.start, entry.stop, cfg):
            invalid.append(entry)
    return invalid


def finalize_shard_generation(cfg: RunConfig, dataset_dir: Path) -> ShardManifest:
    plan = load_generation_plan(dataset_dir)
    _check_plan_matches_config(plan, cfg)
    invalid = validate_planned_shards(cfg, dataset_dir)
    if invalid:
        bad = ", ".join(entry.path for entry in invalid[:5])
        raise ValueError(f"invalid shard file(s): {bad}")
    manifest = ShardManifest(
        schema_version=SCHEMA_VERSION,
        data_mode="sharded",
        config_hash=plan.config_hash,
        config=plan.config,
        split_sizes=plan.split_sizes,
        shard_size=plan.shard_size,
        dtypes=plan.dtypes,
        normalization_stats=plan.normalization_stats,
        shards=[
            ShardInfo(
                split=entry.split,
                path=entry.path,
                start=entry.start,
                stop=entry.stop,
                count=entry.count,
            )
            for entry in plan.shards
        ],
        generated_at=plan.generated_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps(_manifest_to_json(manifest), indent=2, default=str),
        encoding="utf-8",
    )
    return manifest
```

- [ ] **Step 2: Run shard job tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_shard_jobs.py -q
```

Expected: PASS.

- [ ] **Step 3: Run existing shard tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_shards.py tests/test_train.py::test_train_sharded_model_writes_resume_checkpoints_and_csv tests/test_train.py::test_train_sharded_model_does_not_generate_samples_or_heatmaps -q
```

Expected: PASS.

- [ ] **Step 4: Commit implementation**

```bash
git add sali/src/sali/shard_jobs.py sali/tests/test_shard_jobs.py
git commit -m "feat: add parallel-safe shard generation jobs"
```

---

### Task 3: Add Shard Job CLI

**Files:**
- Create: `scripts/shard_jobs.py`
- Create: `tests/test_shard_jobs_script.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_shard_jobs_script.py` with this content:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shard_jobs.py"


def _base_cmd(dataset_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--preset",
        "practical",
        "--field",
        "low",
        "--dataset-dir",
        str(dataset_dir),
        "--train-samples",
        "8",
        "--val-samples",
        "4",
        "--test-samples",
        "4",
        "--shard-size",
        "4",
        "--normalization-samples",
        "4",
    ]


def test_shard_jobs_cli_prepares_writes_and_finalizes_dataset(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"

    prepare = subprocess.run(
        [*_base_cmd(dataset_dir), "prepare"],
        check=True,
        text=True,
        capture_output=True,
    )
    write = subprocess.run(
        [*_base_cmd(dataset_dir), "write", "--shard-id", "0"],
        check=True,
        text=True,
        capture_output=True,
    )
    for shard_id in (1, 2, 3):
        subprocess.run(
            [*_base_cmd(dataset_dir), "write", "--shard-id", str(shard_id)],
            check=True,
            text=True,
            capture_output=True,
        )
    finalize = subprocess.run(
        [*_base_cmd(dataset_dir), "finalize"],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "prepared 4 shards" in prepare.stdout
    assert "wrote train-000000.npz" in write.stdout
    assert "finalized 4 shards" in finalize.stdout
    assert manifest["split_sizes"] == {"train": 8, "val": 4, "test": 4}


def test_shard_jobs_cli_validate_reports_existing_manifest(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    subprocess.run([*_base_cmd(dataset_dir), "prepare"], check=True, text=True, capture_output=True)
    for shard_id in range(4):
        subprocess.run(
            [*_base_cmd(dataset_dir), "write", "--shard-id", str(shard_id)],
            check=True,
            text=True,
            capture_output=True,
        )
    subprocess.run([*_base_cmd(dataset_dir), "finalize"], check=True, text=True, capture_output=True)

    completed = subprocess.run(
        [*_base_cmd(dataset_dir), "validate"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "validated 4 shards" in completed.stdout
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_shard_jobs_script.py -q
```

Expected: FAIL because `scripts/shard_jobs.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/shard_jobs.py` with this content:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if (SRC / "sali").exists():
    sys.path.insert(0, str(SRC))

from sali.config import RunConfig, paper_config, practical_config
from sali.shard_jobs import (
    finalize_shard_generation,
    load_generation_plan,
    prepare_shard_generation,
    validate_planned_shards,
    write_planned_shard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, write, and finalize SALI dataset shards.")
    parser.add_argument("--preset", choices=["practical", "paper"], default="paper")
    parser.add_argument("--field", choices=["low", "high"], default="low")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--normalization-samples", type=int, default=10_000)
    parser.add_argument("--shard-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--raw-signal-dtype", choices=["float32", "float16"], default="float32")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Write generation_plan.json.")
    write = subparsers.add_parser("write", help="Write one shard from generation_plan.json.")
    write.add_argument("--shard-id", type=int, required=True)
    write.add_argument("--no-skip-existing", action="store_true")
    subparsers.add_parser("finalize", help="Validate all planned shards and write manifest.json.")
    subparsers.add_parser("validate", help="Validate all planned shards without rewriting manifest.json.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = practical_config(args.field) if args.preset == "practical" else paper_config(args.field)
    if args.train_samples is not None:
        cfg.data.train_samples = args.train_samples
    if args.val_samples is not None:
        cfg.data.val_samples = args.val_samples
    if args.test_samples is not None:
        cfg.data.test_samples = args.test_samples
    return cfg


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    if args.command == "prepare":
        plan = prepare_shard_generation(
            cfg,
            args.dataset_dir,
            shard_size=args.shard_size,
            normalization_samples=args.normalization_samples,
            shard_dtype=args.shard_dtype,
            raw_signal_dtype=args.raw_signal_dtype,
        )
        print(f"prepared {len(plan.shards)} shards at {args.dataset_dir}")
    elif args.command == "write":
        entry = write_planned_shard(
            cfg,
            args.dataset_dir,
            shard_id=args.shard_id,
            skip_existing=not args.no_skip_existing,
        )
        print(f"wrote {entry.path}")
    elif args.command == "finalize":
        manifest = finalize_shard_generation(cfg, args.dataset_dir)
        print(f"finalized {len(manifest.shards)} shards at {args.dataset_dir / 'manifest.json'}")
    elif args.command == "validate":
        plan = load_generation_plan(args.dataset_dir)
        invalid = validate_planned_shards(cfg, args.dataset_dir)
        if invalid:
            bad = ", ".join(entry.path for entry in invalid[:5])
            raise SystemExit(f"invalid shard file(s): {bad}")
        print(f"validated {len(plan.shards)} shards")
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_shard_jobs_script.py -q
```

Expected: PASS.

- [ ] **Step 5: Run shard job and CLI tests together**

Run:

```bash
.venv/bin/python -m pytest tests/test_shard_jobs.py tests/test_shard_jobs_script.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit CLI**

```bash
git add sali/scripts/shard_jobs.py sali/tests/test_shard_jobs_script.py
git commit -m "feat: add shard generation job CLI"
```

---

### Task 4: Add Hyak Slurm Wrappers And Runbook

**Files:**
- Create: `scripts/hyak/prepare_shards.slurm`
- Create: `scripts/hyak/generate_shards_array.slurm`
- Create: `scripts/hyak/finalize_shards.slurm`
- Create: `scripts/hyak/README.md`

- [ ] **Step 1: Create Hyak script directory**

Run:

```bash
mkdir -p scripts/hyak
```

Expected: directory exists.

- [ ] **Step 2: Add prepare Slurm script**

Create `scripts/hyak/prepare_shards.slurm`:

```bash
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
REPO_DIR="${WORK_ROOT}/repo"
DATASET_DIR="${WORK_ROOT}/datasets/paper-low-full"
VENV="${WORK_ROOT}/venv"

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/datasets"
cd "${REPO_DIR}"
source "${VENV}/bin/activate"

python scripts/shard_jobs.py \
  --preset paper \
  --field low \
  --dataset-dir "${DATASET_DIR}" \
  --shard-size 10000 \
  --normalization-samples 20000 \
  --shard-dtype float32 \
  --raw-signal-dtype float32 \
  prepare
```

- [ ] **Step 3: Add array Slurm script**

Create `scripts/hyak/generate_shards_array.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=sali-shard-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-359%20
#SBATCH --chdir=/gscratch/scrubbed/whe3/sali/repo
#SBATCH --output=/gscratch/scrubbed/whe3/sali/logs/%x-%A_%a.out
#SBATCH --error=/gscratch/scrubbed/whe3/sali/logs/%x-%A_%a.err

set -euo pipefail

WORK_ROOT=/gscratch/scrubbed/whe3/sali
REPO_DIR="${WORK_ROOT}/repo"
DATASET_DIR="${WORK_ROOT}/datasets/paper-low-full"
VENV="${WORK_ROOT}/venv"

mkdir -p "${WORK_ROOT}/logs"
cd "${REPO_DIR}"
source "${VENV}/bin/activate"

python scripts/shard_jobs.py \
  --preset paper \
  --field low \
  --dataset-dir "${DATASET_DIR}" \
  --shard-size 10000 \
  --normalization-samples 20000 \
  --shard-dtype float32 \
  --raw-signal-dtype float32 \
  write \
  --shard-id "${SLURM_ARRAY_TASK_ID}"
```

- [ ] **Step 4: Add finalizer Slurm script**

Create `scripts/hyak/finalize_shards.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=sali-shard-finalize
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
DATASET_DIR="${WORK_ROOT}/datasets/paper-low-full"
VENV="${WORK_ROOT}/venv"

mkdir -p "${WORK_ROOT}/logs"
cd "${REPO_DIR}"
source "${VENV}/bin/activate"

python scripts/shard_jobs.py \
  --preset paper \
  --field low \
  --dataset-dir "${DATASET_DIR}" \
  --shard-size 10000 \
  --normalization-samples 20000 \
  finalize
```

- [ ] **Step 5: Add Hyak runbook**

Create `scripts/hyak/README.md`:

````markdown
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
````

- [ ] **Step 6: Syntax-check shell scripts**

Run:

```bash
bash -n scripts/hyak/prepare_shards.slurm
bash -n scripts/hyak/generate_shards_array.slurm
bash -n scripts/hyak/finalize_shards.slurm
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit Hyak scripts**

```bash
git add sali/scripts/hyak/prepare_shards.slurm sali/scripts/hyak/generate_shards_array.slurm sali/scripts/hyak/finalize_shards.slurm sali/scripts/hyak/README.md
git commit -m "docs: add Hyak shard generation runbook"
```

---

### Task 5: Local End-To-End Smoke Validation

**Files:**
- No code changes expected
- Generated outputs under `/private/tmp` or pytest `tmp_path`; do not commit generated shards

- [ ] **Step 1: Run complete shard job test suite**

Run:

```bash
.venv/bin/python -m pytest tests/test_shard_jobs.py tests/test_shard_jobs_script.py tests/test_shards.py -q
```

Expected: PASS.

- [ ] **Step 2: Run tiny CLI smoke manually**

Run:

```bash
rm -rf /private/tmp/sali-shard-job-smoke
.venv/bin/python scripts/shard_jobs.py --preset practical --field low \
  --dataset-dir /private/tmp/sali-shard-job-smoke \
  --train-samples 8 --val-samples 4 --test-samples 4 \
  --shard-size 4 --normalization-samples 4 prepare
for id in 0 1 2 3; do
  .venv/bin/python scripts/shard_jobs.py --preset practical --field low \
    --dataset-dir /private/tmp/sali-shard-job-smoke \
    --train-samples 8 --val-samples 4 --test-samples 4 \
    --shard-size 4 --normalization-samples 4 write --shard-id "$id"
done
.venv/bin/python scripts/shard_jobs.py --preset practical --field low \
  --dataset-dir /private/tmp/sali-shard-job-smoke \
  --train-samples 8 --val-samples 4 --test-samples 4 \
  --shard-size 4 --normalization-samples 4 finalize
```

Expected output includes:

```text
prepared 4 shards
wrote train-000000.npz
wrote train-000001.npz
wrote val-000000.npz
wrote test-000000.npz
finalized 4 shards
```

- [ ] **Step 3: Verify existing loader can read smoke dataset**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from sali.config import practical_config
from sali.shards import load_shard_manifest, materialize_sharded_samples

cfg = practical_config("low")
cfg.data.train_samples = 8
cfg.data.val_samples = 4
cfg.data.test_samples = 4
cfg.data.max_nuclei = 3
dataset_dir = Path("/private/tmp/sali-shard-job-smoke")
manifest = load_shard_manifest(dataset_dir, cfg)
samples = materialize_sharded_samples(cfg, dataset_dir, "train", max_samples=2)
print(len(manifest.shards), len(samples), samples[0].signals.shape)
PY
```

Expected output:

```text
4 2 (2, 1000)
```

- [ ] **Step 4: Run tiny sharded training smoke**

Run:

```bash
.venv/bin/python scripts/train_colab.py \
  --preset practical \
  --data-mode sharded \
  --field low \
  --output-dir /private/tmp/sali-shard-job-train-smoke \
  --dataset-dir /private/tmp/sali-shard-job-smoke \
  --train-samples 8 \
  --val-samples 4 \
  --test-samples 4 \
  --epochs 1 \
  --batch-size 2 \
  --samples-per-epoch 4 \
  --max-eval-samples 2 \
  --diagnostic-samples 2 \
  --no-final-eval
```

Expected:

```text
Run complete: /private/tmp/sali-shard-job-train-smoke
```

and these files exist:

```text
/private/tmp/sali-shard-job-train-smoke/history.json
/private/tmp/sali-shard-job-train-smoke/checkpoints/latest.pt
```

- [ ] **Step 5: Commit any smoke-test fixes**

If fixes were needed, commit only the changed source/test/script files:

```bash
git status --short
git add sali/src/sali/shard_jobs.py sali/scripts/shard_jobs.py sali/tests/test_shard_jobs.py sali/tests/test_shard_jobs_script.py sali/scripts/hyak
git commit -m "fix: stabilize shard job smoke workflow"
```

If no fixes were needed, do not create an empty commit.

---

### Task 6: Hyak Access And Environment Setup

**Files:**
- No repo code changes expected
- Requires network/SSH approval and user authentication

- [ ] **Step 1: Confirm SSH host**

Use `whe3@klone.hyak.uw.edu` unless the user corrects the host.

- [ ] **Step 2: Request approval and test SSH**

Run with escalated permissions because it needs network access:

```bash
ssh whe3@klone.hyak.uw.edu 'hostname; whoami; pwd'
```

Expected: user completes any password/Duo/2FA prompts, and output includes a Klone hostname and `whe3`.

- [ ] **Step 3: Inspect Hyak allocations and storage**

Run:

```bash
ssh whe3@klone.hyak.uw.edu 'hyakalloc; hyakstorage --home; hyakstorage /gscratch/scrubbed/whe3; df -h /gscratch/scrubbed/whe3'
```

Expected: output lists available accounts/partitions and confirms `/gscratch/scrubbed/whe3` exists or indicates it must be created.

- [ ] **Step 4: Create Hyak working directories**

Run:

```bash
ssh whe3@klone.hyak.uw.edu 'mkdir -p /gscratch/scrubbed/whe3/sali/{repo,logs,datasets}'
```

Expected: command exits 0.

- [ ] **Step 5: Upload repo excluding local large artifacts**

Run with escalated permissions:

```bash
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'runs/' \
  ./ whe3@klone.hyak.uw.edu:/gscratch/scrubbed/whe3/sali/repo/
```

Expected: repo files appear under `/gscratch/scrubbed/whe3/sali/repo`.

- [ ] **Step 6: Create Python environment on Hyak**

Run:

```bash
ssh whe3@klone.hyak.uw.edu 'cd /gscratch/scrubbed/whe3/sali && python3 -m venv venv && source venv/bin/activate && python -m pip install --upgrade pip && python -m pip install -e "repo[dev]"'
```

Expected: dependencies install successfully. If network access from Hyak is blocked, use Hyak-supported Python/module guidance or upload wheels from local cache.

---

### Task 7: Hyak Pilot Run

**Files:**
- No repo code changes expected unless pilot exposes a bug
- Hyak outputs under `/gscratch/scrubbed/whe3/sali`

- [ ] **Step 1: Run tiny Hyak CLI smoke**

Run:

```bash
ssh whe3@klone.hyak.uw.edu 'cd /gscratch/scrubbed/whe3/sali/repo && source /gscratch/scrubbed/whe3/sali/venv/bin/activate && python scripts/shard_jobs.py --preset practical --field low --dataset-dir /gscratch/scrubbed/whe3/sali/datasets/tiny-smoke --train-samples 8 --val-samples 4 --test-samples 4 --shard-size 4 --normalization-samples 4 prepare && for id in 0 1 2 3; do python scripts/shard_jobs.py --preset practical --field low --dataset-dir /gscratch/scrubbed/whe3/sali/datasets/tiny-smoke --train-samples 8 --val-samples 4 --test-samples 4 --shard-size 4 --normalization-samples 4 write --shard-id "$id"; done && python scripts/shard_jobs.py --preset practical --field low --dataset-dir /gscratch/scrubbed/whe3/sali/datasets/tiny-smoke --train-samples 8 --val-samples 4 --test-samples 4 --shard-size 4 --normalization-samples 4 finalize'
```

Expected: prepare, four writes, and finalize all succeed.

- [ ] **Step 2: Choose account and partition**

From `hyakalloc`, set shell variables in the local shell to the exact values reported by Hyak:

```bash
: "${SALI_HYAK_ACCOUNT:?Set SALI_HYAK_ACCOUNT to the exact account shown by hyakalloc}"
: "${SALI_HYAK_PARTITION:?Set SALI_HYAK_PARTITION to the exact partition shown by hyakalloc}"
```

Do not guess these values if `hyakalloc` shows a different account than `whe3`.

- [ ] **Step 3: Submit full prepare job**

Run after setting the local shell variables in Step 2:

```bash
ssh whe3@klone.hyak.uw.edu "cd /gscratch/scrubbed/whe3/sali/repo && sbatch -A \"$SALI_HYAK_ACCOUNT\" -p \"$SALI_HYAK_PARTITION\" scripts/hyak/prepare_shards.slurm"
```

Expected: Slurm prints `Submitted batch job JOBID`.

- [ ] **Step 4: Monitor prepare job**

Run:

```bash
ssh whe3@klone.hyak.uw.edu 'squeue -u whe3; ls -lh /gscratch/scrubbed/whe3/sali/datasets/paper-low-full/generation_plan.json; tail -100 /gscratch/scrubbed/whe3/sali/logs/sali-shard-prepare-*.err'
```

Expected: after completion, `generation_plan.json` exists and logs show no traceback.

- [ ] **Step 5: Submit four-shard pilot array**

Run:

```bash
ssh whe3@klone.hyak.uw.edu "cd /gscratch/scrubbed/whe3/sali/repo && sbatch -A \"$SALI_HYAK_ACCOUNT\" -p \"$SALI_HYAK_PARTITION\" --array=0-3%2 scripts/hyak/generate_shards_array.slurm"
```

Expected: Slurm prints `Submitted batch job JOBID`.

- [ ] **Step 6: Inspect pilot results**

Run:

```bash
ssh whe3@klone.hyak.uw.edu 'squeue -u whe3; ls -lh /gscratch/scrubbed/whe3/sali/datasets/paper-low-full/train-00000*.npz; du -h --max-depth 1 /gscratch/scrubbed/whe3/sali/datasets/paper-low-full; tail -100 /gscratch/scrubbed/whe3/sali/logs/sali-shard-array-*.err'
```

Expected: four shard files exist, logs contain no traceback, and file sizes look plausible.

---

### Task 8: Full Hyak Generation, Finalization, And Transfer

**Files:**
- No repo code changes expected unless production exposes a bug
- Hyak outputs under `/gscratch/scrubbed/whe3/sali/datasets/paper-low-full`

- [ ] **Step 1: Submit full 360-shard array**

Run:

```bash
ssh whe3@klone.hyak.uw.edu "cd /gscratch/scrubbed/whe3/sali/repo && sbatch -A \"$SALI_HYAK_ACCOUNT\" -p \"$SALI_HYAK_PARTITION\" scripts/hyak/generate_shards_array.slurm"
```

Expected: Slurm prints `Submitted batch job JOBID`.

- [ ] **Step 2: Monitor full array**

Run periodically:

```bash
ssh whe3@klone.hyak.uw.edu 'squeue -u whe3; find /gscratch/scrubbed/whe3/sali/datasets/paper-low-full -maxdepth 1 -name "*.npz" | wc -l; du -sh /gscratch/scrubbed/whe3/sali/datasets/paper-low-full'
```

Expected: `.npz` count eventually reaches `360`.

- [ ] **Step 3: Submit finalizer job**

Run:

```bash
ssh whe3@klone.hyak.uw.edu "cd /gscratch/scrubbed/whe3/sali/repo && sbatch -A \"$SALI_HYAK_ACCOUNT\" -p \"$SALI_HYAK_PARTITION\" scripts/hyak/finalize_shards.slurm"
```

Expected: Slurm prints `Submitted batch job JOBID`.

- [ ] **Step 4: Validate final manifest**

Run:

```bash
ssh whe3@klone.hyak.uw.edu 'cd /gscratch/scrubbed/whe3/sali/repo && source /gscratch/scrubbed/whe3/sali/venv/bin/activate && python scripts/shard_jobs.py --preset paper --field low --dataset-dir /gscratch/scrubbed/whe3/sali/datasets/paper-low-full --shard-size 10000 --normalization-samples 20000 validate && python - <<'"'"'PY'"'"'
from pathlib import Path
from sali.config import paper_config
from sali.shards import load_shard_manifest
cfg = paper_config("low")
manifest = load_shard_manifest(Path("/gscratch/scrubbed/whe3/sali/datasets/paper-low-full"), cfg)
print(manifest.split_sizes)
print(len(manifest.shards))
PY'
```

Expected output includes:

```text
validated 360 shards
{'train': 2520000, 'val': 540000, 'test': 540000}
360
```

- [ ] **Step 5: Transfer validated dataset**

Choose one transfer method after checking credential constraints. Preferred local-machine relay:

```bash
rsync -az --partial --progress \
  whe3@klone.hyak.uw.edu:/gscratch/scrubbed/whe3/sali/datasets/paper-low-full/ \
  "/Users/weitao/Library/CloudStorage/GoogleDrive-heweutao@gmail.com/My Drive/02_Research/DQP/sali/datasets/paper-low-full/"
```

Expected: local Google Drive-backed directory contains `manifest.json`, `generation_plan.json`, and 360 `.npz` shard files.

- [ ] **Step 6: Record final artifact summary**

Run:

```bash
ssh whe3@klone.hyak.uw.edu 'du -sh /gscratch/scrubbed/whe3/sali/datasets/paper-low-full; find /gscratch/scrubbed/whe3/sali/datasets/paper-low-full -maxdepth 1 -name "*.npz" | wc -l'
```

Expected: report total Hyak dataset size and `360`.

Add a short note to the final response with:

- Hyak dataset path
- Google Drive path
- shard count
- total dataset size
- any failed/retried Slurm tasks

---

## Plan Self-Review Checklist

- Spec coverage: Tasks 1-5 implement local smoke validation, parallel-safe generation, final manifest writing, and CLI/HPC wrappers. Tasks 6-8 cover SSH access, Hyak setup, pilot array, full array, validation, and Google Drive transfer.
- Placeholder scan: No task contains TBD markers or fake values. Hyak account and partition are runtime shell variables guarded with `:?` checks because they must come from live `hyakalloc` output.
- Type consistency: The plan consistently uses `ShardPlanEntry`, `ShardGenerationPlan`, `generation_plan.json`, `write_planned_shard()`, and `finalize_shard_generation()`.
- Scope check: The plan is one cohesive subsystem: parallel-safe generation of existing shard data. Training changes and model changes are out of scope.
