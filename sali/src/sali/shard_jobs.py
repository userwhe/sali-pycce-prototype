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
