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
from sali.data import NormalizationStats, Sample, _split_count, estimate_normalization_stats, generate_indexed_sample
from sali.physics import Couplings


SCHEMA_VERSION = 1
DEFAULT_SHARD_DTYPE = np.float32
SHARD_METADATA_KEY = "sali_shard_metadata_json"
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
        shards=[
            ShardInfo(
                split=str(item["split"]),
                path=str(item["path"]),
                start=int(item["start"]),
                stop=int(item["stop"]),
                count=int(item["count"]),
            )
            for item in list(payload["shards"])
        ],
        generated_at=str(payload["generated_at"]),
        completed_at=None if payload.get("completed_at") is None else str(payload["completed_at"]),
    )


def _split_names() -> tuple[str, str, str]:
    return ("train", "val", "test")


def _split_sizes(cfg: RunConfig) -> dict[str, int]:
    return {split: _split_count(cfg, split) for split in _split_names()}


def _shard_filename(split: str, shard_index: int) -> str:
    return f"{split}-{shard_index:06d}.npz"


def _validate_shard_arrays(
    path: Path,
    expected_count: int,
    start: int,
    stop: int,
    cfg: RunConfig,
) -> bool:
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
    metadata: dict[str, object] | None = None,
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
    payload: dict[str, np.ndarray] = {
        "signals": signals,
        "raw_signals": raw_signals,
        "heatmaps": heatmaps,
        "nuclei_count": nuclei_count,
        "nuclei_az_khz": nuclei_az,
        "nuclei_aperp_khz": nuclei_aperp,
        "indices": indices,
    }
    if metadata is not None:
        payload[SHARD_METADATA_KEY] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


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
    stats = estimate_normalization_stats(cfg, sample_limit=normalization_samples)
    if not np.isfinite(stats.mean) or not np.isfinite(stats.var):
        raise FloatingPointError("normalization statistics are non-finite")

    manifest = ShardManifest(
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
        shards=[],
        generated_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
    )

    for split in _split_names():
        split_count = _split_count(cfg, split)
        for shard_index, start in enumerate(range(0, split_count, shard_size)):
            stop = min(start + shard_size, split_count)
            count = stop - start
            shard_path = dataset_dir / _shard_filename(split, shard_index)
            if not (skip_existing and _validate_shard_arrays(shard_path, count, start, stop, cfg)):
                _write_shard(
                    cfg,
                    split,
                    start,
                    stop,
                    stats,
                    shard_path,
                    shard_dtype=signal_dtype,
                    raw_signal_dtype=raw_dtype,
                )
            manifest.shards.append(
                ShardInfo(split=split, path=shard_path.name, start=start, stop=stop, count=count)
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


def load_shard_manifest(
    dataset_dir: Path,
    cfg: RunConfig | None = None,
    *,
    validate_files: bool = True,
) -> ShardManifest:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing shard manifest: {path}")
    manifest = _manifest_from_json(json.loads(path.read_text(encoding="utf-8")))
    if manifest.schema_version != SCHEMA_VERSION or manifest.data_mode != "sharded":
        raise ValueError("unsupported shard manifest")
    if cfg is not None and manifest.config_hash != config_hash(cfg):
        raise ValueError("manifest does not match active run config")
    if not validate_files:
        return manifest
    for shard in manifest.shards:
        shard_path = dataset_dir / shard.path
        if cfg is None:
            if not shard_path.exists():
                raise FileNotFoundError(f"missing shard file: {shard_path}")
        elif not _validate_shard_arrays(shard_path, shard.count, shard.start, shard.stop, cfg):
            raise ValueError(f"invalid shard file: {shard_path}")
    return manifest


def _load_shard_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {
            "signals": np.asarray(arrays["signals"], dtype=np.float32),
            "raw_signals": np.asarray(arrays["raw_signals"], dtype=np.float32),
            "heatmaps": np.asarray(arrays["heatmaps"], dtype=np.float32),
            "nuclei_count": np.asarray(arrays["nuclei_count"], dtype=np.uint8),
            "nuclei_az_khz": np.asarray(arrays["nuclei_az_khz"], dtype=np.float32),
            "nuclei_aperp_khz": np.asarray(arrays["nuclei_aperp_khz"], dtype=np.float32),
            "indices": np.asarray(arrays["indices"], dtype=np.int64),
        }


def _sample_from_arrays(arrays: dict[str, np.ndarray], row: int, split: str) -> Sample:
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
        arrays = _load_shard_arrays(dataset_dir / shard.path)
        for row in range(shard.count):
            if max_samples is not None and emitted >= max_samples:
                return
            yield _sample_from_arrays(arrays, row, split)
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


class ShardedSaliDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        cfg: RunConfig,
        dataset_dir: Path,
        split: str,
        *,
        max_samples: int | None = None,
        epoch: int = 0,
        shuffle: bool = False,
        manifest: ShardManifest | None = None,
    ) -> None:
        self.cfg = cfg
        self.dataset_dir = dataset_dir
        self.split = split
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)
        self.manifest = manifest if manifest is not None else load_shard_manifest(dataset_dir, cfg)
        self.shards = [shard for shard in self.manifest.shards if shard.split == split]
        if not self.shards:
            raise ValueError(f"manifest contains no shards for split {split}")
        self.total_count = sum(shard.count for shard in self.shards)
        self.count = self.total_count
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.count = min(self.total_count, int(max_samples))

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        split_ids = {"train": 0, "val": 1, "test": 2}
        seed = int(
            np.random.SeedSequence(
                [self.cfg.data.seed, split_ids[self.split], self.epoch]
            ).generate_state(1)[0]
        )
        rng = np.random.default_rng(seed)
        if self.shuffle and self.count < self.total_count:
            shard_stops = np.array([shard.stop for shard in self.shards], dtype=np.int64)
            loaded_shard_pos: int | None = None
            arrays: dict[str, np.ndarray] | None = None
            for position, global_index in enumerate(rng.permutation(self.total_count)[: self.count]):
                if position % worker_count != worker_id:
                    continue
                shard_pos = int(np.searchsorted(shard_stops, int(global_index), side="right"))
                shard = self.shards[shard_pos]
                if loaded_shard_pos != shard_pos:
                    arrays = _load_shard_arrays(self.dataset_dir / shard.path)
                    loaded_shard_pos = shard_pos
                assert arrays is not None
                row = int(global_index) - shard.start
                signals = arrays["signals"][row]
                heatmap = arrays["heatmaps"][row]
                yield (
                    torch.from_numpy(signals[0:1]),
                    torch.from_numpy(signals[1:2]),
                    torch.from_numpy(heatmap),
                )
            return
        shard_order = np.arange(len(self.shards))
        if self.shuffle:
            rng.shuffle(shard_order)
        position = 0
        for shard_pos in shard_order:
            shard = self.shards[int(shard_pos)]
            arrays = _load_shard_arrays(self.dataset_dir / shard.path)
            row_order = np.arange(shard.count)
            if self.shuffle:
                rng.shuffle(row_order)
            for row in row_order:
                if position >= self.count:
                    return
                if position % worker_count == worker_id:
                    signals = arrays["signals"][int(row)]
                    heatmap = arrays["heatmaps"][int(row)]
                    yield (
                        torch.from_numpy(signals[0:1]),
                        torch.from_numpy(signals[1:2]),
                        torch.from_numpy(heatmap),
                    )
                position += 1
