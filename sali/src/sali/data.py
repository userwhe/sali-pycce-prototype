from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from sali.config import RunConfig
from sali.physics import Couplings, generate_sample_signals
from sali.targets import render_heatmap


MATERIALIZED_SAMPLE_LIMIT = 100_000
_SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@dataclass(slots=True)
class Sample:
    signals: np.ndarray
    raw_signals: np.ndarray
    heatmap: np.ndarray
    nuclei: Couplings
    split: str


@dataclass(slots=True)
class NormalizationStats:
    mean: float
    var: float
    epsilon: float

    def normalize(self, signals: np.ndarray) -> np.ndarray:
        return ((signals - self.mean) / np.sqrt(self.var + self.epsilon)).astype(np.float32)


@dataclass(slots=True)
class DataSplits:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def sample_couplings(cfg: RunConfig, rng: np.random.Generator) -> Couplings:
    count = int(rng.integers(cfg.data.min_nuclei, cfg.data.max_nuclei + 1))
    az = rng.uniform(cfg.data.az_min_khz, cfg.data.az_max_khz, size=count).astype(np.float32)
    aperp = rng.uniform(cfg.data.aperp_min_khz, cfg.data.aperp_max_khz, size=count).astype(np.float32)
    return Couplings(az_khz=az, aperp_khz=aperp)


def _generate_raw_samples(cfg: RunConfig, split: str, count: int, rng: np.random.Generator) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(count):
        nuclei = sample_couplings(cfg, rng)
        raw_signals = generate_sample_signals(nuclei, cfg.physics, rng)
        heatmap = render_heatmap(nuclei, cfg.data, cfg.model)
        samples.append(
            Sample(
                signals=raw_signals.copy(),
                raw_signals=raw_signals,
                heatmap=heatmap,
                nuclei=nuclei,
                split=split,
            )
        )
    return samples


def _split_count(cfg: RunConfig, split: str) -> int:
    if split == "train":
        return cfg.data.train_samples
    if split == "val":
        return cfg.data.val_samples
    if split == "test":
        return cfg.data.test_samples
    raise ValueError("split must be one of: train, val, test")


def _sample_seed(cfg: RunConfig, split: str, index: int) -> np.random.SeedSequence:
    if split not in _SPLIT_IDS:
        raise ValueError("split must be one of: train, val, test")
    if index < 0:
        raise ValueError("sample index must be non-negative")
    return np.random.SeedSequence([int(cfg.data.seed), _SPLIT_IDS[split], int(index)])


def generate_indexed_sample(
    cfg: RunConfig,
    split: str,
    index: int,
    stats: NormalizationStats | None = None,
    *,
    include_heatmap: bool = True,
) -> Sample:
    count = _split_count(cfg, split)
    if index >= count:
        raise IndexError(f"{split} sample index {index} is out of range for count {count}")
    rng = np.random.default_rng(_sample_seed(cfg, split, index))
    nuclei = sample_couplings(cfg, rng)
    raw_signals = generate_sample_signals(nuclei, cfg.physics, rng)
    signals = stats.normalize(raw_signals) if stats is not None else raw_signals.copy()
    heatmap = (
        render_heatmap(nuclei, cfg.data, cfg.model)
        if include_heatmap
        else np.empty((0,), dtype=np.float32)
    )
    return Sample(
        signals=signals,
        raw_signals=raw_signals,
        heatmap=heatmap,
        nuclei=nuclei,
        split=split,
    )


def _iter_indices(count: int, *, seed: int, shuffle: bool) -> Iterator[int]:
    if shuffle:
        for index in np.random.default_rng(seed).permutation(count):
            yield int(index)
    else:
        yield from range(count)


def iter_streamed_samples(
    cfg: RunConfig,
    split: str,
    stats: NormalizationStats,
    *,
    max_samples: int | None = None,
    epoch: int = 0,
    shuffle: bool = False,
) -> Iterator[Sample]:
    total_count = _split_count(cfg, split)
    count = total_count
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        count = min(total_count, int(max_samples))
    seed = int(np.random.SeedSequence([cfg.data.seed, _SPLIT_IDS[split], epoch]).generate_state(1)[0])
    for position, index in enumerate(_iter_indices(total_count, seed=seed, shuffle=shuffle)):
        if position >= count:
            break
        yield generate_indexed_sample(cfg, split, index, stats)


def estimate_normalization_stats(cfg: RunConfig, sample_limit: int) -> NormalizationStats:
    _validate_streaming_config(cfg)
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")
    sample_count = min(int(sample_limit), cfg.data.train_samples)
    total = 0.0
    squared_total = 0.0
    total_values = 0
    for index in range(sample_count):
        sample = generate_indexed_sample(cfg, "train", index, include_heatmap=False)
        raw = sample.raw_signals.astype(np.float64, copy=False)
        total += float(raw.sum())
        squared_total += float(np.square(raw).sum())
        total_values += int(raw.size)
    mean = total / total_values
    var = max(0.0, (squared_total / total_values) - (mean * mean))
    return NormalizationStats(mean=float(mean), var=float(var), epsilon=cfg.data.norm_epsilon)


def materialize_streamed_samples(
    cfg: RunConfig,
    split: str,
    stats: NormalizationStats,
    *,
    max_samples: int,
) -> list[Sample]:
    return list(iter_streamed_samples(cfg, split, stats, max_samples=max_samples))


def _stats_from_train(samples: list[Sample], epsilon: float) -> NormalizationStats:
    stacked = np.stack([sample.signals for sample in samples], axis=0)
    return NormalizationStats(
        mean=float(stacked.mean()),
        var=float(stacked.var()),
        epsilon=epsilon,
    )


def _apply_normalization(samples: list[Sample], stats: NormalizationStats) -> None:
    for sample in samples:
        sample.signals = stats.normalize(sample.signals)


def _validate_common_data_config(cfg: RunConfig) -> None:
    data = cfg.data
    if min(data.train_samples, data.val_samples, data.test_samples) <= 0:
        raise ValueError("train, validation, and test sample counts must be positive")
    if not np.isfinite(data.norm_epsilon) or data.norm_epsilon <= 0.0:
        raise ValueError("norm_epsilon must be finite and greater than zero")
    if data.min_nuclei < 1:
        raise ValueError("min_nuclei must be at least 1")
    if data.max_nuclei < data.min_nuclei:
        raise ValueError("max_nuclei must be greater than or equal to min_nuclei")
    if not np.isfinite(data.az_min_khz) or not np.isfinite(data.az_max_khz) or data.az_max_khz <= data.az_min_khz:
        raise ValueError("Az range must be finite and increasing")
    if (
        not np.isfinite(data.aperp_min_khz)
        or not np.isfinite(data.aperp_max_khz)
        or data.aperp_max_khz <= data.aperp_min_khz
    ):
        raise ValueError("Aperp range must be finite and increasing")


def _validate_streaming_config(cfg: RunConfig) -> None:
    _validate_common_data_config(cfg)


def _validate_generation_config(cfg: RunConfig) -> None:
    _validate_common_data_config(cfg)
    data = cfg.data
    if data.total_samples > MATERIALIZED_SAMPLE_LIMIT:
        raise ValueError(
            "materialized generation is intended for practical runs; "
            f"total_samples={data.total_samples} exceeds limit={MATERIALIZED_SAMPLE_LIMIT}. "
            "Paper-scale generation requires a streaming/sharded pipeline."
        )


def generate_splits(cfg: RunConfig) -> tuple[DataSplits, NormalizationStats]:
    _validate_generation_config(cfg)
    train_seed, val_seed, test_seed = np.random.SeedSequence(cfg.data.seed).spawn(3)
    train = _generate_raw_samples(cfg, "train", cfg.data.train_samples, np.random.default_rng(train_seed))
    val = _generate_raw_samples(cfg, "val", cfg.data.val_samples, np.random.default_rng(val_seed))
    test = _generate_raw_samples(cfg, "test", cfg.data.test_samples, np.random.default_rng(test_seed))
    stats = _stats_from_train(train, cfg.data.norm_epsilon)
    _apply_normalization(train, stats)
    _apply_normalization(val, stats)
    _apply_normalization(test, stats)
    return DataSplits(train=train, val=val, test=test), stats


class SaliDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        signal32 = torch.from_numpy(sample.signals[0:1])
        signal256 = torch.from_numpy(sample.signals[1:2])
        target = torch.from_numpy(sample.heatmap)
        return signal32, signal256, target


class StreamedSaliDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        cfg: RunConfig,
        split: str,
        stats: NormalizationStats,
        *,
        max_samples: int | None = None,
        epoch: int = 0,
        shuffle: bool = False,
    ) -> None:
        _validate_streaming_config(cfg)
        self.cfg = cfg
        self.split = split
        self.stats = stats
        self.total_count = _split_count(cfg, split)
        self.count = self.total_count
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.count = min(self.total_count, int(max_samples))
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        seed = int(
            np.random.SeedSequence(
                [self.cfg.data.seed, _SPLIT_IDS[self.split], self.epoch]
            ).generate_state(1)[0]
        )
        for position, index in enumerate(_iter_indices(self.total_count, seed=seed, shuffle=self.shuffle)):
            if position >= self.count:
                break
            if position % worker_count != worker_id:
                continue
            sample = generate_indexed_sample(self.cfg, self.split, index, self.stats)
            yield (
                torch.from_numpy(sample.signals[0:1]),
                torch.from_numpy(sample.signals[1:2]),
                torch.from_numpy(sample.heatmap),
            )
