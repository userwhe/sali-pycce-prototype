from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from sali.config import RunConfig
from sali.physics import Couplings, generate_sample_signals
from sali.targets import render_heatmap


MATERIALIZED_SAMPLE_LIMIT = 100_000


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


def _validate_generation_config(cfg: RunConfig) -> None:
    data = cfg.data
    if min(data.train_samples, data.val_samples, data.test_samples) <= 0:
        raise ValueError("train, validation, and test sample counts must be positive")
    if data.total_samples > MATERIALIZED_SAMPLE_LIMIT:
        raise ValueError(
            "materialized generation is intended for practical runs; "
            f"total_samples={data.total_samples} exceeds limit={MATERIALIZED_SAMPLE_LIMIT}. "
            "Paper-scale generation requires a streaming/sharded pipeline."
        )
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
