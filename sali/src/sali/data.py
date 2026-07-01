from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from sali.config import RunConfig
from sali.physics import Couplings, generate_sample_signals
from sali.targets import render_heatmap


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


def generate_splits(cfg: RunConfig) -> tuple[DataSplits, NormalizationStats]:
    if min(cfg.data.train_samples, cfg.data.val_samples, cfg.data.test_samples) <= 0:
        raise ValueError("train, validation, and test sample counts must be positive")
    rng = np.random.default_rng(cfg.data.seed)
    train = _generate_raw_samples(cfg, "train", cfg.data.train_samples, rng)
    val = _generate_raw_samples(cfg, "val", cfg.data.val_samples, rng)
    test = _generate_raw_samples(cfg, "test", cfg.data.test_samples, rng)
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
        signals = sample.signals.astype(np.float32)
        heatmap = sample.heatmap.astype(np.float32)
        signal32 = torch.from_numpy(signals[0:1])
        signal256 = torch.from_numpy(signals[1:2])
        target = torch.from_numpy(heatmap)
        return signal32, signal256, target
