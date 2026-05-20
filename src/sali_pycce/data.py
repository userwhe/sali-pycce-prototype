"""Torch datasets for SALI-style synthetic training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap import HeatmapSpec, make_heatmap
from .physics import AnalyticCPMGSimulator, random_spins, spins_to_array


@dataclass
class SyntheticConfig:
    train_samples: int = 10_000
    val_samples: int = 1_000
    max_spins: int = 20
    min_spins: int = 1
    az_range: tuple[float, float] = (-100.0, 100.0)
    aperp_range: tuple[float, float] = (2.0, 102.0)
    seed: int = 1234


class SyntheticSALIDataset(Dataset):
    """On-the-fly synthetic dataset.

    Every integer index deterministically maps to a random spin configuration,
    signal pair, and heatmap. This avoids storing a large dataset while keeping
    validation reproducible.
    """

    def __init__(
        self,
        length: int,
        simulator: AnalyticCPMGSimulator | None = None,
        heatmap_spec: HeatmapSpec | None = None,
        max_spins: int = 20,
        min_spins: int = 1,
        az_range: tuple[float, float] = (-100.0, 100.0),
        aperp_range: tuple[float, float] = (2.0, 102.0),
        seed: int = 1234,
        noisy: bool = True,
    ) -> None:
        self.length = int(length)
        self.simulator = simulator or AnalyticCPMGSimulator()
        self.heatmap_spec = heatmap_spec or HeatmapSpec()
        self.max_spins = int(max_spins)
        self.min_spins = int(min_spins)
        self.az_range = az_range
        self.aperp_range = aperp_range
        self.seed = int(seed)
        self.noisy = bool(noisy)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(idx))
        n_spins = int(rng.integers(self.min_spins, self.max_spins + 1))
        spins = random_spins(rng, n_spins, self.az_range, self.aperp_range)
        sample = self.simulator.sample(spins, rng=rng, noisy=self.noisy)
        heatmap = make_heatmap(spins, self.heatmap_spec)
        spin_array = spins_to_array(spins)

        # Pad variable-length spin list for inspection/evaluation.
        padded = np.full((self.max_spins, 2), np.nan, dtype=np.float32)
        padded[: len(spin_array)] = spin_array

        return {
            "signals": torch.from_numpy(sample["signals"]).float(),
            "heatmap": torch.from_numpy(heatmap[None, :, :]).float(),
            "spins": torch.from_numpy(padded).float(),
            "n_spins": torch.tensor(n_spins, dtype=torch.long),
        }
