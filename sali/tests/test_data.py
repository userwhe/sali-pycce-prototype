from __future__ import annotations

import numpy as np

from sali.data import SaliDataset, generate_splits


def test_generate_splits_sizes_and_shapes(tiny_config) -> None:
    splits, stats = generate_splits(tiny_config)
    assert len(splits.train) == tiny_config.data.train_samples
    assert len(splits.val) == tiny_config.data.val_samples
    assert len(splits.test) == tiny_config.data.test_samples
    sample = splits.train[0]
    assert sample.signals.shape == (2, 1000)
    assert sample.raw_signals.shape == (2, 1000)
    assert sample.heatmap.shape == (1, 204, 104)
    assert sample.nuclei.count >= 1
    assert np.isfinite(stats.mean)
    assert np.isfinite(stats.var)


def test_dataset_returns_torch_ready_arrays(tiny_config) -> None:
    splits, _stats = generate_splits(tiny_config)
    dataset = SaliDataset(splits.train)
    signal32, signal256, heatmap = dataset[0]
    assert tuple(signal32.shape) == (1, 1000)
    assert tuple(signal256.shape) == (1, 1000)
    assert tuple(heatmap.shape) == (1, 204, 104)
