from __future__ import annotations

import numpy as np
import pytest
import torch

import sali.data as data_module
from sali.config import paper_config
from sali.data import (
    NormalizationStats,
    Sample,
    SaliDataset,
    StreamedSaliDataset,
    estimate_normalization_stats,
    generate_splits,
    materialize_streamed_samples,
)
from sali.physics import Couplings


def _raise_if_generating(*_args, **_kwargs) -> None:
    raise AssertionError("sample generation should not start")


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
    raw_train = np.stack([train_sample.raw_signals for train_sample in splits.train], axis=0)
    assert stats.mean == pytest.approx(float(raw_train.mean()))
    assert stats.var == pytest.approx(float(raw_train.var()))
    np.testing.assert_allclose(sample.signals, stats.normalize(sample.raw_signals))
    assert not np.shares_memory(sample.signals, sample.raw_signals)
    assert np.min(sample.raw_signals) >= 0.0
    assert np.max(sample.raw_signals) <= 1.0


def test_dataset_returns_torch_ready_arrays(tiny_config) -> None:
    splits, _stats = generate_splits(tiny_config)
    dataset = SaliDataset(splits.train)
    signal32, signal256, heatmap = dataset[0]
    assert tuple(signal32.shape) == (1, 1000)
    assert tuple(signal256.shape) == (1, 1000)
    assert tuple(heatmap.shape) == (1, 204, 104)
    assert signal32.dtype == torch.float32
    assert signal256.dtype == torch.float32
    assert heatmap.dtype == torch.float32


def test_dataset_tensors_share_memory_with_sample_arrays(tiny_config) -> None:
    splits, _stats = generate_splits(tiny_config)
    sample = splits.train[0]
    signal32, _signal256, heatmap = SaliDataset(splits.train)[0]

    signal_value = float(sample.signals[0, 0]) + 1.0
    heatmap_value = float(sample.heatmap[0, 0, 0]) + 1.0
    signal32[0, 0] = signal_value
    heatmap[0, 0, 0] = heatmap_value

    assert sample.signals[0, 0] == pytest.approx(signal_value)
    assert sample.heatmap[0, 0, 0] == pytest.approx(heatmap_value)


def test_generate_splits_rejects_paper_scale_before_generation(monkeypatch) -> None:
    cfg = paper_config(field="low")
    monkeypatch.setattr(data_module, "_generate_raw_samples", _raise_if_generating)

    with pytest.raises((MemoryError, ValueError), match="materialized.*practical.*streaming/sharded"):
        generate_splits(cfg)


@pytest.mark.parametrize("norm_epsilon", [0.0, -0.1, float("nan")])
def test_generate_splits_validates_norm_epsilon_before_generation(
    tiny_config,
    monkeypatch,
    norm_epsilon: float,
) -> None:
    tiny_config.data.norm_epsilon = norm_epsilon
    monkeypatch.setattr(data_module, "_generate_raw_samples", _raise_if_generating)

    with pytest.raises(ValueError, match="norm_epsilon"):
        generate_splits(tiny_config)


@pytest.mark.parametrize(
    ("min_nuclei", "max_nuclei"),
    [
        (0, 3),
        (4, 3),
    ],
)
def test_generate_splits_validates_nuclei_range_before_generation(
    tiny_config,
    monkeypatch,
    min_nuclei: int,
    max_nuclei: int,
) -> None:
    tiny_config.data.min_nuclei = min_nuclei
    tiny_config.data.max_nuclei = max_nuclei
    monkeypatch.setattr(data_module, "_generate_raw_samples", _raise_if_generating)

    with pytest.raises(ValueError, match="nuclei"):
        generate_splits(tiny_config)


@pytest.mark.parametrize(
    ("low_attr", "high_attr"),
    [
        ("az_min_khz", "az_max_khz"),
        ("aperp_min_khz", "aperp_max_khz"),
    ],
)
def test_generate_splits_validates_coupling_ranges_before_generation(
    tiny_config,
    monkeypatch,
    low_attr: str,
    high_attr: str,
) -> None:
    setattr(tiny_config.data, low_attr, 5.0)
    setattr(tiny_config.data, high_attr, 5.0)
    monkeypatch.setattr(data_module, "_generate_raw_samples", _raise_if_generating)

    with pytest.raises(ValueError, match="range"):
        generate_splits(tiny_config)


def test_validation_and_test_splits_do_not_depend_on_train_sample_count(tiny_config) -> None:
    base_splits, _base_stats = generate_splits(tiny_config)
    changed_config = tiny_config
    changed_config.data.train_samples += 1

    changed_splits, _changed_stats = generate_splits(changed_config)

    for base, changed in zip(base_splits.val, changed_splits.val, strict=True):
        np.testing.assert_array_equal(base.raw_signals, changed.raw_signals)
    for base, changed in zip(base_splits.test, changed_splits.test, strict=True):
        np.testing.assert_array_equal(base.raw_signals, changed.raw_signals)


def test_generate_splits_repeated_seed_is_identical(tiny_config) -> None:
    first_splits, _first_stats = generate_splits(tiny_config)
    second_splits, _second_stats = generate_splits(tiny_config)

    for first, second in zip(first_splits.train, second_splits.train, strict=True):
        np.testing.assert_array_equal(first.raw_signals, second.raw_signals)
        np.testing.assert_array_equal(first.signals, second.signals)
        np.testing.assert_array_equal(first.heatmap, second.heatmap)
        np.testing.assert_array_equal(first.nuclei.az_khz, second.nuclei.az_khz)
        np.testing.assert_array_equal(first.nuclei.aperp_khz, second.nuclei.aperp_khz)


def test_streamed_dataset_returns_torch_ready_arrays(tiny_config) -> None:
    stats = estimate_normalization_stats(tiny_config, sample_limit=4)
    dataset = StreamedSaliDataset(tiny_config, "train", stats)
    signal32, signal256, heatmap = next(iter(dataset))

    assert len(dataset) == tiny_config.data.train_samples
    assert tuple(signal32.shape) == (1, 1000)
    assert tuple(signal256.shape) == (1, 1000)
    assert tuple(heatmap.shape) == (1, 204, 104)
    assert signal32.dtype == torch.float32
    assert signal256.dtype == torch.float32
    assert heatmap.dtype == torch.float32


def test_streamed_dataset_max_samples_shuffles_from_full_split(tiny_config, monkeypatch) -> None:
    tiny_config.data.train_samples = 8
    stats = NormalizationStats(mean=0.0, var=1.0, epsilon=tiny_config.data.norm_epsilon)
    calls: list[int] = []

    def fake_generate_indexed_sample(cfg, split, index, sample_stats):
        calls.append(int(index))
        return Sample(
            signals=np.zeros((2, cfg.physics.signal_points), dtype=np.float32),
            raw_signals=np.zeros((2, cfg.physics.signal_points), dtype=np.float32),
            heatmap=np.zeros((1, cfg.model.output_height, cfg.model.output_width), dtype=np.float32),
            nuclei=Couplings(
                az_khz=np.array([1.0], dtype=np.float32),
                aperp_khz=np.array([2.0], dtype=np.float32),
            ),
            split=split,
        )

    monkeypatch.setattr(data_module, "generate_indexed_sample", fake_generate_indexed_sample)

    dataset = StreamedSaliDataset(tiny_config, "train", stats, max_samples=3, epoch=1, shuffle=True)
    list(dataset)

    assert len(dataset) == 3
    assert calls == [7, 1, 4]


def test_streamed_samples_repeated_seed_is_identical(tiny_config) -> None:
    stats = estimate_normalization_stats(tiny_config, sample_limit=4)

    first = materialize_streamed_samples(tiny_config, "train", stats, max_samples=3)
    second = materialize_streamed_samples(tiny_config, "train", stats, max_samples=3)

    for first_sample, second_sample in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_sample.raw_signals, second_sample.raw_signals)
        np.testing.assert_array_equal(first_sample.signals, second_sample.signals)
        np.testing.assert_array_equal(first_sample.heatmap, second_sample.heatmap)
        np.testing.assert_array_equal(first_sample.nuclei.az_khz, second_sample.nuclei.az_khz)
        np.testing.assert_array_equal(first_sample.nuclei.aperp_khz, second_sample.nuclei.aperp_khz)


def test_streamed_validation_and_test_do_not_depend_on_train_sample_count(tiny_config) -> None:
    stats = estimate_normalization_stats(tiny_config, sample_limit=4)
    base_val = materialize_streamed_samples(tiny_config, "val", stats, max_samples=2)
    base_test = materialize_streamed_samples(tiny_config, "test", stats, max_samples=2)

    changed_config = tiny_config
    changed_config.data.train_samples += 5
    changed_stats = estimate_normalization_stats(changed_config, sample_limit=4)
    changed_val = materialize_streamed_samples(changed_config, "val", changed_stats, max_samples=2)
    changed_test = materialize_streamed_samples(changed_config, "test", changed_stats, max_samples=2)

    for base, changed in zip(base_val, changed_val, strict=True):
        np.testing.assert_array_equal(base.raw_signals, changed.raw_signals)
    for base, changed in zip(base_test, changed_test, strict=True):
        np.testing.assert_array_equal(base.raw_signals, changed.raw_signals)


def test_estimate_normalization_stats_uses_requested_sample_limit(tiny_config, monkeypatch) -> None:
    calls: list[int] = []
    original = data_module.generate_indexed_sample

    def tracking_generate_indexed_sample(*args, **kwargs):
        calls.append(int(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(data_module, "generate_indexed_sample", tracking_generate_indexed_sample)

    stats = estimate_normalization_stats(tiny_config, sample_limit=3)

    assert np.isfinite(stats.mean)
    assert np.isfinite(stats.var)
    assert calls == [0, 1, 2]
