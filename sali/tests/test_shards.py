from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sali.data import generate_indexed_sample
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


def test_sharded_dataset_capped_shuffle_uses_global_sample_order(tiny_config, tmp_path) -> None:
    tiny_config.data.train_samples = 8
    dataset_dir = tmp_path / "dataset"
    manifest = generate_shards(tiny_config, dataset_dir, shard_size=4, normalization_samples=4)

    dataset = ShardedSaliDataset(
        tiny_config,
        dataset_dir,
        "train",
        max_samples=3,
        epoch=1,
        shuffle=True,
    )
    signal32, signal256, heatmap = next(iter(dataset))
    expected = generate_indexed_sample(tiny_config, "train", 7, manifest.normalization_stats)

    assert len(dataset) == 3
    np.testing.assert_array_equal(signal32.numpy(), expected.signals[0:1])
    np.testing.assert_array_equal(signal256.numpy(), expected.signals[1:2])
    np.testing.assert_array_equal(heatmap.numpy(), expected.heatmap)


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
