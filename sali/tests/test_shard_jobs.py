from __future__ import annotations

import json
import os

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
    sentinel_mtime_ns = 946684800_000_000_000
    os.utime(first_path, ns=(sentinel_mtime_ns, sentinel_mtime_ns))
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
