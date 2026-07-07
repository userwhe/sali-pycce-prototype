from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shard_jobs.py"


def _base_cmd(dataset_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--preset",
        "practical",
        "--field",
        "low",
        "--dataset-dir",
        str(dataset_dir),
        "--train-samples",
        "8",
        "--val-samples",
        "4",
        "--test-samples",
        "4",
        "--shard-size",
        "4",
        "--normalization-samples",
        "4",
    ]


def test_shard_jobs_cli_prepares_writes_and_finalizes_dataset(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"

    prepare = subprocess.run(
        [*_base_cmd(dataset_dir), "prepare"],
        check=True,
        text=True,
        capture_output=True,
    )
    write = subprocess.run(
        [*_base_cmd(dataset_dir), "write", "--shard-id", "0"],
        check=True,
        text=True,
        capture_output=True,
    )
    for shard_id in (1, 2, 3):
        subprocess.run(
            [*_base_cmd(dataset_dir), "write", "--shard-id", str(shard_id)],
            check=True,
            text=True,
            capture_output=True,
        )
    finalize = subprocess.run(
        [*_base_cmd(dataset_dir), "finalize"],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "prepared 4 shards" in prepare.stdout
    assert "wrote train-000000.npz" in write.stdout
    assert "finalized 4 shards" in finalize.stdout
    assert manifest["split_sizes"] == {"train": 8, "val": 4, "test": 4}


def test_shard_jobs_cli_validate_reports_existing_manifest(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    subprocess.run([*_base_cmd(dataset_dir), "prepare"], check=True, text=True, capture_output=True)
    for shard_id in range(4):
        subprocess.run(
            [*_base_cmd(dataset_dir), "write", "--shard-id", str(shard_id)],
            check=True,
            text=True,
            capture_output=True,
        )
    subprocess.run([*_base_cmd(dataset_dir), "finalize"], check=True, text=True, capture_output=True)

    completed = subprocess.run(
        [*_base_cmd(dataset_dir), "validate"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "validated 4 shards" in completed.stdout
