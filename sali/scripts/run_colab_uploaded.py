#!/usr/bin/env python
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ARCHIVE = Path("/content/sali-colab.tar.gz")
PROJECT = Path("/content/sali")
OUTPUT = Path("/content/sali-runs/practical-low")


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = (destination_root / member.name).resolve()
            if destination_root != member_path and destination_root not in member_path.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
        try:
            tar.extractall(destination_root, members=members, filter="data")
        except TypeError:
            tar.extractall(destination_root, members=members)


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Expected uploaded archive at {ARCHIVE}")
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    extract_archive(ARCHIVE, Path("/content"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT)], check=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts" / "train_colab.py"),
            "--preset",
            "practical",
            "--field",
            "low",
            "--output-dir",
            str(OUTPUT),
            "--device",
            "auto",
            "--train-samples",
            "64",
            "--val-samples",
            "16",
            "--test-samples",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--max-eval-samples",
            "4",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
