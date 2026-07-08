#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/userwhe/sali-pycce-prototype.git"
BRANCH = "codex/sali-reproduction"
REPO_ROOT = Path("/content/sali-repo")
PROJECT_DIR = REPO_ROOT / "sali"
MY_DRIVE = Path("/content/drive/MyDrive")
DATASET_DIR = MY_DRIVE / "02_Research/DQP/sali/datasets/paper-low-full"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo() -> None:
    if not (REPO_ROOT / ".git").exists():
        run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_ROOT)])
    else:
        print(f"using existing repo clone: {REPO_ROOT}", flush=True)
        run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT)
    if not PROJECT_DIR.exists():
        raise RuntimeError(f"Expected project directory does not exist: {PROJECT_DIR}")


def install_project() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=PROJECT_DIR)


def check_runtime() -> None:
    import torch

    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"torch: {torch.__version__}", flush=True)
    print(f"cuda_available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}", flush=True)


def check_drive_dataset() -> None:
    print(f"drive_mounted: {MY_DRIVE.exists()} ({MY_DRIVE})", flush=True)
    if not MY_DRIVE.exists():
        raise RuntimeError(
            "Google Drive is not mounted in this Colab runtime. "
            "Open the Colab URL and run: from google.colab import drive; drive.mount('/content/drive')"
        )
    print(f"dataset_dir: {DATASET_DIR}", flush=True)
    if not DATASET_DIR.exists():
        raise RuntimeError(f"Dataset directory is not visible from Colab: {DATASET_DIR}")
    npz_count = sum(1 for _ in DATASET_DIR.glob("*.npz"))
    print(f"npz_count: {npz_count}", flush=True)
    print(f"manifest_exists: {(DATASET_DIR / 'manifest.json').exists()}", flush=True)
    print(f"generation_plan_exists: {(DATASET_DIR / 'generation_plan.json').exists()}", flush=True)
    if npz_count != 360:
        raise RuntimeError(f"Expected 360 npz shards, found {npz_count}")


def main() -> None:
    ensure_repo()
    install_project()
    check_runtime()
    check_drive_dataset()


if __name__ == "__main__":
    main()
