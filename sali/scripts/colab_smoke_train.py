#!/usr/bin/env python
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_URL = "https://github.com/userwhe/sali-pycce-prototype.git"
BRANCH = "codex/sali-reproduction"
REPO_ROOT = Path("/content/sali-repo")
PROJECT_DIR = REPO_ROOT / "sali"
MY_DRIVE = Path("/content/drive/MyDrive")
DATASET_DIR = MY_DRIVE / "02_Research/DQP/sali/datasets/paper-low-full"
RUNS_ROOT = MY_DRIVE / "02_Research/DQP/sali/runs"
STAGED_DATASET_DIR = Path("/content/sali-smoke-dataset")
SMOKE_SHARDS = ("train-000000.npz", "val-000000.npz")


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


def preflight() -> None:
    if not MY_DRIVE.exists():
        raise RuntimeError("Google Drive is not mounted at /content/drive/MyDrive")
    if not DATASET_DIR.exists():
        raise RuntimeError(f"Dataset directory is not visible from Colab: {DATASET_DIR}")
    npz_count = sum(1 for _ in DATASET_DIR.glob("*.npz"))
    if npz_count != 360:
        raise RuntimeError(f"Expected 360 npz shards, found {npz_count}")


def stage_smoke_dataset() -> None:
    start = time.monotonic()
    STAGED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for name in SMOKE_SHARDS:
        source = DATASET_DIR / name
        target = STAGED_DATASET_DIR / name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            print(f"copying {source} -> {target}", flush=True)
            shutil.copy2(source, target)
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    keep = set(SMOKE_SHARDS)
    manifest["shards"] = [item for item in manifest["shards"] if item["path"] in keep]
    (STAGED_DATASET_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    elapsed = time.monotonic() - start
    print(f"staged_smoke_dataset: {STAGED_DATASET_DIR} ({elapsed:.1f}s)", flush=True)


def main() -> None:
    ensure_repo()
    preflight()
    stage_smoke_dataset()
    sys.path.insert(0, str(PROJECT_DIR / "src"))

    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    from sali.config import paper_config
    from sali.model import SaliNet
    from sali.shards import ShardedSaliDataset
    from sali.train import _set_torch_seed, choose_device, make_heatmap_loss

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = RUNS_ROOT / f"colab-smoke-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    cfg = paper_config("low")
    cfg.output_dir = output_dir
    cfg.training.batch_size = 8
    cfg.training.device = "auto"
    _set_torch_seed(cfg.data.seed)

    device = choose_device(cfg.training.device)
    model = SaliNet(cfg.model).to(device)
    criterion = make_heatmap_loss(cfg.training)
    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)
    train_loader = DataLoader(
        ShardedSaliDataset(cfg, STAGED_DATASET_DIR, "train", max_samples=16, epoch=1, shuffle=False),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        ShardedSaliDataset(cfg, STAGED_DATASET_DIR, "val", max_samples=8, shuffle=False),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    def run_epoch(loader: DataLoader, *, train: bool) -> float:
        model.train(train)
        total_loss = 0.0
        total_count = 0
        for signal_32, signal_256, heatmap in loader:
            signal_32 = signal_32.to(device)
            signal_256 = signal_256.to(device)
            heatmap = heatmap.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(signal_32, signal_256)
            loss = criterion(prediction, heatmap)
            if train:
                loss.backward()
                optimizer.step()
            batch_count = int(heatmap.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count
        if total_count == 0:
            raise RuntimeError("No batches were emitted by the smoke DataLoader")
        return total_loss / total_count

    train_loss = run_epoch(train_loader, train=True)
    with torch.no_grad():
        val_loss = run_epoch(val_loader, train=False)
    if not math.isfinite(train_loss) or not math.isfinite(val_loss):
        raise RuntimeError(f"Non-finite smoke loss: train={train_loss}, val={val_loss}")

    checkpoint_path = output_dir / "smoke_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    result = {
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_dir": str(DATASET_DIR),
        "staged_dataset_dir": str(STAGED_DATASET_DIR),
        "train_examples": 16,
        "val_examples": 8,
        "batch_size": cfg.training.batch_size,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"smoke_output_dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
