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
STAGED_DATASET_DIR = Path("/content/sali-stability-dataset")
TRAIN_SHARD = "train-000000.npz"
VAL_SHARD = "val-000000.npz"
BATCH_SIZE = 256
LEARNING_RATE = 0.002
TRAIN_EXAMPLES = 8192
VAL_EXAMPLES = 1024


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo() -> None:
    if not (REPO_ROOT / ".git").exists():
        run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_ROOT)])
    else:
        print(f"using existing repo clone: {REPO_ROOT}", flush=True)
        run(["git", "fetch", "origin", BRANCH], cwd=REPO_ROOT)
        run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO_ROOT)
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


def copy_if_needed(source: Path, target: Path) -> None:
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(source, tmp)
    tmp.replace(target)


def stage_dataset() -> None:
    start = time.monotonic()
    STAGED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for name in (TRAIN_SHARD, VAL_SHARD):
        source = DATASET_DIR / name
        target = STAGED_DATASET_DIR / name
        print(f"copying {source} -> {target}", flush=True)
        copy_if_needed(source, target)
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    keep = {TRAIN_SHARD, VAL_SHARD}
    manifest["shards"] = [item for item in manifest["shards"] if item["path"] in keep]
    (STAGED_DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elapsed = time.monotonic() - start
    print(f"staged_stability_dataset: {STAGED_DATASET_DIR} ({elapsed:.1f}s)", flush=True)


def main() -> None:
    ensure_repo()
    preflight()
    stage_dataset()
    sys.path.insert(0, str(PROJECT_DIR / "src"))

    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    from sali.config import paper_config
    from sali.model import SaliNet
    from sali.shards import ShardedSaliDataset, load_shard_manifest
    from sali.train import _set_torch_seed, choose_device, make_heatmap_loss

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = RUNS_ROOT / f"colab-stability-bs256-lr2x-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    cfg = paper_config("low")
    cfg.output_dir = output_dir
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.learning_rate = LEARNING_RATE
    cfg.training.device = "auto"
    _set_torch_seed(cfg.data.seed)

    manifest = load_shard_manifest(STAGED_DATASET_DIR, cfg, validate_files=False)
    device = choose_device(cfg.training.device)
    model = SaliNet(cfg.model).to(device)
    criterion = make_heatmap_loss(cfg.training)
    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)
    train_loader = DataLoader(
        ShardedSaliDataset(
            cfg,
            STAGED_DATASET_DIR,
            "train",
            max_samples=TRAIN_EXAMPLES,
            epoch=1,
            shuffle=True,
            manifest=manifest,
        ),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )
    val_loader = DataLoader(
        ShardedSaliDataset(
            cfg,
            STAGED_DATASET_DIR,
            "val",
            max_samples=VAL_EXAMPLES,
            shuffle=False,
            manifest=manifest,
        ),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=2,
    )

    train_losses: list[float] = []
    start = time.monotonic()
    model.train(True)
    for signal_32, signal_256, heatmap in train_loader:
        signal_32 = signal_32.to(device)
        signal_256 = signal_256.to(device)
        heatmap = heatmap.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(signal_32, signal_256)
        loss = criterion(prediction, heatmap)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    val_losses: list[float] = []
    model.train(False)
    with torch.no_grad():
        for signal_32, signal_256, heatmap in val_loader:
            signal_32 = signal_32.to(device)
            signal_256 = signal_256.to(device)
            heatmap = heatmap.to(device)
            prediction = model(signal_32, signal_256)
            loss = criterion(prediction, heatmap)
            if not torch.isfinite(loss):
                raise FloatingPointError("validation loss became non-finite")
            val_losses.append(float(loss.detach().cpu()))
    elapsed = time.monotonic() - start
    if not train_losses:
        raise RuntimeError("No train batches were emitted")
    if not val_losses:
        raise RuntimeError("No validation batches were emitted")
    if any(not math.isfinite(value) for value in [*train_losses, *val_losses]):
        raise RuntimeError("Non-finite loss detected")

    gpu_status = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint_path = output_dir / "stability_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    result = {
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_dir": str(DATASET_DIR),
        "staged_dataset_dir": str(STAGED_DATASET_DIR),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_examples": TRAIN_EXAMPLES,
        "val_examples": VAL_EXAMPLES,
        "train_batches": len(train_losses),
        "val_batches": len(val_losses),
        "first_train_loss": train_losses[0],
        "last_train_loss": train_losses[-1],
        "mean_train_loss": sum(train_losses) / len(train_losses),
        "mean_val_loss": sum(val_losses) / len(val_losses),
        "elapsed_seconds": elapsed,
        "gpu_status": gpu_status,
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "stability_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"stability_output_dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
