#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path


REPO_URL = "https://github.com/userwhe/sali-pycce-prototype.git"
BRANCH = "codex/sali-reproduction"
REPO_ROOT = Path("/content/sali-repo")
PROJECT_DIR = REPO_ROOT / "sali"
MY_DRIVE = Path("/content/drive/MyDrive")
DATASET_DIR = MY_DRIVE / "02_Research/DQP/sali/datasets/paper-low-full"
STAGED_DATASET_DIR = Path("/content/sali-paper-low-full")
RUNS_ROOT = MY_DRIVE / "02_Research/DQP/sali/runs"
WORKER_PATH = Path("/content/sali_full_train_worker.py")


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


def train_command(output_dir: Path, dataset_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/train_colab.py",
        "--preset",
        "paper",
        "--data-mode",
        "sharded",
        "--field",
        "low",
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "250",
        "--batch-size",
        "64",
        "--checkpoint-every-epochs",
        "1",
        "--max-eval-samples",
        "64",
        "--diagnostic-samples",
        "64",
        "--num-workers",
        "2",
    ]


def write_worker(output_dir: Path, cmd: list[str]) -> None:
    source = f"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path({str(PROJECT_DIR)!r})
SOURCE_DATASET_DIR = Path({str(DATASET_DIR)!r})
STAGED_DATASET_DIR = Path({str(STAGED_DATASET_DIR)!r})
OUTPUT_DIR = Path({str(output_dir)!r})
COMMAND = {cmd!r}


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
    files = [SOURCE_DATASET_DIR / "manifest.json", SOURCE_DATASET_DIR / "generation_plan.json"]
    files.extend(sorted(SOURCE_DATASET_DIR.glob("*.npz")))
    print(f"staging {{len(files)}} files to {{STAGED_DATASET_DIR}}", flush=True)
    for index, source in enumerate(files, start=1):
        target = STAGED_DATASET_DIR / source.name
        copy_if_needed(source, target)
        if index % 20 == 0 or index == len(files):
            elapsed = time.monotonic() - start
            print(f"staged {{index}}/{{len(files)}} files in {{elapsed:.1f}}s", flush=True)
    npz_count = sum(1 for _ in STAGED_DATASET_DIR.glob("*.npz"))
    if npz_count != 360:
        raise RuntimeError(f"Expected 360 staged npz shards, found {{npz_count}}")
    payload = {{
        "source_dataset_dir": str(SOURCE_DATASET_DIR),
        "staged_dataset_dir": str(STAGED_DATASET_DIR),
        "npz_count": npz_count,
        "elapsed_seconds": time.monotonic() - start,
    }}
    (OUTPUT_DIR / "staging_complete.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    stage_dataset()
    print("+ " + " ".join(COMMAND), flush=True)
    return subprocess.run(COMMAND, cwd=PROJECT_DIR).returncode


if __name__ == "__main__":
    raise SystemExit(main())
"""
    WORKER_PATH.write_text(textwrap.dedent(source), encoding="utf-8")


def main() -> None:
    ensure_repo()
    preflight()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = RUNS_ROOT / f"paper-low-full-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "train.log"
    cmd = train_command(output_dir, STAGED_DATASET_DIR)
    write_worker(output_dir, cmd)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            [sys.executable, str(WORKER_PATH)],
            cwd=PROJECT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    launch = {
        "pid": process.pid,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "source_dataset_dir": str(DATASET_DIR),
        "staged_dataset_dir": str(STAGED_DATASET_DIR),
        "worker_path": str(WORKER_PATH),
        "command": cmd,
        "branch": BRANCH,
        "repo_root": str(REPO_ROOT),
        "project_dir": str(PROJECT_DIR),
        "train_examples_per_epoch": 2_520_000,
        "val_examples": 540_000,
        "test_examples": 540_000,
        "total_examples": 3_600_000,
    }
    (output_dir / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(json.dumps(launch, indent=2), flush=True)


if __name__ == "__main__":
    main()
