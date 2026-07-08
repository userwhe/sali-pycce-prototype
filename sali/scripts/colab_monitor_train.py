#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path


MY_DRIVE = Path("/content/drive/MyDrive")
RUNS_ROOT = MY_DRIVE / "02_Research/DQP/sali/runs"


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return f"{path} does not exist"
    text = path.read_text(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def process_status(pid: int) -> str:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return "not-running"
    return "running"


def main() -> None:
    runs = sorted(RUNS_ROOT.glob("paper-low-full-*"), key=lambda path: path.name)
    if not runs:
        print(f"No paper-low-full runs found under {RUNS_ROOT}", flush=True)
        return
    run_dir = runs[-1]
    launch_path = run_dir / "launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
    pid = int(launch.get("pid", -1))
    print(f"run_dir: {run_dir}", flush=True)
    print(f"pid: {pid}", flush=True)
    if pid > 0:
        print(f"process_status: {process_status(pid)}", flush=True)
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        print("run_state:", flush=True)
        print(state_path.read_text(), flush=True)
    history_path = run_dir / "history.csv"
    if history_path.exists():
        print("history_tail:", flush=True)
        print(tail(history_path, 12), flush=True)
    log_path = Path(launch.get("log_path", run_dir / "train.log"))
    print("log_tail:", flush=True)
    print(tail(log_path), flush=True)


if __name__ == "__main__":
    main()
