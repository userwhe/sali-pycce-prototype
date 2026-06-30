from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SALI-PyCCE training/evaluation in Colab.")
    parser.add_argument("--preset", default="colab-medium", choices=["smoke", "colab-medium", "research"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoints/colab_medium.pt")
    parser.add_argument("--history", default="runs/colab_medium_history.csv")
    parser.add_argument("--metrics", default="runs/colab_medium_metrics.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    python = sys.executable
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    run([python, "-m", "pip", "install", "-q", "-e", ".[dev]"])
    run([python, "-m", "pytest", "-q"])
    run(
        [
            python,
            "-m",
            "sali_pycce.train",
            "--preset",
            args.preset,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--out",
            args.checkpoint,
            "--history-out",
            args.history,
        ]
    )
    run(
        [
            python,
            "-m",
            "sali_pycce.evaluate",
            "--checkpoint",
            args.checkpoint,
            "--samples",
            str(args.eval_samples),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--metrics-out",
            args.metrics,
        ]
    )


if __name__ == "__main__":
    main()
