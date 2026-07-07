#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if (SRC / "sali").exists():
    sys.path.insert(0, str(SRC))

from sali.config import RunConfig, paper_config, practical_config
from sali.shard_jobs import (
    finalize_shard_generation,
    load_generation_plan,
    prepare_shard_generation,
    validate_planned_shards,
    write_planned_shard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, write, and finalize SALI dataset shards.")
    parser.add_argument("--preset", choices=["practical", "paper"], default="paper")
    parser.add_argument("--field", choices=["low", "high"], default="low")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--normalization-samples", type=int, default=10_000)
    parser.add_argument("--shard-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--raw-signal-dtype", choices=["float32", "float16"], default="float32")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Write generation_plan.json.")
    write = subparsers.add_parser("write", help="Write one shard from generation_plan.json.")
    write.add_argument("--shard-id", type=int, required=True)
    write.add_argument("--no-skip-existing", action="store_true")
    subparsers.add_parser("finalize", help="Validate all planned shards and write manifest.json.")
    subparsers.add_parser("validate", help="Validate all planned shards without rewriting manifest.json.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = practical_config(args.field) if args.preset == "practical" else paper_config(args.field)
    if args.train_samples is not None:
        cfg.data.train_samples = args.train_samples
    if args.val_samples is not None:
        cfg.data.val_samples = args.val_samples
    if args.test_samples is not None:
        cfg.data.test_samples = args.test_samples
    return cfg


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    if args.command == "prepare":
        plan = prepare_shard_generation(
            cfg,
            args.dataset_dir,
            shard_size=args.shard_size,
            normalization_samples=args.normalization_samples,
            shard_dtype=args.shard_dtype,
            raw_signal_dtype=args.raw_signal_dtype,
        )
        print(f"prepared {len(plan.shards)} shards at {args.dataset_dir}")
    elif args.command == "write":
        entry = write_planned_shard(
            cfg,
            args.dataset_dir,
            shard_id=args.shard_id,
            skip_existing=not args.no_skip_existing,
        )
        print(f"wrote {entry.path}")
    elif args.command == "finalize":
        manifest = finalize_shard_generation(cfg, args.dataset_dir)
        print(f"finalized {len(manifest.shards)} shards at {args.dataset_dir / 'manifest.json'}")
    elif args.command == "validate":
        plan = load_generation_plan(args.dataset_dir)
        invalid = validate_planned_shards(cfg, args.dataset_dir)
        if invalid:
            bad = ", ".join(entry.path for entry in invalid[:5])
            raise SystemExit(f"invalid shard file(s): {bad}")
        print(f"validated {len(plan.shards)} shards")
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
