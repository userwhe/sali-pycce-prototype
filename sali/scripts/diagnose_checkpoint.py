#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "sali").exists():
    sys.path.insert(0, str(ROOT / "src"))

from sali.config import paper_config, practical_config
from sali.data import generate_splits
from sali.diagnostics import DEFAULT_THRESHOLDS, diagnose_splits, select_threshold
from sali.model import SaliNet


def parse_thresholds(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one threshold is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("thresholds must be between 0 and 1")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a trained SALI checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", choices=["practical", "paper"], default="practical")
    parser.add_argument("--field", choices=["low", "high"], default="low")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated post-processing thresholds to sweep.",
    )
    parser.add_argument("--disable-morphology", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = practical_config(args.field) if args.preset == "practical" else paper_config(args.field)
    cfg.training.device = args.device
    thresholds = parse_thresholds(args.thresholds)
    splits, stats = generate_splits(cfg)
    model = SaliNet(cfg.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    diagnostics = diagnose_splits(
        model,
        splits,
        cfg,
        thresholds=thresholds,
        max_samples=args.max_samples,
        disable_morphology=args.disable_morphology,
    )
    selected = select_threshold(diagnostics["val"]["threshold_sweep"])
    save_json(
        args.output_json,
        {
            "config": asdict(cfg),
            "normalization": asdict(stats),
            "checkpoint": str(args.checkpoint),
            "selected_threshold": selected,
            "splits": diagnostics,
        },
    )
    print(f"Diagnostics written: {args.output_json}")


if __name__ == "__main__":
    main()
