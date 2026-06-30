"""Evaluation CLI for the SALI prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PRESETS
from .data import SyntheticSALIDataset
from .heatmap import HeatmapSpec, decode_heatmap
from .metrics import aggregate_detection_metrics, compute_detection_metrics
from .model import SALINet
from .physics import AnalyticCPMGSimulator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a compact SALI-style model.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--signal-points", type=int, default=None)
    p.add_argument("--max-spins", type=int, default=None)
    p.add_argument("--min-spins", type=int, default=None)
    p.add_argument("--b-gauss", type=float, default=None)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--tau-start-us", type=float, default=None)
    p.add_argument("--tau-stop-us", type=float, default=None)
    p.add_argument("--t2-us", type=float, default=None)
    p.add_argument("--t2-stretch", type=float, default=None)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--match-distance-khz", type=float, default=5.0)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=5678)
    p.add_argument("--threads", type=int, default=1)
    return p


def _resolve_from_checkpoint_or_preset(
    args: argparse.Namespace,
    ckpt_args: dict[str, object],
    preset_name: str,
    field: str,
) -> None:
    if getattr(args, field) is not None:
        return
    preset = PRESETS[preset_name]
    if field in ckpt_args:
        setattr(args, field, ckpt_args[field])
    else:
        setattr(args, field, getattr(preset, field))


def resolve_args(args: argparse.Namespace, ckpt: dict[str, object]) -> argparse.Namespace:
    ckpt_args = ckpt.get("args", {})
    ckpt_args = ckpt_args if isinstance(ckpt_args, dict) else {}
    preset = PRESETS[args.preset]
    if args.samples is None:
        args.samples = int(ckpt_args.get("test_samples", preset.test_samples))
    for field in ("signal_points", "max_spins", "min_spins", "b_gauss", "shots"):
        _resolve_from_checkpoint_or_preset(args, ckpt_args, args.preset, field)
    if args.tau_start_us is None:
        args.tau_start_us = float(ckpt_args.get("tau_start_us", preset.tau_ranges_us[0][0]))
    if args.tau_stop_us is None:
        args.tau_stop_us = float(ckpt_args.get("tau_stop_us", preset.tau_ranges_us[0][1]))
    if args.t2_us is None:
        args.t2_us = ckpt_args.get("t2_us", preset.t2_us)
    if args.t2_stretch is None:
        args.t2_stretch = float(ckpt_args.get("t2_stretch", preset.t2_stretch))
    return args


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = str(args.device)
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    args = resolve_args(args, ckpt)
    spec = HeatmapSpec(**ckpt.get("heatmap_spec", {}))
    model = SALINet(n_inputs=2, output_shape=(spec.height, spec.width)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tau_ranges = ((args.tau_start_us, args.tau_stop_us), (args.tau_start_us, args.tau_stop_us))
    simulator = AnalyticCPMGSimulator(
        b_gauss=args.b_gauss,
        pulses=(32, 256),
        tau_ranges_us=tau_ranges,
        signal_points=args.signal_points,
        shots=args.shots,
        t2_us=args.t2_us,
        t2_stretch=args.t2_stretch,
    )
    ds = SyntheticSALIDataset(
        args.samples,
        simulator=simulator,
        heatmap_spec=spec,
        min_spins=args.min_spins,
        max_spins=args.max_spins,
        az_range=spec.az_range,
        aperp_range=spec.aperp_range,
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    rows: list[dict[str, float]] = []
    for batch in loader:
        pred = model(batch["signals"].to(device)).cpu().numpy()
        truth_spins = batch["spins"].numpy()
        for i in range(pred.shape[0]):
            detections = decode_heatmap(pred[i, 0], spec, threshold=args.threshold)
            truth = truth_spins[i]
            truth = truth[np.isfinite(truth).all(axis=1)]
            rows.append(
                compute_detection_metrics(
                    truth,
                    detections,
                    max_dist_khz=args.match_distance_khz,
                )
            )

    metrics = aggregate_detection_metrics(rows)
    tp = metrics["tp"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    mae = float(metrics["mae_khz"]) if np.isfinite(metrics["mae_khz"]) else float("nan")
    payload = {
        "samples": int(args.samples),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "matched_mae_khz": mae,
    }
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"samples={args.samples}")
    print(f"tp={tp:.0f} fp={fp:.0f} fn={fn:.0f}")
    print(f"precision={precision:.3f} recall={recall:.3f} matched_mae_khz={mae:.3f}")


if __name__ == "__main__":
    main()
