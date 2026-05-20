"""Evaluation CLI for the SALI prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SyntheticSALIDataset
from .heatmap import HeatmapSpec, decode_heatmap, nearest_match_errors
from .model import SALINet
from .physics import AnalyticCPMGSimulator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a compact SALI-style model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--signal-points", type=int, default=256)
    p.add_argument("--max-spins", type=int, default=5)
    p.add_argument("--min-spins", type=int, default=1)
    p.add_argument("--b-gauss", type=float, default=500.0)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=5678)
    p.add_argument("--threads", type=int, default=1)
    return p


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = str(args.device)
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    spec = HeatmapSpec(**ckpt.get("heatmap_spec", {}))
    model = SALINet(n_inputs=2, output_shape=(spec.height, spec.width)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    simulator = AnalyticCPMGSimulator(b_gauss=args.b_gauss, signal_points=args.signal_points)
    ds = SyntheticSALIDataset(
        args.samples,
        simulator=simulator,
        heatmap_spec=spec,
        min_spins=args.min_spins,
        max_spins=args.max_spins,
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    tp = fp = fn = 0.0
    maes: list[float] = []
    for batch in loader:
        pred = model(batch["signals"].to(device)).cpu().numpy()
        truth_spins = batch["spins"].numpy()
        for i in range(pred.shape[0]):
            detections = decode_heatmap(pred[i, 0], spec, threshold=args.threshold)
            truth = truth_spins[i]
            truth = truth[~np.isnan(truth[:, 0])]
            metrics = nearest_match_errors(truth, detections)
            tp += metrics["tp"]
            fp += metrics["fp"]
            fn += metrics["fn"]
            if np.isfinite(metrics["mae_khz"]):
                maes.append(metrics["mae_khz"])

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    mae = float(np.mean(maes)) if maes else float("nan")
    print(f"samples={args.samples}")
    print(f"tp={tp:.0f} fp={fp:.0f} fn={fn:.0f}")
    print(f"precision={precision:.3f} recall={recall:.3f} matched_mae_khz={mae:.3f}")


if __name__ == "__main__":
    main()
