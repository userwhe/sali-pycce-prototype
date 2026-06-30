"""Training CLI for the SALI prototype."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PRESETS
from .data import SyntheticSALIDataset
from .heatmap import HeatmapSpec, decode_heatmap
from .losses import make_heatmap_loss
from .metrics import aggregate_detection_metrics, compute_detection_metrics
from .model import SALINet
from .physics import AnalyticCPMGSimulator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a compact SALI-style signal-to-image model.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--train-samples", type=int, default=None)
    p.add_argument("--val-samples", type=int, default=None)
    p.add_argument("--test-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--signal-points", type=int, default=None)
    p.add_argument("--max-spins", type=int, default=None)
    p.add_argument("--min-spins", type=int, default=None)
    p.add_argument("--b-gauss", type=float, default=None)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--tau-start-us", type=float, default=None)
    p.add_argument("--tau-stop-us", type=float, default=None)
    p.add_argument("--heatmap-height", type=int, default=None)
    p.add_argument("--heatmap-width", type=int, default=None)
    p.add_argument("--az-min", type=float, default=None)
    p.add_argument("--az-max", type=float, default=None)
    p.add_argument("--aperp-min", type=float, default=None)
    p.add_argument("--aperp-max", type=float, default=None)
    p.add_argument("--t2-us", type=float, default=None)
    p.add_argument("--t2-stretch", type=float, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=("weighted-mse", "weighted-bce-dice", "weighted-bce", "mse"), default="weighted-mse")
    p.add_argument("--pos-weight", type=float, default=200.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="checkpoints/sali_toy.pt")
    p.add_argument("--history-out", default=None)
    p.add_argument("--metric-threshold", type=float, default=0.25)
    p.add_argument("--match-distance-khz", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=1, help="Torch CPU threads; 1 is often fastest for tiny models.")
    return p


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    for name in (
        "train_samples",
        "val_samples",
        "test_samples",
        "signal_points",
        "max_spins",
        "min_spins",
        "b_gauss",
        "shots",
    ):
        if getattr(args, name) is None:
            setattr(args, name, getattr(preset, name))
    if args.tau_start_us is None:
        args.tau_start_us = preset.tau_ranges_us[0][0]
    if args.tau_stop_us is None:
        args.tau_stop_us = preset.tau_ranges_us[0][1]
    if args.heatmap_height is None:
        args.heatmap_height = preset.heatmap_shape[0]
    if args.heatmap_width is None:
        args.heatmap_width = preset.heatmap_shape[1]
    if args.az_min is None:
        args.az_min = preset.az_range[0]
    if args.az_max is None:
        args.az_max = preset.az_range[1]
    if args.aperp_min is None:
        args.aperp_min = preset.aperp_range[0]
    if args.aperp_max is None:
        args.aperp_max = preset.aperp_range[1]
    if args.t2_us is None:
        args.t2_us = preset.t2_us
    if args.t2_stretch is None:
        args.t2_stretch = preset.t2_stretch
    return args


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, HeatmapSpec]:
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
    spec = HeatmapSpec(
        height=args.heatmap_height,
        width=args.heatmap_width,
        az_range=(args.az_min, args.az_max),
        aperp_range=(args.aperp_min, args.aperp_max),
        sigma_px=1.25 if (args.heatmap_height, args.heatmap_width) == (128, 256) else 1.0,
        patch_radius=3 if (args.heatmap_height, args.heatmap_width) == (128, 256) else 2,
    )
    train_ds = SyntheticSALIDataset(
        args.train_samples,
        simulator=simulator,
        heatmap_spec=spec,
        min_spins=args.min_spins,
        max_spins=args.max_spins,
        az_range=(args.az_min, args.az_max),
        aperp_range=(args.aperp_min, args.aperp_max),
        seed=args.seed,
    )
    val_ds = SyntheticSALIDataset(
        args.val_samples,
        simulator=simulator,
        heatmap_spec=spec,
        min_spins=args.min_spins,
        max_spins=args.max_spins,
        az_range=(args.az_min, args.az_max),
        aperp_range=(args.aperp_min, args.aperp_max),
        seed=args.seed + 1_000_000,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, spec


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        signals = batch["signals"].to(device)
        target = batch["heatmap"].to(device)
        pred = model(signals)
        loss = loss_fn(pred, target)
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    spec: HeatmapSpec,
    device: str,
    threshold: float,
    max_dist_khz: float,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for batch in loader:
        pred = model(batch["signals"].to(device)).cpu().numpy()
        truth_spins = batch["spins"].numpy()
        for i in range(pred.shape[0]):
            detections = decode_heatmap(pred[i, 0], spec, threshold=threshold)
            truth = truth_spins[i]
            truth = truth[np.isfinite(truth).all(axis=1)]
            rows.append(compute_detection_metrics(truth, detections, max_dist_khz=max_dist_khz))
    return aggregate_detection_metrics(rows)


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = resolve_args(build_parser().parse_args(argv))
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = str(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history_out = Path(args.history_out) if args.history_out else out.with_suffix(".history.csv")

    train_loader, val_loader, spec = make_loaders(args)
    model = SALINet(n_inputs=2, output_shape=(spec.height, spec.width)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = make_heatmap_loss(args.loss, pos_weight=args.pos_weight, dice_weight=args.dice_weight)

    best_val = float("inf")
    history_rows: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        running = 0.0
        step = 0
        for step, batch in enumerate(bar, start=1):
            signals = batch["signals"].to(device)
            target = batch["heatmap"].to(device)
            pred = model(signals)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            bar.set_postfix(train_loss=running / step)

        train_loss = running / max(step, 1)
        val_loss = validate(model, val_loader, loss_fn, device)
        metrics = validation_metrics(
            model,
            val_loader,
            spec,
            device,
            threshold=args.metric_threshold,
            max_dist_khz=args.match_distance_khz,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "mae_khz": metrics["mae_khz"],
            }
        )
        write_history(history_out, history_rows)
        print(f"epoch={epoch} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "heatmap_spec": spec.__dict__,
                    "val_loss": best_val,
                },
                out,
            )
            print(f"saved {out} with val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
