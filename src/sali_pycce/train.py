"""Training CLI for the SALI prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SyntheticSALIDataset
from .heatmap import HeatmapSpec
from .model import SALINet
from .physics import AnalyticCPMGSimulator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a compact SALI-style signal-to-image model.")
    p.add_argument("--train-samples", type=int, default=1000)
    p.add_argument("--val-samples", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--signal-points", type=int, default=256)
    p.add_argument("--max-spins", type=int, default=5)
    p.add_argument("--min-spins", type=int, default=1)
    p.add_argument("--b-gauss", type=float, default=500.0)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="checkpoints/sali_toy.pt")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=1, help="Torch CPU threads; 1 is often fastest for tiny models.")
    return p


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, HeatmapSpec]:
    simulator = AnalyticCPMGSimulator(
        b_gauss=args.b_gauss,
        signal_points=args.signal_points,
        shots=args.shots,
    )
    spec = HeatmapSpec()
    train_ds = SyntheticSALIDataset(
        args.train_samples,
        simulator=simulator,
        heatmap_spec=spec,
        min_spins=args.min_spins,
        max_spins=args.max_spins,
        seed=args.seed,
    )
    val_ds = SyntheticSALIDataset(
        args.val_samples,
        simulator=simulator,
        heatmap_spec=spec,
        min_spins=args.min_spins,
        max_spins=args.max_spins,
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = str(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, spec = make_loaders(args)
    model = SALINet(n_inputs=2, output_shape=(spec.height, spec.width)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        running = 0.0
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

        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"epoch={epoch} val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "heatmap_spec": spec.__dict__,
                    "val_mse": best_val,
                },
                out,
            )
            print(f"saved {out} with val_mse={best_val:.6f}")


if __name__ == "__main__":
    main()
