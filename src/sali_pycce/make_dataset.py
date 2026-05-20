"""Generate and save a small synthetic dataset to NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

from .heatmap import HeatmapSpec, make_heatmap
from .physics import AnalyticCPMGSimulator, random_spins, spins_to_array


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--signal-points", type=int, default=256)
    p.add_argument("--max-spins", type=int, default=5)
    p.add_argument("--out", default="data/sali_toy.npz")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sim = AnalyticCPMGSimulator(signal_points=args.signal_points)
    spec = HeatmapSpec()

    signals = np.zeros((args.samples, 2, args.signal_points), dtype=np.float32)
    heatmaps = np.zeros((args.samples, 1, spec.height, spec.width), dtype=np.float32)
    spins_padded = np.full((args.samples, args.max_spins, 2), np.nan, dtype=np.float32)
    n_spins = np.zeros(args.samples, dtype=np.int64)

    for i in trange(args.samples, desc="generating"):
        n = int(rng.integers(1, args.max_spins + 1))
        spins = random_spins(rng, n)
        sample = sim.sample(spins, rng=rng, noisy=True)
        signals[i] = sample["signals"]
        heatmaps[i, 0] = make_heatmap(spins, spec)
        arr = spins_to_array(spins)
        spins_padded[i, :n] = arr
        n_spins[i] = n

    np.savez_compressed(out, signals=signals, heatmaps=heatmaps, spins=spins_padded, n_spins=n_spins)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
