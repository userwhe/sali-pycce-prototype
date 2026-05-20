"""Visualize one prediction from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from sali_pycce.data import SyntheticSALIDataset
from sali_pycce.heatmap import HeatmapSpec, decode_heatmap
from sali_pycce.model import SALINet
from sali_pycce.physics import AnalyticCPMGSimulator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/prediction_demo.png")
    p.add_argument("--signal-points", type=int, default=256)
    p.add_argument("--max-spins", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = str(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    spec = HeatmapSpec(**ckpt.get("heatmap_spec", {}))
    model = SALINet(output_shape=(spec.height, spec.width)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = SyntheticSALIDataset(
        1,
        simulator=AnalyticCPMGSimulator(signal_points=args.signal_points),
        heatmap_spec=spec,
        max_spins=args.max_spins,
        seed=999,
    )
    item = ds[0]
    with torch.no_grad():
        pred = model(item["signals"][None].to(device)).cpu()[0, 0].numpy()
    detections = decode_heatmap(pred, spec, threshold=0.25)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    axes[0].plot(item["signals"][0].numpy(), label="N=32")
    axes[0].plot(item["signals"][1].numpy(), label="N=256")
    axes[0].set_title("Input CPMG signals")
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("P_x")
    axes[0].legend()

    axes[1].imshow(item["heatmap"][0].numpy(), origin="lower", aspect="auto")
    axes[1].set_title("True heatmap")
    axes[1].set_xlabel("A_z pixel")
    axes[1].set_ylabel("A_perp pixel")

    axes[2].imshow(pred, origin="lower", aspect="auto")
    axes[2].set_title(f"Predicted heatmap ({len(detections)} blobs)")
    axes[2].set_xlabel("A_z pixel")
    axes[2].set_ylabel("A_perp pixel")
    for det in detections:
        axes[2].plot(det["col"], det["row"], marker="x")

    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    if detections:
        print("detections:")
        for d in detections:
            print(f"  A_z={d['az_khz']:.2f} kHz, A_perp={d['aperp_khz']:.2f} kHz")


if __name__ == "__main__":
    main()
