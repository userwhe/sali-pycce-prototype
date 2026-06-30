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
from sali_pycce.visualize import plot_heatmap_comparison


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/prediction_demo.png")
    p.add_argument("--heatmap-comparison-out", default=None)
    p.add_argument("--signal-points", type=int, default=None)
    p.add_argument("--max-spins", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = str(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    spec = HeatmapSpec(**ckpt.get("heatmap_spec", {}))
    model = SALINet(output_shape=(spec.height, spec.width)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    signal_points = args.signal_points or int(ckpt_args.get("signal_points", 256))
    max_spins = args.max_spins or int(ckpt_args.get("max_spins", 5))
    min_spins = int(ckpt_args.get("min_spins", 1))
    tau_range = (
        float(ckpt_args.get("tau_start_us", 0.0)),
        float(ckpt_args.get("tau_stop_us", 40.0)),
    )
    ds = SyntheticSALIDataset(
        1,
        simulator=AnalyticCPMGSimulator(
            b_gauss=float(ckpt_args.get("b_gauss", 525.0)),
            pulses=(32, 256),
            tau_ranges_us=(tau_range, tau_range),
            signal_points=signal_points,
            shots=ckpt_args.get("shots", 1000),
            t2_us=ckpt_args.get("t2_us", 800.0),
            t2_stretch=float(ckpt_args.get("t2_stretch", 1.0)),
        ),
        heatmap_spec=spec,
        min_spins=min_spins,
        max_spins=max_spins,
        az_range=spec.az_range,
        aperp_range=spec.aperp_range,
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
    if args.heatmap_comparison_out:
        plot_heatmap_comparison(item["heatmap"][0].numpy(), pred, args.heatmap_comparison_out)
    print(f"wrote {out}")
    if detections:
        print("detections:")
        for d in detections:
            print(f"  A_z={d['az_khz']:.2f} kHz, A_perp={d['aperp_khz']:.2f} kHz")


if __name__ == "__main__":
    main()
