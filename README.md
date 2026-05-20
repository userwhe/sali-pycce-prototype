# SALI-PyCCE Prototype

A small, runnable reproduction of the **SALI-style signal-to-image workflow** for NV-center nuclear-spin detection:

```text
CPMG signal(s) -> 1D CNN feature extraction -> 2D heatmap -> blob detection -> (A_z, A_perp) estimates
```

This is a **functional research prototype**, not the original authors' code or trained weights.  It follows the idea of the SALI paper: use one or more CPMG survival-probability traces as input and predict a 2D image where nuclei appear as Gaussian blobs in hyperfine-parameter space.

## What is included

- Analytic CPMG simulator for fast synthetic data generation.
- Optional PyCCE backend scaffold for replacing the analytic simulator with `pycce.Simulator` runs.
- Synthetic dataset generator with random numbers of nuclei.
- Gaussian heatmap labels in `(A_z, A_perp)` space.
- PyTorch 1D-to-2D CNN model.
- Image postprocessing: thresholding, morphology, connected components, centroids.
- Training, evaluation, and visualization scripts.
- Unit tests for the simulator, heatmaps, and model forward pass.

## Installation

```bash
git clone <your-repo-url>
cd sali-pycce-prototype
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional PyCCE support:

```bash
pip install -e ".[pycce]"
```

PyCCE itself may require a scientific Python environment with NumPy/SciPy/Numba/ASE. If PyCCE is not installed, the default analytic backend still works.

## Quick smoke test

```bash
pytest -q
```

## Train a tiny model on CPU

This trains on a very small toy dataset. It is only meant to verify that the full pipeline works.

```bash
python -m sali_pycce.train \
  --train-samples 256 \
  --val-samples 64 \
  --epochs 2 \
  --batch-size 16 \
  --signal-points 256 \
  --max-spins 5 \
  --device cpu \
  --out checkpoints/toy.pt
```

## Evaluate

```bash
python -m sali_pycce.evaluate \
  --checkpoint checkpoints/toy.pt \
  --samples 32 \
  --signal-points 256 \
  --max-spins 5 \
  --device cpu
```

## Make a visual prediction demo

```bash
python examples/predict_one.py --checkpoint checkpoints/toy.pt --out runs/prediction_demo.png
```

## How this maps to the SALI paper

The paper-scale version uses two CPMG inputs, for example `N=32` and `N=256`, each with 1000 points, and output images where every nucleus is a small Gaussian spot in `(A_z, A_perp)` space. This prototype keeps that design but defaults to smaller datasets and shorter signals so it can run on a laptop.

The default output grid is:

```text
A_z      in [-100, 100] kHz  -> image x-axis
A_perp   in [2, 102] kHz     -> image y-axis
shape    32 x 64 pixels by default for laptop speed
```

## Backend choices

### `analytic`

Fast approximate simulator using the closed-form CPMG expression. This is good for AI pipeline development and debugging.

### `pycce`

PyCCE is designed for spin-bath coherence simulations using cluster-correlation expansion. This repo includes a `pycce_backend.py` adapter scaffold and a clear interface. For serious physics, replace the adapter internals with your exact bath construction, central spin basis, magnetic field, pulse convention, and CCE order.

Example direction:

```python
from sali_pycce.pycce_backend import PyCCESimulator

sim = PyCCESimulator(b_gauss=500, pulses=(32, 256), signal_points=1000)
# sim.sample(...) should return the same dictionary as AnalyticCPMGSimulator.sample(...)
```

## Important limitations

1. The analytic simulator is not a substitute for a carefully validated PyCCE calculation.
2. The network here is intentionally compact and CPU-friendly.
3. The paper used millions of samples; this repo defaults to toy-scale training.
4. The postprocessing thresholds should be tuned for a trained model and target noise level.
5. A faithful experimental pipeline should include calibration, real noise, finite readout contrast, pulse imperfections, and transfer learning/fine-tuning on real data.

## Suggested next steps for your research

- First train with the analytic backend to verify the signal-to-image idea.
- Then replace or augment the simulator with PyCCE-generated CPMG traces.
- Start with `max_spins <= 5`, then increase gradually.
- Track performance by spin count, coupling strength, and distance between nearby nuclei in the output heatmap.
- Use this as a parameter-initialization tool for a later physics-based fit.
