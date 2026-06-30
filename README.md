# SALI-PyCCE Prototype

A small, runnable reproduction of the **SALI-style signal-to-image workflow** for NV-center nuclear-spin detection:

```text
CPMG signal(s) -> 1D CNN feature extraction -> 2D heatmap -> blob detection -> (A_z, A_perp) estimates
```

This is a **functional research prototype**, not the original authors' code or trained weights.  It follows the idea of the SALI paper: use one or more CPMG survival-probability traces as input and predict a 2D image where nuclei appear as Gaussian blobs in hyperfine-parameter space.

## What is included

- Analytic CPMG simulator for fast synthetic data generation.
- PyCCE random-bath benchmark generation for physics-domain testing.
- Synthetic dataset generator with random numbers of nuclei.
- Gaussian heatmap labels in `(A_z, A_perp)` space.
- PyTorch 1D-to-2D CNN model.
- Image postprocessing: thresholding, morphology, connected components, centroids.
- Shared precision, recall, MAE, and threshold-sweep metrics.
- Training, evaluation, Colab, Hyak, and visualization scripts.
- Adapter for the local `dqpmodel/cpmg_model` decomposition baseline.
- Unit tests for the simulator, heatmaps, model, metrics, tensor projection, baseline adapter, PyCCE helpers, CLIs, and visualization.

## Installation

```bash
git clone <your-repo-url>
cd sali-pycce-prototype
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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
  --preset smoke \
  --epochs 1 \
  --batch-size 16 \
  --device cpu \
  --out checkpoints/toy.pt \
  --history-out runs/toy_history.csv \
  --threads 1
```

## Evaluate

```bash
python -m sali_pycce.evaluate \
  --checkpoint checkpoints/toy.pt \
  --samples 32 \
  --batch-size 16 \
  --device cpu \
  --metrics-out runs/toy_metrics.json \
  --threads 1
```

## Make a visual prediction demo

```bash
python examples/predict_one.py --checkpoint checkpoints/toy.pt --out runs/prediction_demo.png
```

## Research Pipeline Defaults

The research workflow trains on analytic noisy CPMG traces, validates during training, evaluates on held-out analytic test data, and benchmarks on PyCCE random spin baths.

- B field: `Bz = 525 G`
- CPMG traces: `N=32` and `N=256`
- Tau grid: `0-40 us`, `4000` points
- Label grid: `128 x 256`
- Label range: `A_z = [-250, 250] kHz`, `A_perp = [2, 250] kHz`
- Noise: binomial shot noise plus configurable `T2`, default `800 us`
- Split counts:
  - smoke: `512/128/128` train/validation/analytic-test
  - Colab medium: `20_000/2_000/2_000` train/validation/analytic-test
  - first research run: `100_000/10_000/10_000` train/validation/analytic-test

Train:

```bash
python -m sali_pycce.train \
  --preset colab-medium \
  --epochs 20 \
  --batch-size 128 \
  --device cuda \
  --out checkpoints/colab_medium.pt \
  --history-out runs/colab_medium_history.csv
```

Evaluate:

```bash
python -m sali_pycce.evaluate \
  --checkpoint checkpoints/colab_medium.pt \
  --samples 500 \
  --device cuda \
  --metrics-out runs/colab_medium_metrics.json
```

PyCCE benchmark:

```bash
python -m sali_pycce.pycce_benchmark \
  --out runs/pycce_sample.npz \
  --bath-number 2000 \
  --signal-points 4000
```

If PyCCE is not installed, the benchmark command raises a clear install error. Install optional support with:

```bash
pip install -e ".[pycce,dev]"
```

Baseline convention for `/Users/weitao/Code/python/dqpmodel/cpmg_model`:

```text
cpmg_model A = -A_par
cpmg_model B = A_perp
comparison uses A_z = -A, A_perp = B
```

## Colab

`notebooks/train_in_colab.ipynb` contains the full Colab workflow and displays the required figures: raw CPMG traces, true spin scatter, ground-truth heatmap, train/validation loss, target-vs-predicted heatmap, precision/recall/MAE, and an overlay of the original CPMG signals with signals regenerated from the identified C13s for `N=32` and `N=256`.

Inside a cloned Colab repo, run:

```bash
python scripts/colab_train.py \
  --preset colab-medium \
  --epochs 20 \
  --batch-size 128 \
  --checkpoint checkpoints/colab_medium.pt \
  --history runs/colab_medium_history.csv \
  --metrics runs/colab_medium_metrics.json
```

When using the local `colab` CLI from outside the VM, upload or clone the repository into `/content/sali-pycce-prototype` before executing `scripts/colab_train.py`; `colab run` sends only the script file, not the whole worktree.

## How this maps to the SALI paper

The paper-scale version uses two CPMG inputs, for example `N=32` and `N=256`, and output images where every nucleus is a small Gaussian spot in `(A_z, A_perp)` space. This prototype keeps that design and exposes smoke, Colab-medium, and research presets so the same code can run on a laptop or a GPU runtime.

The research output grid is:

```text
A_z      in [-250, 250] kHz  -> image x-axis
A_perp   in [2, 250] kHz     -> image y-axis
shape    128 x 256 pixels
```

## Backend choices

### `analytic`

Fast approximate simulator using the closed-form CPMG expression. This is good for AI pipeline development and debugging.

### `pycce`

PyCCE is designed for spin-bath coherence simulations using cluster-correlation expansion. This repo includes a `pycce_benchmark.py` random-bath benchmark generator and a `pycce_backend.py` adapter scaffold. For serious physics, validate the exact bath construction, central spin basis, magnetic field, pulse convention, and CCE order.

Example direction:

```python
from sali_pycce.pycce_backend import PyCCESimulator

sim = PyCCESimulator(b_gauss=525, pulses=(32, 256), signal_points=4000)
# sim.sample(...) should return the same dictionary as AnalyticCPMGSimulator.sample(...)
```

## Important limitations

1. The analytic simulator is not a substitute for a carefully validated PyCCE calculation.
2. The network here is intentionally compact and CPU-friendly.
3. The smoke preset is only for pipeline checks; use Colab-medium or research presets for meaningful experiments.
4. The postprocessing thresholds should be tuned for a trained model and target noise level.
5. A faithful experimental pipeline should include calibration, real noise, finite readout contrast, pulse imperfections, and transfer learning/fine-tuning on real data.

## Suggested next steps for your research

- First train with the analytic backend to verify the signal-to-image idea.
- Then replace or augment the simulator with PyCCE-generated CPMG traces.
- Start with `max_spins <= 5`, then increase gradually.
- Track performance by spin count, coupling strength, and distance between nearby nuclei in the output heatmap.
- Use this as a parameter-initialization tool for a later physics-based fit.
