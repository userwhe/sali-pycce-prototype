# SALI-PyCCE Research Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible SALI-PyCCE research pipeline that trains on analytic noisy CPMG traces, validates during training, evaluates on held-out analytic test data, benchmarks on PyCCE random C13 spin baths, and compares against the local N=32 algorithmic decomposition baseline.

**Architecture:** Add a small configuration layer for research/smoke presets, make the simulator/model/heatmap code shape-configurable, persist train, validation, and analytic-test metrics, and add focused modules for tensor projection, baseline adaptation, PyCCE benchmark generation, visualization, and Colab CLI execution. Keep each module independently testable and route notebook work through reusable Python APIs and CLIs.

**Tech Stack:** Python 3.10+, NumPy, SciPy, PyTorch, Matplotlib, PyCCE, pytest, Google Colab CLI.

---

## File Structure

- Create `src/sali_pycce/config.py`
  - Owns research, Colab-medium, and smoke presets, including train/validation/analytic-test counts.
  - Builds `HeatmapSpec` and `AnalyticCPMGSimulator` from a single config object.
- Modify `src/sali_pycce/physics.py`
  - Adds stretched `T2` decoherence support.
  - Supports research tau defaults through config.
- Modify `src/sali_pycce/heatmap.py`
  - Keeps coordinate mapping generic for `128 x 256` and wider hyperfine ranges.
  - Keeps decoding compatible with older `32 x 64` smoke tests.
- Modify `src/sali_pycce/data.py`
  - Accepts research config fields cleanly.
  - Returns `taus_us` in dataset items for plotting and PyCCE-compatible inspection.
- Modify `src/sali_pycce/model.py`
  - Removes the hard-coded `32 x 64` restriction.
  - Supports any output shape divisible by 8, including `128 x 256`.
- Create `src/sali_pycce/metrics.py`
  - Provides shared matching, precision, recall, MAE, and threshold-sweep helpers.
- Modify `src/sali_pycce/train.py`
  - Adds presets and full research configuration CLI arguments.
  - Writes checkpoint plus CSV history.
- Modify `src/sali_pycce/evaluate.py`
  - Uses shared metrics and supports held-out analytic test evaluation and threshold sweeps.
- Create `src/sali_pycce/visualize.py`
  - Provides notebook/report plotting helpers.
- Create `src/sali_pycce/tensor.py`
  - Converts `3 x 3` hyperfine tensors to `(A_z, A_perp)` in kHz.
- Create `src/sali_pycce/baseline.py`
  - Wraps `/Users/weitao/Code/python/dqpmodel/cpmg_model`.
  - Converts `cpmg_model A = -A_par`, `B = A_perp` into canonical `(A_z, A_perp)`.
- Create `src/sali_pycce/pycce_benchmark.py`
  - Generates random C13 baths with PyCCE, projects tensors, simulates CPMG traces, and saves benchmark samples.
- Create `scripts/colab_train.py`
  - Runs a Colab-friendly train/evaluate/report job.
- Modify `notebooks/train_in_colab.ipynb`
  - Calls package CLIs and displays the required six figures.
- Modify `README.md`
  - Documents research defaults, training counts, Colab command, PyCCE benchmark, and baseline comparison.
- Add tests:
  - `tests/test_config.py`
  - `tests/test_metrics.py`
  - `tests/test_tensor.py`
  - `tests/test_baseline.py`
  - `tests/test_pycce_benchmark.py`
  - Update existing `tests/test_physics.py`, `tests/test_heatmap.py`, and `tests/test_model.py`.

## Dataset Split Counts

Use staged dataset sizes:

- Smoke: `512` train / `128` validation / `128` analytic test
- Colab medium: `20_000` train / `2_000` validation / `2_000` analytic test
- First research run: `100_000` train / `10_000` validation / `10_000` analytic test

All analytic split data remains generated on the fly by deterministic index seeds. Use distinct seed offsets for train, validation, and analytic test splits. Do not store all analytic training traces unless a future experiment explicitly asks for cached data.

The PyCCE random-bath benchmark is a separate physics-domain test set, not a replacement for the held-out analytic test split. Use one PyCCE sample for smoke, ten samples for Colab-medium benchmarking, and one hundred samples for the first research benchmark unless runtime forces a smaller first pass.

## PyCCE API Notes

The PyCCE implementation should use the documented interfaces:

- `pycce.random_bath('13C', config.bath_size_angstrom, number=config.bath_number, seed=config.seed)` for random bath coordinates.
- `BathArray.from_point_dipole(np.zeros(3), inplace=True)` to generate hyperfine tensors from the central spin.
- `BathArray.A` / `bath['A']` for per-spin `3 x 3` hyperfine tensors.
- `pycce.Simulator(1, bath=bath, magnetic_field=np.asarray([0.0, 0.0, 525.0]), pulses=N, as_delay=True, order=1)` for CPMG-style simulations with tau interpreted as pulse delay.

If the installed PyCCE package lacks one of these attributes, `pycce_benchmark.py` should raise a `RuntimeError` naming the missing attribute and keep the public benchmark output schema stable.

---

### Task 1: Add Research Presets And Simulator Defaults

**Files:**
- Create: `src/sali_pycce/config.py`
- Modify: `src/sali_pycce/physics.py`
- Test: `tests/test_config.py`
- Test: `tests/test_physics.py`

- [ ] **Step 1: Write failing config tests**

Add `tests/test_config.py`:

```python
import numpy as np

from sali_pycce.config import COLAB_MEDIUM_CONFIG, RESEARCH_CONFIG, SMOKE_CONFIG


def test_research_config_matches_approved_defaults():
    assert RESEARCH_CONFIG.b_gauss == 525.0
    assert RESEARCH_CONFIG.pulses == (32, 256)
    assert RESEARCH_CONFIG.tau_ranges_us == ((0.0, 40.0), (0.0, 40.0))
    assert RESEARCH_CONFIG.signal_points == 4000
    assert RESEARCH_CONFIG.az_range == (-250.0, 250.0)
    assert RESEARCH_CONFIG.aperp_range == (2.0, 250.0)
    assert RESEARCH_CONFIG.heatmap_shape == (128, 256)
    assert RESEARCH_CONFIG.t2_us == 800.0
    assert RESEARCH_CONFIG.train_samples == 100_000
    assert RESEARCH_CONFIG.val_samples == 10_000
    assert RESEARCH_CONFIG.test_samples == 10_000


def test_presets_build_consistent_simulator_and_heatmap_spec():
    sim = RESEARCH_CONFIG.build_simulator(shots=123)
    spec = RESEARCH_CONFIG.build_heatmap_spec()

    assert sim.b_gauss == 525.0
    assert sim.pulses == (32, 256)
    assert sim.signal_points == 4000
    assert sim.shots == 123
    assert sim.t2_us == 800.0
    assert spec.height == 128
    assert spec.width == 256
    assert spec.az_range == (-250.0, 250.0)
    assert spec.aperp_range == (2.0, 250.0)


def test_smoke_and_colab_medium_have_smaller_counts_than_research():
    assert SMOKE_CONFIG.train_samples < COLAB_MEDIUM_CONFIG.train_samples
    assert COLAB_MEDIUM_CONFIG.train_samples < RESEARCH_CONFIG.train_samples
    assert SMOKE_CONFIG.val_samples < COLAB_MEDIUM_CONFIG.val_samples
    assert COLAB_MEDIUM_CONFIG.val_samples < RESEARCH_CONFIG.val_samples
    assert SMOKE_CONFIG.test_samples < COLAB_MEDIUM_CONFIG.test_samples
    assert COLAB_MEDIUM_CONFIG.test_samples < RESEARCH_CONFIG.test_samples


def test_research_tau_grid_is_zero_to_forty_us():
    sim = RESEARCH_CONFIG.build_simulator(shots=None)
    tau = sim.tau_grid(0)

    assert tau.shape == (4000,)
    np.testing.assert_allclose(tau[0], 0.0)
    np.testing.assert_allclose(tau[-1], 40.0)
```

Add this test to `tests/test_physics.py`:

```python
def test_t2_decoherence_uses_configurable_default_scale():
    sim = AnalyticCPMGSimulator(
        b_gauss=525.0,
        signal_points=5,
        tau_ranges_us=((0.0, 40.0), (0.0, 40.0)),
        t2_us=800.0,
        t2_stretch=1.0,
        shots=None,
    )
    tau = sim.tau_grid(0)
    px = sim.signal([], 32, tau)

    expected = 0.5 + 0.5 * np.exp(-tau / 800.0)
    np.testing.assert_allclose(px, expected, rtol=1e-7, atol=1e-7)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_config.py tests/test_physics.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'sali_pycce.config'
```

or failure because `AnalyticCPMGSimulator` does not accept `t2_stretch`.

- [ ] **Step 3: Implement config presets**

Create `src/sali_pycce/config.py`:

```python
"""Shared configuration presets for SALI-PyCCE workflows."""

from __future__ import annotations

from dataclasses import dataclass

from .heatmap import HeatmapSpec
from .physics import AnalyticCPMGSimulator


_DEFAULT = object()


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    train_samples: int
    val_samples: int
    test_samples: int
    max_spins: int
    min_spins: int
    b_gauss: float = 525.0
    pulses: tuple[int, int] = (32, 256)
    tau_ranges_us: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 40.0), (0.0, 40.0))
    signal_points: int = 4000
    az_range: tuple[float, float] = (-250.0, 250.0)
    aperp_range: tuple[float, float] = (2.0, 250.0)
    heatmap_shape: tuple[int, int] = (128, 256)
    heatmap_sigma_px: float = 1.25
    heatmap_patch_radius: int = 3
    shots: int | None = 1000
    t2_us: float | None = 800.0
    t2_stretch: float = 1.0

    def build_simulator(self, shots: int | None | object = _DEFAULT) -> AnalyticCPMGSimulator:
        resolved_shots = self.shots if shots is _DEFAULT else shots
        return AnalyticCPMGSimulator(
            b_gauss=self.b_gauss,
            pulses=self.pulses,
            tau_ranges_us=self.tau_ranges_us,
            signal_points=self.signal_points,
            t2_us=self.t2_us,
            t2_stretch=self.t2_stretch,
            shots=resolved_shots,
        )

    def build_heatmap_spec(self) -> HeatmapSpec:
        height, width = self.heatmap_shape
        return HeatmapSpec(
            height=height,
            width=width,
            az_range=self.az_range,
            aperp_range=self.aperp_range,
            sigma_px=self.heatmap_sigma_px,
            patch_radius=self.heatmap_patch_radius,
        )


SMOKE_CONFIG = PipelineConfig(
    name="smoke",
    train_samples=512,
    val_samples=128,
    test_samples=128,
    max_spins=5,
    min_spins=1,
    signal_points=256,
    heatmap_shape=(32, 64),
    heatmap_sigma_px=1.0,
    heatmap_patch_radius=2,
)

COLAB_MEDIUM_CONFIG = PipelineConfig(
    name="colab-medium",
    train_samples=20_000,
    val_samples=2_000,
    test_samples=2_000,
    max_spins=10,
    min_spins=1,
)

RESEARCH_CONFIG = PipelineConfig(
    name="research",
    train_samples=100_000,
    val_samples=10_000,
    test_samples=10_000,
    max_spins=20,
    min_spins=1,
)

PRESETS = {
    SMOKE_CONFIG.name: SMOKE_CONFIG,
    COLAB_MEDIUM_CONFIG.name: COLAB_MEDIUM_CONFIG,
    RESEARCH_CONFIG.name: RESEARCH_CONFIG,
}
```

- [ ] **Step 4: Implement configurable stretched T2 decoherence**

Modify `src/sali_pycce/physics.py`:

```python
@dataclass
class AnalyticCPMGSimulator:
    b_gauss: float = 525.0
    pulses: tuple[int, int] = (32, 256)
    tau_ranges_us: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 40.0), (0.0, 40.0))
    signal_points: int = 4000
    gamma_c13_khz_per_g: float = 1.0705
    hyperfine_factor: float = 0.5
    t2_us: float | None = 800.0
    t2_stretch: float = 1.0
    readout_offset: float = 0.0
    readout_visibility: float = 1.0
    shots: int | None = 1000
```

Replace the decoherence block in `signal()` with:

```python
if self.t2_us is not None and self.t2_us > 0:
    stretch = max(float(self.t2_stretch), 1e-12)
    contrast = np.exp(-np.power(tau_us / float(self.t2_us), stretch))
    px = 0.5 + (px - 0.5) * contrast
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_config.py tests/test_physics.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add src/sali_pycce/config.py src/sali_pycce/physics.py tests/test_config.py tests/test_physics.py
git commit -m "feat: add research presets and T2 simulator defaults"
```

---

### Task 2: Wire Dataset And Heatmap Labels To Research Config

**Files:**
- Modify: `src/sali_pycce/data.py`
- Modify: `src/sali_pycce/heatmap.py`
- Test: `tests/test_heatmap.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write failing heatmap and dataset tests**

Add to `tests/test_heatmap.py`:

```python
def test_research_heatmap_roundtrip_has_about_two_khz_resolution():
    spec = HeatmapSpec(
        height=128,
        width=256,
        az_range=(-250.0, 250.0),
        aperp_range=(2.0, 250.0),
        sigma_px=1.25,
        patch_radius=3,
    )
    spin = SpinParams(az_khz=-123.0, aperp_khz=211.0)
    heatmap = make_heatmap([spin], spec)
    det = decode_heatmap(heatmap, spec, threshold=0.2, min_area=1, morph=False)

    assert len(det) == 1
    assert abs(det[0]["az_khz"] - spin.az_khz) < 2.5
    assert abs(det[0]["aperp_khz"] - spin.aperp_khz) < 2.5
```

Create `tests/test_data.py`:

```python
import numpy as np

from sali_pycce.config import SMOKE_CONFIG
from sali_pycce.data import SyntheticSALIDataset


def test_dataset_returns_taus_and_configured_shapes():
    sim = SMOKE_CONFIG.build_simulator(shots=None)
    spec = SMOKE_CONFIG.build_heatmap_spec()
    ds = SyntheticSALIDataset(
        2,
        simulator=sim,
        heatmap_spec=spec,
        min_spins=SMOKE_CONFIG.min_spins,
        max_spins=SMOKE_CONFIG.max_spins,
        az_range=SMOKE_CONFIG.az_range,
        aperp_range=SMOKE_CONFIG.aperp_range,
        seed=7,
        noisy=False,
    )

    item = ds[0]

    assert item["signals"].shape == (2, SMOKE_CONFIG.signal_points)
    assert item["taus_us"].shape == (2, SMOKE_CONFIG.signal_points)
    assert item["heatmap"].shape == (1, *SMOKE_CONFIG.heatmap_shape)
    assert item["spins"].shape == (SMOKE_CONFIG.max_spins, 2)
    assert int(item["n_spins"]) >= SMOKE_CONFIG.min_spins
    assert np.isfinite(item["signals"].numpy()).all()
```

- [ ] **Step 2: Run tests to verify the dataset test fails**

Run:

```bash
python -m pytest tests/test_heatmap.py tests/test_data.py -q
```

Expected:

```text
KeyError: 'taus_us'
```

- [ ] **Step 3: Return tau grids from the dataset**

Modify `src/sali_pycce/data.py` in `SyntheticSALIDataset.__getitem__`:

```python
return {
    "signals": torch.from_numpy(sample["signals"]).float(),
    "taus_us": torch.from_numpy(sample["taus_us"]).float(),
    "heatmap": torch.from_numpy(heatmap[None, :, :]).float(),
    "spins": torch.from_numpy(padded).float(),
    "n_spins": torch.tensor(n_spins, dtype=torch.long),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_heatmap.py tests/test_data.py -q
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/sali_pycce/data.py tests/test_heatmap.py tests/test_data.py
git commit -m "feat: return tau grids from synthetic dataset"
```

---

### Task 3: Make SALINet Output Shape Flexible

**Files:**
- Modify: `src/sali_pycce/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing model shape tests**

Replace `tests/test_model.py` with:

```python
import pytest
import torch

from sali_pycce.model import SALINet


def test_model_forward_shape_smoke_grid():
    model = SALINet(output_shape=(32, 64))
    x = torch.rand(3, 2, 128)
    y = model(x)
    assert y.shape == (3, 1, 32, 64)
    assert torch.isfinite(y).all()


def test_model_forward_shape_research_grid():
    model = SALINet(output_shape=(128, 256))
    x = torch.rand(2, 2, 4000)
    y = model(x)
    assert y.shape == (2, 1, 128, 256)
    assert torch.isfinite(y).all()


def test_model_rejects_output_shape_not_divisible_by_eight():
    with pytest.raises(ValueError, match="divisible by 8"):
        SALINet(output_shape=(130, 256))
```

- [ ] **Step 2: Run tests to verify the research-grid test fails**

Run:

```bash
python -m pytest tests/test_model.py -q
```

Expected:

```text
ValueError: This compact prototype currently expects output_shape=(32, 64).
```

- [ ] **Step 3: Implement shape-flexible decoder**

Modify `src/sali_pycce/model.py`.

Replace `SALINet.__init__` with:

```python
def __init__(
    self,
    n_inputs: int = 2,
    output_shape: tuple[int, int] = (128, 256),
    pooled_len: int = 64,
    decoder_channels: int = 32,
) -> None:
    super().__init__()
    height, width = output_shape
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("output_shape height and width must be divisible by 8.")
    self.n_inputs = int(n_inputs)
    self.output_shape = (int(height), int(width))
    self.base_shape = (height // 8, width // 8)
    self.decoder_channels = int(decoder_channels)
    self.branches = nn.ModuleList([SignalBranch(pooled_len=pooled_len) for _ in range(n_inputs)])
    branch_dim = 16 * int(pooled_len)
    self.fc = nn.Sequential(
        nn.Linear(n_inputs * branch_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, self.decoder_channels * self.base_shape[0] * self.base_shape[1]),
        nn.ReLU(inplace=True),
    )
    self.decoder = nn.Sequential(
        nn.Conv2d(self.decoder_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.Conv2d(4, 1, kernel_size=3, padding=1),
        nn.Sigmoid(),
    )
```

Replace the reshape line in `forward()` with:

```python
z = z.reshape(signals.shape[0], self.decoder_channels, self.base_shape[0], self.base_shape[1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_model.py -q
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/sali_pycce/model.py tests/test_model.py
git commit -m "feat: support configurable SALI heatmap shapes"
```

---

### Task 4: Add Shared Metrics And Threshold Sweeps

**Files:**
- Create: `src/sali_pycce/metrics.py`
- Modify: `src/sali_pycce/heatmap.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_heatmap.py`

- [ ] **Step 1: Write failing metrics tests**

Create `tests/test_metrics.py`:

```python
import numpy as np

from sali_pycce.heatmap import HeatmapSpec, make_heatmap
from sali_pycce.metrics import compute_detection_metrics, threshold_sweep
from sali_pycce.physics import SpinParams


def test_compute_detection_metrics_counts_precision_recall_and_mae():
    truth = np.asarray([[10.0, 20.0], [80.0, 90.0]], dtype=float)
    pred = [
        {"az_khz": 11.0, "aperp_khz": 19.0},
        {"az_khz": 200.0, "aperp_khz": 200.0},
    ]

    metrics = compute_detection_metrics(truth, pred, max_dist_khz=5.0)

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["mae_khz"] == 1.0


def test_threshold_sweep_returns_one_row_per_threshold():
    spec = HeatmapSpec(height=32, width=64)
    spin = SpinParams(az_khz=10.0, aperp_khz=40.0)
    heatmap = make_heatmap([spin], spec)
    truth = np.asarray([[spin.az_khz, spin.aperp_khz]], dtype=float)

    rows = threshold_sweep(
        pred_heatmaps=np.asarray([heatmap]),
        truth_spins=[truth],
        spec=spec,
        thresholds=[0.2, 0.5],
        max_dist_khz=5.0,
    )

    assert [row["threshold"] for row in rows] == [0.2, 0.5]
    assert all(row["precision"] == 1.0 for row in rows)
    assert all(row["recall"] == 1.0 for row in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_metrics.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'sali_pycce.metrics'
```

- [ ] **Step 3: Implement metrics helpers**

Create `src/sali_pycce/metrics.py`:

```python
"""Shared spin-detection metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .heatmap import HeatmapSpec, decode_heatmap, nearest_match_errors


def compute_detection_metrics(
    truth: np.ndarray,
    pred: list[dict[str, float]],
    max_dist_khz: float = 5.0,
) -> dict[str, float]:
    counts = nearest_match_errors(truth, pred, max_dist_khz=max_dist_khz)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        **counts,
        "precision": float(precision),
        "recall": float(recall),
        "mae_khz": float(counts["mae_khz"]),
    }


def aggregate_detection_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    tp = float(sum(row["tp"] for row in rows))
    fp = float(sum(row["fp"] for row in rows))
    fn = float(sum(row["fn"] for row in rows))
    maes = [row["mae_khz"] for row in rows if np.isfinite(row["mae_khz"])]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "mae_khz": float(np.mean(maes)) if maes else float("nan"),
    }


def threshold_sweep(
    pred_heatmaps: np.ndarray,
    truth_spins: Sequence[np.ndarray],
    spec: HeatmapSpec,
    thresholds: Iterable[float],
    max_dist_khz: float = 5.0,
    min_area: int = 3,
    morph: bool = True,
) -> list[dict[str, float]]:
    heatmaps = np.asarray(pred_heatmaps)
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[:, 0]
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        per_sample: list[dict[str, float]] = []
        for heatmap, truth in zip(heatmaps, truth_spins, strict=True):
            detections = decode_heatmap(
                heatmap,
                spec,
                threshold=float(threshold),
                min_area=min_area,
                morph=morph,
            )
            per_sample.append(
                compute_detection_metrics(truth, detections, max_dist_khz=max_dist_khz)
            )
        rows.append({"threshold": float(threshold), **aggregate_detection_metrics(per_sample)})
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_metrics.py tests/test_heatmap.py -q
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/sali_pycce/metrics.py tests/test_metrics.py tests/test_heatmap.py
git commit -m "feat: add shared spin detection metrics"
```

---

### Task 5: Persist Training History And Use Presets In CLI

**Files:**
- Modify: `src/sali_pycce/train.py`
- Modify: `src/sali_pycce/evaluate.py`
- Test: `tests/test_train_cli.py`
- Test: `tests/test_evaluate_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_train_cli.py`:

```python
import csv

from sali_pycce.train import main


def test_train_writes_checkpoint_and_history(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    history = tmp_path / "history.csv"

    main(
        [
            "--preset",
            "smoke",
            "--train-samples",
            "8",
            "--val-samples",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--signal-points",
            "64",
            "--heatmap-height",
            "32",
            "--heatmap-width",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--out",
            str(checkpoint),
            "--history-out",
            str(history),
            "--threads",
            "1",
        ]
    )

    assert checkpoint.exists()
    assert history.exists()
    rows = list(csv.DictReader(history.open()))
    assert len(rows) == 1
    assert {"epoch", "train_loss", "val_loss", "precision", "recall", "mae_khz"}.issubset(
        rows[0].keys()
    )
```

Create `tests/test_evaluate_cli.py`:

```python
import json

from sali_pycce.evaluate import main as evaluate_main
from sali_pycce.train import main as train_main


def test_evaluate_writes_json_metrics(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    metrics = tmp_path / "metrics.json"

    train_main(
        [
            "--preset",
            "smoke",
            "--train-samples",
            "8",
            "--val-samples",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--signal-points",
            "64",
            "--heatmap-height",
            "32",
            "--heatmap-width",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--out",
            str(checkpoint),
            "--threads",
            "1",
        ]
    )
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--samples",
            "4",
            "--batch-size",
            "2",
            "--signal-points",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--metrics-out",
            str(metrics),
            "--threads",
            "1",
        ]
    )

    payload = json.loads(metrics.read_text())
    assert {"samples", "tp", "fp", "fn", "precision", "recall", "matched_mae_khz"}.issubset(
        payload.keys()
    )


def test_evaluate_uses_preset_test_samples_when_samples_omitted(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    metrics = tmp_path / "metrics.json"

    train_main(
        [
            "--preset",
            "smoke",
            "--train-samples",
            "8",
            "--val-samples",
            "4",
            "--test-samples",
            "128",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--signal-points",
            "64",
            "--heatmap-height",
            "32",
            "--heatmap-width",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--out",
            str(checkpoint),
            "--threads",
            "1",
        ]
    )
    evaluate_main(
        [
            "--preset",
            "smoke",
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            "64",
            "--signal-points",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--metrics-out",
            str(metrics),
            "--threads",
            "1",
        ]
    )

    payload = json.loads(metrics.read_text())
    assert payload["samples"] == 128
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_train_cli.py tests/test_evaluate_cli.py -q
```

Expected:

```text
SystemExit: 2
```

because `--preset`, `--history-out`, or `--metrics-out` is not recognized.

- [ ] **Step 3: Add preset and shape arguments to `train.py`**

Modify imports in `src/sali_pycce/train.py`:

```python
import csv

from .config import PRESETS
from .metrics import aggregate_detection_metrics, compute_detection_metrics
```

Add parser arguments:

```python
p.add_argument("--train-samples", type=int, default=None)
p.add_argument("--val-samples", type=int, default=None)
p.add_argument("--test-samples", type=int, default=None)
p.add_argument("--signal-points", type=int, default=None)
p.add_argument("--max-spins", type=int, default=None)
p.add_argument("--min-spins", type=int, default=None)
p.add_argument("--b-gauss", type=float, default=None)
p.add_argument("--shots", type=int, default=None)
p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
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
p.add_argument("--history-out", default=None)
p.add_argument("--metric-threshold", type=float, default=0.25)
p.add_argument("--match-distance-khz", type=float, default=5.0)
```

Add this helper near `make_loaders()`:

```python
def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    for name in ("train_samples", "val_samples", "test_samples", "signal_points", "max_spins", "min_spins", "b_gauss", "shots"):
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
```

In `main()`, call `args = resolve_args(args)` before `make_loaders(args)`.

- [ ] **Step 4: Update loader creation and history writing**

In `make_loaders(args)`, construct:

```python
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
```

Add a helper:

```python
@torch.no_grad()
def validation_metrics(model, loader, spec, device: str, threshold: float, max_dist_khz: float) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for batch in loader:
        pred = model(batch["signals"].to(device)).cpu().numpy()
        truth_spins = batch["spins"].numpy()
        for i in range(pred.shape[0]):
            detections = decode_heatmap(pred[i, 0], spec, threshold=threshold)
            truth = truth_spins[i]
            truth = truth[~np.isnan(truth[:, 0])]
            rows.append(compute_detection_metrics(truth, detections, max_dist_khz=max_dist_khz))
    return aggregate_detection_metrics(rows)
```

After each epoch:

```python
metrics = validation_metrics(
    model,
    val_loader,
    spec,
    device,
    threshold=args.metric_threshold,
    max_dist_khz=args.match_distance_khz,
)
history_row = {
    "epoch": epoch,
    "train_loss": running / max(step, 1),
    "val_loss": val_loss,
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "mae_khz": metrics["mae_khz"],
}
history_rows.append(history_row)
```

Write history after each epoch:

```python
history_out = Path(args.history_out) if args.history_out else out.with_suffix(".history.csv")
history_out.parent.mkdir(parents=True, exist_ok=True)
with history_out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
    writer.writeheader()
    writer.writerows(history_rows)
```

- [ ] **Step 5: Add metrics JSON support to `evaluate.py`**

Modify imports in `src/sali_pycce/evaluate.py`:

```python
import json

from .config import PRESETS
```

Add parser argument:

```python
p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
p.add_argument("--metrics-out", default=None)
```

Change the existing `--samples` argument default to `None`:

```python
p.add_argument("--samples", type=int, default=None)
```

After loading the checkpoint in `evaluate.py`, resolve the held-out analytic test sample count:

```python
preset = PRESETS[args.preset]
if args.samples is None:
    ckpt_args = ckpt.get("args", {})
    args.samples = int(ckpt_args.get("test_samples", preset.test_samples))
```

After computing metrics:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_train_cli.py tests/test_evaluate_cli.py -q
```

Expected:

```text
passed
```

- [ ] **Step 7: Commit**

Run:

```bash
git add src/sali_pycce/train.py src/sali_pycce/evaluate.py tests/test_train_cli.py tests/test_evaluate_cli.py
git commit -m "feat: persist training and evaluation metrics"
```

---

### Task 6: Add Tensor Projection Utilities

**Files:**
- Create: `src/sali_pycce/tensor.py`
- Test: `tests/test_tensor.py`

- [ ] **Step 1: Write failing tensor tests**

Create `tests/test_tensor.py`:

```python
import numpy as np
import pytest

from sali_pycce.tensor import project_hyperfine_tensor, project_hyperfine_tensors


def test_project_hyperfine_tensor_khz_with_z_axis():
    tensor = np.asarray(
        [
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )

    az, aperp = project_hyperfine_tensor(tensor, axis=(0.0, 0.0, 1.0), units="kHz")

    assert az == 5.0
    np.testing.assert_allclose(aperp, 5.0)


def test_project_hyperfine_tensor_hz_to_khz():
    tensor_hz = np.diag([0.0, 0.0, 12_000.0])

    az, aperp = project_hyperfine_tensor(tensor_hz, units="Hz")

    assert az == 12.0
    assert aperp == 0.0


def test_project_hyperfine_tensors_batch_shape():
    tensors = np.asarray([np.diag([0.0, 0.0, 3.0]), np.diag([0.0, 0.0, -7.0])])

    out = project_hyperfine_tensors(tensors, units="kHz")

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[:, 0], [3.0, -7.0])
    np.testing.assert_allclose(out[:, 1], [0.0, 0.0])


def test_project_hyperfine_tensor_rejects_unknown_units():
    with pytest.raises(ValueError, match="Unsupported hyperfine tensor units"):
        project_hyperfine_tensor(np.eye(3), units="cycles")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_tensor.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'sali_pycce.tensor'
```

- [ ] **Step 3: Implement tensor projection**

Create `src/sali_pycce/tensor.py`:

```python
"""Hyperfine tensor projection utilities."""

from __future__ import annotations

import numpy as np


def _to_khz(tensor: np.ndarray, units: str) -> np.ndarray:
    normalized = units.strip().lower()
    if normalized == "khz":
        return tensor.astype(float, copy=False)
    if normalized == "hz":
        return tensor.astype(float, copy=False) / 1_000.0
    if normalized == "mhz":
        return tensor.astype(float, copy=False) * 1_000.0
    if normalized in {"rad/us", "rad_per_us"}:
        return tensor.astype(float, copy=False) / (2.0 * np.pi * 1e-3)
    raise ValueError(f"Unsupported hyperfine tensor units: {units!r}")


def _unit_axis(axis: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    vec = np.asarray(axis, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"axis must have shape (3,), got {vec.shape}")
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("axis must be nonzero")
    return vec / norm


def project_hyperfine_tensor(
    tensor: np.ndarray,
    axis: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
    units: str = "kHz",
) -> tuple[float, float]:
    arr = _to_khz(np.asarray(tensor, dtype=float), units)
    if arr.shape != (3, 3):
        raise ValueError(f"tensor must have shape (3, 3), got {arr.shape}")
    b_hat = _unit_axis(axis)
    column = arr @ b_hat
    az = float(b_hat @ column)
    aperp_sq = float(column @ column - az * az)
    aperp = float(np.sqrt(max(aperp_sq, 0.0)))
    return az, aperp


def project_hyperfine_tensors(
    tensors: np.ndarray,
    axis: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
    units: str = "kHz",
) -> np.ndarray:
    arr = np.asarray(tensors, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"tensors must have shape (n, 3, 3), got {arr.shape}")
    return np.asarray(
        [project_hyperfine_tensor(tensor, axis=axis, units=units) for tensor in arr],
        dtype=np.float32,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_tensor.py -q
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/sali_pycce/tensor.py tests/test_tensor.py
git commit -m "feat: project hyperfine tensors to model coordinates"
```

---

### Task 7: Add Algorithmic Decomposition Baseline Adapter

**Files:**
- Create: `src/sali_pycce/baseline.py`
- Test: `tests/test_baseline.py`

- [ ] **Step 1: Write failing baseline tests**

Create `tests/test_baseline.py`:

```python
from types import SimpleNamespace

import numpy as np
import pytest

from sali_pycce.baseline import baseline_spin_to_detection, convert_baseline_spins, import_cpmg_model


def test_baseline_spin_conversion_uses_cpmg_model_sign_convention():
    spin = SimpleNamespace(a_khz=15.0, b_khz=40.0, rmse=0.02)

    det = baseline_spin_to_detection(spin)

    assert det["az_khz"] == -15.0
    assert det["aperp_khz"] == 40.0
    assert det["source_a_khz"] == 15.0
    assert det["source_b_khz"] == 40.0
    assert det["rmse"] == 0.02


def test_convert_baseline_spins_returns_detection_list():
    spins = [SimpleNamespace(a_khz=1.0, b_khz=2.0), SimpleNamespace(a_khz=-3.0, b_khz=4.0)]

    out = convert_baseline_spins(spins)

    assert out == [
        {"az_khz": -1.0, "aperp_khz": 2.0, "source_a_khz": 1.0, "source_b_khz": 2.0},
        {"az_khz": 3.0, "aperp_khz": 4.0, "source_a_khz": -3.0, "source_b_khz": 4.0},
    ]


def test_import_cpmg_model_missing_path_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="cpmg_model path does not exist"):
        import_cpmg_model(tmp_path / "missing")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_baseline.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'sali_pycce.baseline'
```

- [ ] **Step 3: Implement baseline adapter**

Create `src/sali_pycce/baseline.py`:

```python
"""Adapter for the local dqpmodel/cpmg_model decomposition baseline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CPMG_MODEL_PATH = Path("/Users/weitao/Code/python/dqpmodel/cpmg_model")


@dataclass(frozen=True)
class BaselineConfig:
    repo_path: Path = DEFAULT_CPMG_MODEL_PATH
    n_pulses: int = 32
    magnetic_field_gauss: float = 525.0
    threshold: float = 0.05
    dmax_us: float = 0.01
    clone_layers: int = 3
    beam_width: int = 20
    max_fit_evals: int = 4000


def import_cpmg_model(repo_path: str | Path = DEFAULT_CPMG_MODEL_PATH):
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"cpmg_model path does not exist: {path}")
    src = path / "src"
    if not src.exists():
        raise FileNotFoundError(f"cpmg_model src path does not exist: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cpmg_analysis import ExtractionConfig, extract_hyperfine_parameters

    return ExtractionConfig, extract_hyperfine_parameters


def baseline_spin_to_detection(spin) -> dict[str, float]:
    row = {
        "az_khz": -float(spin.a_khz),
        "aperp_khz": float(spin.b_khz),
        "source_a_khz": float(spin.a_khz),
        "source_b_khz": float(spin.b_khz),
    }
    if hasattr(spin, "rmse"):
        row["rmse"] = float(spin.rmse)
    return row


def convert_baseline_spins(spins) -> list[dict[str, float]]:
    return [baseline_spin_to_detection(spin) for spin in spins]


def run_cpmg_model_baseline(
    tau_us: np.ndarray,
    signal: np.ndarray,
    config: BaselineConfig = BaselineConfig(),
) -> list[dict[str, float]]:
    ExtractionConfig, extract_hyperfine_parameters = import_cpmg_model(config.repo_path)
    extractor_config = ExtractionConfig(
        n_pulses=config.n_pulses,
        magnetic_field_gauss=config.magnetic_field_gauss,
        threshold=config.threshold,
        dmax_us=config.dmax_us,
        clone_layers=config.clone_layers,
        beam_width=config.beam_width,
        max_fit_evals=config.max_fit_evals,
    )
    result = extract_hyperfine_parameters(
        np.asarray(tau_us, dtype=float),
        np.asarray(signal, dtype=float),
        config=extractor_config,
    )
    return convert_baseline_spins(result.spins)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_baseline.py -q
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/sali_pycce/baseline.py tests/test_baseline.py
git commit -m "feat: adapt cpmg decomposition baseline"
```

---

### Task 8: Add PyCCE Random-Bath Benchmark Generation

**Files:**
- Create: `src/sali_pycce/pycce_benchmark.py`
- Modify: `src/sali_pycce/pycce_backend.py`
- Test: `tests/test_pycce_benchmark.py`

- [ ] **Step 1: Write tests that do not require PyCCE installation**

Create `tests/test_pycce_benchmark.py`:

```python
import numpy as np

from sali_pycce.pycce_benchmark import (
    PyCCEBenchmarkConfig,
    bath_arrays_to_spin_table,
    detectable_spin_mask,
)


def test_detectable_spin_mask_uses_range_and_depth():
    spin_table = np.asarray(
        [
            [0.0, 50.0, 0.10],
            [300.0, 50.0, 0.20],
            [0.0, 1.0, 0.20],
            [0.0, 50.0, 0.001],
        ],
        dtype=float,
    )
    config = PyCCEBenchmarkConfig(
        az_range=(-250.0, 250.0),
        aperp_range=(2.0, 250.0),
        min_signal_depth=0.01,
    )

    mask = detectable_spin_mask(spin_table, config)

    assert mask.tolist() == [True, False, False, False]


def test_bath_arrays_to_spin_table_uses_projected_parameters():
    xyz = np.asarray([[1.0, 2.0, 3.0]])
    tensors = np.asarray([[[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [3.0, 4.0, 5.0]]])
    names = np.asarray(["13C"])

    table = bath_arrays_to_spin_table(names, xyz, tensors, tensor_units="kHz")

    assert table.shape == (1, 9)
    assert table[0, 0] == 0.0
    assert table[0, 1] == 1.0
    assert table[0, 2] == 2.0
    assert table[0, 3] == 3.0
    assert table[0, 4] == 5.0
    assert table[0, 5] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_pycce_benchmark.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'sali_pycce.pycce_benchmark'
```

- [ ] **Step 3: Implement benchmark data helpers**

Create `src/sali_pycce/pycce_benchmark.py`:

```python
"""PyCCE random-bath benchmark generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import RESEARCH_CONFIG
from .heatmap import make_heatmap
from .physics import SpinParams
from .tensor import project_hyperfine_tensors


@dataclass(frozen=True)
class PyCCEBenchmarkConfig:
    b_gauss: float = 525.0
    pulses: tuple[int, int] = (32, 256)
    tau_range_us: tuple[float, float] = (0.0, 40.0)
    signal_points: int = 4000
    bath_size_angstrom: float = 100.0
    bath_number: int = 2000
    seed: int = 1234
    az_range: tuple[float, float] = (-250.0, 250.0)
    aperp_range: tuple[float, float] = (2.0, 250.0)
    min_signal_depth: float = 0.01
    tensor_units: str = "kHz"
    pycce_time_units: str = "ms"


def bath_arrays_to_spin_table(
    names: np.ndarray,
    xyz: np.ndarray,
    tensors: np.ndarray,
    tensor_units: str = "kHz",
) -> np.ndarray:
    projected = project_hyperfine_tensors(tensors, units=tensor_units)
    table = np.zeros((projected.shape[0], 9), dtype=np.float32)
    table[:, 0] = np.arange(projected.shape[0], dtype=np.float32)
    table[:, 1:4] = np.asarray(xyz, dtype=np.float32)
    table[:, 4:6] = projected
    table[:, 6] = np.linalg.norm(np.asarray(xyz, dtype=float), axis=1)
    table[:, 7] = np.asarray([1.0 if str(name) == "13C" else 0.0 for name in names], dtype=np.float32)
    table[:, 8] = 0.0
    return table


def detectable_spin_mask(spin_table: np.ndarray, config: PyCCEBenchmarkConfig) -> np.ndarray:
    if spin_table.ndim != 2 or spin_table.shape[1] != 9:
        raise ValueError(f"spin_table must have shape (n, 9), got {spin_table.shape}")
    az = spin_table[:, 4]
    aperp = spin_table[:, 5]
    depth = spin_table[:, 8]
    return (
        (az >= config.az_range[0])
        & (az <= config.az_range[1])
        & (aperp >= config.aperp_range[0])
        & (aperp <= config.aperp_range[1])
        & (depth >= config.min_signal_depth)
    )
```

Add an analytic single-spin depth helper for the detectable mask:

```python
from .physics import AnalyticCPMGSimulator


def add_signal_depths(spin_table: np.ndarray, config: PyCCEBenchmarkConfig) -> np.ndarray:
    table = np.asarray(spin_table, dtype=np.float32).copy()
    sim = AnalyticCPMGSimulator(
        b_gauss=config.b_gauss,
        pulses=config.pulses,
        tau_ranges_us=(config.tau_range_us, config.tau_range_us),
        signal_points=config.signal_points,
        t2_us=None,
        shots=None,
    )
    tau = sim.tau_grid(0)
    for row in table:
        spin = SpinParams(az_khz=float(row[4]), aperp_khz=float(row[5]))
        depths = []
        for n_pulses in config.pulses:
            signal = sim.signal([spin], n_pulses, tau)
            depths.append(float(1.0 - np.min(signal)))
        row[8] = max(depths)
    return table
```

- [ ] **Step 4: Implement PyCCE generation entry points**

In `src/sali_pycce/pycce_benchmark.py`, add:

```python
def require_pycce():
    try:
        import pycce as pc
    except Exception as exc:
        raise ImportError("PyCCE is required for PyCCE benchmark generation. Install with `pip install pycce`.") from exc
    return pc


def generate_random_c13_bath(config: PyCCEBenchmarkConfig):
    pc = require_pycce()
    bath = pc.random_bath("13C", config.bath_size_angstrom, number=config.bath_number, seed=config.seed)
    bath.from_point_dipole(np.zeros(3), inplace=True)
    return bath


def simulate_bath_traces(bath, config: PyCCEBenchmarkConfig) -> dict[str, np.ndarray]:
    pc = require_pycce()
    taus = np.linspace(config.tau_range_us[0], config.tau_range_us[1], config.signal_points)
    signals = []
    for n_pulses in config.pulses:
        sim = pc.Simulator(
            1,
            bath=bath,
            magnetic_field=np.asarray([0.0, 0.0, config.b_gauss]),
            pulses=int(n_pulses),
            as_delay=True,
            order=1,
        )
        pycce_times = taus * 1e-3
        coherence = sim.compute(pycce_times)
        px = 0.5 * (1.0 + np.real(coherence))
        signals.append(np.clip(px, 0.0, 1.0).astype(np.float32))
    return {
        "taus_us": np.stack([taus.astype(np.float32), taus.astype(np.float32)], axis=0),
        "signals": np.stack(signals, axis=0),
        "pycce_times": np.stack([pycce_times.astype(np.float32), pycce_times.astype(np.float32)], axis=0),
    }
```

- [ ] **Step 5: Add CLI save path**

Add:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a PyCCE random-bath SALI benchmark sample.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bath-number", type=int, default=2000)
    parser.add_argument("--bath-size-angstrom", type=float, default=100.0)
    parser.add_argument("--signal-points", type=int, default=4000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PyCCEBenchmarkConfig(
        seed=args.seed,
        bath_number=args.bath_number,
        bath_size_angstrom=args.bath_size_angstrom,
        signal_points=args.signal_points,
    )
    bath = generate_random_c13_bath(config)
    traces = simulate_bath_traces(bath, config)
    spin_table = bath_arrays_to_spin_table(bath.N, bath.xyz, bath.A, tensor_units=config.tensor_units)
    spin_table = add_signal_depths(spin_table, config)
    mask = detectable_spin_mask(spin_table, config)
    spec = RESEARCH_CONFIG.build_heatmap_spec()
    spins = [SpinParams(float(row[4]), float(row[5])) for row in spin_table[mask]]
    heatmap = make_heatmap(spins, spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        signals=traces["signals"],
        taus_us=traces["taus_us"],
        spin_table=spin_table,
        detectable_mask=mask,
        heatmap=heatmap[None],
        metadata=json.dumps(asdict(config)),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

Add the console script in `pyproject.toml`:

```toml
sali-pycce-benchmark = "sali_pycce.pycce_benchmark:main"
```

- [ ] **Step 6: Run non-PyCCE tests**

Run:

```bash
python -m pytest tests/test_pycce_benchmark.py -q
```

Expected:

```text
passed
```

- [ ] **Step 7: Run PyCCE smoke command when PyCCE is installed**

Run:

```bash
python -m sali_pycce.pycce_benchmark --out runs/pycce_smoke.npz --bath-number 20 --signal-points 64
```

Expected with PyCCE installed:

```text
wrote runs/pycce_smoke.npz
```

Expected without PyCCE:

```text
ImportError: PyCCE is required for PyCCE benchmark generation. Install with `pip install pycce`.
```

- [ ] **Step 8: Commit**

Run:

```bash
git add src/sali_pycce/pycce_benchmark.py src/sali_pycce/pycce_backend.py pyproject.toml tests/test_pycce_benchmark.py
git commit -m "feat: add PyCCE benchmark generation"
```

---

### Task 9: Add Visualization Helpers For Required Notebook Figures

**Files:**
- Create: `src/sali_pycce/visualize.py`
- Modify: `examples/predict_one.py`
- Test: `tests/test_visualize.py`

- [ ] **Step 1: Write failing visualization tests**

Create `tests/test_visualize.py`:

```python
import csv

import numpy as np

from sali_pycce.heatmap import HeatmapSpec
from sali_pycce.visualize import (
    plot_heatmap_comparison,
    plot_loss_history,
    plot_metric_history,
    plot_raw_traces_and_spins,
)


def test_visualization_helpers_write_pngs(tmp_path):
    taus = np.stack([np.linspace(0.0, 40.0, 32), np.linspace(0.0, 40.0, 32)])
    signals = np.stack([np.linspace(1.0, 0.8, 32), np.linspace(1.0, 0.7, 32)])
    spins = np.asarray([[10.0, 20.0], [-30.0, 90.0]])
    spec = HeatmapSpec(height=32, width=64)
    target = np.zeros((32, 64), dtype=np.float32)
    pred = np.zeros((32, 64), dtype=np.float32)
    history = tmp_path / "history.csv"
    with history.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "precision", "recall", "mae_khz"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "train_loss": 0.2,
                "val_loss": 0.3,
                "precision": 0.4,
                "recall": 0.5,
                "mae_khz": 6.0,
            }
        )

    paths = [
        plot_raw_traces_and_spins(taus, signals, spins, spec, tmp_path / "raw.png"),
        plot_heatmap_comparison(target, pred, tmp_path / "heatmaps.png"),
        plot_loss_history(history, tmp_path / "loss.png"),
        plot_metric_history(history, tmp_path / "metrics.png"),
    ]

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_visualize.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'sali_pycce.visualize'
```

- [ ] **Step 3: Implement visualization helpers**

Create `src/sali_pycce/visualize.py`:

```python
"""Plot helpers for SALI-PyCCE notebooks and reports."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .heatmap import HeatmapSpec


def _finish(fig, out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_raw_traces_and_spins(
    taus_us: np.ndarray,
    signals: np.ndarray,
    spins: np.ndarray,
    spec: HeatmapSpec,
    out: str | Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(taus_us[0], signals[0], label="N=32")
    axes[0].plot(taus_us[1], signals[1], label="N=256")
    axes[0].set_xlabel("tau (us)")
    axes[0].set_ylabel("P_x")
    axes[0].set_title("Raw CPMG traces")
    axes[0].legend()
    axes[1].scatter(spins[:, 0], spins[:, 1], s=28)
    axes[1].set_xlim(*spec.az_range)
    axes[1].set_ylim(*spec.aperp_range)
    axes[1].set_xlabel("A_z (kHz)")
    axes[1].set_ylabel("A_perp (kHz)")
    axes[1].set_title("True spins")
    return _finish(fig, out)


def plot_heatmap_comparison(target: np.ndarray, pred: np.ndarray, out: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].imshow(np.asarray(target).squeeze(), origin="lower", aspect="auto")
    axes[0].set_title("Ground-truth heatmap")
    axes[1].imshow(np.asarray(pred).squeeze(), origin="lower", aspect="auto")
    axes[1].set_title("Predicted heatmap")
    return _finish(fig, out)


def _read_history(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open() as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def plot_loss_history(history_csv: str | Path, out: str | Path) -> Path:
    rows = _read_history(history_csv)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]
    ax.plot(epochs, [row["train_loss"] for row in rows], label="train")
    ax.plot(epochs, [row["val_loss"] for row in rows], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train/validation loss")
    ax.legend()
    return _finish(fig, out)


def plot_metric_history(history_csv: str | Path, out: str | Path) -> Path:
    rows = _read_history(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]
    axes[0].plot(epochs, [row["precision"] for row in rows], label="precision")
    axes[0].plot(epochs, [row["recall"] for row in rows], label="recall")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    axes[1].plot(epochs, [row["mae_khz"] for row in rows], label="MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("kHz")
    axes[1].legend()
    return _finish(fig, out)
```

- [ ] **Step 4: Update `examples/predict_one.py` to use shared plotting**

Keep its CLI, but call `plot_heatmap_comparison` for the target/predicted heatmap panel or leave the existing custom figure if it remains simpler. Ensure it still writes `runs/prediction_demo.png`.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_visualize.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add src/sali_pycce/visualize.py examples/predict_one.py tests/test_visualize.py
git commit -m "feat: add report visualization helpers"
```

---

### Task 10: Update Colab Notebook And Add Colab Runner Script

**Files:**
- Create: `scripts/colab_train.py`
- Modify: `notebooks/train_in_colab.ipynb`
- Test: `tests/test_colab_script.py`

- [ ] **Step 1: Write script import test**

Create `tests/test_colab_script.py`:

```python
import importlib.util
from pathlib import Path


def test_colab_train_script_is_importable():
    path = Path("scripts/colab_train.py")
    spec = importlib.util.spec_from_file_location("colab_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "build_parser")
    assert hasattr(module, "main")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_colab_script.py -q
```

Expected:

```text
FileNotFoundError
```

- [ ] **Step 3: Create Colab runner script**

Create `scripts/colab_train.py`:

```python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SALI-PyCCE training/evaluation in Colab.")
    parser.add_argument("--preset", default="colab-medium", choices=["smoke", "colab-medium", "research"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoints/colab_medium.pt")
    parser.add_argument("--history", default="runs/colab_medium_history.csv")
    parser.add_argument("--metrics", default="runs/colab_medium_metrics.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)
    run(["python", "-m", "pip", "install", "-q", "-e", ".[dev]"])
    run(["python", "-m", "pytest", "-q"])
    run(
        [
            "python",
            "-m",
            "sali_pycce.train",
            "--preset",
            args.preset,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--out",
            args.checkpoint,
            "--history-out",
            args.history,
        ]
    )
    run(
        [
            "python",
            "-m",
            "sali_pycce.evaluate",
            "--checkpoint",
            args.checkpoint,
            "--samples",
            str(args.eval_samples),
            "--device",
            args.device,
            "--metrics-out",
            args.metrics,
        ]
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update notebook cells**

Modify `notebooks/train_in_colab.ipynb` so it contains sections and executable cells for:

```text
1. Environment and GPU check
2. Repository setup and install
3. Display one synthetic research sample:
   - raw N=32 and N=256 traces
   - true spin scatter plot
   - ground-truth heatmap label
4. Colab smoke training command
5. Colab medium/research training command through scripts/colab_train.py
6. Loss curve from history CSV
7. Target heatmap versus predicted heatmap
8. Precision/recall/MAE plot from history CSV or threshold sweep
9. Save checkpoints and runs to Drive
```

Use these command cells:

```python
!python scripts/colab_train.py \
  --preset colab-medium \
  --epochs 20 \
  --batch-size 128 \
  --checkpoint checkpoints/colab_medium.pt \
  --history runs/colab_medium_history.csv \
  --metrics runs/colab_medium_metrics.json
```

and display plots with:

```python
from IPython.display import Image, display
display(Image("runs/raw_traces_and_spins.png"))
display(Image("runs/loss_curve.png"))
display(Image("runs/metric_curve.png"))
display(Image("runs/heatmap_comparison.png"))
```

- [ ] **Step 5: Run import test**

Run:

```bash
python -m pytest tests/test_colab_script.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/colab_train.py notebooks/train_in_colab.ipynb tests/test_colab_script.py
git commit -m "feat: add Colab training workflow"
```

---

### Task 11: Run Colab CLI Training Smoke

**Files:**
- No source edits expected.
- Generated artifacts under `runs/` and `checkpoints/` are ignored unless the user asks to preserve a small example.

- [ ] **Step 1: Verify Colab CLI authentication**

Run:

```bash
colab sessions
```

Expected:

```text
[any existing session list, or no sessions]
```

The exact session list can be empty. A 401 or 403 means authentication scopes need repair before training.

- [ ] **Step 2: Run a self-cleaning Colab smoke job**

Run:

```bash
colab run --gpu T4 -s sali-pycce-smoke scripts/colab_train.py --preset smoke --epochs 1 --batch-size 16 --checkpoint checkpoints/colab_smoke.pt --history runs/colab_smoke_history.csv --metrics runs/colab_smoke_metrics.json
```

Expected:

```text
[colab] provisioning session sali-pycce-smoke
```

The script output should include successful package installation, `pytest`, training, and evaluation metrics. `colab run` should stop the VM automatically when the script exits.

- [ ] **Step 3: If T4 allocation fails, retry CPU smoke**

Run:

```bash
colab run -s sali-pycce-smoke-cpu scripts/colab_train.py --preset smoke --epochs 1 --batch-size 16 --device cpu --checkpoint checkpoints/colab_smoke_cpu.pt --history runs/colab_smoke_cpu_history.csv --metrics runs/colab_smoke_cpu_metrics.json
```

Expected:

```text
[colab] provisioning session sali-pycce-smoke-cpu
```

The script output should include successful package installation, `pytest`, training, and evaluation metrics.

- [ ] **Step 4: Record Colab result in final implementation summary**

Do not commit generated checkpoints by default. Report:

```text
Colab smoke: passed/failed
Accelerator: T4 or CPU
Checkpoint path inside Colab job
History path inside Colab job
Metrics path inside Colab job
```

---

### Task 12: Add README Usage And Run Full Local Verification

**Files:**
- Modify: `README.md`
- Possibly modify: `docs/hyak.md` if references conflict with new defaults.

- [ ] **Step 1: Update README**

Add a section:

```markdown
## Research Pipeline Defaults

The research workflow trains on analytic/noisy CPMG traces, validates during training, evaluates on held-out analytic test data, and benchmarks on PyCCE random spin baths.

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
python -m sali_pycce.train --preset colab-medium --epochs 20 --batch-size 128 --device cuda --out checkpoints/colab_medium.pt --history-out runs/colab_medium_history.csv
```

Evaluate:

```bash
python -m sali_pycce.evaluate --checkpoint checkpoints/colab_medium.pt --samples 500 --device cuda --metrics-out runs/colab_medium_metrics.json
```

PyCCE benchmark:

```bash
python -m sali_pycce.pycce_benchmark --out runs/pycce_sample.npz --bath-number 2000 --signal-points 4000
```

Baseline convention:

```text
cpmg_model A = -A_par
cpmg_model B = A_perp
comparison uses A_z = -A, A_perp = B
```
```

- [ ] **Step 2: Run full local tests**

Run:

```bash
python -m pytest -q
```

Expected:

```text
passed
```

- [ ] **Step 3: Run local smoke training**

Run:

```bash
python -m sali_pycce.train --preset smoke --epochs 1 --batch-size 16 --device cpu --out checkpoints/local_smoke.pt --history-out runs/local_smoke_history.csv --threads 1
```

Expected:

```text
saved checkpoints/local_smoke.pt
```

- [ ] **Step 4: Run local smoke evaluation**

Run:

```bash
python -m sali_pycce.evaluate --checkpoint checkpoints/local_smoke.pt --samples 8 --batch-size 4 --device cpu --metrics-out runs/local_smoke_metrics.json --threads 1
```

Expected:

```text
precision=
```

and `runs/local_smoke_metrics.json` exists.

- [ ] **Step 5: Commit docs and final fixes**

Run:

```bash
git add README.md docs/hyak.md src tests notebooks scripts pyproject.toml examples
git commit -m "docs: document research training workflow"
```

---

## Final Verification Checklist

- [ ] `python -m pytest -q` passes locally.
- [ ] `python -m sali_pycce.train --preset smoke --epochs 1 --batch-size 16 --device cpu --out checkpoints/local_smoke.pt --history-out runs/local_smoke_history.csv --threads 1` writes a checkpoint and history CSV.
- [ ] `python -m sali_pycce.evaluate --checkpoint checkpoints/local_smoke.pt --samples 8 --batch-size 4 --device cpu --metrics-out runs/local_smoke_metrics.json --threads 1` writes JSON metrics.
- [ ] `notebooks/train_in_colab.ipynb` includes all six required visual outputs.
- [ ] PyCCE benchmark command either writes an NPZ when PyCCE is installed or raises the documented install error.
- [ ] Baseline adapter converts `A_z = -A`, `A_perp = B`.
- [ ] Colab CLI smoke run has been attempted and reported.
