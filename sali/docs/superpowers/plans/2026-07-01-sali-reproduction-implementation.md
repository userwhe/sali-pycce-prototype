# SALI Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch reproduction of the SALI paper with synthetic CPMG data generation, target heatmaps, training/evaluation, requested notebook plots, and Colab CLI execution.

**Architecture:** The implementation is a focused Python package under `src/sali`, with one module per responsibility: config, physics, targets, data, model, post-processing, metrics, plots, and training. The notebook and Colab script call package APIs rather than duplicating logic.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, SciPy, scikit-image, Matplotlib, pytest, nbformat, Google Colab CLI.

---

## File Structure

- Create `pyproject.toml`: package metadata, dependencies, pytest configuration.
- Create `README.md`: quickstart, local smoke run, Colab CLI run, paper-scale notes.
- Create `src/sali/__init__.py`: package exports and version.
- Create `src/sali/config.py`: dataclasses for physics, data, model, training, post-processing, run config, and presets.
- Create `src/sali/physics.py`: CPMG equations, signal generation, shot noise, reconstruction helper.
- Create `src/sali/targets.py`: coupling-to-pixel mapping, Gaussian heatmap rendering, box generation.
- Create `src/sali/data.py`: synthetic sample generation, train/val/test split, normalization, PyTorch dataset.
- Create `src/sali/model.py`: PyTorch SALI 1D-to-2D CNN.
- Create `src/sali/train.py`: training loop, validation loop, scheduler, early stopping, checkpoints.
- Create `src/sali/postprocess.py`: morphology, thresholding, components, local maxima, centroid extraction.
- Create `src/sali/metrics.py`: IoU matching, precision/recall, coupling MAE, reconstructed-signal MAE.
- Create `src/sali/plots.py`: plot helpers for spectra, heatmaps, overlays, loss, precision/recall, MAE.
- Create `scripts/train_colab.py`: command-line entrypoint for practical and paper-config runs.
- Create `notebooks/sali_reproduction_colab.ipynb`: Colab-ready visual workflow.
- Create `tests/`: focused tests for each non-expensive unit plus a tiny training smoke test.

## Task 1: Project Scaffold

**Files:**
- Create: `/Users/weitao/Code/python/sali/pyproject.toml`
- Create: `/Users/weitao/Code/python/sali/README.md`
- Create: `/Users/weitao/Code/python/sali/src/sali/__init__.py`
- Create: `/Users/weitao/Code/python/sali/tests/conftest.py`

- [ ] **Step 1: Write package metadata**

Create `pyproject.toml` with this content:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sali-reproduction"
version = "0.1.0"
description = "PyTorch reproduction of the SALI signal-to-image nuclear spin detection model."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "torch>=2.1",
  "scipy>=1.10",
  "scikit-image>=0.22",
  "matplotlib>=3.7",
  "pandas>=2.0",
  "tqdm>=4.66",
  "nbformat>=5.9",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```

- [ ] **Step 2: Write quickstart documentation**

Create `README.md` with this content:

```markdown
# SALI Reproduction

PyTorch reproduction of the SALI signal-to-image model for automatic C13 nuclear-spin detection around NV centers.

The paper reports TensorFlow/Keras training, but this reproduction translates the architecture and training recipe to PyTorch while preserving the described physics simulator, target heatmap representation, post-processing, and evaluation workflow.

## Local Setup

```bash
python -m pip install -e ".[dev]"
pytest
```

## Practical Smoke Run

```bash
python scripts/train_colab.py --preset practical --field low --output-dir runs/practical-low
```

## Colab CLI Run

```bash
colab new -s sali-practical --gpu T4
colab exec -s sali-practical -f scripts/train_colab.py
colab stop -s sali-practical
```

## Paper-Scale Settings

The `paper` preset exposes the paper's sample count, split ratios, signal lengths, batch size, optimizer, learning-rate schedule, and early-stopping patience. It is expected to require substantial accelerator time and storage.
```

- [ ] **Step 3: Add package initializer**

Create `src/sali/__init__.py` with this content:

```python
"""PyTorch reproduction utilities for the SALI signal-to-image model."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Add pytest fixtures**

Create `tests/conftest.py` with this content:

```python
from __future__ import annotations

import numpy as np
import pytest

from sali.config import RunConfig, practical_config


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture()
def tiny_config() -> RunConfig:
    cfg = practical_config(field="low")
    cfg.data.train_samples = 8
    cfg.data.val_samples = 4
    cfg.data.test_samples = 4
    cfg.data.max_nuclei = 3
    cfg.training.batch_size = 4
    cfg.training.max_epochs = 1
    cfg.training.early_stopping_patience = 2
    cfg.training.lr_plateau_patience = 1
    return cfg
```

- [ ] **Step 5: Run scaffold check**

Run:

```bash
python -m compileall src
```

Expected: command exits with status 0.

## Task 2: Configuration Presets

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/config.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config.py` with this content:

```python
from __future__ import annotations

import pytest

from sali.config import FieldConfig, paper_config, practical_config


def test_practical_config_uses_low_field_value() -> None:
    cfg = practical_config(field="low")
    assert cfg.physics.bz_tesla == pytest.approx(0.0056)
    assert cfg.data.train_samples == 2048
    assert cfg.data.val_samples == 512
    assert cfg.data.test_samples == 512
    assert cfg.model.output_height == 204
    assert cfg.model.output_width == 104


def test_paper_config_matches_reported_training_recipe() -> None:
    cfg = paper_config(field="high")
    assert cfg.physics.bz_tesla == pytest.approx(0.056)
    assert cfg.data.total_samples == 3_600_000
    assert cfg.data.train_samples == 2_520_000
    assert cfg.data.val_samples == 540_000
    assert cfg.data.test_samples == 540_000
    assert cfg.training.batch_size == 64
    assert cfg.training.max_epochs == 250
    assert cfg.training.learning_rate == pytest.approx(0.001)
    assert cfg.training.lr_reduction_factor == pytest.approx(0.7)
    assert cfg.training.lr_plateau_patience == 5
    assert cfg.training.early_stopping_patience == 20


def test_invalid_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="field must be"):
        FieldConfig.from_name("medium")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_config.py -q
```

Expected: FAIL because `sali.config` does not exist.

- [ ] **Step 3: Implement config module**

Create `src/sali/config.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path


@dataclass(slots=True)
class FieldConfig:
    name: str
    bz_tesla: float

    @classmethod
    def from_name(cls, field: str) -> "FieldConfig":
        if field == "high":
            return cls(name="high", bz_tesla=0.056)
        if field == "low":
            return cls(name="low", bz_tesla=0.0056)
        raise ValueError("field must be either 'high' or 'low'")


@dataclass(slots=True)
class PhysicsConfig:
    bz_tesla: float
    gamma_mhz_per_t: float = 10.705
    t2_us: float = 200.0
    measurements: int = 1000
    signal_points: int = 1000
    tau_32_min_us: float = 6.0
    tau_32_max_us: float = 50.0
    tau_256_min_us: float = 10.0
    tau_256_max_us: float = 40.0
    add_decoherence: bool = True
    add_shot_noise: bool = True


@dataclass(slots=True)
class DataConfig:
    train_samples: int
    val_samples: int
    test_samples: int
    min_nuclei: int = 1
    max_nuclei: int = 20
    az_min_khz: float = -100.0
    az_max_khz: float = 100.0
    aperp_min_khz: float = 2.0
    aperp_max_khz: float = 102.0
    norm_epsilon: float = 0.001
    seed: int = 12345

    @property
    def total_samples(self) -> int:
        return self.train_samples + self.val_samples + self.test_samples


@dataclass(slots=True)
class ModelConfig:
    signal_length: int = 1000
    conv1_filters: int = 16
    conv2_filters: int = 32
    dense_height: int = 102
    dense_width: int = 52
    output_height: int = 204
    output_width: int = 104
    dropout: float = 0.2


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 250
    learning_rate: float = 0.001
    lr_reduction_factor: float = 0.7
    lr_plateau_patience: int = 5
    early_stopping_patience: int = 20
    min_delta: float = 0.0
    device: str = "auto"


@dataclass(slots=True)
class PostprocessConfig:
    threshold: float = 0.25
    min_area: int = 3
    local_max_min_distance_px: int = 3
    erosion_size: int = 1
    dilation_size: int = 1


@dataclass(slots=True)
class RunConfig:
    physics: PhysicsConfig
    data: DataConfig
    model: ModelConfig = dataclass_field(default_factory=ModelConfig)
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)
    postprocess: PostprocessConfig = dataclass_field(default_factory=PostprocessConfig)
    output_dir: Path = Path("runs/practical")


def practical_config(field: str = "low") -> RunConfig:
    field_cfg = FieldConfig.from_name(field)
    return RunConfig(
        physics=PhysicsConfig(bz_tesla=field_cfg.bz_tesla),
        data=DataConfig(train_samples=2048, val_samples=512, test_samples=512),
        training=TrainingConfig(max_epochs=10),
        output_dir=Path(f"runs/practical-{field}"),
    )


def paper_config(field: str = "low") -> RunConfig:
    field_cfg = FieldConfig.from_name(field)
    return RunConfig(
        physics=PhysicsConfig(bz_tesla=field_cfg.bz_tesla),
        data=DataConfig(
            train_samples=2_520_000,
            val_samples=540_000,
            test_samples=540_000,
        ),
        training=TrainingConfig(),
        output_dir=Path(f"runs/paper-{field}"),
    )
```

- [ ] **Step 4: Run config tests**

Run:

```bash
pytest tests/test_config.py -q
```

Expected: PASS.

## Task 3: Physics Signal Generation

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/physics.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_physics.py`

- [ ] **Step 1: Write failing physics tests**

Create `tests/test_physics.py` with this content:

```python
from __future__ import annotations

import numpy as np

from sali.config import PhysicsConfig
from sali.physics import Couplings, generate_sample_signals, simulate_cpmg_signal, tau_grid_us


def test_tau_grid_uses_requested_bounds() -> None:
    tau = tau_grid_us(6.0, 50.0, 1000)
    assert tau.shape == (1000,)
    assert tau[0] == np.float32(6.0)
    assert tau[-1] == np.float32(50.0)


def test_simulate_cpmg_signal_is_finite_and_probability_like(rng: np.random.Generator) -> None:
    cfg = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([-20.0, 35.0], dtype=np.float32),
        aperp_khz=np.array([12.0, 50.0], dtype=np.float32),
    )
    tau = tau_grid_us(6.0, 50.0, 1000)
    signal = simulate_cpmg_signal(nuclei, tau, n_pulses=32, cfg=cfg, rng=rng)
    assert signal.shape == (1000,)
    assert np.isfinite(signal).all()
    assert signal.min() >= 0.0
    assert signal.max() <= 1.0


def test_generate_sample_signals_returns_two_inputs(rng: np.random.Generator) -> None:
    cfg = PhysicsConfig(bz_tesla=0.056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([10.0], dtype=np.float32),
        aperp_khz=np.array([25.0], dtype=np.float32),
    )
    signals = generate_sample_signals(nuclei, cfg, rng)
    assert signals.shape == (2, 1000)
    assert np.isfinite(signals).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_physics.py -q
```

Expected: FAIL because `sali.physics` does not exist.

- [ ] **Step 3: Implement physics module**

Create `src/sali/physics.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sali.config import PhysicsConfig


@dataclass(frozen=True, slots=True)
class Couplings:
    az_khz: np.ndarray
    aperp_khz: np.ndarray

    def __post_init__(self) -> None:
        if self.az_khz.shape != self.aperp_khz.shape:
            raise ValueError("az_khz and aperp_khz must have matching shapes")
        if self.az_khz.ndim != 1:
            raise ValueError("coupling arrays must be one-dimensional")

    @property
    def count(self) -> int:
        return int(self.az_khz.size)


def tau_grid_us(start_us: float, stop_us: float, points: int) -> np.ndarray:
    if points <= 1:
        raise ValueError("points must be greater than 1")
    if stop_us <= start_us:
        raise ValueError("tau stop must be greater than tau start")
    return np.linspace(start_us, stop_us, points, dtype=np.float32)


def _ideal_signal(nuclei: Couplings, tau_us: np.ndarray, n_pulses: int, cfg: PhysicsConfig) -> np.ndarray:
    if nuclei.count == 0:
        return np.ones_like(tau_us, dtype=np.float32)
    tau_s = tau_us.astype(np.float64) * 1e-6
    gamma_rad_s_t = 2.0 * np.pi * cfg.gamma_mhz_per_t * 1e6
    omega_l = gamma_rad_s_t * cfg.bz_tesla
    az = nuclei.az_khz.astype(np.float64) * 2.0 * np.pi * 1e3
    aperp = nuclei.aperp_khz.astype(np.float64) * 2.0 * np.pi * 1e3
    shifted = az + omega_l
    omega_tilde = np.sqrt(shifted[:, None] ** 2 + aperp[:, None] ** 2)
    mz = shifted[:, None] / omega_tilde
    mx = aperp[:, None] / omega_tilde
    alpha = omega_tilde * tau_s[None, :]
    beta = omega_l * tau_s[None, :]
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)
    cos_phi = cos_alpha * cos_beta - mz * sin_alpha * sin_beta
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    denom = 1.0 + cos_alpha * cos_beta - mz * sin_alpha * sin_beta
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + 1e-12, denom)
    modulation = ((1.0 - cos_alpha) * (1.0 - cos_beta)) / denom
    sin_term = np.sin(n_pulses * phi / 2.0) ** 2
    mj = 1.0 - (mx**2) * modulation * sin_term
    product = np.prod(mj, axis=0)
    px = 0.5 * (1.0 + product)
    return np.clip(px, 0.0, 1.0).astype(np.float32)


def _apply_decoherence(signal: np.ndarray, tau_us: np.ndarray, cfg: PhysicsConfig) -> np.ndarray:
    if not cfg.add_decoherence:
        return signal
    decay = np.exp(-tau_us.astype(np.float32) / np.float32(cfg.t2_us))
    return np.clip(signal * decay, 0.0, 1.0).astype(np.float32)


def _apply_shot_noise(signal: np.ndarray, cfg: PhysicsConfig, rng: np.random.Generator) -> np.ndarray:
    if not cfg.add_shot_noise:
        return signal.astype(np.float32)
    counts = rng.binomial(cfg.measurements, np.clip(signal, 0.0, 1.0))
    noisy = counts.astype(np.float32) / np.float32(cfg.measurements)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def simulate_cpmg_signal(
    nuclei: Couplings,
    tau_us: np.ndarray,
    n_pulses: int,
    cfg: PhysicsConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_pulses <= 0:
        raise ValueError("n_pulses must be positive")
    if tau_us.shape != (cfg.signal_points,):
        raise ValueError(f"tau_us must have shape ({cfg.signal_points},)")
    signal = _ideal_signal(nuclei, tau_us, n_pulses, cfg)
    signal = _apply_decoherence(signal, tau_us, cfg)
    signal = _apply_shot_noise(signal, cfg, rng)
    if not np.isfinite(signal).all():
        raise FloatingPointError("generated signal contains non-finite values")
    return signal.astype(np.float32)


def generate_sample_signals(
    nuclei: Couplings,
    cfg: PhysicsConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    tau32 = tau_grid_us(cfg.tau_32_min_us, cfg.tau_32_max_us, cfg.signal_points)
    tau256 = tau_grid_us(cfg.tau_256_min_us, cfg.tau_256_max_us, cfg.signal_points)
    signal32 = simulate_cpmg_signal(nuclei, tau32, 32, cfg, rng)
    signal256 = simulate_cpmg_signal(nuclei, tau256, 256, cfg, rng)
    signals = np.stack([signal32, signal256], axis=0).astype(np.float32)
    if signals.shape != (2, cfg.signal_points):
        raise AssertionError("signals must have shape (2, signal_points)")
    return signals
```

- [ ] **Step 4: Run physics tests**

Run:

```bash
pytest tests/test_physics.py -q
```

Expected: PASS.

## Task 4: Target Heatmaps

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/targets.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_targets.py`

- [ ] **Step 1: Write failing target tests**

Create `tests/test_targets.py` with this content:

```python
from __future__ import annotations

import numpy as np

from sali.config import DataConfig, ModelConfig
from sali.physics import Couplings
from sali.targets import coupling_to_pixel, render_heatmap, true_boxes


def test_coupling_to_pixel_maps_ranges_with_border() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    row, col = coupling_to_pixel(az_khz=-100.0, aperp_khz=2.0, data=data, model=model)
    assert (row, col) == (2, 2)
    row, col = coupling_to_pixel(az_khz=100.0, aperp_khz=102.0, data=data, model=model)
    assert (row, col) == (201, 101)


def test_render_heatmap_shape_and_peak() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    heatmap = render_heatmap(nuclei, data, model)
    assert heatmap.shape == (1, 204, 104)
    assert heatmap.max() <= 1.0
    row, col = coupling_to_pixel(0.0, 52.0, data, model)
    assert heatmap[0, row, col] == heatmap.max()


def test_true_boxes_are_five_by_five() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    boxes = true_boxes(nuclei, data, model)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.height == 5
    assert box.width == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_targets.py -q
```

Expected: FAIL because `sali.targets` does not exist.

- [ ] **Step 3: Implement target module**

Create `src/sali/targets.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sali.config import DataConfig, ModelConfig
from sali.physics import Couplings


@dataclass(frozen=True, slots=True)
class Box:
    row_min: int
    col_min: int
    row_max: int
    col_max: int

    @property
    def height(self) -> int:
        return self.row_max - self.row_min + 1

    @property
    def width(self) -> int:
        return self.col_max - self.col_min + 1


def coupling_to_pixel(
    az_khz: float,
    aperp_khz: float,
    data: DataConfig,
    model: ModelConfig,
) -> tuple[int, int]:
    y = (az_khz - data.az_min_khz) / (data.az_max_khz - data.az_min_khz)
    x = (aperp_khz - data.aperp_min_khz) / (data.aperp_max_khz - data.aperp_min_khz)
    effective_h = model.output_height - 4
    effective_w = model.output_width - 4
    row = int(np.rint(y * (effective_h - 1))) + 2
    col = int(np.rint(x * (effective_w - 1))) + 2
    row = int(np.clip(row, 2, model.output_height - 3))
    col = int(np.clip(col, 2, model.output_width - 3))
    return row, col


def pixel_to_coupling(
    row: float,
    col: float,
    data: DataConfig,
    model: ModelConfig,
) -> tuple[float, float]:
    effective_h = model.output_height - 4
    effective_w = model.output_width - 4
    y = (row - 2.0) / float(effective_h - 1)
    x = (col - 2.0) / float(effective_w - 1)
    az = data.az_min_khz + y * (data.az_max_khz - data.az_min_khz)
    aperp = data.aperp_min_khz + x * (data.aperp_max_khz - data.aperp_min_khz)
    return float(az), float(aperp)


def _five_by_five_box(row: int, col: int, model: ModelConfig) -> Box:
    return Box(
        row_min=max(0, row - 2),
        col_min=max(0, col - 2),
        row_max=min(model.output_height - 1, row + 2),
        col_max=min(model.output_width - 1, col + 2),
    )


def render_heatmap(nuclei: Couplings, data: DataConfig, model: ModelConfig) -> np.ndarray:
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)
    offsets = np.arange(-2, 3, dtype=np.float32)
    rr, cc = np.meshgrid(offsets, offsets, indexing="ij")
    kernel = np.exp(-((rr**2 + cc**2) / 2.0)).astype(np.float32)
    for az, aperp in zip(nuclei.az_khz, nuclei.aperp_khz, strict=True):
        row, col = coupling_to_pixel(float(az), float(aperp), data, model)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                target_row = row + dr
                target_col = col + dc
                if 0 <= target_row < model.output_height and 0 <= target_col < model.output_width:
                    heatmap[0, target_row, target_col] += kernel[dr + 2, dc + 2]
    np.clip(heatmap, 0.0, 1.0, out=heatmap)
    return heatmap


def true_boxes(nuclei: Couplings, data: DataConfig, model: ModelConfig) -> list[Box]:
    boxes: list[Box] = []
    for az, aperp in zip(nuclei.az_khz, nuclei.aperp_khz, strict=True):
        row, col = coupling_to_pixel(float(az), float(aperp), data, model)
        boxes.append(_five_by_five_box(row, col, model))
    return boxes
```

- [ ] **Step 4: Run target tests**

Run:

```bash
pytest tests/test_targets.py -q
```

Expected: PASS.

## Task 5: Synthetic Dataset And Normalization

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/data.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_data.py`

- [ ] **Step 1: Write failing data tests**

Create `tests/test_data.py` with this content:

```python
from __future__ import annotations

import numpy as np

from sali.data import SaliDataset, generate_splits


def test_generate_splits_sizes_and_shapes(tiny_config) -> None:
    splits, stats = generate_splits(tiny_config)
    assert len(splits.train) == tiny_config.data.train_samples
    assert len(splits.val) == tiny_config.data.val_samples
    assert len(splits.test) == tiny_config.data.test_samples
    sample = splits.train[0]
    assert sample.signals.shape == (2, 1000)
    assert sample.raw_signals.shape == (2, 1000)
    assert sample.heatmap.shape == (1, 204, 104)
    assert sample.nuclei.count >= 1
    assert np.isfinite(stats.mean)
    assert np.isfinite(stats.var)


def test_dataset_returns_torch_ready_arrays(tiny_config) -> None:
    splits, _stats = generate_splits(tiny_config)
    dataset = SaliDataset(splits.train)
    signal32, signal256, heatmap = dataset[0]
    assert tuple(signal32.shape) == (1, 1000)
    assert tuple(signal256.shape) == (1, 1000)
    assert tuple(heatmap.shape) == (1, 204, 104)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_data.py -q
```

Expected: FAIL because `sali.data` does not exist.

- [ ] **Step 3: Implement data module**

Create `src/sali/data.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from sali.config import RunConfig
from sali.physics import Couplings, generate_sample_signals
from sali.targets import render_heatmap


@dataclass(slots=True)
class Sample:
    signals: np.ndarray
    raw_signals: np.ndarray
    heatmap: np.ndarray
    nuclei: Couplings
    split: str


@dataclass(slots=True)
class NormalizationStats:
    mean: float
    var: float
    epsilon: float

    def normalize(self, signals: np.ndarray) -> np.ndarray:
        return ((signals - self.mean) / np.sqrt(self.var + self.epsilon)).astype(np.float32)


@dataclass(slots=True)
class DataSplits:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def sample_couplings(cfg: RunConfig, rng: np.random.Generator) -> Couplings:
    count = int(rng.integers(cfg.data.min_nuclei, cfg.data.max_nuclei + 1))
    az = rng.uniform(cfg.data.az_min_khz, cfg.data.az_max_khz, size=count).astype(np.float32)
    aperp = rng.uniform(cfg.data.aperp_min_khz, cfg.data.aperp_max_khz, size=count).astype(np.float32)
    return Couplings(az_khz=az, aperp_khz=aperp)


def _generate_raw_samples(cfg: RunConfig, split: str, count: int, rng: np.random.Generator) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(count):
        nuclei = sample_couplings(cfg, rng)
        raw_signals = generate_sample_signals(nuclei, cfg.physics, rng)
        heatmap = render_heatmap(nuclei, cfg.data, cfg.model)
        samples.append(
            Sample(
                signals=raw_signals.copy(),
                raw_signals=raw_signals,
                heatmap=heatmap,
                nuclei=nuclei,
                split=split,
            )
        )
    return samples


def _stats_from_train(samples: list[Sample], epsilon: float) -> NormalizationStats:
    stacked = np.stack([sample.signals for sample in samples], axis=0)
    return NormalizationStats(
        mean=float(stacked.mean()),
        var=float(stacked.var()),
        epsilon=epsilon,
    )


def _apply_normalization(samples: list[Sample], stats: NormalizationStats) -> None:
    for sample in samples:
        sample.signals = stats.normalize(sample.signals)


def generate_splits(cfg: RunConfig) -> tuple[DataSplits, NormalizationStats]:
    if min(cfg.data.train_samples, cfg.data.val_samples, cfg.data.test_samples) <= 0:
        raise ValueError("train, validation, and test sample counts must be positive")
    rng = np.random.default_rng(cfg.data.seed)
    train = _generate_raw_samples(cfg, "train", cfg.data.train_samples, rng)
    val = _generate_raw_samples(cfg, "val", cfg.data.val_samples, rng)
    test = _generate_raw_samples(cfg, "test", cfg.data.test_samples, rng)
    stats = _stats_from_train(train, cfg.data.norm_epsilon)
    _apply_normalization(train, stats)
    _apply_normalization(val, stats)
    _apply_normalization(test, stats)
    return DataSplits(train=train, val=val, test=test), stats


class SaliDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        signals = sample.signals.astype(np.float32)
        heatmap = sample.heatmap.astype(np.float32)
        signal32 = torch.from_numpy(signals[0:1])
        signal256 = torch.from_numpy(signals[1:2])
        target = torch.from_numpy(heatmap)
        return signal32, signal256, target
```

- [ ] **Step 4: Run data tests**

Run:

```bash
pytest tests/test_data.py -q
```

Expected: PASS.

## Task 6: PyTorch Model

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/model.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_model.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_model.py` with this content:

```python
from __future__ import annotations

import torch

from sali.config import ModelConfig
from sali.model import SaliNet


def test_sali_net_forward_shape() -> None:
    model = SaliNet(ModelConfig())
    signal32 = torch.zeros((2, 1, 1000), dtype=torch.float32)
    signal256 = torch.zeros((2, 1, 1000), dtype=torch.float32)
    output = model(signal32, signal256)
    assert tuple(output.shape) == (2, 1, 204, 104)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_model.py -q
```

Expected: FAIL because `sali.model` does not exist.

- [ ] **Step 3: Implement model module**

Create `src/sali/model.py` with this content:

```python
from __future__ import annotations

import torch
from torch import nn

from sali.config import ModelConfig


class SignalBranch(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, cfg.conv1_filters, kernel_size=3, padding=1),
            nn.Conv1d(cfg.conv1_filters, cfg.conv2_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return torch.flatten(x, start_dim=1)


class SaliNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.branch32 = SignalBranch(cfg)
        self.branch256 = SignalBranch(cfg)
        branch_features = cfg.conv2_filters * (cfg.signal_length // 2)
        merged_features = branch_features * 2
        bottleneck = (cfg.dense_height * cfg.dense_width) // 2
        dense_size = cfg.dense_height * cfg.dense_width
        self.fc = nn.Sequential(
            nn.Linear(merged_features, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(bottleneck, dense_size),
            nn.BatchNorm1d(dense_size),
            nn.ReLU(),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(cfg.dropout),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, signal32: torch.Tensor, signal256: torch.Tensor) -> torch.Tensor:
        x32 = self.branch32(signal32)
        x256 = self.branch256(signal256)
        merged = torch.cat([x32, x256], dim=1)
        x = self.fc(merged)
        x = x.view(-1, 1, self.cfg.dense_height, self.cfg.dense_width)
        out = self.conv2d(x)
        expected = (self.cfg.output_height, self.cfg.output_width)
        if out.shape[-2:] != expected:
            raise RuntimeError(f"model produced {out.shape[-2:]}, expected {expected}")
        return out
```

- [ ] **Step 4: Run model tests**

Run:

```bash
pytest tests/test_model.py -q
```

Expected: PASS.

## Task 7: Post-Processing

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/postprocess.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_postprocess.py`

- [ ] **Step 1: Write failing post-processing tests**

Create `tests/test_postprocess.py` with this content:

```python
from __future__ import annotations

import numpy as np

from sali.config import DataConfig, ModelConfig, PostprocessConfig
from sali.postprocess import postprocess_heatmap
from sali.targets import render_heatmap
from sali.physics import Couplings


def test_postprocess_detects_synthetic_blob() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig(threshold=0.2, min_area=1)
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    heatmap = render_heatmap(nuclei, data, model)
    predictions = postprocess_heatmap(heatmap, data, model, pp)
    assert len(predictions) == 1
    pred = predictions[0]
    assert abs(pred.az_khz - 0.0) < 2.0
    assert abs(pred.aperp_khz - 52.0) < 2.0


def test_postprocess_handles_empty_heatmap() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig(threshold=0.2)
    heatmap = np.zeros((1, 204, 104), dtype=np.float32)
    assert postprocess_heatmap(heatmap, data, model, pp) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_postprocess.py -q
```

Expected: FAIL because `sali.postprocess` does not exist.

- [ ] **Step 3: Implement post-processing module**

Create `src/sali/postprocess.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, square

from sali.config import DataConfig, ModelConfig, PostprocessConfig
from sali.targets import Box, pixel_to_coupling


@dataclass(frozen=True, slots=True)
class Prediction:
    az_khz: float
    aperp_khz: float
    row: float
    col: float
    box: Box
    confidence: float


def _box_around(row: int, col: int, model: ModelConfig) -> Box:
    return Box(
        row_min=max(0, row - 2),
        col_min=max(0, col - 2),
        row_max=min(model.output_height - 1, row + 2),
        col_max=min(model.output_width - 1, col + 2),
    )


def _region_box(region) -> Box:
    row_min, col_min, row_max, col_max = region.bbox
    return Box(row_min=row_min, col_min=col_min, row_max=row_max - 1, col_max=col_max - 1)


def postprocess_heatmap(
    heatmap: np.ndarray,
    data: DataConfig,
    model: ModelConfig,
    cfg: PostprocessConfig,
) -> list[Prediction]:
    if heatmap.shape != (1, model.output_height, model.output_width):
        raise ValueError(f"heatmap must have shape (1, {model.output_height}, {model.output_width})")
    image = np.asarray(heatmap[0], dtype=np.float32)
    if not np.isfinite(image).all():
        raise FloatingPointError("heatmap contains non-finite values")
    mask = image >= cfg.threshold
    if cfg.erosion_size > 0:
        mask = binary_erosion(mask, square(2 * cfg.erosion_size + 1))
    if cfg.dilation_size > 0:
        mask = binary_dilation(mask, square(2 * cfg.dilation_size + 1))
    labels = label(mask, connectivity=2)
    predictions: list[Prediction] = []
    for region in regionprops(labels, intensity_image=image):
        if region.area < cfg.min_area:
            continue
        region_mask = labels == region.label
        local_image = np.where(region_mask, image, 0.0)
        peaks = peak_local_max(
            local_image,
            min_distance=cfg.local_max_min_distance_px,
            threshold_abs=cfg.threshold,
            exclude_border=False,
        )
        if len(peaks) > 1:
            for row, col in peaks:
                az, aperp = pixel_to_coupling(float(row), float(col), data, model)
                predictions.append(
                    Prediction(
                        az_khz=az,
                        aperp_khz=aperp,
                        row=float(row),
                        col=float(col),
                        box=_box_around(int(row), int(col), model),
                        confidence=float(image[row, col]),
                    )
                )
            continue
        centroid_row, centroid_col = ndi.center_of_mass(image, labels, region.label)
        if not np.isfinite(centroid_row) or not np.isfinite(centroid_col):
            centroid_row, centroid_col = region.centroid
        az, aperp = pixel_to_coupling(float(centroid_row), float(centroid_col), data, model)
        predictions.append(
            Prediction(
                az_khz=az,
                aperp_khz=aperp,
                row=float(centroid_row),
                col=float(centroid_col),
                box=_region_box(region),
                confidence=float(region.max_intensity),
            )
        )
    return predictions
```

- [ ] **Step 4: Run post-processing tests**

Run:

```bash
pytest tests/test_postprocess.py -q
```

Expected: PASS.

## Task 8: Metrics

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/metrics.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_metrics.py`

- [ ] **Step 1: Write failing metrics tests**

Create `tests/test_metrics.py` with this content:

```python
from __future__ import annotations

import numpy as np

from sali.config import DataConfig, ModelConfig, PhysicsConfig
from sali.metrics import box_iou, evaluate_sample
from sali.physics import Couplings, generate_sample_signals
from sali.postprocess import Prediction
from sali.targets import Box, true_boxes


def test_box_iou_detects_overlap() -> None:
    a = Box(0, 0, 4, 4)
    b = Box(2, 2, 6, 6)
    assert box_iou(a, b) > 0.0
    c = Box(10, 10, 12, 12)
    assert box_iou(a, c) == 0.0


def test_evaluate_sample_counts_tp_fp_fn(rng: np.random.Generator) -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    physics = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    box = true_boxes(nuclei, data, model)[0]
    predictions = [
        Prediction(az_khz=0.5, aperp_khz=52.5, row=102.0, col=52.0, box=box, confidence=0.9)
    ]
    raw_signals = generate_sample_signals(nuclei, physics, rng)
    result = evaluate_sample(predictions, nuclei, raw_signals, data, model, physics, rng)
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.mae_az_khz >= 0.0
    assert result.mae_aperp_khz >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_metrics.py -q
```

Expected: FAIL because `sali.metrics` does not exist.

- [ ] **Step 3: Implement metrics module**

Create `src/sali/metrics.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sali.config import DataConfig, ModelConfig, PhysicsConfig
from sali.physics import Couplings, generate_sample_signals
from sali.postprocess import Prediction
from sali.targets import Box, true_boxes


@dataclass(slots=True)
class SampleMetrics:
    true_nuclei: int
    predicted_nuclei: int
    true_positives: int
    false_positives: int
    false_negatives: int
    mae_az_khz: float
    mae_aperp_khz: float
    signal_mae_32: float
    signal_mae_256: float

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return 0.0 if denom == 0 else self.true_positives / denom

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return 0.0 if denom == 0 else self.true_positives / denom


def box_iou(a: Box, b: Box) -> float:
    row_min = max(a.row_min, b.row_min)
    col_min = max(a.col_min, b.col_min)
    row_max = min(a.row_max, b.row_max)
    col_max = min(a.col_max, b.col_max)
    if row_max < row_min or col_max < col_min:
        return 0.0
    intersection = (row_max - row_min + 1) * (col_max - col_min + 1)
    area_a = a.height * a.width
    area_b = b.height * b.width
    union = area_a + area_b - intersection
    return float(intersection / union)


def _match_predictions(predictions: list[Prediction], true: Couplings, data: DataConfig, model: ModelConfig) -> list[tuple[int, int]]:
    boxes = true_boxes(true, data, model)
    matches: list[tuple[int, int]] = []
    used_true: set[int] = set()
    for pred_idx, pred in enumerate(predictions):
        best_idx = -1
        best_iou = 0.0
        for true_idx, box in enumerate(boxes):
            if true_idx in used_true:
                continue
            iou = box_iou(pred.box, box)
            if iou > best_iou:
                best_iou = iou
                best_idx = true_idx
        if best_idx >= 0 and best_iou > 0.0:
            matches.append((pred_idx, best_idx))
            used_true.add(best_idx)
    return matches


def evaluate_sample(
    predictions: list[Prediction],
    true: Couplings,
    raw_signals: np.ndarray,
    data: DataConfig,
    model: ModelConfig,
    physics: PhysicsConfig,
    rng: np.random.Generator,
) -> SampleMetrics:
    matches = _match_predictions(predictions, true, data, model)
    tp = len(matches)
    fp = len(predictions) - tp
    fn = true.count - tp
    if matches:
        az_errors = [
            abs(predictions[pred_idx].az_khz - float(true.az_khz[true_idx]))
            for pred_idx, true_idx in matches
        ]
        aperp_errors = [
            abs(predictions[pred_idx].aperp_khz - float(true.aperp_khz[true_idx]))
            for pred_idx, true_idx in matches
        ]
        mae_az = float(np.mean(az_errors))
        mae_aperp = float(np.mean(aperp_errors))
    else:
        mae_az = float("nan")
        mae_aperp = float("nan")
    pred_couplings = Couplings(
        az_khz=np.array([pred.az_khz for pred in predictions], dtype=np.float32),
        aperp_khz=np.array([pred.aperp_khz for pred in predictions], dtype=np.float32),
    )
    clean_physics = PhysicsConfig(
        bz_tesla=physics.bz_tesla,
        gamma_mhz_per_t=physics.gamma_mhz_per_t,
        t2_us=physics.t2_us,
        measurements=physics.measurements,
        signal_points=physics.signal_points,
        tau_32_min_us=physics.tau_32_min_us,
        tau_32_max_us=physics.tau_32_max_us,
        tau_256_min_us=physics.tau_256_min_us,
        tau_256_max_us=physics.tau_256_max_us,
        add_decoherence=physics.add_decoherence,
        add_shot_noise=False,
    )
    reconstructed = generate_sample_signals(pred_couplings, clean_physics, rng)
    return SampleMetrics(
        true_nuclei=true.count,
        predicted_nuclei=len(predictions),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        mae_az_khz=mae_az,
        mae_aperp_khz=mae_aperp,
        signal_mae_32=float(np.mean(np.abs(reconstructed[0] - raw_signals[0]))),
        signal_mae_256=float(np.mean(np.abs(reconstructed[1] - raw_signals[1]))),
    )


def aggregate_by_true_count(results: list[SampleMetrics]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[SampleMetrics]] = {}
    for result in results:
        grouped.setdefault(result.true_nuclei, []).append(result)
    summary: dict[int, dict[str, float]] = {}
    for count, items in grouped.items():
        summary[count] = {
            "precision": float(np.mean([item.precision for item in items])),
            "recall": float(np.mean([item.recall for item in items])),
            "mae_az_khz": float(np.nanmean([item.mae_az_khz for item in items])),
            "mae_aperp_khz": float(np.nanmean([item.mae_aperp_khz for item in items])),
            "signal_mae_32": float(np.mean([item.signal_mae_32 for item in items])),
            "signal_mae_256": float(np.mean([item.signal_mae_256 for item in items])),
        }
    return summary
```

- [ ] **Step 4: Run metrics tests**

Run:

```bash
pytest tests/test_metrics.py -q
```

Expected: PASS.

## Task 9: Training Loop

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/train.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_train.py`

- [ ] **Step 1: Write failing training tests**

Create `tests/test_train.py` with this content:

```python
from __future__ import annotations

from sali.data import generate_splits
from sali.train import choose_device, train_model


def test_choose_device_accepts_cpu() -> None:
    assert str(choose_device("cpu")) == "cpu"


def test_train_model_smoke(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "run"
    splits, _stats = generate_splits(tiny_config)
    result = train_model(tiny_config, splits)
    assert result.best_checkpoint.exists()
    assert len(result.history["train_loss"]) == 1
    assert len(result.history["val_loss"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_train.py -q
```

Expected: FAIL because `sali.train` does not exist.

- [ ] **Step 3: Implement training module**

Create `src/sali/train.py` with this content:

```python
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from sali.config import RunConfig
from sali.data import DataSplits, SaliDataset
from sali.model import SaliNet


@dataclass(slots=True)
class TrainResult:
    model: SaliNet
    history: dict[str, list[float]]
    best_checkpoint: Path


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _run_epoch(
    model: SaliNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    for signal32, signal256, target in loader:
        signal32 = signal32.to(device)
        signal256 = signal256.to(device)
        target = target.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(signal32, signal256)
            loss = criterion(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss became non-finite")
            if is_train:
                loss.backward()
                optimizer.step()
        batch_size = signal32.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def _save_history(path: Path, history: dict[str, list[float]]) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def train_model(cfg: RunConfig, splits: DataSplits) -> TrainResult:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.training.device)
    model = SaliNet(cfg.model).to(device)
    train_loader = DataLoader(
        SaliDataset(splits.train),
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SaliDataset(splits.val),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_reduction_factor,
        patience=cfg.training.lr_plateau_patience,
        min_lr=1e-8,
    )
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_checkpoint = cfg.output_dir / "best_model.pt"
    stale_epochs = 0
    for _epoch in range(cfg.training.max_epochs):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device, None)
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(lr)
        if val_loss < best_loss - cfg.training.min_delta:
            best_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_checkpoint)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.training.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    if not best_checkpoint.exists():
        torch.save(best_state, best_checkpoint)
    _save_history(cfg.output_dir / "history.json", history)
    return TrainResult(model=model, history=history, best_checkpoint=best_checkpoint)
```

- [ ] **Step 4: Run training tests**

Run:

```bash
pytest tests/test_train.py -q
```

Expected: PASS.

## Task 10: Plotting Helpers

**Files:**
- Create: `/Users/weitao/Code/python/sali/src/sali/plots.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_plots.py`

- [ ] **Step 1: Write failing plot tests**

Create `tests/test_plots.py` with this content:

```python
from __future__ import annotations

import numpy as np

from sali.plots import plot_heatmap, plot_loss


def test_plot_loss_writes_file(tmp_path) -> None:
    path = tmp_path / "loss.png"
    plot_loss({"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}, path)
    assert path.exists()


def test_plot_heatmap_writes_file(tmp_path) -> None:
    path = tmp_path / "heatmap.png"
    heatmap = np.zeros((1, 204, 104), dtype=np.float32)
    heatmap[0, 100, 50] = 1.0
    plot_heatmap(heatmap, "True Heatmap", path)
    assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_plots.py -q
```

Expected: FAIL because `sali.plots` does not exist.

- [ ] **Step 3: Implement plotting module**

Create `src/sali/plots.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sali.metrics import SampleMetrics
from sali.physics import tau_grid_us


def _prepare_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_loss(history: dict[str, list[float]], path: Path) -> None:
    _prepare_path(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heatmap(heatmap: np.ndarray, title: str, path: Path) -> None:
    _prepare_path(path)
    fig, ax = plt.subplots(figsize=(6, 8))
    image = ax.imshow(heatmap[0], origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Aperp pixel")
    ax.set_ylabel("Az pixel")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spectra(signals: np.ndarray, path: Path) -> None:
    _prepare_path(path)
    tau32 = tau_grid_us(6.0, 50.0, signals.shape[1])
    tau256 = tau_grid_us(10.0, 40.0, signals.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
    axes[0].plot(tau32, signals[0])
    axes[0].set_title("Generated spectrum N=32")
    axes[0].set_xlabel("tau (us)")
    axes[0].set_ylabel("P_x")
    axes[1].plot(tau256, signals[1])
    axes[1].set_title("Generated spectrum N=256")
    axes[1].set_xlabel("tau (us)")
    axes[1].set_ylabel("P_x")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_signal_overlay(original: np.ndarray, reconstructed: np.ndarray, path: Path) -> None:
    _prepare_path(path)
    tau32 = tau_grid_us(6.0, 50.0, original.shape[1])
    tau256 = tau_grid_us(10.0, 40.0, original.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
    axes[0].plot(tau32, original[0], label="original", color="black")
    axes[0].plot(tau32, reconstructed[0], label="identified C13", color="tab:orange")
    axes[0].set_title("Original vs identified signal N=32")
    axes[0].legend()
    axes[1].plot(tau256, original[1], label="original", color="black")
    axes[1].plot(tau256, reconstructed[1], label="identified C13", color="tab:blue")
    axes[1].set_title("Original vs identified signal N=256")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_precision_recall(results: list[SampleMetrics], path: Path) -> None:
    _prepare_path(path)
    counts = sorted({result.true_nuclei for result in results})
    precision = []
    recall = []
    for count in counts:
        selected = [result for result in results if result.true_nuclei == count]
        precision.append(float(np.mean([item.precision for item in selected])))
        recall.append(float(np.mean([item.recall for item in selected])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(counts, precision, marker="o", label="precision")
    ax.plot(counts, recall, marker="o", label="recall")
    ax.set_xlabel("True nuclei")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mae(results: list[SampleMetrics], path: Path) -> None:
    _prepare_path(path)
    counts = sorted({result.true_nuclei for result in results})
    mae_az = []
    mae_aperp = []
    signal32 = []
    signal256 = []
    for count in counts:
        selected = [result for result in results if result.true_nuclei == count]
        mae_az.append(float(np.nanmean([item.mae_az_khz for item in selected])))
        mae_aperp.append(float(np.nanmean([item.mae_aperp_khz for item in selected])))
        signal32.append(float(np.mean([item.signal_mae_32 for item in selected])))
        signal256.append(float(np.mean([item.signal_mae_256 for item in selected])))
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(counts, mae_az, marker="o", label="Az")
    axes[0].plot(counts, mae_aperp, marker="o", label="Aperp")
    axes[0].set_ylabel("Coupling MAE (kHz)")
    axes[0].legend()
    axes[1].plot(counts, signal32, marker="o", label="N=32")
    axes[1].plot(counts, signal256, marker="o", label="N=256")
    axes[1].set_xlabel("True nuclei")
    axes[1].set_ylabel("Signal MAE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
```

- [ ] **Step 4: Run plotting tests**

Run:

```bash
pytest tests/test_plots.py -q
```

Expected: PASS.

## Task 11: Evaluation Pipeline

**Files:**
- Modify: `/Users/weitao/Code/python/sali/src/sali/train.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_evaluate_pipeline.py`

- [ ] **Step 1: Write failing evaluation pipeline test**

Create `tests/test_evaluate_pipeline.py` with this content:

```python
from __future__ import annotations

from sali.data import generate_splits
from sali.train import evaluate_model, train_model


def test_evaluate_model_returns_metrics(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "run"
    splits, _stats = generate_splits(tiny_config)
    result = train_model(tiny_config, splits)
    metrics = evaluate_model(result.model, tiny_config, splits.test, max_samples=2)
    assert len(metrics) == 2
    assert metrics[0].true_nuclei >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_evaluate_pipeline.py -q
```

Expected: FAIL because `evaluate_model` is not defined.

- [ ] **Step 3: Add evaluation function to training module**

Append this function to `src/sali/train.py`:

```python
import numpy as np

from sali.data import Sample
from sali.metrics import SampleMetrics, evaluate_sample
from sali.postprocess import postprocess_heatmap


def evaluate_model(
    model: SaliNet,
    cfg: RunConfig,
    samples: list[Sample],
    max_samples: int | None = None,
) -> list[SampleMetrics]:
    device = choose_device(cfg.training.device)
    model.to(device)
    model.eval()
    rng = np.random.default_rng(cfg.data.seed + 999)
    selected = samples if max_samples is None else samples[:max_samples]
    results: list[SampleMetrics] = []
    for sample in selected:
        signal32 = torch.from_numpy(sample.signals[0:1]).unsqueeze(0).to(device)
        signal256 = torch.from_numpy(sample.signals[1:2]).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(signal32, signal256).cpu().numpy()[0]
        predicted_nuclei = postprocess_heatmap(prediction, cfg.data, cfg.model, cfg.postprocess)
        results.append(
            evaluate_sample(
                predicted_nuclei,
                sample.nuclei,
                sample.raw_signals,
                cfg.data,
                cfg.model,
                cfg.physics,
                rng,
            )
        )
    return results
```

Also ensure these imports appear at module top rather than below existing code after the first local test pass:

```python
import numpy as np

from sali.data import DataSplits, SaliDataset, Sample
from sali.metrics import SampleMetrics, evaluate_sample
from sali.postprocess import postprocess_heatmap
```

- [ ] **Step 4: Run evaluation pipeline test**

Run:

```bash
pytest tests/test_evaluate_pipeline.py -q
```

Expected: PASS.

## Task 12: Colab Script

**Files:**
- Create: `/Users/weitao/Code/python/sali/scripts/train_colab.py`
- Create: `/Users/weitao/Code/python/sali/scripts/run_colab_uploaded.py`
- Create: `/Users/weitao/Code/python/sali/tests/test_train_colab_script.py`

- [ ] **Step 1: Write failing script smoke test**

Create `tests/test_train_colab_script.py` with this content:

```python
from __future__ import annotations

import subprocess
import sys


def test_train_colab_help_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/train_colab.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--preset" in completed.stdout
    assert "--field" in completed.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_train_colab_script.py -q
```

Expected: FAIL because `scripts/train_colab.py` does not exist.

- [ ] **Step 3: Implement Colab script**

Create `scripts/train_colab.py` with this content:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", Path("/content/sali/src")):
    if (candidate / "sali").exists():
        sys.path.insert(0, str(candidate))
        break

from sali.config import paper_config, practical_config
from sali.data import generate_splits
from sali.metrics import aggregate_by_true_count
from sali.physics import Couplings, generate_sample_signals
from sali.plots import (
    plot_heatmap,
    plot_loss,
    plot_mae,
    plot_precision_recall,
    plot_signal_overlay,
    plot_spectra,
)
from sali.postprocess import postprocess_heatmap
from sali.train import evaluate_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SALI PyTorch reproduction.")
    parser.add_argument("--preset", choices=["practical", "paper"], default="practical")
    parser.add_argument("--field", choices=["low", "high"], default="low")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-eval-samples", type=int, default=64)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace):
    cfg = practical_config(args.field) if args.preset == "practical" else paper_config(args.field)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    cfg.training.device = args.device
    return cfg


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def make_example_plots(cfg, splits, result) -> None:
    figures = cfg.output_dir / "figures"
    sample = splits.test[0]
    plot_spectra(sample.raw_signals, figures / "generated_spectra.png")
    plot_heatmap(sample.heatmap, "True heatmap", figures / "true_heatmap.png")
    device = next(result.model.parameters()).device
    result.model.eval()
    with torch.no_grad():
        pred = result.model(
            torch.from_numpy(sample.signals[0:1]).unsqueeze(0).to(device),
            torch.from_numpy(sample.signals[1:2]).unsqueeze(0).to(device),
        ).cpu().numpy()[0]
    plot_heatmap(pred, "Predicted heatmap", figures / "predicted_heatmap.png")
    predictions = postprocess_heatmap(pred, cfg.data, cfg.model, cfg.postprocess)
    post = np.zeros_like(pred)
    for item in predictions:
        row = int(round(item.row))
        col = int(round(item.col))
        post[0, max(0, row - 2): row + 3, max(0, col - 2): col + 3] = 1.0
    plot_heatmap(post, "Post-processed heatmap", figures / "postprocessed_heatmap.png")
    pred_couplings = Couplings(
        az_khz=np.array([item.az_khz for item in predictions], dtype=np.float32),
        aperp_khz=np.array([item.aperp_khz for item in predictions], dtype=np.float32),
    )
    clean_physics = replace(cfg.physics, add_shot_noise=False)
    reconstructed = generate_sample_signals(pred_couplings, clean_physics, np.random.default_rng(cfg.data.seed + 202))
    plot_signal_overlay(sample.raw_signals, reconstructed, figures / "signal_overlay.png")


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg.output_dir / "config.json", asdict(cfg))
    splits, stats = generate_splits(cfg)
    save_json(cfg.output_dir / "normalization.json", asdict(stats))
    result = train_model(cfg, splits)
    plot_loss(result.history, cfg.output_dir / "figures" / "loss.png")
    metrics = evaluate_model(result.model, cfg, splits.test, max_samples=args.max_eval_samples)
    summary = aggregate_by_true_count(metrics)
    save_json(cfg.output_dir / "metrics_by_true_count.json", summary)
    plot_precision_recall(metrics, cfg.output_dir / "figures" / "precision_recall.png")
    plot_mae(metrics, cfg.output_dir / "figures" / "mae.png")
    make_example_plots(cfg, splits, result)
    print(f"Run complete: {cfg.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement uploaded Colab runner**

Create `scripts/run_colab_uploaded.py` with this content:

```python
#!/usr/bin/env python
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ARCHIVE = Path("/content/sali-colab.tar.gz")
PROJECT = Path("/content/sali")
OUTPUT = Path("/content/sali-runs/practical-low")


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Expected uploaded archive at {ARCHIVE}")
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall("/content")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT)], check=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts" / "train_colab.py"),
            "--preset",
            "practical",
            "--field",
            "low",
            "--output-dir",
            str(OUTPUT),
            "--device",
            "auto",
            "--max-eval-samples",
            "64",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run script help test**

Run:

```bash
pytest tests/test_train_colab_script.py -q
```

Expected: PASS.

- [ ] **Step 6: Run practical local smoke command**

Run:

```bash
python scripts/train_colab.py --preset practical --field low --output-dir runs/local-smoke --device cpu --max-eval-samples 4
```

Expected: command exits with status 0 and creates `runs/local-smoke/figures/loss.png`, `precision_recall.png`, `mae.png`, `generated_spectra.png`, `true_heatmap.png`, `predicted_heatmap.png`, `postprocessed_heatmap.png`, and `signal_overlay.png`.

## Task 13: Notebook Generation

**Files:**
- Create: `/Users/weitao/Code/python/sali/scripts/create_notebook.py`
- Create: `/Users/weitao/Code/python/sali/notebooks/sali_reproduction_colab.ipynb`
- Create: `/Users/weitao/Code/python/sali/tests/test_notebook.py`

- [ ] **Step 1: Write failing notebook test**

Create `tests/test_notebook.py` with this content:

```python
from __future__ import annotations

import nbformat


def test_notebook_contains_requested_plot_sections() -> None:
    notebook = nbformat.read("notebooks/sali_reproduction_colab.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "Generated spectra" in source
    assert "True heatmap" in source
    assert "Original vs identified C13 reconstructed signals" in source
    assert "Training and validation loss" in source
    assert "Precision and recall" in source
    assert "MAE" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_notebook.py -q
```

Expected: FAIL because the notebook does not exist.

- [ ] **Step 3: Implement notebook generator**

Create `scripts/create_notebook.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/sali_reproduction_colab.ipynb")


def code(source: str):
    return nbf.v4.new_code_cell(source)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown("# SALI PyTorch Reproduction\n\nColab-ready practical reproduction of the SALI signal-to-image model."),
        code("%pip install -q -e ."),
        markdown("## Configuration"),
        code(
            "from pathlib import Path\n"
            "from sali.config import practical_config\n\n"
            "cfg = practical_config(field='low')\n"
            "cfg.output_dir = Path('runs/notebook-practical-low')\n"
            "cfg.training.device = 'auto'\n"
            "cfg.training.max_epochs = 5\n"
            "cfg.data.train_samples = 512\n"
            "cfg.data.val_samples = 128\n"
            "cfg.data.test_samples = 128\n"
            "cfg.output_dir.mkdir(parents=True, exist_ok=True)\n"
            "cfg\n"
        ),
        markdown("## Generate training, validation and testing dataset"),
        code(
            "from sali.data import generate_splits\n\n"
            "splits, stats = generate_splits(cfg)\n"
            "len(splits.train), len(splits.val), len(splits.test), stats\n"
        ),
        markdown("## Generated spectra"),
        code(
            "from sali.plots import plot_spectra\n"
            "from IPython.display import Image, display\n\n"
            "sample = splits.test[0]\n"
            "path = cfg.output_dir / 'figures' / 'generated_spectra.png'\n"
            "plot_spectra(sample.raw_signals, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## True heatmap"),
        code(
            "from sali.plots import plot_heatmap\n\n"
            "path = cfg.output_dir / 'figures' / 'true_heatmap.png'\n"
            "plot_heatmap(sample.heatmap, 'True heatmap', path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## Train PyTorch SALI model"),
        code(
            "from sali.train import train_model\n\n"
            "result = train_model(cfg, splits)\n"
            "result.best_checkpoint\n"
        ),
        markdown("## Training and validation loss"),
        code(
            "from sali.plots import plot_loss\n\n"
            "path = cfg.output_dir / 'figures' / 'loss.png'\n"
            "plot_loss(result.history, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## Predicted and post-processed heatmaps"),
        code(
            "import numpy as np\n"
            "import torch\n"
            "from sali.postprocess import postprocess_heatmap\n\n"
            "device = next(result.model.parameters()).device\n"
            "result.model.eval()\n"
            "with torch.no_grad():\n"
            "    pred = result.model(\n"
            "        torch.from_numpy(sample.signals[0:1]).unsqueeze(0).to(device),\n"
            "        torch.from_numpy(sample.signals[1:2]).unsqueeze(0).to(device),\n"
            "    ).cpu().numpy()[0]\n"
            "predictions = postprocess_heatmap(pred, cfg.data, cfg.model, cfg.postprocess)\n"
            "pred_path = cfg.output_dir / 'figures' / 'predicted_heatmap.png'\n"
            "plot_heatmap(pred, 'Predicted heatmap', pred_path)\n"
            "display(Image(filename=str(pred_path)))\n"
            "post = np.zeros_like(pred)\n"
            "for item in predictions:\n"
            "    row = int(round(item.row)); col = int(round(item.col))\n"
            "    post[0, max(0, row-2):row+3, max(0, col-2):col+3] = 1.0\n"
            "post_path = cfg.output_dir / 'figures' / 'postprocessed_heatmap.png'\n"
            "plot_heatmap(post, 'Post-processed heatmap', post_path)\n"
            "display(Image(filename=str(post_path)))\n"
            "[(p.az_khz, p.aperp_khz, p.confidence) for p in predictions[:10]]\n"
        ),
        markdown("## Original vs identified C13 reconstructed signals"),
        code(
            "from dataclasses import replace\n"
            "from sali.physics import Couplings, generate_sample_signals\n"
            "from sali.plots import plot_signal_overlay\n\n"
            "pred_couplings = Couplings(\n"
            "    az_khz=np.array([item.az_khz for item in predictions], dtype=np.float32),\n"
            "    aperp_khz=np.array([item.aperp_khz for item in predictions], dtype=np.float32),\n"
            ")\n"
            "clean_physics = replace(cfg.physics, add_shot_noise=False)\n"
            "reconstructed = generate_sample_signals(pred_couplings, clean_physics, np.random.default_rng(cfg.data.seed + 202))\n"
            "path = cfg.output_dir / 'figures' / 'signal_overlay.png'\n"
            "plot_signal_overlay(sample.raw_signals, reconstructed, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## Precision and recall"),
        code(
            "from sali.train import evaluate_model\n"
            "from sali.plots import plot_precision_recall\n\n"
            "metrics = evaluate_model(result.model, cfg, splits.test, max_samples=64)\n"
            "path = cfg.output_dir / 'figures' / 'precision_recall.png'\n"
            "plot_precision_recall(metrics, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## MAE"),
        code(
            "from sali.plots import plot_mae\n\n"
            "path = cfg.output_dir / 'figures' / 'mae.png'\n"
            "plot_mae(metrics, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate notebook**

Run:

```bash
python scripts/create_notebook.py
```

Expected: command exits with status 0 and creates `notebooks/sali_reproduction_colab.ipynb`.

- [ ] **Step 5: Run notebook test**

Run:

```bash
pytest tests/test_notebook.py -q
```

Expected: PASS.

## Task 14: Full Verification And Colab CLI Run

**Files:**
- Modify only if verification exposes a concrete bug.

- [ ] **Step 1: Run full local unit suite**

Run:

```bash
pytest
```

Expected: PASS.

- [ ] **Step 2: Run local practical smoke**

Run:

```bash
python scripts/train_colab.py --preset practical --field low --output-dir runs/local-smoke --device cpu --max-eval-samples 4
```

Expected: command exits with status 0 and prints `Run complete: runs/local-smoke`.

- [ ] **Step 3: Check Colab CLI authentication**

Run:

```bash
colab whoami
```

Expected: command prints the authenticated account and usable scopes. If it reports missing scopes, stop and report the exact remediation from the command output.

- [ ] **Step 4: Start Colab session**

Run:

```bash
colab new -s sali-practical --gpu T4
```

Expected: session is created. If GPU allocation fails due to quota, retry with CPU:

```bash
colab new -s sali-practical
```

- [ ] **Step 5: Package and upload the project to Colab**

Run:

```bash
tar -czf /private/tmp/sali-colab.tar.gz -C /Users/weitao/Code/python sali
colab upload -s sali-practical /private/tmp/sali-colab.tar.gz /content/sali-colab.tar.gz
```

Expected: the archive is created locally and uploaded to `/content/sali-colab.tar.gz` on the Colab VM.

- [ ] **Step 6: Execute practical run in Colab**

Run:

```bash
colab exec -s sali-practical -f scripts/run_colab_uploaded.py
```

Expected: command extracts `/content/sali`, installs the package in editable mode, runs `scripts/train_colab.py`, and completes or reports a concrete dependency/runtime issue.

- [ ] **Step 7: Inspect Colab status and logs**

Run:

```bash
colab status -s sali-practical
colab log -s sali-practical -n 20
```

Expected: status/logs show the run completed or identify a specific failure.

- [ ] **Step 8: Stop Colab session**

Run:

```bash
colab stop -s sali-practical
```

Expected: session stops so compute units are not wasted.

- [ ] **Step 9: Final git status review**

Run:

```bash
git status --short sali
```

Expected: shows only intended SALI files. Do not modify unrelated sibling repository changes.

## Self-Review Checklist

- Spec coverage: tasks cover scaffold, config, physics, target heatmaps, data splits and normalization, PyTorch model, training recipe, post-processing, evaluation metrics, requested plots, notebook, tests, and Colab CLI.
- Type consistency: `RunConfig`, `PhysicsConfig`, `DataConfig`, `ModelConfig`, `Couplings`, `Sample`, `Prediction`, and `SampleMetrics` are introduced before downstream use.
- Path consistency: all project files are under `/Users/weitao/Code/python/sali`; Git commands are scoped to `sali` when checking status from the parent repository.
- Verification: local tests, local smoke run, Colab authentication, Colab run, log inspection, and Colab stop are explicit.
