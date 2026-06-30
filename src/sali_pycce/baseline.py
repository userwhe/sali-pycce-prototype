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
    for name in ("rmse", "a_error_khz", "b_error_khz"):
        if hasattr(spin, name):
            value = getattr(spin, name)
            if value is not None:
                row[name] = float(value)
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
