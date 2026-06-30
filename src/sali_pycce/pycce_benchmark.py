"""PyCCE random-bath benchmark generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import RESEARCH_CONFIG
from .heatmap import make_heatmap
from .physics import AnalyticCPMGSimulator, SpinParams
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


def _is_c13_name(name: object) -> bool:
    if isinstance(name, bytes):
        name = name.decode(errors="ignore")
    return str(name) == "13C"


def bath_arrays_to_spin_table(
    names: np.ndarray,
    xyz: np.ndarray,
    tensors: np.ndarray,
    tensor_units: str = "kHz",
) -> np.ndarray:
    projected = project_hyperfine_tensors(tensors, units=tensor_units)
    coordinates = np.asarray(xyz, dtype=np.float32)
    if coordinates.shape != (projected.shape[0], 3):
        raise ValueError(f"xyz must have shape {(projected.shape[0], 3)}, got {coordinates.shape}")
    names_arr = np.asarray(names)
    if names_arr.shape[0] != projected.shape[0]:
        raise ValueError(f"names must have length {projected.shape[0]}, got {names_arr.shape[0]}")

    table = np.zeros((projected.shape[0], 9), dtype=np.float32)
    table[:, 0] = np.arange(projected.shape[0], dtype=np.float32)
    table[:, 1:4] = coordinates
    table[:, 4:6] = projected
    table[:, 6] = np.linalg.norm(coordinates.astype(float), axis=1)
    table[:, 7] = np.asarray([1.0 if _is_c13_name(name) else 0.0 for name in names_arr], dtype=np.float32)
    table[:, 8] = 0.0
    return table


def _spin_columns(spin_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.asarray(spin_table, dtype=float)
    if table.ndim != 2:
        raise ValueError(f"spin_table must be 2D, got {table.shape}")
    if table.shape[1] == 3:
        return table[:, 0], table[:, 1], table[:, 2]
    if table.shape[1] == 9:
        return table[:, 4], table[:, 5], table[:, 8]
    raise ValueError(f"spin_table must have shape (n, 3) or (n, 9), got {table.shape}")


def detectable_spin_mask(spin_table: np.ndarray, config: PyCCEBenchmarkConfig) -> np.ndarray:
    az, aperp, depth = _spin_columns(spin_table)
    return (
        (az >= config.az_range[0])
        & (az <= config.az_range[1])
        & (aperp >= config.aperp_range[0])
        & (aperp <= config.aperp_range[1])
        & (depth >= config.min_signal_depth)
    )


def add_signal_depths(spin_table: np.ndarray, config: PyCCEBenchmarkConfig) -> np.ndarray:
    table = np.asarray(spin_table, dtype=np.float32).copy()
    if table.ndim != 2 or table.shape[1] != 9:
        raise ValueError(f"spin_table must have shape (n, 9), got {table.shape}")
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


def require_pycce():
    try:
        import pycce as pc
    except Exception as exc:
        raise ImportError(
            "PyCCE is required for PyCCE benchmark generation. Install with `pip install pycce`."
        ) from exc
    return pc


def _require_attr(obj: Any, attr: str) -> Any:
    if not hasattr(obj, attr):
        raise RuntimeError(f"PyCCE object is missing required attribute: {attr}")
    return getattr(obj, attr)


def _bath_array(bath: Any, attr: str) -> np.ndarray:
    if hasattr(bath, attr):
        return np.asarray(getattr(bath, attr))
    try:
        return np.asarray(bath[attr])
    except Exception as exc:
        raise RuntimeError(f"PyCCE bath is missing required attribute: {attr}") from exc


def generate_random_c13_bath(config: PyCCEBenchmarkConfig):
    pc = require_pycce()
    random_bath = _require_attr(pc, "random_bath")
    bath = random_bath("13C", config.bath_size_angstrom, number=config.bath_number, seed=config.seed)
    from_point_dipole = _require_attr(bath, "from_point_dipole")
    from_point_dipole(np.zeros(3), inplace=True)
    return bath


def _to_pycce_times(taus_us: np.ndarray, units: str) -> np.ndarray:
    normalized = units.strip().lower()
    if normalized == "ms":
        return taus_us * 1e-3
    if normalized in {"us", "microsecond", "microseconds"}:
        return taus_us
    raise ValueError(f"Unsupported PyCCE time units: {units!r}")


def simulate_bath_traces(bath, config: PyCCEBenchmarkConfig) -> dict[str, np.ndarray]:
    pc = require_pycce()
    simulator_cls = _require_attr(pc, "Simulator")
    taus = np.linspace(config.tau_range_us[0], config.tau_range_us[1], config.signal_points)
    pycce_times = _to_pycce_times(taus, config.pycce_time_units)
    signals = []
    for n_pulses in config.pulses:
        sim = simulator_cls(
            1,
            bath=bath,
            magnetic_field=np.asarray([0.0, 0.0, config.b_gauss]),
            pulses=int(n_pulses),
            as_delay=True,
            order=1,
        )
        coherence = sim.compute(pycce_times)
        px = 0.5 * (1.0 + np.real(coherence))
        signals.append(np.clip(px, 0.0, 1.0).astype(np.float32))
    return {
        "taus_us": np.stack([taus.astype(np.float32)] * len(config.pulses), axis=0),
        "signals": np.stack(signals, axis=0),
        "pycce_times": np.stack([pycce_times.astype(np.float32)] * len(config.pulses), axis=0),
    }


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
    spin_table = bath_arrays_to_spin_table(
        _bath_array(bath, "N"),
        _bath_array(bath, "xyz"),
        _bath_array(bath, "A"),
        tensor_units=config.tensor_units,
    )
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
