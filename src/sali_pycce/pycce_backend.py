"""Optional PyCCE backend adapter.

This file intentionally keeps the interface small.  PyCCE bath construction is
highly experiment-specific, so the adapter is a scaffold rather than a hidden
black box.  Replace ``simulate_trace`` with your validated PyCCE setup once you
fix central-spin basis, bath coordinates, hyperfine tensors, magnetic field, and
pulse convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .physics import AnalyticCPMGSimulator, SpinParams


@dataclass
class PyCCESimulator:
    """PyCCE-backed simulator interface with analytic fallback for development.

    Parameters
    ----------
    use_fallback:
        If True, returns analytic traces when PyCCE is unavailable or the PyCCE
        adapter is not fully configured.  If False, raises a clear error.
    """

    b_gauss: float = 500.0
    pulses: tuple[int, int] = (32, 256)
    tau_ranges_us: tuple[tuple[float, float], tuple[float, float]] = ((6.0, 50.0), (10.0, 40.0))
    signal_points: int = 1000
    use_fallback: bool = True

    def __post_init__(self) -> None:
        try:
            import pycce as _pycce  # type: ignore
        except Exception:  # pragma: no cover - depends on optional package
            _pycce = None
        self.pycce = _pycce
        self.fallback = AnalyticCPMGSimulator(
            b_gauss=self.b_gauss,
            pulses=self.pulses,
            tau_ranges_us=self.tau_ranges_us,
            signal_points=self.signal_points,
        )

    def sample(
        self,
        spins: Sequence[SpinParams],
        rng: np.random.Generator | None = None,
        noisy: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return SALI-compatible traces.

        Current behavior:
        - If PyCCE is not installed, use analytic fallback when allowed.
        - If PyCCE is installed, this still raises unless you implement
          ``simulate_trace`` for your exact bath model.
        """
        if self.pycce is None:
            if self.use_fallback:
                return self.fallback.sample(spins, rng=rng, noisy=noisy)
            raise ImportError("PyCCE is not installed. Run `pip install pycce` or enable fallback.")

        if self.use_fallback:
            # Keep the prototype runnable until the experiment-specific bath
            # construction is provided.
            return self.fallback.sample(spins, rng=rng, noisy=noisy)

        raise NotImplementedError(
            "PyCCE is installed, but you must implement the experiment-specific "
            "bath construction and CPMG compute call in pycce_backend.py. "
            "See PyCCE Simulator(pulses=N, as_delay=...) documentation."
        )

    def simulate_trace(self, spins: Sequence[SpinParams], n_pulses: int, tau_us: np.ndarray) -> np.ndarray:
        """Template for a real PyCCE trace implementation.

        Pseudocode sketch:

        ```python
        import pycce as pc
        bath = build_bath_array_from_spins(spins)  # positions or tensors required
        calc = pc.Simulator(
            1,
            bath=bath,
            r_bath=..., r_dipole=..., order=...,
            D=2.87e6,
            magnetic_field=self.b_gauss,
            pulses=n_pulses,
            as_delay=True,
        )
        coherence = calc.compute(tau_us * 1e-3)  # check PyCCE time units
        px = 0.5 * (1 + np.real(coherence))
        ```
        """
        raise NotImplementedError
