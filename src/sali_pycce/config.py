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
