import json
import math

import numpy as np

from sali_pycce.physics import AnalyticCPMGSimulator, SpinParams
from scripts.colab_train_report import (
    finite_or_none,
    reconstruct_signals_from_detections,
    strict_json,
)


def test_strict_json_converts_nonfinite_metrics_to_null():
    payload = {
        "matched_mae_khz": finite_or_none(float("nan")),
        "finite_metric": finite_or_none(1.25),
    }

    text = strict_json(payload)
    decoded = json.loads(text)

    assert "NaN" not in text
    assert decoded["matched_mae_khz"] is None
    assert decoded["finite_metric"] == 1.25


def test_finite_or_none_rejects_infinities():
    assert finite_or_none(math.inf) is None
    assert finite_or_none(-math.inf) is None


def test_reconstruct_signals_from_detections_uses_identified_spin_parameters():
    simulator = AnalyticCPMGSimulator(
        pulses=(32, 256),
        tau_ranges_us=((0.0, 4.0), (0.0, 4.0)),
        signal_points=16,
        shots=1000,
        t2_us=800.0,
    )
    detections = [{"az_khz": 12.0, "aperp_khz": 34.0}]

    reconstructed = reconstruct_signals_from_detections(simulator, detections)
    expected = simulator.sample([SpinParams(12.0, 34.0)], noisy=False)["signals"]

    assert reconstructed.shape == (2, 16)
    np.testing.assert_allclose(reconstructed, expected)
