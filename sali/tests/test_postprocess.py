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
