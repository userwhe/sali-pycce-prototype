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
