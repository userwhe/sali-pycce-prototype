from __future__ import annotations

import numpy as np

from sali.config import DataConfig, ModelConfig
from sali.physics import Couplings
from sali.targets import coupling_to_pixel, pixel_to_coupling, render_heatmap, true_boxes


def test_coupling_to_pixel_maps_ranges_with_border() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    row, col = coupling_to_pixel(az_khz=-100.0, aperp_khz=2.0, data=data, model=model)
    assert (row, col) == (2, 2)
    row, col = coupling_to_pixel(az_khz=100.0, aperp_khz=102.0, data=data, model=model)
    assert (row, col) == (201, 101)


def test_coupling_to_pixel_and_inverse_handle_asymmetric_axes() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    row, col = coupling_to_pixel(az_khz=-50.0, aperp_khz=77.0, data=data, model=model)
    assert (row, col) == (52, 76)

    az_khz, aperp_khz = pixel_to_coupling(row, col, data, model)
    az_half_bin = (data.az_max_khz - data.az_min_khz) / (model.output_height - 5) / 2.0
    aperp_half_bin = (data.aperp_max_khz - data.aperp_min_khz) / (model.output_width - 5) / 2.0
    assert abs(az_khz - -50.0) <= az_half_bin
    assert abs(aperp_khz - 77.0) <= aperp_half_bin


def test_pixel_to_coupling_clamps_border_pixels_to_coupling_ranges() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()

    min_az, min_aperp = pixel_to_coupling(0, 0, data, model)
    max_az, max_aperp = pixel_to_coupling(model.output_height - 1, model.output_width - 1, data, model)

    np.testing.assert_allclose((min_az, min_aperp), (data.az_min_khz, data.aperp_min_khz))
    np.testing.assert_allclose((max_az, max_aperp), (data.az_max_khz, data.aperp_max_khz))


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


def test_render_heatmap_clips_overlapping_nuclei() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    nuclei = Couplings(
        az_khz=np.array([0.0, 0.0], dtype=np.float32),
        aperp_khz=np.array([52.0, 52.0], dtype=np.float32),
    )
    heatmap = render_heatmap(nuclei, data, model)
    assert heatmap.max() <= 1.0


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


def test_true_boxes_clip_to_heatmap_edges() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    nuclei = Couplings(
        az_khz=np.array([-100.0, 100.0], dtype=np.float32),
        aperp_khz=np.array([2.0, 102.0], dtype=np.float32),
    )
    boxes = true_boxes(nuclei, data, model)
    assert boxes[0].row_min == 0
    assert boxes[0].col_min == 0
    assert boxes[0].row_max == 4
    assert boxes[0].col_max == 4
    assert boxes[1].row_min == 199
    assert boxes[1].col_min == 99
    assert boxes[1].row_max == 203
    assert boxes[1].col_max == 103
