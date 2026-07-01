from __future__ import annotations

import numpy as np
import pytest

from sali.config import DataConfig, ModelConfig, PostprocessConfig
import sali.postprocess as postprocess_module
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


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_postprocess_rejects_invalid_threshold(threshold: float) -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig(threshold=threshold)
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)

    with pytest.raises(ValueError, match="threshold"):
        postprocess_heatmap(heatmap, data, model, pp)


def test_postprocess_rejects_invalid_local_max_distance() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig(threshold=0.2, local_max_min_distance_px=0)
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)

    with pytest.raises(ValueError, match="local_max_min_distance_px"):
        postprocess_heatmap(heatmap, data, model, pp)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (PostprocessConfig(min_area=0), "min_area"),
        (PostprocessConfig(erosion_size=-1), "erosion_size"),
        (PostprocessConfig(dilation_size=-1), "dilation_size"),
    ],
)
def test_postprocess_rejects_invalid_integer_config(config: PostprocessConfig, message: str) -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        postprocess_heatmap(heatmap, data, model, config)


def test_postprocess_rejects_bad_heatmap_shape() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig()
    heatmap = np.zeros((model.output_height, model.output_width), dtype=np.float32)

    with pytest.raises(ValueError, match="heatmap must have shape"):
        postprocess_heatmap(heatmap, data, model, pp)


def test_postprocess_rejects_nonfinite_heatmap() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig()
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)
    heatmap[0, 12, 14] = np.nan

    with pytest.raises(FloatingPointError, match="non-finite"):
        postprocess_heatmap(heatmap, data, model, pp)


def test_postprocess_preserves_one_pixel_detection_with_min_area_one() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig(threshold=0.2, min_area=1)
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)
    heatmap[0, 100, 50] = 1.0

    predictions = postprocess_heatmap(heatmap, data, model, pp)

    assert len(predictions) == 1
    assert predictions[0].row == 100.0
    assert predictions[0].col == 50.0


def test_postprocess_preserves_one_pixel_detection_next_to_surviving_blob() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig(threshold=0.5, min_area=1, erosion_size=1, dilation_size=1)
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)
    heatmap[0, 39:42, 39:42] = np.array(
        [
            [0.6, 0.7, 0.6],
            [0.7, 1.0, 0.7],
            [0.6, 0.7, 0.6],
        ],
        dtype=np.float32,
    )
    heatmap[0, 100, 80] = 0.9

    predictions = postprocess_heatmap(heatmap, data, model, pp)

    assert len(predictions) == 2
    assert {(round(prediction.row), round(prediction.col)) for prediction in predictions} == {
        (40, 40),
        (100, 80),
    }


def test_postprocess_splits_connected_region_with_two_separated_peaks() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig(
        threshold=0.2,
        min_area=1,
        local_max_min_distance_px=3,
        erosion_size=0,
        dilation_size=0,
    )
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)
    heatmap[0, 60, 30:39] = np.array([1.0, 0.8, 0.6, 0.4, 0.21, 0.4, 0.6, 0.8, 0.9])

    predictions = postprocess_heatmap(heatmap, data, model, pp)

    assert len(predictions) == 2
    assert {(prediction.row, prediction.col) for prediction in predictions} == {(60.0, 30.0), (60.0, 38.0)}


def test_postprocess_border_prediction_stays_clamped_to_ranges_and_valid_box() -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    pp = PostprocessConfig(threshold=0.2, min_area=1)
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)
    heatmap[0, 0, 0] = 1.0

    predictions = postprocess_heatmap(heatmap, data, model, pp)

    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.az_khz == data.az_min_khz
    assert pred.aperp_khz == data.aperp_min_khz
    assert 0 <= pred.box.row_min <= pred.box.row_max < model.output_height
    assert 0 <= pred.box.col_min <= pred.box.col_max < model.output_width


def test_square_footprint_falls_back_to_skimage_022_square(monkeypatch: pytest.MonkeyPatch) -> None:
    class LegacyMorphology:
        @staticmethod
        def square(size: int) -> np.ndarray:
            return np.ones((size, size), dtype=bool)

    monkeypatch.setattr(postprocess_module, "morphology", LegacyMorphology)

    np.testing.assert_array_equal(postprocess_module._square_footprint(3), np.ones((3, 3), dtype=bool))
