from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import dilation, erosion, footprint_rectangle

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


def _region_max_intensity(region) -> float:
    intensity_max = getattr(region, "intensity_max", None)
    if intensity_max is None:
        intensity_max = region.max_intensity
    return float(intensity_max)


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
        mask = erosion(mask, footprint_rectangle((2 * cfg.erosion_size + 1, 2 * cfg.erosion_size + 1)))
    if cfg.dilation_size > 0:
        mask = dilation(mask, footprint_rectangle((2 * cfg.dilation_size + 1, 2 * cfg.dilation_size + 1)))
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
                confidence=_region_max_intensity(region),
            )
        )
    return predictions
