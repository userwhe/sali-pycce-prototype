from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sali.config import DataConfig, ModelConfig
from sali.physics import Couplings


@dataclass(frozen=True, slots=True)
class Box:
    row_min: int
    col_min: int
    row_max: int
    col_max: int

    @property
    def height(self) -> int:
        return self.row_max - self.row_min + 1

    @property
    def width(self) -> int:
        return self.col_max - self.col_min + 1


def coupling_to_pixel(
    az_khz: float,
    aperp_khz: float,
    data: DataConfig,
    model: ModelConfig,
) -> tuple[int, int]:
    y = (az_khz - data.az_min_khz) / (data.az_max_khz - data.az_min_khz)
    x = (aperp_khz - data.aperp_min_khz) / (data.aperp_max_khz - data.aperp_min_khz)
    effective_h = model.output_height - 4
    effective_w = model.output_width - 4
    row = int(np.rint(y * (effective_h - 1))) + 2
    col = int(np.rint(x * (effective_w - 1))) + 2
    row = int(np.clip(row, 2, model.output_height - 3))
    col = int(np.clip(col, 2, model.output_width - 3))
    return row, col


def pixel_to_coupling(
    row: float,
    col: float,
    data: DataConfig,
    model: ModelConfig,
) -> tuple[float, float]:
    row = float(np.clip(row, 2, model.output_height - 3))
    col = float(np.clip(col, 2, model.output_width - 3))
    effective_h = model.output_height - 4
    effective_w = model.output_width - 4
    y = (row - 2.0) / float(effective_h - 1)
    x = (col - 2.0) / float(effective_w - 1)
    az = data.az_min_khz + y * (data.az_max_khz - data.az_min_khz)
    aperp = data.aperp_min_khz + x * (data.aperp_max_khz - data.aperp_min_khz)
    return float(az), float(aperp)


def _five_by_five_box(row: int, col: int, model: ModelConfig) -> Box:
    return Box(
        row_min=max(0, row - 2),
        col_min=max(0, col - 2),
        row_max=min(model.output_height - 1, row + 2),
        col_max=min(model.output_width - 1, col + 2),
    )


def render_heatmap(nuclei: Couplings, data: DataConfig, model: ModelConfig) -> np.ndarray:
    heatmap = np.zeros((1, model.output_height, model.output_width), dtype=np.float32)
    offsets = np.arange(-2, 3, dtype=np.float32)
    rr, cc = np.meshgrid(offsets, offsets, indexing="ij")
    kernel = np.exp(-((rr**2 + cc**2) / 2.0)).astype(np.float32)
    for az, aperp in zip(nuclei.az_khz, nuclei.aperp_khz, strict=True):
        row, col = coupling_to_pixel(float(az), float(aperp), data, model)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                target_row = row + dr
                target_col = col + dc
                if 0 <= target_row < model.output_height and 0 <= target_col < model.output_width:
                    heatmap[0, target_row, target_col] += kernel[dr + 2, dc + 2]
    np.clip(heatmap, 0.0, 1.0, out=heatmap)
    return heatmap


def true_boxes(nuclei: Couplings, data: DataConfig, model: ModelConfig) -> list[Box]:
    boxes: list[Box] = []
    for az, aperp in zip(nuclei.az_khz, nuclei.aperp_khz, strict=True):
        row, col = coupling_to_pixel(float(az), float(aperp), data, model)
        boxes.append(_five_by_five_box(row, col, model))
    return boxes
