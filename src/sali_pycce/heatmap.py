"""Heatmap label generation and decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .physics import SpinParams


@dataclass(frozen=True)
class HeatmapSpec:
    """Coordinate system for the output image."""

    height: int = 32
    width: int = 64
    az_range: tuple[float, float] = (-100.0, 100.0)
    aperp_range: tuple[float, float] = (2.0, 102.0)
    sigma_px: float = 1.0
    patch_radius: int = 2

    def coord_to_pixel(self, az_khz: float, aperp_khz: float) -> tuple[float, float]:
        """Map physical coordinates to fractional image coordinates ``(row, col)``."""
        col = (az_khz - self.az_range[0]) / (self.az_range[1] - self.az_range[0]) * (self.width - 1)
        row = (aperp_khz - self.aperp_range[0]) / (self.aperp_range[1] - self.aperp_range[0]) * (self.height - 1)
        return float(row), float(col)

    def pixel_to_coord(self, row: float, col: float) -> tuple[float, float]:
        """Map fractional image coordinates ``(row, col)`` to ``(A_z, A_perp)``."""
        az = self.az_range[0] + col / (self.width - 1) * (self.az_range[1] - self.az_range[0])
        aperp = self.aperp_range[0] + row / (self.height - 1) * (self.aperp_range[1] - self.aperp_range[0])
        return float(az), float(aperp)


def make_heatmap(spins: list[SpinParams], spec: HeatmapSpec = HeatmapSpec()) -> np.ndarray:
    """Render spins as Gaussian blobs on a 2D heatmap."""
    image = np.zeros((spec.height, spec.width), dtype=np.float32)
    r = spec.patch_radius
    for spin in spins:
        row_f, col_f = spec.coord_to_pixel(spin.az_khz, spin.aperp_khz)
        row0 = int(round(row_f))
        col0 = int(round(col_f))
        for rr in range(row0 - r, row0 + r + 1):
            if rr < 0 or rr >= spec.height:
                continue
            for cc in range(col0 - r, col0 + r + 1):
                if cc < 0 or cc >= spec.width:
                    continue
                value = np.exp(-((rr - row_f) ** 2 + (cc - col_f) ** 2) / (2.0 * spec.sigma_px**2))
                image[rr, cc] = max(image[rr, cc], float(value))
    return image


def decode_heatmap(
    heatmap: np.ndarray,
    spec: HeatmapSpec = HeatmapSpec(),
    threshold: float = 0.25,
    min_area: int = 3,
    morph: bool = True,
) -> list[dict[str, float]]:
    """Decode a heatmap into a list of detected nuclei.

    Returns dictionaries containing estimated ``az_khz``, ``aperp_khz``, pixel
    area, and mean score.
    """
    h = np.asarray(heatmap)
    if h.ndim == 3:
        h = h.squeeze()
    if h.shape != (spec.height, spec.width):
        raise ValueError(f"Expected heatmap shape {(spec.height, spec.width)}, got {h.shape}")

    mask = h >= threshold
    if morph:
        mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
        mask = ndimage.binary_dilation(mask, structure=np.ones((2, 2), dtype=bool))

    labels, n = ndimage.label(mask)
    detections: list[dict[str, float]] = []
    for label_id in range(1, n + 1):
        region = labels == label_id
        area = int(region.sum())
        if area < min_area:
            continue
        weights = np.where(region, h, 0.0)
        total = float(weights.sum())
        if total <= 0:
            rows, cols = np.nonzero(region)
            row = float(rows.mean())
            col = float(cols.mean())
        else:
            row, col = ndimage.center_of_mass(weights)
            row = float(row)
            col = float(col)
        az, aperp = spec.pixel_to_coord(row, col)
        detections.append(
            {
                "az_khz": az,
                "aperp_khz": aperp,
                "area": float(area),
                "score": float(h[region].mean()),
                "row": row,
                "col": col,
            }
        )
    return detections


def nearest_match_errors(
    truth: np.ndarray,
    pred: list[dict[str, float]],
    max_dist_khz: float = 5.0,
) -> dict[str, float]:
    """Simple one-to-one nearest-neighbor matching for quick evaluation."""
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    pred_arr = np.asarray([[p["az_khz"], p["aperp_khz"]] for p in pred], dtype=float).reshape(-1, 2)
    if len(truth) == 0 and len(pred_arr) == 0:
        return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "mae_khz": float("nan")}
    if len(truth) == 0:
        return {"tp": 0.0, "fp": float(len(pred_arr)), "fn": 0.0, "mae_khz": float("nan")}
    if len(pred_arr) == 0:
        return {"tp": 0.0, "fp": 0.0, "fn": float(len(truth)), "mae_khz": float("nan")}

    dists = np.linalg.norm(truth[:, None, :] - pred_arr[None, :, :], axis=-1)
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    errors: list[float] = []

    while True:
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if not np.isfinite(dists[i, j]) or dists[i, j] > max_dist_khz:
            break
        used_truth.add(int(i))
        used_pred.add(int(j))
        errors.append(float(np.mean(np.abs(truth[i] - pred_arr[j]))))
        dists[i, :] = np.inf
        dists[:, j] = np.inf

    tp = len(errors)
    fp = len(pred_arr) - len(used_pred)
    fn = len(truth) - len(used_truth)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "mae_khz": float(np.mean(errors)) if errors else float("nan"),
    }
