from __future__ import annotations

import numpy as np

from sali.plots import plot_heatmap, plot_loss


def test_plot_loss_writes_file(tmp_path) -> None:
    path = tmp_path / "loss.png"
    plot_loss({"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}, path)
    assert path.exists()


def test_plot_heatmap_writes_file(tmp_path) -> None:
    path = tmp_path / "heatmap.png"
    heatmap = np.zeros((1, 204, 104), dtype=np.float32)
    heatmap[0, 100, 50] = 1.0
    plot_heatmap(heatmap, "True Heatmap", path)
    assert path.exists()
