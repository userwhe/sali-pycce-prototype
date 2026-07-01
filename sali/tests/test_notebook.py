from __future__ import annotations

import nbformat


def test_notebook_contains_requested_plot_sections() -> None:
    notebook = nbformat.read("notebooks/sali_reproduction_colab.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "Generated spectra" in source
    assert "True heatmap" in source
    assert "Original vs identified C13 reconstructed signals" in source
    assert "Training and validation loss" in source
    assert "Precision and recall" in source
    assert "MAE" in source
