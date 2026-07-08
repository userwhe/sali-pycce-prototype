from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/sali_reproduction_colab.ipynb")


def code(source: str):
    return nbf.v4.new_code_cell(source)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown("# SALI PyTorch Reproduction\n\nColab-ready practical reproduction of the SALI signal-to-image model."),
        code("%pip install -q -e ."),
        markdown("## Configuration"),
        code(
            "from pathlib import Path\n"
            "from sali.config import practical_config\n\n"
            "cfg = practical_config(field='low')\n"
            "cfg.output_dir = Path('runs/notebook-practical-low')\n"
            "cfg.training.device = 'auto'\n"
            "cfg.training.max_epochs = 1\n"
            "cfg.training.batch_size = 4\n"
            "cfg.data.train_samples = 64\n"
            "cfg.data.val_samples = 16\n"
            "cfg.data.test_samples = 16\n"
            "cfg.output_dir.mkdir(parents=True, exist_ok=True)\n"
            "cfg\n"
        ),
        markdown("## Generate training, validation and testing dataset"),
        code(
            "from sali.data import generate_splits\n\n"
            "splits, stats = generate_splits(cfg)\n"
            "len(splits.train), len(splits.val), len(splits.test), stats\n"
        ),
        markdown("## Generated spectra"),
        code(
            "from IPython.display import Image, display\n"
            "from sali.plots import plot_spectra\n\n"
            "sample = splits.test[0]\n"
            "path = cfg.output_dir / 'figures' / 'generated_spectra.png'\n"
            "plot_spectra(sample.raw_signals, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## True heatmap"),
        code(
            "from sali.plots import plot_heatmap\n\n"
            "path = cfg.output_dir / 'figures' / 'true_heatmap.png'\n"
            "plot_heatmap(sample.heatmap, 'True heatmap', path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## Train PyTorch SALI model"),
        code(
            "from sali.train import train_model\n\n"
            "result = train_model(cfg, splits)\n"
            "result.best_checkpoint\n"
        ),
        markdown("## Training and validation loss"),
        code(
            "from sali.plots import plot_loss\n\n"
            "path = cfg.output_dir / 'figures' / 'loss.png'\n"
            "plot_loss(result.history, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## Predicted and post-processed heatmaps"),
        code(
            "import numpy as np\n"
            "import torch\n"
            "from sali.postprocess import postprocess_heatmap\n\n"
            "device = next(result.model.parameters()).device\n"
            "result.model.eval()\n"
            "with torch.no_grad():\n"
            "    pred = result.model(\n"
            "        torch.from_numpy(sample.signals[0:1]).unsqueeze(0).to(device),\n"
            "        torch.from_numpy(sample.signals[1:2]).unsqueeze(0).to(device),\n"
            "    ).cpu().numpy()[0]\n"
            "predictions = postprocess_heatmap(pred, cfg.data, cfg.model, cfg.postprocess)\n"
            "pred_path = cfg.output_dir / 'figures' / 'predicted_heatmap.png'\n"
            "plot_heatmap(pred, 'Predicted heatmap', pred_path)\n"
            "display(Image(filename=str(pred_path)))\n"
            "post = np.zeros_like(pred)\n"
            "for item in predictions:\n"
            "    row = int(round(item.row))\n"
            "    col = int(round(item.col))\n"
            "    post[0, max(0, row - 2):row + 3, max(0, col - 2):col + 3] = 1.0\n"
            "post_path = cfg.output_dir / 'figures' / 'postprocessed_heatmap.png'\n"
            "plot_heatmap(post, 'Post-processed heatmap', post_path)\n"
            "display(Image(filename=str(post_path)))\n"
            "[(p.az_khz, p.aperp_khz, p.confidence) for p in predictions[:10]]\n"
        ),
        markdown("## Original vs identified C13 reconstructed signals"),
        code(
            "from dataclasses import replace\n"
            "from sali.physics import Couplings, generate_sample_signals\n"
            "from sali.plots import plot_signal_overlay\n\n"
            "pred_couplings = Couplings(\n"
            "    az_khz=np.array([item.az_khz for item in predictions], dtype=np.float32),\n"
            "    aperp_khz=np.array([item.aperp_khz for item in predictions], dtype=np.float32),\n"
            ")\n"
            "clean_physics = replace(cfg.physics, add_shot_noise=False)\n"
            "reconstructed = generate_sample_signals(\n"
            "    pred_couplings,\n"
            "    clean_physics,\n"
            "    np.random.default_rng(cfg.data.seed + 202),\n"
            ")\n"
            "path = cfg.output_dir / 'figures' / 'signal_overlay.png'\n"
            "plot_signal_overlay(sample.raw_signals, reconstructed, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## Precision and recall"),
        code(
            "from sali.plots import plot_precision_recall\n"
            "from sali.train import evaluate_model\n\n"
            "metrics = evaluate_model(result.model, cfg, splits.test, max_samples=16)\n"
            "path = cfg.output_dir / 'figures' / 'precision_recall.png'\n"
            "plot_precision_recall(metrics, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
        markdown("## MAE"),
        code(
            "from sali.plots import plot_mae, plot_mae_by_nuclei\n\n"
            "path = cfg.output_dir / 'figures' / 'mae.png'\n"
            "plot_mae(metrics, path)\n"
            "display(Image(filename=str(path)))\n"
            "path = cfg.output_dir / 'figures' / 'mae_by_nuclei.png'\n"
            "plot_mae_by_nuclei(metrics, path)\n"
            "display(Image(filename=str(path)))\n"
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK)


if __name__ == "__main__":
    main()
