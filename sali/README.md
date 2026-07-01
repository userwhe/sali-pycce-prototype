# SALI Reproduction

PyTorch reproduction of the SALI signal-to-image model for automatic C13 nuclear-spin detection around NV centers.

The paper reports TensorFlow/Keras training, but this reproduction translates the architecture and training recipe to PyTorch while preserving the described physics simulator, target heatmap representation, post-processing, and evaluation workflow.

## Local Setup

```bash
python -m pip install -e ".[dev]"
pytest
```

## Practical Smoke Run

```bash
python scripts/train_colab.py --preset practical --field low --output-dir runs/practical-low
```

## Colab CLI Run

```bash
colab new -s sali-practical --gpu T4
colab exec -s sali-practical -f scripts/train_colab.py
colab stop -s sali-practical
```

## Paper-Scale Settings

The `paper` preset exposes the paper's sample count, split ratios, signal lengths, batch size, optimizer, learning-rate schedule, and early-stopping patience. It is expected to require substantial accelerator time and storage.
