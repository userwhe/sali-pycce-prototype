from __future__ import annotations

from sali.data import generate_splits
from sali.train import evaluate_model, train_model


def test_evaluate_model_returns_metrics(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "run"
    splits, _stats = generate_splits(tiny_config)
    result = train_model(tiny_config, splits)

    metrics = evaluate_model(result.model, tiny_config, splits.test, max_samples=2)

    assert len(metrics) == 2
    assert metrics[0].true_nuclei >= 1
