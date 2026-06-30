import json

from sali_pycce.evaluate import main as evaluate_main
from sali_pycce.train import main as train_main


def test_evaluate_writes_json_metrics(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    metrics = tmp_path / "metrics.json"

    train_main(
        [
            "--preset",
            "smoke",
            "--train-samples",
            "8",
            "--val-samples",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--signal-points",
            "64",
            "--heatmap-height",
            "32",
            "--heatmap-width",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--out",
            str(checkpoint),
            "--threads",
            "1",
        ]
    )
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--samples",
            "4",
            "--batch-size",
            "2",
            "--signal-points",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--metrics-out",
            str(metrics),
            "--threads",
            "1",
        ]
    )

    payload = json.loads(metrics.read_text())
    assert {"samples", "tp", "fp", "fn", "precision", "recall", "matched_mae_khz"}.issubset(
        payload.keys()
    )


def test_evaluate_uses_preset_test_samples_when_samples_omitted(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    metrics = tmp_path / "metrics.json"

    train_main(
        [
            "--preset",
            "smoke",
            "--train-samples",
            "8",
            "--val-samples",
            "4",
            "--test-samples",
            "128",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--signal-points",
            "64",
            "--heatmap-height",
            "32",
            "--heatmap-width",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--out",
            str(checkpoint),
            "--threads",
            "1",
        ]
    )
    evaluate_main(
        [
            "--preset",
            "smoke",
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            "64",
            "--signal-points",
            "64",
            "--max-spins",
            "2",
            "--device",
            "cpu",
            "--metrics-out",
            str(metrics),
            "--threads",
            "1",
        ]
    )

    payload = json.loads(metrics.read_text())
    assert payload["samples"] == 128
