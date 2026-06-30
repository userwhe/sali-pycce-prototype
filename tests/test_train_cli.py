import csv

from sali_pycce.train import main


def test_train_writes_checkpoint_and_history(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    history = tmp_path / "history.csv"

    main(
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
            "--history-out",
            str(history),
            "--threads",
            "1",
        ]
    )

    assert checkpoint.exists()
    assert history.exists()
    rows = list(csv.DictReader(history.open()))
    assert len(rows) == 1
    assert {"epoch", "train_loss", "val_loss", "precision", "recall", "mae_khz"}.issubset(
        rows[0].keys()
    )
