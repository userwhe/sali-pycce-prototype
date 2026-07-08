from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


def test_train_colab_help_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/train_colab.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--preset" in completed.stdout
    assert "--field" in completed.stdout
    assert "--loss-type" in completed.stdout
    assert "--positive-weight" in completed.stdout
    assert "--border-penalty-weight" in completed.stdout
    assert "--threshold-mode" in completed.stdout
    assert "--data-mode" in completed.stdout
    assert "--resume" in completed.stdout
    assert "--checkpoint-every-epochs" in completed.stdout
    assert "--calibrate-every-epochs" in completed.stdout
    assert "--sample-plots-every-epochs" in completed.stdout
    assert "--normalization-samples" in completed.stdout
    assert "--run-root" in completed.stdout
    assert "--full-test-eval" in completed.stdout
    assert "--no-final-eval" in completed.stdout
    assert "--dataset-dir" in completed.stdout
    assert "--shard-size" in completed.stdout
    assert "--generate-shards" in completed.stdout
    assert "--shard-dtype" in completed.stdout
    assert "--raw-signal-dtype" in completed.stdout
    assert "--cache-dataset-dir" in completed.stdout
    assert "--skip-existing-shards" in completed.stdout
    assert "--samples-per-epoch" in completed.stdout
    assert "--learning-rate" in completed.stdout


def test_train_colab_defaults_paper_to_sharded_mode() -> None:
    from scripts.train_colab import default_data_mode

    assert default_data_mode("paper") == "sharded"
    assert default_data_mode("practical") == "materialized"


def test_train_colab_applies_samples_per_epoch(monkeypatch) -> None:
    from scripts.train_colab import config_from_args, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_colab.py", "--preset", "paper", "--samples-per-epoch", "12345"],
    )

    args = parse_args()
    cfg = config_from_args(args)

    assert cfg.training.samples_per_epoch == 12345


def test_train_colab_applies_learning_rate(monkeypatch) -> None:
    from scripts.train_colab import config_from_args, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_colab.py", "--preset", "paper", "--learning-rate", "0.002"],
    )

    args = parse_args()
    cfg = config_from_args(args)

    assert cfg.training.learning_rate == pytest.approx(0.002)


def test_train_colab_passes_learning_rate_as_sharded_resume_override(
    monkeypatch,
    tmp_path,
) -> None:
    from scripts import train_colab

    cfg = train_colab.paper_config("low")
    cfg.output_dir = tmp_path / "run"
    cfg.training.learning_rate = 0.002
    captured: dict[str, float | None] = {}

    def fake_train_sharded_model(
        _cfg,
        _dataset_dir,
        *,
        checkpoint_every_epochs,
        resume_from,
        num_workers,
        epoch_callback,
        resume_learning_rate,
    ):
        captured["resume_learning_rate"] = resume_learning_rate
        return SimpleNamespace(model=object(), history={"step": [], "train_loss": [], "val_loss": [], "lr": []})

    args = SimpleNamespace(
        generate_shards=False,
        checkpoint_every_epochs=1,
        resume="latest",
        learning_rate=0.002,
        num_workers=4,
        diagnostic_samples=64,
        threshold_mode="fixed",
        disable_diagnostic_morphology=True,
        no_final_eval=True,
        full_test_eval=False,
        max_eval_samples=64,
    )
    monkeypatch.setattr(train_colab, "resolve_dataset_dir", lambda _cfg, _args: tmp_path / "dataset")
    monkeypatch.setattr(train_colab, "train_sharded_model", fake_train_sharded_model)
    monkeypatch.setattr(train_colab, "plot_loss", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_colab, "sharded_sample_views", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_colab, "diagnose_splits", lambda *_args, **_kwargs: {"val": {"threshold_sweep": []}})
    monkeypatch.setattr(train_colab, "save_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_colab, "materialize_sharded_samples", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(train_colab, "make_example_plots_for_sample", lambda *_args, **_kwargs: None)

    train_colab.run_sharded(cfg, args, thresholds=[0.5])

    assert captured["resume_learning_rate"] == pytest.approx(0.002)


def test_train_colab_sets_writable_matplotlib_config(monkeypatch) -> None:
    sys.modules.pop("scripts.train_colab", None)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    importlib.import_module("scripts.train_colab")

    assert os.environ["MPLCONFIGDIR"].endswith("sali-matplotlib")
    assert os.environ["XDG_CACHE_HOME"].endswith("sali-cache")


def test_colab_uploaded_runner_rejects_unsafe_archive_member(tmp_path) -> None:
    from scripts.run_colab_uploaded import extract_archive

    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"bad"
        info = tarfile.TarInfo("../bad.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe archive member"):
        extract_archive(archive, tmp_path / "extract")
