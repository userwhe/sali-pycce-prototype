from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
import tarfile

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
