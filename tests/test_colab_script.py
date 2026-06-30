import importlib.util
from pathlib import Path


def test_colab_train_script_is_importable():
    path = Path("scripts/colab_train.py")
    spec = importlib.util.spec_from_file_location("colab_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "build_parser")
    assert hasattr(module, "main")
