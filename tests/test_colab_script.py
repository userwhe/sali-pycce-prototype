import importlib.util
from pathlib import Path


def load_colab_train_module():
    path = Path("scripts/colab_train.py")
    spec = importlib.util.spec_from_file_location("colab_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_colab_train_script_is_importable():
    module = load_colab_train_module()
    assert hasattr(module, "build_parser")
    assert hasattr(module, "main")


def test_colab_train_resolves_project_root_from_env(tmp_path, monkeypatch):
    project = tmp_path / "project"
    package = project / "src" / "sali_pycce"
    package.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    monkeypatch.setenv("SALI_PYCCE_PROJECT_ROOT", str(project))
    module = load_colab_train_module()

    assert module.resolve_project_root() == project.resolve()
