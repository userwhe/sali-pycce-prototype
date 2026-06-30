from types import SimpleNamespace

import pytest

from sali_pycce.baseline import baseline_spin_to_detection, convert_baseline_spins, import_cpmg_model


def test_baseline_spin_conversion_uses_cpmg_model_sign_convention():
    spin = SimpleNamespace(a_khz=15.0, b_khz=40.0, rmse=0.02)

    det = baseline_spin_to_detection(spin)

    assert det["az_khz"] == -15.0
    assert det["aperp_khz"] == 40.0
    assert det["source_a_khz"] == 15.0
    assert det["source_b_khz"] == 40.0
    assert det["rmse"] == 0.02


def test_convert_baseline_spins_returns_detection_list():
    spins = [SimpleNamespace(a_khz=1.0, b_khz=2.0), SimpleNamespace(a_khz=-3.0, b_khz=4.0)]

    out = convert_baseline_spins(spins)

    assert out == [
        {"az_khz": -1.0, "aperp_khz": 2.0, "source_a_khz": 1.0, "source_b_khz": 2.0},
        {"az_khz": 3.0, "aperp_khz": 4.0, "source_a_khz": -3.0, "source_b_khz": 4.0},
    ]


def test_import_cpmg_model_missing_path_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="cpmg_model path does not exist"):
        import_cpmg_model(tmp_path / "missing")
