from sali_pycce.heatmap import HeatmapSpec, decode_heatmap, make_heatmap
from sali_pycce.physics import SpinParams


def test_heatmap_roundtrip_single_blob():
    spec = HeatmapSpec(height=32, width=64)
    spin = SpinParams(az_khz=10.0, aperp_khz=40.0)
    heatmap = make_heatmap([spin], spec)
    det = decode_heatmap(heatmap, spec, threshold=0.2, min_area=1, morph=False)
    assert len(det) == 1
    assert abs(det[0]["az_khz"] - spin.az_khz) < 3.0
    assert abs(det[0]["aperp_khz"] - spin.aperp_khz) < 3.0


def test_research_heatmap_roundtrip_has_about_two_khz_resolution():
    spec = HeatmapSpec(
        height=128,
        width=256,
        az_range=(-250.0, 250.0),
        aperp_range=(2.0, 250.0),
        sigma_px=1.25,
        patch_radius=3,
    )
    spin = SpinParams(az_khz=-123.0, aperp_khz=211.0)
    heatmap = make_heatmap([spin], spec)
    det = decode_heatmap(heatmap, spec, threshold=0.2, min_area=1, morph=False)

    assert len(det) == 1
    assert abs(det[0]["az_khz"] - spin.az_khz) < 2.5
    assert abs(det[0]["aperp_khz"] - spin.aperp_khz) < 2.5
