import unittest
from pathlib import Path

import numpy as np

from direct_fit import (
    GaussianComponent,
    extract_revival_segment,
    fit_spins_from_segment,
    gaussian_component_curves,
    gaussian_to_spin_estimate,
    load_cpmg_csv,
)


DATA_PATH = Path(__file__).with_name("cpmg_data_spinset2_525G_simulated.csv")
B0_GAUSS = 525.0
REVIVAL_INDEX = 6
N_PULSES = 32


class DirectFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tau_us, cls.signal = load_cpmg_csv(DATA_PATH)
        cls.segment_tau, cls.segment_signal = extract_revival_segment(
            cls.tau_us,
            cls.signal,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
        )

    def test_extract_revival_segment_has_expected_bounds(self) -> None:
        self.assertGreater(len(self.segment_tau), 100)
        self.assertAlmostEqual(self.segment_tau[0], 4.45, places=2)
        self.assertAlmostEqual(self.segment_tau[-1], 5.335, places=2)

    def test_direct_fit_on_sample_revival_returns_stable_spins(self) -> None:
        result = fit_spins_from_segment(
            self.segment_tau,
            self.segment_signal,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
            refine=False,
        )

        self.assertEqual(len(result.gaussians), 5)
        self.assertLess(result.gaussian_rmse, 0.02)

        strongest_spin = max(result.initial_spins, key=lambda spin: spin.amplitude)
        self.assertTrue(28.0 <= strongest_spin.a_khz <= 36.0)
        self.assertTrue(40.0 <= strongest_spin.b_khz <= 48.0)
        self.assertAlmostEqual(strongest_spin.peak_tau_us, 4.7507, places=3)

    def test_refinement_reduces_model_error(self) -> None:
        direct_result = fit_spins_from_segment(
            self.segment_tau,
            self.segment_signal,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
            refine=False,
        )
        refined_result = fit_spins_from_segment(
            self.segment_tau,
            self.segment_signal,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
            n_pulses=N_PULSES,
            refine=True,
        )

        self.assertIsNotNone(refined_result.refined_spins)
        self.assertIsNotNone(refined_result.physical_rmse)
        self.assertEqual(len(refined_result.refined_spins), len(direct_result.initial_spins))
        self.assertLess(refined_result.physical_rmse, direct_result.gaussian_rmse)

        strongest_refined = max(refined_result.refined_spins, key=lambda spin: spin.b_khz)
        self.assertTrue(55.0 <= strongest_refined.b_khz <= 75.0)
        self.assertTrue(20.0 <= strongest_refined.a_khz <= 40.0)

    def test_refine_requires_pulse_count(self) -> None:
        with self.assertRaises(ValueError):
            fit_spins_from_segment(
                self.segment_tau,
                self.segment_signal,
                revival_index=REVIVAL_INDEX,
                b0_gauss=B0_GAUSS,
                refine=True,
            )

    def test_exact_component_count_can_be_forced(self) -> None:
        result = fit_spins_from_segment(
            self.segment_tau,
            self.segment_signal,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
            n_components=3,
            refine=False,
        )

        self.assertEqual(len(result.gaussians), 3)
        centers = [component.center_us for component in result.gaussians]
        self.assertTrue(any(abs(center - 4.7507) < 0.01 for center in centers))

    def test_invalid_component_request_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_spins_from_segment(
                self.segment_tau,
                self.segment_signal,
                revival_index=REVIVAL_INDEX,
                b0_gauss=B0_GAUSS,
                n_components=0,
                refine=False,
            )

    def test_branch_selection_exposes_alternate_a(self) -> None:
        gaussian = GaussianComponent(
            amplitude=1.0,
            center_us=4.750736028832706,
            sigma_us=0.00869318929271718,
            fwhm_us=2.0 * np.sqrt(2.0 * np.log(2.0)) * 0.00869318929271718,
            prominence=1.0,
        )

        weak = gaussian_to_spin_estimate(
            gaussian,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
            branch="weak_coupling",
        )
        positive = gaussian_to_spin_estimate(
            gaussian,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
            branch="positive",
        )
        negative = gaussian_to_spin_estimate(
            gaussian,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
            branch="negative",
        )

        self.assertAlmostEqual(weak.a_khz, positive.a_khz, places=6)
        self.assertAlmostEqual(weak.alternate_a_khz, negative.a_khz, places=6)
        self.assertGreater(positive.a_khz, -100.0)
        self.assertLess(negative.a_khz, -1000.0)

    def test_gaussian_component_curves_sum_to_reconstruction(self) -> None:
        result = fit_spins_from_segment(
            self.segment_tau,
            self.segment_signal,
            revival_index=REVIVAL_INDEX,
            b0_gauss=B0_GAUSS,
            refine=False,
        )

        background, dip_curves, signal_curves = gaussian_component_curves(
            self.segment_tau,
            baseline=result.baseline,
            slope=result.slope,
            gaussians=result.gaussians,
        )

        dip_sum = np.sum(dip_curves, axis=0)
        signal_sum = background - dip_sum
        self.assertTrue(np.allclose(signal_sum, result.gaussian_reconstruction))

        for curve in signal_curves:
            self.assertEqual(curve.shape, self.segment_tau.shape)


if __name__ == "__main__":
    unittest.main()
