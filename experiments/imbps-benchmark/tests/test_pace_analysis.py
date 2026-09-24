import unittest

from imbps_bench.pace_analysis import bootstrap_median_ci, fit_empirical_surrogate


class PaceAnalysisTests(unittest.TestCase):
    def test_fit_recovers_known_surrogate(self):
        points = {
            split_k: 10.0 + 20.0 / split_k + 3.0 * split_k
            for split_k in (1, 2, 4, 8, 16)
        }
        fit = fit_empirical_surrogate(points)
        self.assertAlmostEqual(fit.c, 10.0)
        self.assertAlmostEqual(fit.a, 20.0)
        self.assertAlmostEqual(fit.b, 3.0)
        self.assertAlmostEqual(fit.continuous_optimum, (20.0 / 3.0) ** 0.5)

    def test_bootstrap_interval_contains_constant(self):
        low, high = bootstrap_median_ci([7.0] * 10, samples=100, seed=1)
        self.assertEqual(low, 7.0)
        self.assertEqual(high, 7.0)

    def test_fit_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            fit_empirical_surrogate({1: 1.0, 2: 2.0})


if __name__ == "__main__":
    unittest.main()
