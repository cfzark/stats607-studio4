"""Student A's tests, plus a proposed shared integration test.

Team assumptions: percentile CI; ordinary least squares with an intercept.
Discuss these with Student B before adopting this test suite.
"""

import unittest
import numpy as np

from bootstrap import bootstrap_ci, bootstrap_sample, r_squared


class TestBootstrapCI(unittest.TestCase):
    def test_default_95_percent_interval(self):
        stats = np.arange(101, dtype=float)
        result = bootstrap_ci(stats)
        self.assertIsInstance(result, tuple)
        np.testing.assert_allclose(result, (2.5, 97.5))

    def test_custom_alpha_and_unsorted_input(self):
        np.testing.assert_allclose(
            bootstrap_ci(np.array([4., 0., 3., 1., 2.]), alpha=0.5),
            (1., 3.),
        )

    def test_constant_and_singleton(self):
        for stats in (np.array([7.]), np.full(10, 7.)):
            with self.subTest(stats=stats):
                np.testing.assert_allclose(bootstrap_ci(stats), (7., 7.))

    def test_invalid_alpha(self):
        for alpha in (-0.1, 0., 1., 1.1, np.nan):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    bootstrap_ci(np.array([1., 2.]), alpha=alpha)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            bootstrap_ci(np.array([]))


class TestRSquared(unittest.TestCase):
    def test_perfect_positive_and_negative_relationship(self):
        x = np.arange(5, dtype=float)
        for slope in (2., -3.):
            with self.subTest(slope=slope):
                self.assertAlmostEqual(
                    r_squared(np.column_stack((x, 5 + slope * x))), 1.
                )

    def test_known_nonperfect_fit(self):
        # OLS with intercept: fitted y = x + 1/3; SSE=2/3, SST=8/3.
        result = r_squared([[0., 0.], [1., 2.], [2., 2.]])
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 0.75)

    def test_zero_linear_association(self):
        self.assertAlmostEqual(r_squared([[-1., 1.], [0., 0.], [1., 1.]]), 0.)

    def test_two_distinct_points(self):
        self.assertAlmostEqual(r_squared([[0., 2.], [1., 4.]]), 1.)

    def test_invalid_shapes(self):
        for data in ([], [1., 2.], [[1., 2.]], [[1.], [2.]],
                     [[1., 2., 3.], [4., 5., 6.]], np.zeros((2, 2, 2))):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    r_squared(data)


class TestIntegration(unittest.TestCase):
    def test_bootstrap_r_squared_interval(self):
        # Reproducible continuous data avoids degenerate tiny samples.
        state = np.random.get_state()
        try:
            np.random.seed(607)
            rng = np.random.default_rng(607)
            x = rng.normal(size=100)
            data = np.column_stack((x, 1 + 2 * x + rng.normal(size=100)))
            stats = bootstrap_sample(data, r_squared, n_bootstrap=200)
            self.assertEqual(stats.shape, (200,))
            self.assertTrue(np.all(np.isfinite(stats)))
            self.assertTrue(np.all((0 <= stats) & (stats <= 1)))
            lower, upper = bootstrap_ci(stats)
            self.assertTrue(0 <= lower <= upper <= 1)
            np.testing.assert_allclose(
                (lower, upper), np.quantile(stats, [0.025, 0.975])
            )
        finally:
            np.random.set_state(state)


if __name__ == '__main__':
    unittest.main()
