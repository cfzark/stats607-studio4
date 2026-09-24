"""Student B's tests for bootstrap_sample.

Student A tests and the shared integration test follow below.
"""

import numpy as np
import pytest

from scipy.stats import beta

from bootstrap import bootstrap_ci, bootstrap_sample, r_squared


def test_returns_one_statistic_per_replicate():
    data = np.array([1.0, 2.0, 4.0, 8.0])

    result = bootstrap_sample(data, np.mean, n_bootstrap=25)

    assert isinstance(result, np.ndarray)
    assert result.shape == (25,)
    assert np.all(np.isfinite(result))
    assert np.all((result >= data.min()) & (result <= data.max()))


def test_default_number_of_replicates():
    result = bootstrap_sample(np.ones(4), np.mean)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1000,)
    np.testing.assert_array_equal(result, np.ones(1000))


def test_constant_data_has_constant_bootstrap_mean():
    result = bootstrap_sample(np.full(5, 7.0), np.mean, n_bootstrap=20)

    np.testing.assert_array_equal(result, np.full(20, 7.0))


def test_single_observation():
    result = bootstrap_sample(np.array([3.0]), np.mean, n_bootstrap=5)

    np.testing.assert_array_equal(result, np.full(5, 3.0))


def test_one_replicate():
    result = bootstrap_sample(np.ones(4), np.mean, n_bootstrap=1)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    np.testing.assert_array_equal(result, np.array([1.0]))


def test_accepts_list_and_uses_supplied_statistic():
    # A sum of four ones is 4, whereas their mean would be 1.
    result = bootstrap_sample([1.0, 1.0, 1.0, 1.0], np.sum, n_bootstrap=6)

    np.testing.assert_array_equal(result, np.full(6, 4.0))


def test_samples_have_original_size_and_values():
    data = np.array([2.0, 5.0, 9.0, 12.0])
    samples = []

    def record_mean(sample):
        samples.append(np.array(sample, copy=True))
        return np.mean(sample)

    result = bootstrap_sample(data, record_mean, n_bootstrap=12)

    assert len(samples) == 12
    for sample in samples:
        assert sample.shape == data.shape
        assert np.all(np.isin(sample, data))
    np.testing.assert_allclose(result, [np.mean(sample) for sample in samples])


def test_regression_rows_stay_paired_and_input_is_unchanged():
    data = np.array([[1.0, 10.0], [2.0, 40.0], [3.0, 90.0]])
    original = data.copy()
    samples = []

    def record_y_mean(sample):
        samples.append(np.array(sample, copy=True))
        return np.mean(sample[:, 1])

    result = bootstrap_sample(data, record_y_mean, n_bootstrap=15)

    assert len(samples) == 15
    for sample in samples:
        assert sample.shape == original.shape
        for row in sample:
            assert np.any(np.all(original == row, axis=1))
    np.testing.assert_allclose(result, [np.mean(s[:, 1]) for s in samples])
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("data", [[], np.array([]), np.empty((0, 2))])
def test_empty_data_raises_value_error(data):
    with pytest.raises(ValueError):
        bootstrap_sample(data, np.mean, n_bootstrap=5)


@pytest.mark.parametrize("n_bootstrap", [0, -1, -10])
def test_nonpositive_replicates_raise_value_error(n_bootstrap):
    with pytest.raises(ValueError):
        bootstrap_sample([1.0, 2.0], np.mean, n_bootstrap=n_bootstrap)


@pytest.mark.parametrize("compute_stat", [None, 42, "mean"])
def test_noncallable_statistic_raises_type_error(compute_stat):
    with pytest.raises(TypeError):
        bootstrap_sample([1.0, 2.0], compute_stat, n_bootstrap=5)


@pytest.mark.parametrize("data", [np.array(1.0), np.ones((2, 2, 2))])
def test_invalid_dimensions_raise_value_error(data):
    with pytest.raises(ValueError):
        bootstrap_sample(data, np.mean, n_bootstrap=5)


# Student A: tests for bootstrap_ci and r_squared, plus integration.


@pytest.mark.parametrize(
    "stats, alpha, expected",
    [
        (np.arange(101, dtype=float), 0.05, (2.5, 97.5)),
        (np.array([4., 0., 3., 1., 2.]), 0.5, (1., 3.)),
        (np.array([7.]), 0.05, (7., 7.)),
        (np.full(10, 7.), 0.05, (7., 7.)),
    ],
)
def test_ci_percentile_interval(stats, alpha, expected):
    result = bootstrap_ci(stats, alpha=alpha)
    assert isinstance(result, tuple)
    assert result == pytest.approx(expected)


def test_ci_default_is_95_percent_interval():
    assert bootstrap_ci(np.arange(101, dtype=float)) == pytest.approx((2.5, 97.5))


@pytest.mark.parametrize("alpha", [-0.1, 0., 1., 1.1, np.nan])
def test_ci_invalid_alpha_raises_value_error(alpha):
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([1., 2.]), alpha=alpha)


def test_ci_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


@pytest.mark.parametrize("slope", [2., -3.])
def test_r_squared_perfect_relationship(slope):
    x = np.arange(5, dtype=float)
    assert r_squared(np.column_stack((x, 5 + slope * x))) == pytest.approx(1.)


def test_r_squared_known_nonperfect_fit():
    # OLS with intercept: fitted y = x + 1/3; SSE=2/3, SST=8/3.
    result = r_squared([[0., 0.], [1., 2.], [2., 2.]])
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(0.75)


def test_r_squared_zero_linear_association():
    assert r_squared([[-1., 1.], [0., 0.], [1., 1.]]) == pytest.approx(0.)


def test_r_squared_two_distinct_points():
    assert r_squared([[0., 2.], [1., 4.]]) == pytest.approx(1.)


@pytest.mark.parametrize(
    "data",
    [[], [1., 2.], [[1., 2.]], [[1.], [2.]],
     [[1., 2., 3.], [4., 5., 6.]], np.zeros((2, 2, 2))],
)
def test_r_squared_invalid_shapes_raise_value_error(data):
    with pytest.raises(ValueError):
        r_squared(data)


def test_integration_bootstrap_r_squared_interval():
    # Restore random state so this test does not affect other tests.
    state = np.random.get_state()
    try:
        np.random.seed(607)
        rng = np.random.default_rng(607)
        x = rng.normal(size=100)
        data = np.column_stack((x, 1 + 2 * x + rng.normal(size=100)))
        stats = bootstrap_sample(data, r_squared, n_bootstrap=200)
        assert stats.shape == (200,)
        assert np.all(np.isfinite(stats))
        assert np.all((0 <= stats) & (stats <= 1))
        lower, upper = bootstrap_ci(stats)
        assert 0 <= lower <= upper <= 1
        np.testing.assert_allclose(
            (lower, upper), np.quantile(stats, [0.025, 0.975])
        )
    finally:
        np.random.set_state(state)


# Shared checks and statistical validation (Bonus).
@pytest.mark.parametrize("stats", [[np.nan], [np.inf], [[1., 2.]]])
def test_ci_rejects_nonfinite_or_nonvector_statistics(stats):
    with pytest.raises(ValueError):
        bootstrap_ci(stats)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_r_squared_rejects_nonfinite_data(bad):
    with pytest.raises(ValueError):
        r_squared([[0., 1.], [1., bad]])


@pytest.mark.parametrize(
    "data, expected",
    [([[0., 4.], [1., 4.]], 1.), ([[2., 0.], [2., 1.]], 0.)],
)
def test_r_squared_documented_constant_conventions(data, expected):
    assert r_squared(data) == pytest.approx(expected)


def test_bootstrap_uses_replacement(monkeypatch):
    def repeated_indices(a, size, replace):
        assert a == 3
        assert size == 3
        assert replace is True
        return np.array([2, 2, 0])

    monkeypatch.setattr(np.random, "choice", repeated_indices)
    result = bootstrap_sample([1., 2., 9.], np.mean, n_bootstrap=2)
    np.testing.assert_allclose(result, [19 / 3, 19 / 3])


def test_r_squared_normal_null_distribution():
    """Under independent normals, R² ~ Beta(1/2, (n-2)/2).

    This follows by squaring the null Pearson correlation distribution:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    The empirical CDF uses a DKW bound with failure probability 1e-6.
    """
    rng = np.random.default_rng(60704)
    n, repetitions = 30, 4000
    statistics = np.array([
        r_squared(rng.normal(size=(n, 2))) for _ in range(repetitions)
    ])
    probabilities = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    cutoffs = beta.ppf(probabilities, 0.5, (n - 2) / 2)
    observed = np.mean(statistics[:, None] <= cutoffs, axis=0)
    tolerance = np.sqrt(np.log(2 / 1e-6) / (2 * repetitions))
    np.testing.assert_allclose(observed, probabilities, atol=tolerance, rtol=0)


def test_null_centered_bootstrap_approximately_matches_theory():
    """Check the pairs bootstrap against a Gaussian-null benchmark.

    Remove the sample linear association first: bootstrapping uncentered
    pairs would preserve its accidental correlation, not impose the null.
    For this large, nearly Gaussian empirical sample the Beta reference
    is approximate, not an exact conditional bootstrap law. Allow 0.03
    CDF approximation error in addition to the Monte Carlo DKW bound.
    This does not assert percentile-CI coverage at the R²=0 boundary.
    """
    rng = np.random.default_rng(60705)
    n, repetitions = 2000, 4000
    x, y = rng.normal(size=(2, n))
    x -= x.mean()
    y -= y.mean()
    y -= x * (np.dot(x, y) / np.dot(x, x))
    data = np.column_stack((x, y))
    assert r_squared(data) < 1e-20

    state = np.random.get_state()
    try:
        np.random.seed(60706)
        statistics = bootstrap_sample(data, r_squared, repetitions)
    finally:
        np.random.set_state(state)

    probabilities = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    cutoffs = beta.ppf(probabilities, 0.5, (n - 2) / 2)
    observed = np.mean(statistics[:, None] <= cutoffs, axis=0)
    mc_tolerance = np.sqrt(np.log(2 / 1e-6) / (2 * repetitions))
    np.testing.assert_allclose(
        observed, probabilities, atol=mc_tolerance + 0.03, rtol=0
    )
    assert statistics.mean() == pytest.approx(1 / (n - 1), rel=0.2)
