"""Student B's tests for bootstrap_sample.

Student A tests and the shared integration test follow below.
"""

import numpy as np
import pytest

from bootstrap import bootstrap_sample


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
# The Student B section above is preserved unchanged.
from bootstrap import bootstrap_ci, r_squared
from bootstrap import bootstrap_sample as bootstrap_sample_for_integration


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
        stats = bootstrap_sample_for_integration(data, r_squared, n_bootstrap=200)
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
