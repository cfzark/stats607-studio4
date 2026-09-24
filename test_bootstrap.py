"""Student B's tests for bootstrap_sample.

After merging implementations into bootstrap.py, change the import below
from studio04 to bootstrap.
"""

import numpy as np
import pytest

from studio04 import bootstrap_sample


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
