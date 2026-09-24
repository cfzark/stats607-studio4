import numpy as np


def bootstrap_sample(data, compute_stat, n_bootstrap=1000):
    """
    Generate the bootstrap distribution of a statistic

    Parameters
    ----------
    data : array-like
        original sample (for regression: 2D array with columns [x, y])

    compute_stat : callable
        function that computes a univariate statistic from data

    n_bootstrap : int, default 1000
        number of bootstrap replicates to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    ValueError
        If data is empty, n_bootstrap < 1, or data has wrong shape
    TypeError
        If compute_stat is not callable


    Examples
    --------
    >>> bootstrap_sample([7., 7., 7.], np.mean, n_bootstrap=3)
    array([7., 7., 7.])

    """

    if not callable(compute_stat):
        raise TypeError("compute_stat must be callable")
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(
        n_bootstrap, (int, np.integer)
    ):
        raise ValueError("n_bootstrap must be a positive integer")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")

    values = np.asarray(data)
    # Accept 1D samples and 2D tables; the statistic checks its own columns.
    if values.ndim not in (1, 2) or values.size == 0:
        raise ValueError("data must be a nonempty 1D sample or 2D table")

    statistics = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        indices = np.random.choice(len(values), size=len(values), replace=True)
        # Index rows together so regression x/y pairs remain paired.
        statistic = np.asarray(compute_stat(values[indices]))
        if statistic.ndim != 0:
            raise ValueError("compute_stat must return a scalar statistic")
        statistics[i] = statistic
    return statistics


def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate a CI from bootstrap distribution

    Parameters
    ----------
    bootstrap_stats : numpy.ndarray
        bootstrap statistics from bootstrap_sample(...)

    alpha : float, default 0.05
        significance level

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the CI

    Raises
    ------
    ValueError
        If alpha not in (0, 1) or if bootstrap_stats is empty

    Notes
    -----
    Uses the percentile interval with quantiles alpha / 2 and 1 - alpha / 2.
    Statistics must be a finite, one-dimensional array.

    Examples
    --------
    >>> bootstrap_ci(np.arange(101, dtype=float))
    (2.5, 97.5)

    """
    stats = np.asarray(bootstrap_stats, dtype=float)
    if stats.ndim != 1 or stats.size == 0:
        raise ValueError("bootstrap_stats must be a nonempty 1D array")
    if not np.all(np.isfinite(stats)):
        raise ValueError("bootstrap_stats must contain only finite values")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number between 0 and 1")

    lower, upper = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def r_squared(data):
    """
    Calculate R^2 from a linear regression

    Parameters
    ----------
    data : array-like, shape (n, 2)
        Data with columns [x, y]

    Returns
    -------
    float
        R-squared value between 0 and 1

    Raises
    ------
    ValueError
        If data doesn't have exactly 2 columns or < 2 rows

    Notes
    -----
    Fits a simple linear regression with an intercept. Values must be finite.
    For constant y, returns 1.0 by convention (the intercept fits y exactly).
    For constant x and nonconstant y, returns 0.0.

    Examples
    --------
    >>> round(r_squared([[0., 0.], [1., 2.], [2., 2.]]), 2)
    0.75
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("data must have shape (n, 2) with at least two rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values")

    x, y = values[:, 0], values[:, 1]
    if np.all(y == y[0]):
        return 1.0
    if np.all(x == x[0]):
        return 0.0

    # Scaling before centering avoids overflow for large finite observations.
    x_centered = x / np.max(np.abs(x))
    y_centered = y / np.max(np.abs(y))
    x_centered = x_centered - np.mean(x_centered)
    y_centered = y_centered - np.mean(y_centered)
    x_centered = x_centered / np.max(np.abs(x_centered))
    y_centered = y_centered / np.max(np.abs(y_centered))

    # With an intercept
    correlation = np.dot(x_centered, y_centered) / (
        np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    )
    return float(np.clip(correlation ** 2, 0.0, 1.0))

