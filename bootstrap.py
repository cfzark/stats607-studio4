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
    

    Example
    -------
    TBA

    """

    import numpy as np

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
    
    Example
    -------
    TBA
    
    """

    raise NotImplementedError("Student B: implement bootstrap_ci")

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
    """
    raise NotImplementedError("Student B: implement r_squared")
