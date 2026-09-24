# STATS 607 Studio 04: Testing & Exceptions for the Bootstrap

## Files to submit

- `bootstrap.py`: all three functions, combining Student A and B implementations.
- `test_bootstrap.py`: both students' pytest tests, integration, and statistical validation.

If Canvas requests two Python files, upload these two. The studio PDF also asks
for the shared GitHub repository URL. Follow Canvas's submission fields for the
URL and individual/group submission setting. For a private repository, ensure
`jake-soloff` has access.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -v --doctest-modules
```

Dependencies can also be installed with `python -m pip install numpy scipy pytest`
if only the two submitted Python files are available.

Validation: 56 checks pass (53 pytest cases and 3 docstring examples).

## Responsibilities

- Student A: bootstrap_sample implementation; tests for bootstrap_ci and r_squared.
- Student B: bootstrap_ci and r_squared implementations; bootstrap_sample tests.
- Shared: integration test, running/debugging tests, Bonus validation, and GitHub submission.

## Statistical conventions and Bonus

The CI is the percentile interval. R² uses simple linear regression with an
intercept. As documented in the implementation, constant y returns 1 and constant
x with nonconstant y returns 0. The tests verify these conventions.

For n independent Gaussian pairs with population correlation zero,
R² follows Beta(1/2, (n-2)/2). The first Bonus test checks simulated R² values
against this distribution. The second removes sample linear association from a
large nearly Gaussian dataset before running the pairs bootstrap, then checks its
approximate agreement with the same reference. Ordinary pairs resampling of an
uncentered sample retains its accidental correlation; its conditional law is not
exactly the null Beta distribution. These tests do not establish percentile-CI
coverage at the R²=0 boundary.

Both statistical tests use fixed seeds and an empirical-CDF tolerance based on
the Dvoretzky-Kiefer-Wolfowitz bound; the bootstrap test includes an explicitly
stated approximation allowance. Random state is restored afterward.

Reference: [SciPy Pearson correlation null distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).
