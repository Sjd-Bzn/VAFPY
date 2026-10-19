"""Statistical error of the mean from the integrated autocorrelation time (exactly known test cases)."""
import os
import sys

import numpy as np
import pytest
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(__file__))
import functions as new  # noqa: E402


def _ar1(n, tau, rng):
    r = np.exp(-1.0 / tau)
    e = rng.standard_normal(n + 5000) * np.sqrt(1 - r * r)
    return lfilter([1.0], [1.0, -r], e)[5000:]


def _exact_error(n, tau):
    k = np.arange(1, n)
    return np.sqrt((1.0 + 2.0 * np.sum((1 - k / n) * np.exp(-k / tau))) / n)


def test_autocorr_error_white_noise():
    rng = np.random.default_rng(3)
    n = 20000
    est = [new.autocorr_error(rng.standard_normal(n))[0] for _ in range(40)]
    assert np.median(est) == pytest.approx(1.0 / np.sqrt(n), rel=0.1)


def test_autocorr_error_correlated_series():
    rng = np.random.default_rng(5)
    n, tau = 20000, 30.0
    res = [new.autocorr_error(_ar1(n, tau, rng)) for _ in range(60)]
    assert np.median([r[0] for r in res]) == pytest.approx(_exact_error(n, tau), rel=0.1)
    assert np.median([r[1] for r in res]) == pytest.approx(tau, rel=0.15)
    assert all(r[2] for r in res)


def test_autocorr_error_series_much_slower_than_its_length_is_flagged():
    x = _ar1(200, 5000.0, np.random.default_rng(1))
    assert new.autocorr_error(x)[2] is False
