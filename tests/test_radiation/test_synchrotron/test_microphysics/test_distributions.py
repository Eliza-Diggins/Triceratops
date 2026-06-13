"""
Tests for the electron distribution callable factories.

These distributions serve as numerical ground truth in the normalization and emissivity
tests, so they are validated first and independently here.

Covers per distribution (PL, BPL, MJD):
  - Normalization: int N(gamma) d(gamma) equals the supplied norm
  - Point values: N(gamma) matches the analytic formula at specific gamma
  - Compact support: N = 0 outside [gamma_min, gamma_max] (PL and BPL only)
  - Continuity at the break (BPL only)
  - Unit flag: with_units=True -> astropy.Quantity; with_units=False -> ndarray
"""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad

from trilobite.radiation.synchrotron.microphysics import (
    _opt_BPL_moment,
    _opt_PL_moment,
    get_BPL_distribution,
    get_MJD_distribution,
    get_PL_distribution,
)

RTOL_QUADRATURE = 1e-6
RTOL_EXACT = 1e-10


# ---------------------------------------------------------------------------
# Power-law distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,gamma_min,gamma_max,N0",
    [
        (2.5, 10.0, 1e6, 1.0),
        (3.0, 1.0, 1e5, 1e3),
    ],
)
def test_PL_distribution_normalization(p, gamma_min, gamma_max, N0):
    N = get_PL_distribution(p, norm=N0, gamma_min=gamma_min, gamma_max=gamma_max, with_units=False)
    result, _ = quad(N, gamma_min, gamma_max, epsrel=1e-8, limit=200)
    expected = N0 * _opt_PL_moment(p, gamma_min, gamma_max, order=0)
    assert np.isclose(result, expected, rtol=RTOL_QUADRATURE)


@pytest.mark.parametrize(
    "p,gamma_min,gamma_max,N0,gamma_test",
    [
        (3.0, 1.0, 1e6, 1.0, 100.0),
        (2.5, 10.0, 1e5, 1e2, 500.0),
    ],
)
def test_PL_distribution_single_point(p, gamma_min, gamma_max, N0, gamma_test):
    N = get_PL_distribution(p, norm=N0, gamma_min=gamma_min, gamma_max=gamma_max, with_units=False)
    assert np.isclose(N(gamma_test), N0 * gamma_test ** (-p), rtol=RTOL_EXACT)


def test_PL_distribution_compact_support():
    N = get_PL_distribution(3.0, norm=1.0, gamma_min=10.0, gamma_max=1e4, with_units=False)
    assert N(9.99) == 0.0
    assert N(1e4 + 1.0) == 0.0


def test_PL_distribution_with_units_returns_quantity():
    N = get_PL_distribution(3.0, norm=1.0 * u.cm**-3, gamma_min=1.0, gamma_max=1e6)
    result = N(100.0)
    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.cm**-3)


def test_PL_distribution_without_units_returns_ndarray():
    N = get_PL_distribution(3.0, norm=1.0, gamma_min=1.0, gamma_max=1e6, with_units=False)
    result = N(np.array([10.0, 100.0]))
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Broken power-law distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a1,a2,gamma_b,gamma_min,gamma_max,N0",
    [
        (-2.5, -3.5, 1e3, 1.0, 1e6, 1.0),
        (-2.0, -3.0, 1e4, 10.0, 1e7, 1e2),
    ],
)
def test_BPL_distribution_normalization(a1, a2, gamma_b, gamma_min, gamma_max, N0):
    N = get_BPL_distribution(a1, a2, gamma_b, norm=N0, gamma_min=gamma_min, gamma_max=gamma_max, with_units=False)
    result, _ = quad(N, gamma_min, gamma_max, epsrel=1e-8, limit=500, points=[gamma_b])
    expected = N0 * gamma_b * _opt_BPL_moment(a1, a2, gamma_min / gamma_b, gamma_max / gamma_b, order=0)
    assert np.isclose(result, expected, rtol=RTOL_QUADRATURE)


def test_BPL_distribution_continuity_at_break():
    a1, a2, gamma_b, N0 = -2.5, -3.5, 1e3, 1.0
    N = get_BPL_distribution(a1, a2, gamma_b, norm=N0, with_units=False)
    eps = 1e-6 * gamma_b
    # Both sides of the break approach N0 continuously
    assert np.isclose(N(gamma_b - eps), N0, rtol=1e-4)
    assert np.isclose(N(gamma_b + eps), N0, rtol=1e-4)
    # The break point itself equals N0 exactly
    assert np.isclose(N(gamma_b), N0, rtol=RTOL_EXACT)


def test_BPL_distribution_single_point_below_break():
    a1, a2, gamma_b, N0 = -2.5, -3.5, 1e3, 1.0
    N = get_BPL_distribution(a1, a2, gamma_b, norm=N0, with_units=False)
    gamma_test = 100.0
    assert np.isclose(N(gamma_test), N0 * (gamma_test / gamma_b) ** a1, rtol=RTOL_EXACT)


def test_BPL_distribution_single_point_above_break():
    a1, a2, gamma_b, N0 = -2.5, -3.5, 1e3, 1.0
    N = get_BPL_distribution(a1, a2, gamma_b, norm=N0, with_units=False)
    gamma_test = 1e4
    assert np.isclose(N(gamma_test), N0 * (gamma_test / gamma_b) ** a2, rtol=RTOL_EXACT)


def test_BPL_distribution_compact_support():
    N = get_BPL_distribution(-2.5, -3.5, 1e3, norm=1.0, gamma_min=10.0, gamma_max=1e5, with_units=False)
    assert N(5.0) == 0.0
    assert N(2e5) == 0.0


def test_BPL_distribution_with_units_returns_quantity():
    N = get_BPL_distribution(-2.5, -3.5, 1e3, norm=1.0 * u.cm**-3)
    result = N(1000.0)
    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.cm**-3)


def test_BPL_distribution_without_units_returns_ndarray():
    N = get_BPL_distribution(-2.5, -3.5, 1e3, norm=1.0, with_units=False)
    result = N(np.array([100.0, 1000.0, 1e4]))
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Maxwell-Juttner distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Theta,N_therm",
    [
        (1.0, 1.0),
        (10.0, 1e3),
        (100.0, 1.0),
    ],
)
def test_MJD_distribution_normalization(Theta, N_therm):
    N = get_MJD_distribution(Theta, norm=N_therm, with_units=False)
    upper = 50.0 * Theta
    result, _ = quad(N, 0.0, upper, epsrel=1e-8, limit=500)
    assert np.isclose(result, N_therm, rtol=RTOL_QUADRATURE)


@pytest.mark.parametrize(
    "Theta,N_therm,gamma_test",
    [
        (10.0, 1.0, 30.0),
        (1.0, 1e3, 3.0),
    ],
)
def test_MJD_distribution_single_point(Theta, N_therm, gamma_test):
    N = get_MJD_distribution(Theta, norm=N_therm, with_units=False)
    expected = (N_therm / (2.0 * Theta**3)) * gamma_test**2 * np.exp(-gamma_test / Theta)
    assert np.isclose(N(gamma_test), expected, rtol=RTOL_EXACT)


def test_MJD_distribution_with_units_returns_quantity():
    N = get_MJD_distribution(10.0, norm=1.0 * u.cm**-3)
    result = N(30.0)
    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.cm**-3)


def test_MJD_distribution_without_units_returns_ndarray():
    N = get_MJD_distribution(10.0, norm=1.0, with_units=False)
    result = N(np.array([10.0, 30.0, 100.0]))
    assert isinstance(result, np.ndarray)
