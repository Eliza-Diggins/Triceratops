"""
Tests for analytic moment calculations of the three electron distribution families.

Background
----------
All normalizations and emissivities in the microphysics package reduce to integrals of the form

    M^(l) = int gamma^l N(gamma) d(gamma)

called the *order-l moment* of the distribution.  Three families are supported:

* **PL** (power-law): N(gamma) = N0 * gamma^{-p}
* **BPL** (broken power-law): N(gamma) ~ (gamma/gamma_b)^{a1} below the break, (gamma/gamma_b)^{a2} above
* **MJD** (ultra-relativistic Maxwell-Juttner): N(gamma) ~ gamma**2 * exp(-gamma/Theta)

The private ``_opt_*`` backends compute these moments analytically.  Each has a special
logarithmic branch that fires when the standard power formula would produce 0/0 (i.e.
when the exponent of the integrand is exactly zero).  That branch is the primary source of
bugs and is explicitly exercised here alongside the regular branch.

Test philosophy
---------------
* The private backends are the correctness anchors.  They are verified against independent
  ``scipy.quad`` quadrature at ``RTOL_QUADRATURE = 1e-6``.
* The public wrappers (``compute_PL_moment`` etc.) add input validation on
  top of the same backend; they are tested only for their API contracts -- error paths and
  agreement with the backend -- not re-validated against quadrature.
* Vectorization (array inputs broadcasting to array output) is a regression risk because
  the backends use NumPy masking instead of Python conditionals; one smoke test per family.

Tolerance tiers
---------------
* ``RTOL_QUADRATURE = 1e-6`` -- analytic formula vs. ``scipy.quad`` at ``epsrel=1e-10``.
  Tighter than the previous 1e-3 used in ``test_microphysics.py``; the analytic formulas
  are exact so the gap between them and quadrature is pure quadrature error.
* ``RTOL_EXACT = 1e-10`` -- properties that hold exactly by construction (vectorized output
  must match a scalar loop to floating-point precision).
"""

import numpy as np
import pytest
from scipy.integrate import quad

from trilobite.radiation.synchrotron.microphysics import (
    _opt_BPL_moment,
    _opt_MJD_moment,
    _opt_PL_moment,
    compute_BPL_moment,
    compute_MJD_mean_gamma,
    compute_PL_moment,
)

# Analytic formula vs. quadrature: limited by quadrature error, not the formula.
RTOL_QUADRATURE = 1e-6
# Exact-by-construction checks (vectorization, public-vs-private agreement).
RTOL_EXACT = 1e-10


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------
# These implement the same integrals as the _opt_* backends via scipy.quad so
# they can serve as independent references.  They use epsrel=1e-10 to stay well
# below the RTOL_QUADRATURE threshold.


def _numeric_PL_moment(p, gamma_min, gamma_max, order):
    """Compute int_{gamma_min}^{gamma_max} gamma^{order - p} d(gamma) via adaptive quadrature."""
    result, _ = quad(
        lambda g: g ** (order - p),
        gamma_min,
        gamma_max,
        epsabs=0,
        epsrel=1e-10,
        limit=200,
    )
    return result


def _numeric_BPL_moment(a1, a2, x_min, x_max, order):
    """
    Compute the order-th moment of a unit-break BPL distribution via quadrature.

    Evaluates int_{x_min}^{x_max} x^order N(x) dx where N(x) = x^{a1} for x < 1
    and N(x) = x^{a2} for x >= 1.  The domain is split at x=1 so each piece is
    smooth and the integrator can achieve high accuracy.
    """
    result = 0.0
    if x_min < 1.0:
        upper = min(1.0, x_max)
        result += quad(lambda x: x ** (order + a1), x_min, upper, epsrel=1e-10)[0]
    if x_max > 1.0:
        lower = max(1.0, x_min)
        result += quad(lambda x: x ** (order + a2), lower, x_max, epsrel=1e-10)[0]
    return result


def _numeric_MJD_moment(Theta, order):
    """
    Compute the MJD shape factor S^(order)(Theta) via quadrature.

    For N(gamma) = N_therm / (2*Theta**3) * gamma**2 * exp(-gamma/Theta), the shape
    factor is defined by M^(order) = N_therm * S^(order)(Theta), i.e.

        S^(order)(Theta) = 1/(2*Theta**3) * int_0^inf gamma^{order+2} exp(-gamma/Theta) d(gamma).

    The integrand decays as exp(-gamma/Theta), so integrating to 1000*Theta captures all
    significant probability mass.
    """
    upper = 1000.0 * Theta
    result, _ = quad(
        lambda g: g ** (order + 2) * np.exp(-g / Theta),
        0.0,
        upper,
        epsabs=0,
        epsrel=1e-10,
        limit=500,
    )
    return result / (2.0 * Theta**3)


# ---------------------------------------------------------------------------
# PL moment tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,gamma_min,gamma_max,order",
    [
        # --- Regular branch (exponent = order+1-p != 0) ---
        # p=3 with orders 0,1,2 covers the three most-used moments at a
        # typical spectral index.
        (3.0, 1.0, 1e6, 0),
        (3.0, 1.0, 1e6, 1),
        (3.0, 1.0, 1e6, 2),
        # Softer index with non-unity gamma_min exercises the lower-limit term.
        (2.5, 10.0, 1e5, 0),
        (2.5, 10.0, 1e5, 1),
        # p=2 with large gamma_max tests near-divergent integrals (order=0, p=2
        # gives exponent -1, which is finite but the upper limit dominates).
        (2.0, 100.0, 1e8, 0),
        # --- Logarithmic branch (exponent = order+1-p = 0 -> result = ln(gamma_max/gamma_min)) ---
        # The implementation switches to log(x_max/x_min) when |exponent| < 1e-15.
        # Each row satisfies p = order+1 so exponent is exactly zero.
        (1.0, 1.0, 1e6, 0),  # order 0, p=1
        (2.0, 10.0, 1e4, 1),  # order 1, p=2
        (3.0, 1.0, 1e8, 2),  # order 2, p=3
    ],
)
def test_PL_moment_matches_quadrature(p, gamma_min, gamma_max, order):
    """Analytic PL moment agrees with independent quadrature to RTOL_QUADRATURE.

    The parametrize cases cover both the standard power formula and the special
    logarithmic branch (triggered when p = order + 1).  A failure on a log-branch
    case while the regular cases pass indicates a bug in the branch-switch logic.
    """
    analytic = _opt_PL_moment(p, gamma_min, gamma_max, order=order)
    numeric = _numeric_PL_moment(p, gamma_min, gamma_max, order)
    assert np.isclose(analytic, numeric, rtol=RTOL_QUADRATURE)


def test_PL_moment_vectorization():
    """Array input to the PL backend broadcasts correctly over the p axis.

    The backend uses NumPy masking (not Python if/else) to handle the regular
    and logarithmic branches simultaneously for array inputs.  This test
    catches regressions where masking is applied to the wrong elements or the
    scalar-return reshape loses values.
    """
    p_arr = np.array([2.0, 2.5, 3.0, 3.5])
    gamma_min, gamma_max, order = 10.0, 1e6, 1
    result_vec = _opt_PL_moment(p_arr, gamma_min, gamma_max, order=order)
    result_scalar = np.array([_opt_PL_moment(p, gamma_min, gamma_max, order=order) for p in p_arr])
    np.testing.assert_allclose(result_vec, result_scalar, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# BPL moment tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a1,a2,x_min,x_max,order",
    [
        # --- Regular branch (k1 != 0 and k2 != 0, where k_i = order+1+a_i) ---
        (-2.0, -3.0, 1e-3, 1e3, 0),
        (-2.0, -3.0, 1e-3, 1e3, 1),
        (-1.5, -2.5, 1e-2, 1e2, 2),
        (-2.5, -3.5, 1e-1, 1e4, 0),
        (-2.5, -3.5, 1e-1, 1e4, 1),
        # --- Logarithmic k1=0 branch (a1 = -(order+1) -> k1 = 0) ---
        # The low-energy piece switches to -ln(x_min).
        (-1.0, -3.0, 1e-2, 1e2, 0),  # k1 = 0+1+(-1) = 0
        (-2.0, -3.0, 1e-2, 1e2, 1),  # k1 = 1+1+(-2) = 0
        # --- Logarithmic k2=0 branch (a2 = -(order+1) -> k2 = 0) ---
        # The high-energy piece switches to ln(x_max).
        (-2.0, -1.0, 1e-2, 1e2, 0),  # k2 = 0+1+(-1) = 0
    ],
)
def test_BPL_moment_matches_quadrature(a1, a2, x_min, x_max, order):
    """Analytic BPL moment agrees with independent quadrature to RTOL_QUADRATURE.

    The BPL implementation has two independent logarithmic branches -- one for
    the low-energy arm (k1=0) and one for the high-energy arm (k2=0).  Each is
    tested explicitly; a failure on only the k1 or k2 cases isolates which arm
    has a bug.
    """
    analytic = _opt_BPL_moment(a1, a2, x_min, x_max, order=order)
    numeric = _numeric_BPL_moment(a1, a2, x_min, x_max, order)
    assert np.isclose(analytic, numeric, rtol=RTOL_QUADRATURE)


def test_BPL_moment_vectorization():
    """Array input to the BPL backend broadcasts correctly over the a1 axis."""
    a1_arr = np.array([-1.5, -2.0, -2.5])
    a2, x_min, x_max, order = -3.5, 0.01, 100.0, 1
    result_vec = _opt_BPL_moment(a1_arr, a2, x_min, x_max, order=order)
    result_scalar = np.array([_opt_BPL_moment(a1, a2, x_min, x_max, order=order) for a1 in a1_arr])
    np.testing.assert_allclose(result_vec, result_scalar, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# MJD moment tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Theta,order",
    [
        # Sub-relativistic (Theta=0.1), mildly relativistic (Theta=1), and
        # ultra-relativistic (Theta=10, 100) temperatures covering orders 0-2.
        # The closed-form result is S^(l)(Theta) = (l+2)!/2 * Theta^l.
        (0.1, 0),
        (0.1, 1),
        (0.1, 2),
        (1.0, 0),
        (1.0, 1),
        (10.0, 0),
        (10.0, 2),
        (100.0, 1),
    ],
)
def test_MJD_moment_matches_quadrature(Theta, order):
    """Analytic MJD shape factor agrees with independent quadrature to RTOL_QUADRATURE.

    The shape factor S^(l)(Theta) = (l+2)!/2 * Theta^l is an exact closed form derived
    from the Gamma integral; this test confirms the factorial coefficient and
    the power of Theta are both implemented correctly.
    """
    analytic = _opt_MJD_moment(Theta, order=order)
    numeric = _numeric_MJD_moment(Theta, order)
    assert np.isclose(analytic, numeric, rtol=RTOL_QUADRATURE)


@pytest.mark.parametrize("Theta", [0.1, 1.0, 10.0, 100.0])
def test_MJD_mean_gamma_is_3_Theta(Theta):
    """Mean Lorentz factor of the MJD distribution is exactly 3*Theta.

    For the ultra-relativistic MJD, <gamma> = M^(1)/M^(0) = (3!/2 * Theta) / (2!/2) = 3*Theta.
    This is a closed-form result with no approximation, so RTOL_EXACT is appropriate.
    A failure here while the moment tests pass indicates a bug in how
    ``compute_MJD_mean_gamma`` combines the two moments.
    """
    assert np.isclose(compute_MJD_mean_gamma(Theta), 3.0 * Theta, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# Public API contract tests
# ---------------------------------------------------------------------------


def test_public_PL_moment_raises_on_gamma_min_zero():
    """Public PL moment wrapper rejects gamma_min = 0 before calling the backend.

    The power-law integrand gamma^{order-p} diverges at gamma=0 for any p > order,
    so gamma_min must be strictly positive.  The private backend does not check
    this; the public wrapper is responsible for the guard.
    """
    with pytest.raises(ValueError, match="gamma_min must be strictly positive"):
        compute_PL_moment(p=3.0, gamma_min=0.0, gamma_max=1e6, order=1)


def test_public_PL_moment_raises_on_divergent_integral():
    """Public PL moment wrapper rejects divergent integrals before calling the backend.

    When p < order + 1 (i.e. exponent = order+1-p > 0) the integral diverges
    at gamma_max -> inf.  The private backend would silently return inf or NaN;
    the public wrapper raises ValueError instead.  Here p=1.5, order=2 gives
    exponent = 1.5 > 0, which is divergent.
    """
    with pytest.raises(ValueError):
        compute_PL_moment(p=1.5, gamma_min=1.0, gamma_max=np.inf, order=2)


def test_public_PL_moment_agrees_with_private_backend():
    """Public PL moment wrapper delegates to the private backend for valid inputs.

    Confirms that the public wrapper does not introduce any numerical transformation
    on top of the backend for a well-conditioned case.
    """
    p, gamma_min, gamma_max, order = 3.0, 10.0, 1e6, 1
    public = compute_PL_moment(p=p, gamma_min=gamma_min, gamma_max=gamma_max, order=order)
    private = _opt_PL_moment(p, gamma_min, gamma_max, order=order)
    assert np.isclose(public, private, rtol=RTOL_EXACT)


def test_public_BPL_moment_raises_on_gamma_min_zero():
    """Public BPL moment wrapper rejects gamma_min = 0."""
    with pytest.raises(ValueError, match="gamma_min must be strictly positive"):
        compute_BPL_moment(a1=-2.0, a2=-3.0, gamma_b=1e3, gamma_min=0.0, gamma_max=1e6, order=1)
