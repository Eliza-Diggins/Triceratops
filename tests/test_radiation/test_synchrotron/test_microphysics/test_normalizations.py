"""
Tests for electron distribution normalization closures.

Background
----------
Every synchrotron normalization reduces to an *equipartition condition* of the form

    U_e = (eps_E / eps_B) * B**2 / (8*pi)

where U_e is the electron energy density.  For each distribution family this
condition sets the normalization amplitude N0 (or N_therm):

* **PL**: U_e = m_e c**2 * N0 * M1  where M1 = int_{gamma_min}^{gamma_max} gamma^{1-p} d(gamma)
* **BPL**: U_e = m_e c**2 * N0 * gamma_b**2 * M_tilde1  where M_tilde1 is the rescaled BPL moment
* **MJD**: U_e = N_therm * m_e c**2 * Theta * (6 + 15*Theta) / (4 + 5*Theta)   (Ghisellini fitting formula)

The Ghisellini factor (6+15*Theta)/(4+5*Theta) interpolates between the classical non-relativistic
limit 3/2 (at Theta -> 0, kT << m_e c**2) and the ultra-relativistic limit 3 (at Theta -> inf).

Two independent code paths produce the same normalization:

* **B-field path**: B is given; N0 is set from the equipartition condition.
* **Thermal path**: u_therm is given; B = sqrt(8*pi * eps_B * u_therm) then N0 follows.

Because these paths invoke different helper functions but must agree, they form a
powerful consistency check: if either helper has a sign error or missing factor of 8*pi,
the paths diverge.

There is also a gamma-space <-> energy-space normalization conversion.  Some literature
parameterizes N(E) = K_E * E^{1-p} instead of N(gamma) = N0 * gamma^{-p}.  The conversion
K_E = N0 * (m_e c**2)^{p-1} must be exactly invertible.

Test philosophy
---------------
* Normalization correctness is verified by re-deriving U_e from N0 using the moment
  integral, and checking it equals the target.  This catches coefficient errors in the
  normalization formula without depending on any other subsystem.
* B-field and thermal paths are tested to agree with RTOL_EXACT because they are
  algebraically identical when B is the equipartition field; any discrepancy is a bug.
* The Ghisellini limits (NR and UR) are tested at RTOL_APPROX because they are
  approximate fitting-formula limits, not exact identities.
* The mixed MJD+PL normalization (delta parameter) is tested at the degenerate endpoints
  delta=0 (pure PL) and delta=1 (pure MJD), both of which must exactly reproduce the
  single-component normalizations.

Tolerance tiers
---------------
* ``RTOL_EXACT = 1e-10`` -- algebraic identities that must hold to floating-point precision,
  such as path consistency and norm round-trips.
* ``RTOL_QUADRATURE = 1e-6`` -- used only where a numerical integration is the reference;
  here the normalization tests are purely algebraic so this tier is imported for
  completeness but not applied in the main normalization checks.
* ``RTOL_APPROX = 1e-2`` -- Ghisellini limiting-case tests and UR distribution integration,
  where 1% is the physical accuracy of the fitting formula at the tested temperature.
"""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad

from trilobite.radiation.synchrotron.microphysics import (
    _opt_BPL_moment,
    _opt_BPL_norm_energy_to_gamma,
    _opt_BPL_norm_from_magnetic_field,
    _opt_BPL_norm_from_thermal,
    _opt_BPL_norm_gamma_to_energy,
    _opt_MJD_norm_from_magnetic_field,
    _opt_PL_moment,
    _opt_PL_norm_energy_to_gamma,
    _opt_PL_norm_from_magnetic_field,
    _opt_PL_norm_from_thermal,
    _opt_PL_norm_gamma_to_energy,
    _opt_equipart_magnetic_field,
    _opt_mixed_norm_from_magnetic_field,
    compute_BPL_norm_from_magnetic_field,
    compute_PL_norm_from_magnetic_field,
    compute_equipartition_magnetic_field,
    electron_rest_energy_cgs,
    get_MJD_distribution,
    swap_PL_normalization,
)

RTOL_QUADRATURE = 1e-6
RTOL_EXACT = 1e-10
RTOL_APPROX = 1e-2


# ---------------------------------------------------------------------------
# PL normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,p,eps_B,eps_E,gamma_min,gamma_max",
    [
        # Typical astrophysical spectral index with equal microphysical fractions
        (1.0, 3.0, 0.1, 0.1, 1.0, 1e6),
        # Softer index and unequal fractions; exercises the eps_E/eps_B prefactor
        (0.5, 2.5, 0.01, 0.1, 10.0, 1e7),
        # Strong-B regime with high fractions close to equipartition
        (2.0, 3.5, 0.3, 0.3, 1.0, 1e5),
    ],
)
def test_PL_norm_from_B_gives_correct_energy_density(B, p, eps_B, eps_E, gamma_min, gamma_max):
    """N0 from the B-field path satisfies the equipartition energy density condition.

    After normalizing, the electron energy density U_e = m_e c**2 * N0 * M1 must equal
    (eps_E / eps_B) * B**2 / (8*pi).  M1 is the first moment int gamma^{1-p} d(gamma).

    A failure here indicates a coefficient error in ``_opt_PL_norm_from_magnetic_field``
    -- most commonly a missing factor of 8*pi or a wrong power of B.
    """
    N0 = _opt_PL_norm_from_magnetic_field(B, p, eps_B, eps_E, gamma_min, gamma_max)
    M1 = _opt_PL_moment(p, gamma_min, gamma_max, order=1)
    u_e = electron_rest_energy_cgs * N0 * M1
    u_target = (eps_E / eps_B) * B**2 / (8.0 * np.pi)
    assert np.isclose(u_e, u_target, rtol=RTOL_EXACT)


@pytest.mark.parametrize(
    "u_therm,p,eps_E,gamma_min,gamma_max",
    [
        # Moderate thermal energy density with standard spectral index
        (1e12, 3.0, 0.1, 1.0, 1e6),
        # Lower u_therm and unequal eps_E; exercises the linear u_therm scaling
        (1e10, 2.5, 0.2, 10.0, 1e7),
    ],
)
def test_PL_norm_from_thermal_gives_correct_energy_density(u_therm, p, eps_E, gamma_min, gamma_max):
    """N0 from the thermal path satisfies U_e = eps_E * u_therm.

    Unlike the B-field path, this version takes the total thermal energy as input
    and sets U_e directly to the eps_E fraction.  Checks that the eps_E prefactor is
    applied correctly and that the moment integral is not double-applied.
    """
    N0 = _opt_PL_norm_from_thermal(u_therm, p, eps_E, gamma_min, gamma_max)
    M1 = _opt_PL_moment(p, gamma_min, gamma_max, order=1)
    u_e = electron_rest_energy_cgs * N0 * M1
    assert np.isclose(u_e, eps_E * u_therm, rtol=RTOL_EXACT)


def test_PL_B_and_thermal_paths_agree():
    """The B-field and thermal normalization paths give identical N0 when B = equipartition field.

    At the equipartition field B = sqrt(8*pi * eps_B * u_therm) the two paths reduce to the same
    expression.  Any disagreement at RTOL_EXACT indicates that the two functions implement
    the same formula with different constants, sign errors, or unit factors.
    """
    u_therm, eps_B, eps_E = 1e12, 0.1, 0.1
    p, gamma_min, gamma_max = 3.0, 10.0, 1e6
    B = _opt_equipart_magnetic_field(u_therm, eps_B)
    N0_B = _opt_PL_norm_from_magnetic_field(B, p, eps_B, eps_E, gamma_min, gamma_max)
    N0_therm = _opt_PL_norm_from_thermal(u_therm, p, eps_E, gamma_min, gamma_max)
    assert np.isclose(N0_B, N0_therm, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# BPL normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,a1,a2,gamma_b,eps_B,eps_E,gamma_min,gamma_max",
    [
        # Standard BPL with moderate break at gamma_b=10**3; equal microphysical fractions
        (1.0, -2.5, -3.5, 1e3, 0.1, 0.1, 1.0, 1e6),
        # Shallower slopes and large break; exercises rescaling by gamma_b**2
        (0.5, -2.0, -3.0, 1e4, 0.01, 0.1, 10.0, 1e7),
    ],
)
def test_BPL_norm_from_B_gives_correct_energy_density(B, a1, a2, gamma_b, eps_B, eps_E, gamma_min, gamma_max):
    """N0 from the BPL B-field path satisfies the equipartition energy density condition.

    The BPL normalization is normalized so that the distribution equals N0 at the break
    (gamma = gamma_b), so the energy density involves an extra gamma_b**2 factor:

        U_e = m_e c**2 * N0 * gamma_b**2 * M_tilde1(a1, a2, gamma_min/gamma_b, gamma_max/gamma_b)

    where M_tilde1 is the rescaled BPL moment.  A failure here often traces to the gamma_b**2
    prefactor being absent, squared, or mixed up with the moment scaling.
    """
    N0 = _opt_BPL_norm_from_magnetic_field(B, a1, a2, gamma_b, eps_B, eps_E, gamma_min, gamma_max)
    M1 = gamma_b**2 * _opt_BPL_moment(a1, a2, gamma_min / gamma_b, gamma_max / gamma_b, order=1)
    u_e = electron_rest_energy_cgs * N0 * M1
    u_target = (eps_E / eps_B) * B**2 / (8.0 * np.pi)
    assert np.isclose(u_e, u_target, rtol=RTOL_EXACT)


def test_BPL_B_and_thermal_paths_agree():
    """The BPL B-field and thermal normalization paths give identical N0 at equipartition.

    Same consistency check as the PL case, applied to the BPL family.  The extra gamma_b**2
    factor must cancel when deriving one path from the other; if it doesn't the paths diverge.
    """
    u_therm, eps_B, eps_E = 1e12, 0.1, 0.1
    a1, a2, gamma_b = -2.5, -3.5, 1e3
    gamma_min, gamma_max = 1.0, 1e6
    B = _opt_equipart_magnetic_field(u_therm, eps_B)
    N0_B = _opt_BPL_norm_from_magnetic_field(B, a1, a2, gamma_b, eps_B, eps_E, gamma_min, gamma_max)
    N0_therm = _opt_BPL_norm_from_thermal(u_therm, a1, a2, gamma_b, eps_E, gamma_min, gamma_max)
    assert np.isclose(N0_B, N0_therm, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# MJD normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,Theta,eps_B,eps_E",
    [
        # Mildly relativistic temperature (Theta=10, intermediate between NR and UR)
        (1.0, 10.0, 0.1, 0.1),
        # NR-like temperature (Theta=1) with unequal fractions
        (0.5, 1.0, 0.01, 0.1),
        # UR temperature (Theta=100) with large microphysical fractions
        (2.0, 100.0, 0.3, 0.3),
    ],
)
def test_MJD_norm_from_B_satisfies_equipartition(B, Theta, eps_B, eps_E):
    """N_therm from the MJD B-field path satisfies the Ghisellini equipartition formula.

    The MJD mean energy per particle is approximated by the Ghisellini fitting formula:

        <eps> = m_e c**2 * Theta * (6 + 15*Theta) / (4 + 5*Theta)

    so the normalization condition is N_therm * <eps> = (eps_E / eps_B) * B**2 / (8*pi).

    Tests at Theta=1, 10, 100 span the fitting formula from near-NR to deep-UR, verifying
    the formula coefficients and the B**2 prefactor.  A failure here points to a wrong
    coefficient in the Ghisellini factor or a missing 8*pi.
    """
    N_therm = _opt_MJD_norm_from_magnetic_field(B, Theta, eps_B, eps_E)
    ghisellini_factor = (6 + 15 * Theta) / (4 + 5 * Theta)
    u_e = N_therm * electron_rest_energy_cgs * Theta * ghisellini_factor
    u_target = (eps_E / eps_B) * B**2 / (8.0 * np.pi)
    assert np.isclose(u_e, u_target, rtol=RTOL_EXACT)


def test_MJD_ghisellini_approaches_NR_limit():
    """At Theta << 1 the Ghisellini mean-energy factor -> 3/2 (classical non-relativistic gas).

    The NR limit of the relativistic Maxwell-Juttner distribution is the Maxwell-Boltzmann
    distribution with mean kinetic energy (3/2) kT.  In dimensionless units this is
    (3/2) Theta, so the Ghisellini factor (which is the mean energy in units of m_e c**2 * Theta)
    must approach 3/2.
    """
    Theta = 0.001
    ghisellini_factor = (6 + 15 * Theta) / (4 + 5 * Theta)
    assert np.isclose(ghisellini_factor, 1.5, rtol=RTOL_APPROX)


def test_MJD_NR_limit_normalization_matches_classical_equipartition():
    """At Theta=0.001, the Ghisellini-based N_therm satisfies U_e ~= (3/2) N_therm m_e c**2 Theta.

    In the non-relativistic limit the Ghisellini formula gives <eps> ~= (3/2) m_e c**2 * Theta,
    which equals kT for kT << m_e c**2.  This test verifies that the NR approximation
    U_e_classical = (3/2) N_therm m_e c**2 Theta is consistent with the equipartition target
    to within RTOL_APPROX.  RTOL_EXACT is not appropriate here because the Ghisellini
    formula is only an approximation at Theta=0.001 (not exactly 3/2).
    """
    B, Theta, eps_B, eps_E = 1.0, 0.001, 0.1, 0.1
    N_therm = _opt_MJD_norm_from_magnetic_field(B, Theta, eps_B, eps_E)
    u_e_classical = 1.5 * N_therm * electron_rest_energy_cgs * Theta
    u_target = (eps_E / eps_B) * B**2 / (8.0 * np.pi)
    assert np.isclose(u_e_classical, u_target, rtol=RTOL_APPROX)


def test_MJD_ghisellini_approaches_UR_limit():
    """At Theta >> 1 the Ghisellini mean-energy factor -> 3 (ultra-relativistic thermal gas).

    The UR limit of the Maxwell-Juttner distribution has <gamma> = 3*Theta, so the mean energy
    per particle is 3 m_e c**2 * Theta and the Ghisellini factor must approach 3.
    """
    Theta = 100.0
    ghisellini_factor = (6 + 15 * Theta) / (4 + 5 * Theta)
    assert np.isclose(ghisellini_factor, 3.0, rtol=RTOL_APPROX)


def test_MJD_UR_limit_normalization_consistent_with_distribution():
    """At Theta=100, integrating the UR MJD distribution gives U_e ~= (eps_E/eps_B) u_B.

    This is the only MJD test that uses numerical integration.  At Theta=100 the ultra-
    relativistic Maxwell-Juttner distribution (gamma**2 * exp(-gamma/Theta)) is physically
    valid and the Ghisellini factor is accurate to ~1%.  The test verifies that the
    normalization N_therm, when used to build the actual distribution callable, reproduces
    the target energy density to within RTOL_APPROX.

    Note: This uses the UR distribution (which assumes gamma >> 1 always) and so is only
    meaningful at Theta >> 1.  At Theta=0.001 the same integrand would give U_e ~= 3*Theta
    per particle (UR limit), not 3/2*Theta (NR limit), making it an inconsistent reference.
    """
    B, Theta, eps_B, eps_E = 1.0, 100.0, 0.1, 0.1
    N_therm = _opt_MJD_norm_from_magnetic_field(B, Theta, eps_B, eps_E)
    N_dist = get_MJD_distribution(Theta, norm=N_therm, with_units=False)
    upper = 20.0 * Theta
    u_e_numeric, _ = quad(
        lambda g: electron_rest_energy_cgs * g * N_dist(g),
        0.0,
        upper,
        epsrel=1e-8,
        limit=500,
    )
    u_target = (eps_E / eps_B) * B**2 / (8.0 * np.pi)
    assert np.isclose(u_e_numeric, u_target, rtol=RTOL_APPROX)


# ---------------------------------------------------------------------------
# Mixed MJD + PL normalization
# ---------------------------------------------------------------------------


def test_mixed_delta_zero_gives_pure_PL():
    """When delta=0 the mixed normalization reduces to pure PL: N_therm ~= 0, N0 = PL norm.

    The delta parameter controls the fraction of thermal energy in the MJD component.
    At delta=0 there is no MJD component, so N_therm must be zero and N0 must equal
    the pure-PL normalization from the same B, eps_B, eps_E.

    This is the limiting case that validates the linear mixing formula without needing
    to know the exact split at intermediate delta values.
    """
    B, Theta, p, eps_B, eps_E = 1.0, 10.0, 3.0, 0.1, 0.1
    gamma_min, gamma_max = 1.0, 1e6
    N_therm, N0_pl = _opt_mixed_norm_from_magnetic_field(B, Theta, p, 0.0, eps_B, eps_E, gamma_min, gamma_max)
    N0_pure_pl = _opt_PL_norm_from_magnetic_field(B, p, eps_B, eps_E, gamma_min, gamma_max)
    assert np.isclose(N_therm, 0.0, atol=1e-30)
    assert np.isclose(N0_pl, N0_pure_pl, rtol=RTOL_EXACT)


def test_mixed_delta_one_gives_pure_MJD():
    """When delta=1 the mixed normalization reduces to pure MJD: N0 ~= 0, N_therm = MJD norm.

    At delta=1 all of the electron energy budget is in the MJD component, so N0 must be
    zero and N_therm must equal the pure-MJD normalization from the same parameters.
    """
    B, Theta, p, eps_B, eps_E = 1.0, 10.0, 3.0, 0.1, 0.1
    N_therm, N0_pl = _opt_mixed_norm_from_magnetic_field(B, Theta, p, 1.0, eps_B, eps_E)
    N_therm_pure = _opt_MJD_norm_from_magnetic_field(B, Theta, eps_B, eps_E)
    assert np.isclose(N0_pl, 0.0, atol=1e-30)
    assert np.isclose(N_therm, N_therm_pure, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# gamma-space <-> energy-space normalization round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N0,p", [(1e3, 2.5), (1e5, 3.0), (1.0, 3.5)])
def test_PL_norm_gamma_to_energy_to_gamma_roundtrip(N0, p):
    """The gamma->energy->gamma normalization conversion is exactly invertible.

    The two parameterizations N(gamma) = N0 * gamma^{-p} and N(E) = K_E * E^{1-p} are related
    by K_E = N0 * (m_e c**2)^{p-1}.  Converting N0 -> K_E -> N0 must recover the original
    value to floating-point precision.  Tests at different p values check that the
    correct power of (m_e c**2) is applied in each direction.
    """
    K_E = _opt_PL_norm_gamma_to_energy(N0, p)
    N0_back = _opt_PL_norm_energy_to_gamma(K_E, p)
    assert np.isclose(N0_back, N0, rtol=RTOL_EXACT)


@pytest.mark.parametrize("N0", [1e3, 1e5, 1.0])
def test_BPL_norm_gamma_to_energy_to_gamma_roundtrip(N0):
    """The BPL gamma->energy->gamma normalization conversion is exactly invertible.

    For the BPL, the conversion factor is p-independent (it applies at the break
    where the distribution equals N0 by construction).  The round-trip must recover
    N0 regardless of the amplitude.
    """
    K_E = _opt_BPL_norm_gamma_to_energy(N0)
    N0_back = _opt_BPL_norm_energy_to_gamma(K_E)
    assert np.isclose(N0_back, N0, rtol=RTOL_EXACT)


def test_public_swap_PL_norm_roundtrip():
    """The public ``swap_PL_normalization`` is a self-inverse pair.

    Calls the public unit-bearing wrapper in both directions and verifies that
    N0 (in cm**-3) is recovered after mode='gamma' -> mode='energy' conversion.
    This catches sign-flip bugs in the exponent or wrong power of m_e c**2 in the
    public wrapper that would not be visible through the private backends alone.
    """
    N0 = 1e4 * u.cm**-3
    p = 3.0
    K_E = swap_PL_normalization(N0, p, mode="gamma")
    N0_back = swap_PL_normalization(K_E, p, mode="energy")
    assert np.isclose(N0_back.to_value(u.cm**-3), N0.to_value(u.cm**-3), rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# Public API contracts
# ---------------------------------------------------------------------------


def test_compute_PL_norm_returns_quantity_with_correct_units():
    """The public PL normalization wrapper returns a positive Quantity in cm**-3.

    Verifies the full public pipeline: unit conversion of B (to Gauss), dispatch to
    the private backend, and wrapping the result as an astropy.Quantity.
    """
    result = compute_PL_norm_from_magnetic_field(1.0 * u.G, p=3.0, epsilon_B=0.1, epsilon_E=0.1)
    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.cm**-3)
    assert result.to_value(u.cm**-3) > 0


def test_compute_BPL_norm_raises_on_gamma_min_zero():
    """The public BPL normalization wrapper rejects gamma_min = 0.

    The BPL moment integral int x^{a1+order} dx diverges at x=0 for any a1 < -(order+1),
    so gamma_min must be strictly positive.  The private backend does not validate this;
    the public wrapper is responsible for the guard.
    """
    with pytest.raises(ValueError, match="gamma_min must be strictly positive"):
        compute_BPL_norm_from_magnetic_field(
            1.0,
            a1=-2.5,
            a2=-3.5,
            gamma_b=1e3,
            epsilon_B=0.1,
            epsilon_E=0.1,
            gamma_min=0.0,
        )


def test_compute_PL_norm_mode_gamma_vs_energy_consistent():
    """The ``mode='gamma'`` and ``mode='energy'`` outputs of the public PL norm are consistent.

    ``compute_PL_norm_from_magnetic_field`` with mode='energy' must return the same value
    as converting the mode='gamma' result through ``swap_PL_normalization``.
    A failure here means the mode dispatch is calling the wrong branch or applying the
    conversion factor a second time.
    """
    B, p, eps_B, eps_E = 1.0 * u.G, 3.0, 0.1, 0.1
    N0_gamma = compute_PL_norm_from_magnetic_field(B, p=p, epsilon_B=eps_B, epsilon_E=eps_E, mode="gamma")
    K_E_direct = compute_PL_norm_from_magnetic_field(B, p=p, epsilon_B=eps_B, epsilon_E=eps_E, mode="energy")
    K_E_from_swap = swap_PL_normalization(N0_gamma, p=p, mode="gamma")
    target_unit = u.cm**-3 * u.erg ** (p - 1)
    assert np.isclose(
        K_E_direct.to_value(target_unit),
        K_E_from_swap.to_value(target_unit),
        rtol=RTOL_EXACT,
    )


def test_compute_equipartition_B_formula():
    """The public equipartition B-field wrapper implements B = sqrt(8*pi * eps_B * u_therm).

    Verifies the formula against the analytical expectation in CGS.  A wrong prefactor
    (e.g., 4*pi instead of 8*pi) is the most common error in this formula and would produce
    a sqrt(2) error in B, propagating as a factor of 2 into all normalization and
    emissivity calculations.
    """
    u_therm = 1e12 * u.erg / u.cm**3
    eps_B = 0.1
    B = compute_equipartition_magnetic_field(u_therm, eps_B)
    expected_cgs = np.sqrt(8.0 * np.pi * eps_B * u_therm.to_value(u.erg / u.cm**3))
    assert np.isclose(B.to_value(u.G), expected_cgs, rtol=RTOL_EXACT)
