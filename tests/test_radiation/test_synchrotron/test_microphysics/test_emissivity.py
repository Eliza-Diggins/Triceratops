"""
Tests for bolometric synchrotron emissivity calculations.

Background
----------
The bolometric synchrotron emissivity (power per unit volume) is proportional to the
product of the magnetic energy density and the effective electron number density:

    j_sync = (4/3) sigma_T c * U_B * n_eff

where U_B = B**2 / (8*pi) and n_eff = N0 * M2 (the second moment of the distribution,
int gamma**2 N(gamma) d(gamma), weighted by the distribution amplitude N0).

This leads to two important scaling laws that hold regardless of the distribution shape:

* **B**2 scaling**: doubling B at fixed N0 quadruples the emissivity.
  ``j ~ U_B ~ B**2``
* **N0 linear scaling**: doubling N0 at fixed B doubles the emissivity.
  ``j ~ n_eff ~ N0``

The module exposes two parallel computation paths:

* **B-field path**: takes B and N0 directly (``_opt_*_bol_emissivity_from_magnetic_field``).
* **Full thermal path**: takes u_therm, eps_B, eps_E and internally calls the equipartition
  field and normalization functions before computing the emissivity
  (``_opt_*_bol_emissivity_from_thermal_full``).

Because the full thermal path is a composition of the B-field path with the equipartition
and normalization helpers, it must agree with an explicit two-step calculation
(B = equipartition(u_therm), N0 = norm(B, ...), j = emiss(B, N0, ...)).

Test philosophy
---------------
* The B**2 and N0 scaling laws are tested at RTOL_EXACT because they are exact algebraic
  identities -- if the implementation has the right formula, ratios must be exact to
  floating-point precision.
* The full-thermal vs. two-step agreement is tested at RTOL_EXACT for the same reason:
  the two paths evaluate identical formulas in a different order.
* The public wrappers (``compute_*_bol_emissivity``) are tested only for their API contracts
  (units, positivity) rather than for exact values, since their numerical correctness is
  verified through the private backend tests.

Tolerance tiers
---------------
* ``RTOL_EXACT = 1e-10`` -- all tests here; the scaling and path-consistency properties
  are algebraic identities with no approximation.
"""

import numpy as np
import pytest
from astropy import units as u

from trilobite.radiation.synchrotron.microphysics import (
    _opt_BPL_bol_emissivity_from_magnetic_field,
    _opt_BPL_bol_emissivity_from_thermal_full,
    _opt_BPL_norm_from_magnetic_field,
    _opt_MJD_bol_emissivity_from_magnetic_field,
    _opt_MJD_bol_emissivity_from_thermal_full,
    _opt_MJD_norm_from_magnetic_field,
    _opt_PL_bol_emissivity_from_magnetic_field,
    _opt_PL_bol_emissivity_from_thermal_full,
    _opt_PL_norm_from_magnetic_field,
    _opt_equipart_magnetic_field,
    compute_BPL_bol_emissivity,
    compute_MJD_bol_emissivity,
    compute_PL_bol_emissivity,
)

RTOL_EXACT = 1e-10


# ---------------------------------------------------------------------------
# PL emissivity
# ---------------------------------------------------------------------------


def test_PL_emissivity_scales_as_B_squared():
    """Doubling B at fixed N0 quadruples the PL bolometric emissivity.

    The emissivity j ~ U_B ~ B**2, so j(2B) / j(B) must be exactly 4.
    A failure here indicates either a missing B**2 term or a spurious B-dependent
    factor in the formula (e.g., using B instead of B**2 for the energy density).
    """
    B, N0, p = 1.0, 1e3, 3.0
    e1 = _opt_PL_bol_emissivity_from_magnetic_field(B, N0, p, 1.0, 1e6)
    e2 = _opt_PL_bol_emissivity_from_magnetic_field(2.0 * B, N0, p, 1.0, 1e6)
    assert np.isclose(e2 / e1, 4.0, rtol=RTOL_EXACT)


def test_PL_emissivity_scales_linearly_with_N0():
    """Doubling N0 at fixed B doubles the PL bolometric emissivity.

    The emissivity j ~ n_eff ~ N0 * M2, so j(2*N0) / j(N0) must be exactly 2.
    A failure here indicates that N0 is incorrectly squared, or that the moment
    integral M2 is being recomputed as a function of N0 rather than held fixed.
    """
    B, N0, p = 1.0, 1e3, 3.0
    e1 = _opt_PL_bol_emissivity_from_magnetic_field(B, N0, p, 1.0, 1e6)
    e2 = _opt_PL_bol_emissivity_from_magnetic_field(B, 2.0 * N0, p, 1.0, 1e6)
    assert np.isclose(e2 / e1, 2.0, rtol=RTOL_EXACT)


def test_PL_full_thermal_agrees_with_two_step():
    """The full-thermal PL path agrees with the explicit two-step calculation.

    Two-step: compute B from u_therm, compute N0 from B, compute j from B and N0.
    Full path: ``_opt_PL_bol_emissivity_from_thermal_full`` does all
    three steps internally.  Any disagreement at RTOL_EXACT indicates a transcription
    error in the full path (e.g., wrong argument order or missing intermediate result).
    """
    u_therm, eps_B, eps_E = 1e12, 0.1, 0.1
    p, gamma_min, gamma_max = 3.0, 10.0, 1e6
    B = _opt_equipart_magnetic_field(u_therm, eps_B)
    N0 = _opt_PL_norm_from_magnetic_field(B, p, eps_B, eps_E, gamma_min, gamma_max)
    e_explicit = _opt_PL_bol_emissivity_from_magnetic_field(B, N0, p, gamma_min, gamma_max)
    e_full = _opt_PL_bol_emissivity_from_thermal_full(u_therm, p, eps_B, eps_E, gamma_min, gamma_max)
    assert np.isclose(e_explicit, e_full, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# BPL emissivity
# ---------------------------------------------------------------------------


def test_BPL_emissivity_scales_as_B_squared():
    """Doubling B at fixed N0 quadruples the BPL bolometric emissivity.

    Same B**2 scaling check as for PL, applied to the BPL formula.  The BPL second
    moment M2 = gamma_b**3 * M_tilde2 has extra gamma_b factors that must not introduce
    any additional B-dependence.
    """
    B, N0 = 1.0, 1e3
    a1, a2, gamma_b = -2.5, -3.5, 1e3
    e1 = _opt_BPL_bol_emissivity_from_magnetic_field(B, N0, a1, a2, gamma_b, 1.0, 1e6)
    e2 = _opt_BPL_bol_emissivity_from_magnetic_field(2.0 * B, N0, a1, a2, gamma_b, 1.0, 1e6)
    assert np.isclose(e2 / e1, 4.0, rtol=RTOL_EXACT)


def test_BPL_full_thermal_agrees_with_two_step():
    """The full-thermal BPL path agrees with the explicit two-step calculation.

    Same path-consistency check as the PL case, applied to the BPL family.
    The BPL normalization internally scales by gamma_b**2, so errors in the gamma_b
    prefactor of the normalization step would propagate into the emissivity only
    in the two-step but not the full path (or vice versa), causing disagreement.
    """
    u_therm, eps_B, eps_E = 1e12, 0.1, 0.1
    a1, a2, gamma_b = -2.5, -3.5, 1e3
    gamma_min, gamma_max = 1.0, 1e6
    B = _opt_equipart_magnetic_field(u_therm, eps_B)
    N0 = _opt_BPL_norm_from_magnetic_field(B, a1, a2, gamma_b, eps_B, eps_E, gamma_min, gamma_max)
    e_explicit = _opt_BPL_bol_emissivity_from_magnetic_field(B, N0, a1, a2, gamma_b, gamma_min, gamma_max)
    e_full = _opt_BPL_bol_emissivity_from_thermal_full(u_therm, a1, a2, gamma_b, eps_B, eps_E, gamma_min, gamma_max)
    assert np.isclose(e_explicit, e_full, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# MJD emissivity
# ---------------------------------------------------------------------------


def test_MJD_emissivity_scales_as_B_squared():
    """Doubling B at fixed N_therm quadruples the MJD bolometric emissivity.

    Same B**2 scaling check applied to the MJD family.  The MJD emissivity uses the
    second-moment shape factor S2(Theta) = 12 * Theta**2, so the B**2 scaling must not
    be contaminated by any Theta-dependent factor.
    """
    B, N_therm, Theta = 1.0, 1e3, 10.0
    e1 = _opt_MJD_bol_emissivity_from_magnetic_field(B, N_therm, Theta)
    e2 = _opt_MJD_bol_emissivity_from_magnetic_field(2.0 * B, N_therm, Theta)
    assert np.isclose(e2 / e1, 4.0, rtol=RTOL_EXACT)


def test_MJD_full_thermal_agrees_with_two_step():
    """The full-thermal MJD path agrees with the explicit two-step calculation.

    Same path-consistency check as PL and BPL, applied to the MJD family.
    The MJD normalization uses the Ghisellini formula, so any inconsistency between
    how the Ghisellini factor enters the normalization step versus the full path
    would be caught here.
    """
    u_therm, eps_B, eps_E, Theta = 1e12, 0.1, 0.1, 10.0
    B = _opt_equipart_magnetic_field(u_therm, eps_B)
    N_therm = _opt_MJD_norm_from_magnetic_field(B, Theta, eps_B, eps_E)
    e_explicit = _opt_MJD_bol_emissivity_from_magnetic_field(B, N_therm, Theta)
    e_full = _opt_MJD_bol_emissivity_from_thermal_full(u_therm, Theta, eps_B, eps_E)
    assert np.isclose(e_explicit, e_full, rtol=RTOL_EXACT)


# ---------------------------------------------------------------------------
# Public API contracts
# ---------------------------------------------------------------------------


def test_compute_PL_bol_emissivity_returns_correct_units():
    """The public PL emissivity wrapper returns a positive Quantity in erg s**-1 cm**-3.

    Checks the full public pipeline for the PL distribution: unit conversion of B and N0,
    dispatch to the private backend, and wrapping as astropy.Quantity with the correct
    volumetric power unit.
    """
    result = compute_PL_bol_emissivity(1.0 * u.G, 1e3 * u.cm**-3, p=3.0, gamma_min=1.0, gamma_max=1e6)
    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.erg / u.s / u.cm**3)
    assert result.to_value(u.erg / u.s / u.cm**3) > 0


def test_compute_BPL_bol_emissivity_returns_correct_units():
    """The public BPL emissivity wrapper returns a positive Quantity in erg s**-1 cm**-3."""
    result = compute_BPL_bol_emissivity(
        1.0 * u.G,
        1e3 * u.cm**-3,
        a1=-2.5,
        a2=-3.5,
        gamma_b=1e3,
        gamma_min=1.0,
        gamma_max=1e6,
    )
    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.erg / u.s / u.cm**3)
    assert result.to_value(u.erg / u.s / u.cm**3) > 0


def test_compute_MJD_bol_emissivity_returns_correct_units():
    """The public MJD emissivity wrapper returns a positive Quantity in erg s**-1 cm**-3."""
    result = compute_MJD_bol_emissivity(1.0 * u.G, 1e3 * u.cm**-3, Theta=10.0)
    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.erg / u.s / u.cm**3)
    assert result.to_value(u.erg / u.s / u.cm**3) > 0
