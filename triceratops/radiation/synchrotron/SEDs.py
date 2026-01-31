"""
Synchrotron Spectral Energy Distributions (SEDs).

This module implements a comprehensive library of **phenomenological synchrotron
spectral energy distributions (SEDs)** for power-law electron populations, following
the standard theoretical framework developed in the literature (e.g. :footcite:t:`GranotSari2002SpectralBreaks`)

The core design philosophy is to construct SEDs via **log-space SED surgery**:
complex spectra are assembled by multiplying (adding in log-space) a sequence of
*scale-free, smoothed broken power-law* factors. Each factor introduces a controlled
change in spectral slope at a characteristic frequency without altering the overall
normalization. This approach ensures:

- numerical stability over many decades in frequency,
- clean separation of spectral segments,
- correct asymptotic slopes,
- and composability of multiple spectral breaks.

The module supports:

- non-cooling, slow-cooling, and fast-cooling synchrotron regimes,
- synchrotron self-absorption (SSA), including stratified SSA cases,
- hidden or absorbed cooling breaks,
- smooth or sharp spectral transitions,
- automatic regime determination from physical parameters,
- and analytic closure relations linking phenomenological SED parameters to
  physical quantities (e.g. magnetic field strength, radius).

.. note::

    A complete theoretical discussion of SED construction, including derivations
    and physical interpretations, is provided in the :ref:`synch_sed_theory`. A
    user-guide description of this module can be found in :ref:`synchrotron_seds`, including
    usage examples.

"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from astropy import units as u

from triceratops.profiles import smoothed_BPL
from triceratops.radiation.constants import (
    electron_rest_energy_cgs,
    electron_rest_mass_cgs,
)
from triceratops.radiation.synchrotron.utils import (
    c_1_cgs,
    compute_c5_parameter,
    compute_c6_parameter,
)
from triceratops.utils.misc_utils import ensure_in_units

# Type checking imports
if TYPE_CHECKING:
    from triceratops._typing import (
        _ArrayLike,
        _UnitBearingArrayLike,
        _UnitBearingScalarLike,
    )


# ============================================================= #
# SED Functions                                                 #
# ============================================================= #
# This is where we implement all of the low-level SED functions that
# can be used to draw the correct SED. These functions should all be
# implemented in log-space for numerical stability.
#
# These can be used directly, but are typically wrapped in the corresponding
# SED class. PLEASE DOCUMENT ALL OF THESE FUNCTIONS THOROUGHLY. The corresponding
# theory is complex.
def log_smoothed_SFBPL(
    log_x: "_ArrayLike",
    a1: float,
    a2: float,
    smoothing: float,
):
    r"""
    Logarithm of a scale-free smoothed broken power law (SFBPL).

    This function implements a *scale-free*, multiplicative spectral break
    suitable for constructing complex synchrotron SEDs via **log-space SED
    surgery**. The functional form is

    .. math::

        \tilde f(x)
        =
        \left[
            1 + x^{(a_2 - a_1)/s}
        \right]^s,

    where :math:`x` is a dimensionless ratio (e.g. :math:`\nu / \nu_{\rm brk}`)
    and :math:`s` controls the smoothness and direction of the transition.
    The logarithm of this factor is

    .. math::

        \log \tilde f(x)
        =
        s \log \left[ 1 + x^{(a_2 - a_1)/s} \right].

    Parameters
    ----------
    log_x : array-like
        Logarithm of the dimensionless ratio :math:`x` (e.g.
        :math:`\log(\nu / \nu_{\rm brk})`).
    a1 : float
        Spectral slope on the *high-frequency* side of the break.
    a2 : float
        Spectral slope on the *low-frequency* side of the break.
    smoothing : float
        Smoothness parameter controlling the width and direction of the
        transition. The magnitude :math:`|s|` sets the sharpness, while the
        **sign** of ``s`` determines whether the slope increases or decreases
        across the break. In the limit :math:`|s| \to 0`, the break becomes
        sharp.

    Returns
    -------
    array-like
        Logarithm of the scale-free smoothed broken power-law factor.

    Notes
    -----
    - This function is **scale-free** and introduces no additional
      normalization factors.
    - Designed to be *added* to an existing SED in log-space
      (i.e. multiplied in linear space).
    - Multiple SFBPL factors may be composed, provided each acts on a
      distinct frequency range.

    .. rubric:: Asymptotic Behavior

    - For :math:`x \ll 1`:
      :math:`\tilde f(x) \propto x^{\,a_2 - a_1}`
    - For :math:`x \gg 1`:
      :math:`\tilde f(x) \to 1`

    When added to a baseline power law with slope :math:`a_1`, this produces
    a spectrum with slope :math:`a_2` below the break and slope :math:`a_1`
    above the break.
    """
    return smoothing * np.logaddexp(0, ((a2 - a1) / smoothing) * log_x)


def log_exp_cutoff_sed(
    log_x: "_ArrayLike",
):
    r"""
    Logarithm of a smooth high-frequency exponential cutoff.

    This function implements a *scale-free*, phenomenological cutoff factor

    .. math::

        \Phi(x)
        =
        x^{1/2}\,\exp(1 - x),

    where :math:`x \equiv \nu / \nu_{\max}`. The logarithm of this factor is

    .. math::

        \log \Phi(x)
        =
        \frac{1}{2}\log x + (1 - x).

    The cutoff is normalized such that :math:`\Phi(1) = 1`, allowing it to be
    applied multiplicatively to an existing SED without altering its
    normalization below the cutoff frequency.

    Parameters
    ----------
    log_x : array-like
        Logarithm of the dimensionless frequency ratio
        :math:`\log(\nu / \nu_{\max})`.

    Returns
    -------
    array-like
        Logarithm of the exponential cutoff factor.

    Notes
    -----
    - For :math:`\nu \ll \nu_{\max}`, the cutoff approaches unity.
    - For :math:`\nu > \nu_{\max}`, the spectrum is exponentially suppressed.
    - The conditional expression ensures that the cutoff contributes only
      above :math:`\nu_{\max}`, preserving exact scale-freeness below the
      cutoff.
    - Implemented entirely in log-space for numerical stability.
    """
    return np.where(log_x > 0, 0.5 * log_x + (1.0 - np.exp(log_x)), 0.0)


# --- Power-Law No-Cooling No-SSA SEDs --- #
def _log_powerlaw_sbpl_sed(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED for a non-cooling power-law electron population.

    This function implements the **non-cooling**, optically thin synchrotron
    spectrum for a power-law electron distribution, with no synchrotron
    self-absorption (SSA) and including a high-frequency exponential cutoff.

    The optically thin spectral slopes are:

    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu > \nu_m`.

    The spectrum is constructed using *scale-free smoothed broken power laws*
    (SFBPLs) via log-space SED surgery and is **normalized at the injection
    frequency** :math:`\nu_m`.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection (minimum electron) frequency :math:`\nu_m`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transition. The magnitude controls
        the sharpness of the break, while the sign controls the direction of
        the slope change.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - No synchrotron self-absorption (SSA) is included.
    - The high-frequency exponential cutoff is applied multiplicatively.
    """
    x_m = log_nu - log_nu_m
    x_max = log_nu - log_nu_max

    # Normalize at the injection frequency ν_m
    log_sed = (1.0 / 3.0) * x_m

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, 1.0 / 3.0, -(p - 1.0) / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


# --- Power-Law Cooling SEDs --- #
def _log_powerlaw_sbpl_sed_cool_1(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED in the **fast-cooling** regime.

    This function implements the fast-cooling synchrotron spectrum for a
    power-law electron distribution, assuming the frequency ordering

    .. math::

        \nu_c < \nu_m.

    The optically thin spectral slopes are:

    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-1/2}` for :math:`\nu_c < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_m`.

    The spectrum is constructed via log-space SED surgery using SFBPL kernels
    and is **normalized at the cooling frequency** :math:`\nu_c`.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - Assumes :math:`\nu_c < \nu_m`; no internal validation is performed.
    - No synchrotron self-absorption (SSA) is included.
    """
    x_c = log_nu - log_nu_c
    x_m = log_nu - log_nu_m
    x_max = log_nu - log_nu_max

    # Normalize at the cooling frequency ν_c
    log_sed = (1.0 / 3.0) * x_c

    # Cooling break
    log_sed += log_smoothed_SFBPL(x_c, 1.0 / 3.0, -1.0 / 2.0, s)

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, -1.0 / 2.0, -p / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


def _log_powerlaw_sbpl_sed_cool_2(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED in the **slow-cooling** regime.

    This function implements the slow-cooling synchrotron spectrum for a
    power-law electron distribution, assuming the frequency ordering

    .. math::

        \nu_m < \nu_c.

    The optically thin spectral slopes are:

    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for
      :math:`\nu_m < \nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_c`.

    The spectrum is constructed via log-space SED surgery using SFBPL kernels
    and is **normalized at the injection frequency** :math:`\nu_m`.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - Assumes :math:`\nu_m < \nu_c`; no internal validation is performed.
    - No synchrotron self-absorption (SSA) is included.
    """
    x_m = log_nu - log_nu_m
    x_c = log_nu - log_nu_c
    x_max = log_nu - log_nu_max

    # Normalize at the injection frequency ν_m
    log_sed = (1.0 / 3.0) * x_m

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, 1.0 / 3.0, -(p - 1.0) / 2.0, s)

    # Cooling break
    log_sed += log_smoothed_SFBPL(x_c, -(p - 1.0) / 2.0, -p / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


class _SynchrotronCoolingSEDFunctions(Enum):
    SPECTRUM_1 = _log_powerlaw_sbpl_sed_cool_1  # fast-cooling
    SPECTRUM_2 = _log_powerlaw_sbpl_sed_cool_2  # slow-cooling
    SPECTRUM_3 = _log_powerlaw_sbpl_sed  # non-cooling


# --- Power-Law SSA SEDs --- #
def _log_powerlaw_sbpl_sed_ssa_1(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED with SSA, assuming the ordering :math:`\nu < \nu_a < \nu_m`.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu_a < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu > \nu_m`.

    In this configuration, the SSA break lies below the injection frequency.
    The spectrum is therefore **normalized at the injection break**
    :math:`\nu_m`, where the optically thin synchrotron slope is well defined.
    The SSA turnover is introduced at lower frequencies using a scale-free
    smoothed broken power-law (SFBPL) factor.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection (minimum electron) frequency
        :math:`\nu_m`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency
        :math:`\nu_a`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency
        :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - Assumes :math:`\nu < \nu_a < \nu_m`; no internal validation is performed.
    - The spectrum is constructed via log-space SED surgery using scale-free
      SFBPL kernels.
    """
    x_a = log_nu - log_nu_a
    x_m = log_nu - log_nu_m
    x_max = log_nu - log_nu_max

    # Normalize at the injection frequency ν_m
    log_sed = 2.0 * x_a + (1.0 / 3.0) * (log_nu_a - log_nu_m)

    # SSA break (optically thick → optically thin)
    log_sed += log_smoothed_SFBPL(x_a, 2.0, 1.0 / 3.0, s)

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, 1.0 / 3.0, -(p - 1.0) / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


def _log_powerlaw_sbpl_sed_ssa_2(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED with SSA, assuming the ordering :math:`\nu < \nu_m < \nu_a`.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{5/2}` for :math:`\nu_a < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu > \nu_a`.

    In this case, the SSA break lies above the injection frequency and the
    intermediate :math:`\nu^{1/3}` segment is absent. The spectrum is therefore
    **normalized at the injection frequency** :math:`\nu_m` and propagated
    downward to the SSA break using the appropriate optically thick slope.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency
        :math:`\nu_a`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency
        :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - Assumes :math:`\nu < \nu_m < \nu_a`; no internal validation is performed.
    - The increase in slope from :math:`\nu^2` to :math:`\nu^{5/2}` requires
      flipping the sign of the smoothing parameter due to the adopted SFBPL
      convention.
    """
    x_m = log_nu - log_nu_m
    x_a = log_nu - log_nu_a
    x_max = log_nu - log_nu_max

    # Normalize at the injection frequency ν_m
    log_sed = 2.0 * x_m + (5.0 / 2.0) * (log_nu_m - log_nu_a)

    # SSA break (slope increase requires -s in this convention)
    log_sed += log_smoothed_SFBPL(x_m, 2.0, 5.0 / 2.0, -s)

    # Injection break
    log_sed += log_smoothed_SFBPL(x_a, 5.0 / 2.0, -(p - 1.0) / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


class _SynchrotronSSASEDFunctions(Enum):
    SPECTRUM_1 = _log_powerlaw_sbpl_sed_ssa_1
    SPECTRUM_2 = _log_powerlaw_sbpl_sed_ssa_2


# --- Power-Law SSA + Cooling SEDs --- #
# NOTE: Spectra 1 and 2 are identical to the non-cooling SSA cases because
# the cooling break lies above the maximum synchrotron frequency.
# They are therefore not implemented separately here.
def _log_powerlaw_sbpl_sed_ssa_cool_3(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA in the **slow-cooling** regime, assuming the ordering :math:`\nu < \nu_a < \nu_m < \nu_c`.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu_a < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for
      :math:`\nu_m < \nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_c`.

    The spectrum is **normalized at the injection frequency** :math:`\nu_m`,
    which is the dominant optically thin break in the slow-cooling regime.
    Synchrotron self-absorption and cooling breaks are then applied via
    scale-free smoothed broken power-law (SFBPL) factors.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency :math:`\nu_a`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency.
    p : float
        Electron energy distribution index.
    s : float
        Smoothness parameter for SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.
    """
    x_a = log_nu - log_nu_a
    x_m = log_nu - log_nu_m
    x_c = log_nu - log_nu_c
    x_max = log_nu - log_nu_max

    # Normalize at ν_m
    log_sed = 2.0 * x_a + (1.0 / 3.0) * (log_nu_a - log_nu_m)

    # SSA break
    log_sed += log_smoothed_SFBPL(x_a, 2.0, 1.0 / 3.0, s)

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, 1.0 / 3.0, -(p - 1.0) / 2.0, s)

    # Cooling break
    log_sed += log_smoothed_SFBPL(x_c, -(p - 1.0) / 2.0, -p / 2.0, s)

    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


def _log_powerlaw_sbpl_sed_ssa_cool_4(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA in the **slow-cooling** regime, assuming the ordering :math:`\nu < \nu_m < \nu_a < \nu_c`.

    The spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{5/2}` for :math:`\nu_m < \nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for
      :math:`\nu_a < \nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_c`.

    The spectrum is **normalized at the injection frequency** :math:`\nu_m`.
    The increase in slope from :math:`\nu^2` to :math:`\nu^{5/2}` requires
    flipping the sign of the smoothing parameter due to the adopted SFBPL
    convention.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.
    """
    x_m = log_nu - log_nu_m
    x_a = log_nu - log_nu_a
    x_c = log_nu - log_nu_c
    x_max = log_nu - log_nu_max

    log_sed = 2.0 * x_m + (5.0 / 2.0) * (log_nu_m - log_nu_a)

    # RJ → optically thick SSA (slope increase)
    log_sed += log_smoothed_SFBPL(x_m, 2.0, 5.0 / 2.0, -s)

    # SSA → optically thin uncooled
    log_sed += log_smoothed_SFBPL(x_a, 5.0 / 2.0, -(p - 1.0) / 2.0, s)

    # Cooling break
    log_sed += log_smoothed_SFBPL(x_c, -(p - 1.0) / 2.0, -p / 2.0, s)

    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


def _log_powerlaw_sbpl_sed_ssa_cool_5(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_a: float,
    log_nu_ac: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with synchrotron self-absorption (SSA) in the **fast-cooling** regime.

    Includes stratified absorption, assuming the ordering

    .. math::

        \nu < \nu_{ac} < \nu_a < \nu_c < \nu_m.

    In this regime, electrons cool efficiently below the injection energy, and
    synchrotron self-absorption occurs within a stratified, cooling electron
    population behind the shock. This produces an additional low-frequency break
    at :math:`\nu_{ac}`, separating optically thick emission from uncooled and
    cooled electron layers.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_{ac}`,
    - :math:`F_\nu \propto \nu^{11/8}` for :math:`\nu_{ac} < \nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu_a < \nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-1/2}` for :math:`\nu_c < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_m`.

    The spectrum is **anchored at the cooling frequency** :math:`\nu_c`, which is
    the dominant physical break in the fast-cooling regime. Lower-frequency
    structure is constructed via scale-free smoothed broken power laws (SFBPLs)
    to ensure continuity and correct asymptotic behavior.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency :math:`\nu_a`.
    log_nu_ac : float
        Logarithm of the stratified absorption break frequency :math:`\nu_{ac}`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency.
    p : float
        Electron energy distribution index.
    s : float
        Smoothness parameter for SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.

    Notes
    -----
    - Assumes fast cooling (:math:`\nu_c \ll \nu_m`) and the stated frequency
      ordering; no internal validation is performed.
    - The :math:`11/8` slope arises from stratified synchrotron self-absorption in
      a cooling electron population.
    """
    # Determine the nu/nu_a, nu/nu_m, and nu/nu_c ratios
    x_a, _, x_m, x_ac, x_max = (
        log_nu - log_nu_a,
        log_nu - log_nu_c,
        log_nu - log_nu_m,
        log_nu - log_nu_ac,
        log_nu - log_nu_max,
    )

    # This is a fast-cooled spectrum, so the dominant break is at nu_c and that
    # is where we anchor the spectrum. The transition at the cooling break is from SPL
    # E to SPL F (1/3 to -1/2) corresponding from the transition from the uncooled RJ tail to
    # the cooled population.
    log_sed = 2 * x_ac + (11 / 8) * (log_nu_ac - log_nu_a) + (1 / 3) * (log_nu_a - log_nu_c)

    # We now add the nu_ac break which corresponds to the transition from SPL B to SPL C
    # (2 to 11/8). This is the transition from the RJ tail to the cooled SSA segment.
    log_sed += log_smoothed_SFBPL(x_ac, 2.0, 11 / 8, s)

    # Add the transition from the 11/8 nu_ac stratified segment to the RJ tail at the
    # absorption break nu_a (11/8 to 1/3; SPL C to SPL E).
    log_sed += log_smoothed_SFBPL(x_a, 11 / 8, 1 / 3, s)

    # Finally, we need to add in the injection break at nu_m. This is SPL F -> SPL H
    # (-1/2 to -p/2)
    log_sed += log_smoothed_SFBPL(x_m, -1 / 2, -p / 2, s)
    # Truncate
    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


def _log_powerlaw_sbpl_sed_ssa_cool_6(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_ac: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA assuming the ordering :math:`\nu < \nu_{ac} < \nu_a < \nu_m`.

    In this configuration, the cooling break lies *above* the self-absorption
    photosphere and is therefore not directly visible in the emergent spectrum.
    The observed emission is dominated by optically thick and marginally thin
    synchrotron radiation from a cooling electron population.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_{ac}`,
    - :math:`F_\nu \propto \nu^{11/8}` for :math:`\nu_{ac} < \nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{-1/2}` for :math:`\nu_a < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_m`.

    Although the system is fast cooling, the cooling break itself does not appear
    explicitly because it is obscured by synchrotron self-absorption. The spectrum
    is therefore constructed by anchoring at the absorption break and propagating
    to higher frequencies using the fast-cooling optically thin slopes.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency :math:`\nu_a`.
    log_nu_ac : float
        Logarithm of the stratified absorption break frequency :math:`\nu_{ac}`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency.
    p : float
        Electron energy distribution index.
    s : float
        Smoothness parameter for SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.

    Notes
    -----
    - Assumes fast cooling with the cooling break hidden by SSA.
    - The :math:`11/8` segment reflects stratified absorption in a cooling flow.
    """
    # Determine the nu/nu_a, nu/nu_m, and nu/nu_c ratios.
    x_a, x_m, x_ac, x_max = log_nu - log_nu_a, log_nu - log_nu_m, log_nu - log_nu_ac, log_nu - log_nu_max

    # This is a fast-cooling spectrum, so we normalize at the cooling break, but use
    # the power-law propagation technique to actually place the anchor point at nu_a instead.
    # This is SPL C to SPL F (11/8 to -1/2).
    log_sed = 2 * x_ac + (11 / 8) * (log_nu_ac - log_nu_a)

    # We now add the nu_ac break which corresponds to the transition from SPL B to SPL C
    # (2 to 11/8). This is the transition from the RJ tail to the cooled SSA segment.
    log_sed += log_smoothed_SFBPL(x_ac, 2.0, 11 / 8, s)

    # Add the transition from 11/8 to -1/2 at the absorption break nu_a (SPL C to SPL F).
    log_sed += log_smoothed_SFBPL(x_a, 11 / 8, -1 / 2, s)

    # Now add on the injection break at nu_m. This is SPL F -> SPL H. Because we are
    # fast cooling, the high frequency slope is -p/2.
    log_sed += log_smoothed_SFBPL(x_m, -1 / 2, -p / 2, s)
    # Truncate
    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


def _log_powerlaw_sbpl_sed_ssa_cool_7(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA in the **fast-cooling** regime, assuming the ordering :math:`\nu < \nu_m < \nu_a`.

    In this case, both the cooling break and any stratified absorption structure
    are hidden beneath the synchrotron self-absorption photosphere. The observed
    spectrum transitions directly from optically thick emission to optically thin
    fast-cooled synchrotron radiation.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{5/2}` for :math:`\nu_m < \nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_a`.

    The spectrum is **anchored at the injection frequency** :math:`\nu_m`, since
    all cooling-related breaks occur at higher frequencies and do not affect the
    low-frequency emission. The SSA break at :math:`\nu_a` directly connects the
    optically thick synchrotron emission to the fast-cooled optically thin regime.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency :math:`\nu_a`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency.
    p : float
        Electron energy distribution index.
    s : float
        Smoothness parameter for SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.

    Notes
    -----
    - This regime corresponds to extreme self-absorption in a fast-cooling system.
    - No explicit cooling break appears in the observable spectrum.
    """
    # Determine the nu/nu_a and nu/nu_m ratios
    x_a, x_m, x_max = log_nu - log_nu_a, log_nu - log_nu_m, log_nu - log_nu_max

    # We anchor at the injection break because the cooling break (if present) is not visible behind the
    # absorption photosphere at the shock. We therefore start with the SPL B -> SPL A transition (2 to 5/2).
    log_sed = 2 * x_m + (5 / 2) * (log_nu_m - log_nu_a)

    # Now add the SSA break at nu_a which correspond to the transition from optically
    # thick SSA (A) to optically thin cooled (H) (5/2 to -p/2).
    log_sed += log_smoothed_SFBPL(x_a, 5 / 2, -p / 2, s)

    # Now we need to add in the injection break at nu_m.
    log_sed += log_smoothed_SFBPL(x_m, 2, 5 / 2, -s)

    # Truncate
    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


class _SynchrotronSSACoolingSEDFunctions(Enum):
    SPECTRUM_1 = _log_powerlaw_sbpl_sed_ssa_1
    SPECTRUM_2 = _log_powerlaw_sbpl_sed_ssa_2
    SPECTRUM_3 = _log_powerlaw_sbpl_sed_ssa_cool_3
    SPECTRUM_4 = _log_powerlaw_sbpl_sed_ssa_cool_4
    SPECTRUM_5 = _log_powerlaw_sbpl_sed_ssa_cool_5
    SPECTRUM_6 = _log_powerlaw_sbpl_sed_ssa_cool_6
    SPECTRUM_7 = _log_powerlaw_sbpl_sed_ssa_cool_7


# =============================================================
# SED Base Class
# =============================================================
# To help compartmentalize the name space and prevent any issues
# with clarity, we provide a small base class for SEDs to serve as a
# guide both for SED implementation and for documentation.
class SynchrotronSED(ABC):
    """
    Base class for synchrotron SED implementation.

    The :class:`SynchrotronSED` is a simple compartment for defining the
    structure of a specific spectral energy distribution. For each SED,
    one needs to provide

    1. The :meth:`sed` method (vis-a-vis the low-level ``_log_opt_sed`` method), which
       simply provides the phenomenological SED shape as a function of frequency and parameters. For example
       a broken power-law SED would implement the appropriate power-law segments and breaks in this method.

    2. The :meth:`from_params_to_physics` (vis-a-vis the low-level ``_opt_from_params_to_physics`` method), which
       provides the mapping from phenomenological SED parameters (e.g., break frequencies, flux normalizations, etc.)
       to physical parameters (e.g., magnetic field strength, radius, etc.) based on closure relations.
    3. The :meth:`from_physics_to_params` (vis-a-vis the low-level ``_opt_from_physics_to_params`` method), which
       provides the mapping from physical parameters (e.g., magnetic field strength, radius, etc.) to phenomenological
       SED parameters (e.g., break frequencies, flux normalizations, etc.) based on closure relations.

    Components (2) and (3) are **NOT REQUIRED** for an SED to be functional; however, they are generally necessary
    for inference. The necessary extent of the infrastructure is left to the implementer.

    .. hint::

        All of the existing SEDs are implemented using this base class as a guide. If you're
        in need of check out the other classes in this module.

    Instantiation
    --------------
    Instantiation of the SED class may be used to pre-compute class-wide constants, evaluate or construct kernels,
    etc; however, it should **NOT** be used to store SED-specific parameters. All SED-specific parameters
    should be passed directly to the relevant methods. This ensures that, during high-load inference tasks,
    no unnecessary re-instantiation of SED objects is required.
    """

    # ============================================================ #
    # Instantiation and basic structure                            #
    # ============================================================ #
    def __init__(self, *args, **kwargs):
        r"""
        Instantiate the SED object.

        Parameters
        ----------
        args:
            Positional arguments for SED instantiation.
        kwargs:
            Keyword arguments for SED instantiation.

        Notes
        -----
        The SED instantiation should NOT be used to store SED-specific parameters. All SED-specific parameters
        should be passed directly to the relevant methods. This ensures that, during high-load inference tasks,
        no unnecessary re-instantiation of SED objects is required.
        """
        pass

    def __call__(self, nu, **parameters):
        r"""
        Evaluate the SED at the given frequency.

        This is a thin wrapper around :meth:`sed` and exists purely for
        convenience, allowing SED objects to be used as callables.
        """
        return self.sed(nu, **parameters)

    def __repr__(self):
        return f"<{self.__class__.__name__}()>"

    # ============================================================ #
    # SED Function Implementation                                  #
    # ============================================================ #
    # Here should be the implementation of the SED function itself,
    # which is a function of nu and some set of additional parameters.
    @abstractmethod
    def _log_opt_sed(self, nu: "_ArrayLike", **parameters):
        r"""
        Low-level optimized log-space SED evaluation.

        This method implements the core numerical kernel for the synchrotron
        spectral energy distribution (SED) in **logarithmic space**. It is intended
        for performance-critical use and therefore assumes that all inputs are
        already provided in a validated, unit-consistent form.

        No unit handling, type checking, or safety checks are performed.

        Parameters
        ----------
        nu : float or array-like
            Natural logarithm of the frequency at which to evaluate the SED,

            .. math::

                \nu \equiv \ln\!\left(\frac{\nu_{\rm phys}}{\mathrm{Hz}}\right),

            where :math:`\nu_{\rm phys}` is the physical frequency expressed in
            Hz-equivalent CGS units.
        **parameters
            Additional dimensionless or CGS-valued parameters required for the
            SED calculation. The exact set of required parameters is
            implementation-specific.

        Returns
        -------
        float or array-like
            Natural logarithm of the synchrotron SED evaluated at the specified
            frequency.
        """
        raise NotImplementedError

    def _opt_sed(self, nu: "_ArrayLike", **parameters):
        r"""
        Low-level optimized linear-space SED evaluation.

        This method is a thin convenience wrapper around
        :meth:`_log_opt_sed`, exponentiating the log-space SED to produce
        linear-space values:

        .. math::

            F_\nu = \exp\!\left[\log F_\nu\right].

        Like :meth:`_log_opt_sed`, this method assumes that all inputs are
        provided as dimensionless scalars or NumPy arrays in consistent
        Hz-equivalent CGS units. No unit validation, type checking, or safety
        checks are performed.

        Parameters
        ----------
        nu : float or array-like
            Natural logarithm of the frequency at which to evaluate the SED,
            defined as in :meth:`_log_opt_sed`.
        **parameters
            Additional parameters required for the SED calculation. The required
            parameters are implementation-specific.

        Returns
        -------
        float or array-like
            Synchrotron SED evaluated in linear space at the specified frequency.
        """
        log_sed = self._log_opt_sed(nu, **parameters)
        return np.exp(log_sed)

    @abstractmethod
    def sed(self, nu: "_UnitBearingArrayLike", **parameters):
        r"""
        User-facing synchrotron SED evaluation.

        This method provides a high-level, user-friendly interface for computing
        the synchrotron spectral energy distribution (SED). It is responsible for
        handling unit validation and coercion, basic shape checking, and any other
        user-facing conveniences before dispatching to the low-level optimized
        backend.

        Internally, this method should convert inputs into the dimensionless,
        log-space form expected by the optimized implementation
        (:meth:`_log_opt_sed`).

        Parameters
        ----------
        nu : float, array-like, or astropy.units.Quantity
            Frequency at which to evaluate the SED. If provided without units,
            frequencies are assumed to be in Hz. If provided as an
            :class:`astropy.units.Quantity`, the value will be converted to
            Hz-equivalent CGS units before evaluation.

            The frequency may be specified as a scalar (to evaluate a single
            spectrum) or as a one-dimensional array (to evaluate the SED over
            a frequency grid).
        **parameters
            Additional parameters required for the SED calculation. These may
            include phenomenological SED parameters (e.g. break frequencies,
            normalization constants) or physical model parameters, depending
            on the specific SED implementation.

        Returns
        -------
        float, array-like, or astropy.units.Quantity
            The synchrotron SED evaluated at the specified frequency. Implementations
            may return either plain numerical values or
            :class:`astropy.units.Quantity` objects with appropriate physical units.

        Notes
        -----
        - Subclasses must implement this method.
        - The returned SED should be consistent with the conventions and units
          adopted by the corresponding low-level implementation.
        """
        raise NotImplementedError

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def from_params_to_physics(self, **parameters):
        r"""
        Convert phenomenological SED parameters into physical parameters.

        This method provides a **user-facing interface** for mapping phenomenological
        SED parameters—such as break frequencies, peak fluxes, or normalization
        constants—into underlying physical quantities like magnetic field strength,
        emitting radius, characteristic electron energies, or energy densities.

        The mapping implemented by this method is **model-dependent** and typically
        relies on analytic closure relations derived from synchrotron theory, often
        supplemented by additional microphysical assumptions. Examples include:

        - assumptions about particle acceleration efficiency,
        - equipartition or near-equipartition between fields and particles,
        - prescriptions for the minimum Lorentz factor :math:`\gamma_m`,
        - geometric assumptions encoded via solid angle or filling factor terms.

        As a result, the inferred physical parameters should be interpreted within
        the context of the specific microphysical model adopted by the implementing
        subclass.

        This method is optional: an SED implementation is fully functional without
        it. However, closure relations are generally required for inference workflows,
        parameter estimation, and for coupling SEDs to dynamical or microphysical
        models.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying phenomenological SED parameters. The exact
            set of required parameters is model-dependent and determined by the
            implementing subclass.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters. The contents,
            naming conventions, and units are implementation-specific.

        Notes
        -----
        - This method may perform unit validation, coercion, or shape checking
          before dispatching to the low-level optimized implementation
          :meth:`_opt_from_params_to_physics`.
        - The mapping is not guaranteed to be unique; degeneracies may be present
          depending on the assumed microphysics.
        """
        raise NotImplementedError

    def _opt_from_params_to_physics(self, **parameters):
        r"""
        Low-level optimized conversion from SED parameters to physical parameters.

        This method implements the same phenomenological-to-physical parameter
        mapping as :meth:`from_params_to_physics`, but assumes that all inputs are
        provided as dimensionless scalars or NumPy arrays in consistent CGS units.

        The mapping may encode analytic closure relations and microphysical
        assumptions specific to the SED model, but **no validation or safety checks**
        are performed at this level.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying phenomenological SED parameters in CGS or
            dimensionless form. The exact set of required parameters is
            implementation-specific.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters in CGS units.

        Notes
        -----
        - This method is intended for internal use in performance-critical contexts.
        - It should be called only after unit handling and basic validation have been
          performed by :meth:`from_params_to_physics`.
        """
        raise NotImplementedError

    def _opt_from_physics_to_params(self, **parameters):
        r"""
        Low-level optimized conversion from physical parameters to SED parameters.

        This method implements the inverse mapping of
        :meth:`_opt_from_params_to_physics`, converting physical quantities—such as
        magnetic field strength, system size, particle energy scales, or energy
        densities—into phenomenological SED parameters like break frequencies or
        peak fluxes.

        The mapping assumes a specific set of synchrotron closure relations and
        microphysical prescriptions adopted by the implementing subclass.

        All inputs are assumed to be provided in CGS units or as dimensionless
        scalars. No unit validation, consistency checks, or physical sanity checks
        are performed.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying physical parameters in CGS units. The exact
            set of required parameters is implementation-specific.

        Returns
        -------
        dict
            Dictionary containing phenomenological SED parameters in CGS or
            dimensionless form.

        Notes
        -----
        - This method is intended for internal use in performance-critical contexts.
        - Inverse mappings may not exist or may not be unique for all SED models.
        """
        raise NotImplementedError

    def from_physics_to_params(self, **parameters):
        r"""
        Convert physical parameters into phenomenological SED parameters.

        This method provides a **user-facing interface** for mapping physical
        quantities—such as magnetic field strength, emitting radius, particle
        energy scales, or energy densities—into phenomenological SED parameters
        like break frequencies, normalization constants, or spectral amplitudes.

        The conversion is based on analytic closure relations from synchrotron
        theory and typically incorporates additional microphysical assumptions
        (e.g., acceleration efficiency, geometry, or equipartition conditions)
        defined by the implementing subclass.

        This functionality is primarily used in inference workflows, where
        physical model parameters are sampled and must be translated into
        observable SED quantities.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying physical parameters. The exact set of
            required parameters is model-dependent and determined by the
            implementing subclass.

        Returns
        -------
        dict
            Dictionary containing phenomenological SED parameters.

        Notes
        -----
        - This method may perform unit validation, coercion, or shape checking
          before dispatching to the low-level optimized implementation
          :meth:`_opt_from_physics_to_params`.
        - Subclasses that do not support inversion of closure relations may leave
          this method unimplemented.
        """
        raise NotImplementedError


class MultiSpectrumSynchrotronSED(SynchrotronSED, ABC):
    r"""
    Base class for synchrotron SEDs with multiple discrete spectral regimes.

    Many synchrotron models admit multiple global spectral "regimes" defined by
    the ordering of characteristic frequencies (e.g. :math:`\nu_a, \nu_m, \nu_c`).
    This class provides a standard pattern:

    1. Determine a regime label from the (global) model parameters.
    2. Optionally compute *derived* parameters needed to evaluate that regime
       (e.g. :math:`\nu_a` inferred from :math:`F_{\nu,\mathrm{pk}}`, geometry, etc.).
    3. Dispatch to a regime-specific optimized kernel.

    Subclasses implement:

    - :meth:`_compute_sed_regime`
    - :meth:`determine_sed_regime`
    - :meth:`_log_opt_sed_from_regime`

    Notes
    -----
    All "opt" methods operate on **unitless CGS scalars / NumPy arrays** and
    should not perform validation. Concrete subclasses must define whether
    the optimized backend expects linear frequencies ``nu`` or log-frequencies
    ``log_nu`` (see :meth:`_expects_log_frequency`).
    """

    #: Optional mapping/enum describing available regime functions.
    SPECTRUM_FUNCTIONS = None

    # ============================================================ #
    # Regime Management                                            #
    # ============================================================ #
    @abstractmethod
    def _compute_sed_regime(self, **parameters) -> tuple[Any, dict[str, Any]]:
        r"""
        Determine the global SED regime and any derived parameters.

        This method encodes the logic used to classify the SED into a discrete
        physical regime based on the ordering of characteristic frequencies.
        It may additionally compute *derived* parameters required to evaluate
        the spectrum (e.g. an inferred SSA frequency :math:`\nu_a`).

        The returned regime applies **globally** and does not depend on the
        sampling frequency grid.

        Parameters
        ----------
        **parameters
            Model parameters required to determine the regime. Typical examples
            include characteristic frequencies (or their log-values), peak flux
            normalizations, geometric factors, and microphysical parameters.

        Returns
        -------
        (regime, derived)
            regime : int or Enum-like
                Regime identifier. The interpretation is defined by the subclass.
            derived : dict
                Derived parameters needed for evaluation in the selected regime.
                This may include values such as ``log_nu_a`` or other cached
                quantities used by the regime-specific kernel.

        Notes
        -----
        - This method should be *fast* and free of allocations when possible.
        - No unit checking or validation should occur here.
        """
        raise NotImplementedError

    @abstractmethod
    def determine_sed_regime(self, **parameters) -> Any:
        r"""
        Determine the physical synchrotron spectral regime (public API).

        This method is intended for diagnostic and introspection use. It should
        perform any necessary unit handling and validation, then delegate to the
        optimized implementation (:meth:`_compute_sed_regime`).

        Parameters
        ----------
        **parameters
            User-facing model parameters. Typical examples include quantities
            such as :math:`\nu_m`, :math:`\nu_c`, :math:`F_{\nu,\mathrm{pk}}`, etc.

        Returns
        -------
        regime : int or Enum-like
            Regime identifier defined by the subclass.

        Notes
        -----
        - The regime does not depend on the frequency grid used for evaluation.
        - Subclasses should ensure that the regime returned here is consistent
          with the behavior of :meth:`sed`.
        """
        raise NotImplementedError

    # ============================================================ #
    # Regime-Specific SED Kernel                                   #
    # ============================================================ #
    @abstractmethod
    def _log_opt_sed_from_regime(
        self,
        nu: "_ArrayLike",
        regime: Any,
        **parameters,
    ):
        r"""
        Evaluate the log-space SED for a pre-determined regime.

        This method computes the logarithm of the SED for a single regime.
        All branching on ``regime`` should occur here (or above), and this
        method should assume that any derived parameters required for the regime
        have already been computed.

        Parameters
        ----------
        nu : float or array-like
            Frequency grid in optimized form. If :meth:`_expects_log_frequency`
            returns True, this is ``log_nu = log(ν)``. Otherwise it is linear ν.
        regime : int or Enum-like
            Regime identifier returned by :meth:`_compute_sed_regime`.
        **parameters
            Parameters required by the kernel, including any derived values
            produced by :meth:`_compute_sed_regime`.

        Returns
        -------
        float or array-like
            Logarithm of the SED evaluated at the given frequencies.

        Notes
        -----
        This is the performance-critical kernel. Implementations should avoid
        allocations and unnecessary branching when possible.
        """
        raise NotImplementedError

    # ============================================================ #
    # Log-Space Orchestration                                      #
    # ============================================================ #
    def _log_opt_sed(self, nu: "_ArrayLike", **parameters):
        r"""
        Log-space optimized SED evaluation with regime dispatch.

        This method determines the appropriate SED regime from the input
        parameters and dispatches to the corresponding regime-specific kernel.

        Subclasses should generally **not override** this method. Instead,
        implement :meth:`_compute_sed_regime` and :meth:`_log_opt_sed_from_regime`.

        Parameters
        ----------
        nu : float or array-like
            Frequency grid in optimized form. If :meth:`_expects_log_frequency`
            returns True, this is ``log_nu = log(ν)``. Otherwise it is linear ν.
        **parameters
            Parameters required for both regime determination and SED evaluation.

        Returns
        -------
        float or array-like
            Logarithm of the SED evaluated at the given frequencies.
        """
        regime, derived = self._compute_sed_regime(**parameters)
        merged = dict(parameters)
        merged.update(derived)
        return self._log_opt_sed_from_regime(nu, regime, **merged)


# ============================================================ #
# SED Implementations                                          #
# ============================================================ #
# Now we can include concrete implementations of various SEDs. Not
# all of the SEDs we plan to implement are currently implemented in the
# codebase, but we provide a few examples here to illustrate the structure.
class PowerLaw_Cooling_SSA_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Synchrotron spectral energy distribution with cooling and self-absorption.

    This class implements the **full piecewise synchrotron spectrum** for a
    power-law electron population, including:

    - Radiative cooling (fast, slow, and non-cooling regimes),
    - Synchrotron self-absorption (SSA),
    - Stratified SSA corrections where applicable,
    - A high-frequency exponential cutoff.

    The implementation follows the standard GRB / supernova afterglow
    formalism (e.g. :footcite:t:`GranotSari2002SpectralBreaks`, :footcite:t:`2020MNRAS.493.3521B`,
    :footcite:t:`duran2013radius`, :footcite:t:`2025ApJ...992L..18S`, :footcite:t:`GaoSynchrotronReview2013`,
    etc.)

    Specifically we implement spectra using **log-space SED surgery** with scale-free smoothed broken
    power laws (SFBPLs). The spectrum is assembled by:

    1. Determining the **global spectral regime** from characteristic frequencies,
    2. Computing any **derived break frequencies** required by that regime,
    3. Dispatching to a **regime-specific optimized kernel**,
    4. Applying an overall normalization via the peak flux density.

    For a detailed derivation of the spectral segments and break orderings, see
    :ref:`synchrotron_theory` and :ref:`synchrotron_ssa_theory`.

    Spectral Regimes
    -----------------

    The SED is globally classified into one of several discrete regimes based on
    the ordering of the characteristic frequencies:

    - Injection frequency :math:`\nu_m`,
    - Cooling frequency :math:`\nu_c`,
    - Self-absorption frequency :math:`\nu_a` (computed internally),
    - Maximum synchrotron frequency :math:`\nu_{\max}`.

    Each regime corresponds to a specific ordering (e.g.
    :math:`\nu_a < \nu_c < \nu_m < \nu_{\max}`) and therefore to a unique set of
    spectral slopes. Internally, these regimes are enumerated by
    :class:`~_SynchrotronSSACoolingSEDFunctions`.

    The regime selection is **global** and does not depend on the frequency grid
    used for evaluation.

    SED Parameters
    ------------------

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Peak flux density
                  - :math:`F_{\nu,\mathrm{pk}}`
                  - Flux normalization at the spectral peak
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron frequency of minimum-energy electrons
                * - Cooling frequency
                  - :math:`\nu_c`
                  - Frequency corresponding to the cooling Lorentz factor
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-energy cutoff frequency
                * - (Optional) stratified SSA frequency
                  - :math:`\nu_{ac}`
                  - Transition frequency for stratified SSA regimes

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* and smoothness of the spectrum
            but are not usually directly inferred from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron distribution
                * - Smoothing parameter
                  - :math:`s`
                  - Controls sharpness of spectral breaks
                * - Emission solid angle
                  - :math:`\Omega`
                  - Effective emitting area divided by distance squared
                * - Minimum Lorentz factor
                  - :math:`\gamma_m`
                  - Minimum electron Lorentz factor

        .. tab-item:: Derived parameters (internal)

            These quantities are **not user inputs**, but are computed internally
            from the free parameters and microphysical assumptions.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - SSA frequency
                  - :math:`\nu_a`
                  - Self-absorption break frequency
                * - Cooling regime
                  - —
                  - Fast, slow, or non-cooling classification
                * - Regime identifier
                  - —
                  - Discrete index selecting the SED kernel

    Implementation Notes
    ----------------------

    - All internal calculations are performed in **logarithmic space** for
      numerical stability.
    - No unit checking occurs in optimized methods; units are enforced only
      in the public :meth:`sed` interface.
    - The SSA frequency :math:`\nu_a` is computed self-consistently using
      analytic scalings appropriate to each regime.
    - The implementation assumes isotropic pitch-angle distributions and
      standard synchrotron emissivity expressions.

    Example
    ---------------------

    Compute a synchrotron SED in the slow-cooling regime:

    .. code-block:: python

        from astropy import units as u
        from triceratops.radiation.synchrotron import (
            PowerLaw_Cooling_SSA_SynchrotronSED,
        )

        sed = PowerLaw_Cooling_SSA_SynchrotronSED()

        nu = u.logspace(8, 20, 500, unit="Hz")

        flux = sed.sed(
            nu=nu,
            nu_m=1e12 * u.Hz,
            nu_c=1e15 * u.Hz,
            nu_max=1e18 * u.Hz,
            F_peak=1 * u.mJy,
            p=2.5,
            s=-0.05,
        )

    References
    ----------

    .. footbibliography::
    """

    # ============================================================ #
    # Declare the spectrum functions mapping                       #
    # ============================================================ #
    SPECTRUM_FUNCTIONS = _SynchrotronSSACoolingSEDFunctions

    # ============================================================ #
    # Instantiation                                                #
    # ============================================================ #
    def __init__(self):
        super().__init__()

    # ============================================================ #
    # Regime Management                                            #
    # ============================================================ #
    def _compute_sed_regime(
        self,
        log_F_peak: float,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_omega: float,
        log_gamma_m: float,
    ):
        r"""
        Determine the synchrotron spectral regime and SSA frequency.

        This low-level method classifies the synchrotron spectrum into a **single,
        global spectral regime** based on the ordering of the characteristic
        frequencies and returns both:

        1. A discrete **regime identifier**, and
        2. The corresponding **self-absorption frequency** :math:`\nu_a`
           appropriate to that regime.

        The classification proceeds in two stages:

        1. Determine whether the system is **fast-cooling**, **slow-cooling**, or
           **non-cooling** by comparing :math:`\nu_c` and :math:`\nu_m`.
        2. Within that cooling class, select the specific SSA spectrum by comparing
           the candidate self-absorption frequencies to :math:`\nu_m` and
           :math:`\nu_c`.

        All inputs and outputs are assumed to be in **natural logarithmic space**.

        Parameters
        ----------
        log_F_peak : float
            Natural logarithm of the peak flux density
            :math:`\log F_{\nu,\mathrm{pk}}`.
        log_nu_m : float
            Natural logarithm of the injection frequency
            :math:`\log \nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency
            :math:`\log \nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\log \nu_{\max}`.
        log_omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\log \Omega`.
        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\log \gamma_m`.

        Returns
        -------
        tuple
            A two-element tuple ``(regime, log_nu_a)`` where:

            - ``regime`` is a member of
              :class:`~_SynchrotronSSACoolingSEDFunctions` identifying the spectral
              branch to evaluate.
            - ``log_nu_a`` is the natural logarithm of the self-absorption frequency
              appropriate to that regime.

        Notes
        -----
        - This method performs **no unit checking** and assumes all inputs are valid.
        - The returned regime applies **globally** to the spectrum.
        - Errors are raised if the frequency ordering is inconsistent with any
          supported spectral configuration.
        """
        # Begin by calculating the SSA frequency from the other parameters. This also
        # tells some regime information. 0=Fast cooling, 1=Slow cooling, 2=No cooling.
        log_nu_ssa, cooling_regime = self._compute_log_ssa_frequencies(
            log_F_peak,
            log_nu_m,
            log_nu_c,
            log_nu_max,
            log_omega,
            log_gamma_m,
        )

        # From the cooling regime, we need to determine which specific spectrum we're in.
        if cooling_regime == 0:
            # Fast cooling: S5, S6, or S7
            if log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_5] < log_nu_c:
                # This is the extremely fast cooling case where nu_a < nu_c < nu_m < nu_max.
                # We get the 2, 11/8, 1/3, -1/2, -p/2 spectrum.
                return self.SPECTRUM_FUNCTIONS.SPECTRUM_5, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_5]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_6] > log_nu_m:
                # This is the moderate fast cooling case with extreme absorption nu_c < nu_m < nu_a < nu_max.
                return self.SPECTRUM_FUNCTIONS.SPECTRUM_7, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_6] < log_nu_m:
                # This is the intermediate fast cooling case with nu_c < nu_a < nu_m < nu_max.
                # NOTE: the catch on nu_c above allows us to avoid ambiguity with the extreme fast cooling case.
                return self.SPECTRUM_FUNCTIONS.SPECTRUM_6, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_6]
            else:
                raise RuntimeError(
                    "Could not determine regime in fast cooling scenario:\n"
                    f"  log_nu_c          = {log_nu_c:0.3f}\n"
                    f"  log_nu_m          = {log_nu_m:0.3f}\n"
                    f"  log_nu_max        = {log_nu_max:0.3f}\n"
                    f"  log_nu_a (S5)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_5]:0.3f}\n"
                    f"  log_nu_a (S6)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_6]:0.3f}\n"
                    f"  log_nu_a (S7)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7]:0.3f}\n"
                )

        # Slow cooling: S3 or S4
        elif cooling_regime == 1:
            if log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_3] < log_nu_m:
                # This is the optically thin at peak slow cooling case with nu_a < nu_m < nu_c < nu_max.
                return self.SPECTRUM_FUNCTIONS.SPECTRUM_3, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_3]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_4] < log_nu_c:
                # This is the case where nu_m < nu_a < nu_c < nu_max. Note that we catch nu_a < nu_m above.
                return self.SPECTRUM_FUNCTIONS.SPECTRUM_4, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_4]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7] > log_nu_c:
                # This is the extreme case where nu_m < nu_c < nu_a < nu_max and the cooling break is hidden.
                return self.SPECTRUM_FUNCTIONS.SPECTRUM_7, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7]
            else:
                raise RuntimeError(
                    "Could not determine regime in slow cooling scenario:\n"
                    f"  log_nu_c          = {log_nu_c:0.3f}\n"
                    f"  log_nu_m          = {log_nu_m:0.3f}\n"
                    f"  log_nu_max        = {log_nu_max:0.3f}\n"
                    f"  log_nu_a (S3)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_3]:0.3f}\n"
                    f"  log_nu_a (S4)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_4]:0.3f}\n"
                    f"  log_nu_a (S7)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7]:0.3f}\n"
                )

        # No cooling: S1 or S2
        elif cooling_regime == 2:
            if log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_1] < log_nu_m:
                # This is the optically thin at peak no cooling case with nu_a < nu_m < nu_max.
                return self.SPECTRUM_FUNCTIONS.SPECTRUM_1, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_1]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_2] > log_nu_m:
                # This is the extreme case where nu_m < nu_a < nu_max.
                return self.SPECTRUM_FUNCTIONS.SPECTRUM_2, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_2]
            else:
                raise RuntimeError(
                    "Could not determine regime in no cooling scenario:\n"
                    f"  log_nu_m          = {log_nu_m:0.3f}\n"
                    f"  log_nu_max        = {log_nu_max:0.3f}\n"
                    f"  log_nu_c          = {log_nu_c:0.3f}\n"
                    f"  log_nu_a (S1)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_1]:0.3f}\n"
                    f"  log_nu_a (S2)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_2]:0.3f}\n"
                )

        # Catch failures
        else:
            raise RuntimeError("Unable to determine SSA spectrum regime: unrecognized cooling regime.")

    def determine_sed_regime(
        self,
        F_peak: "_UnitBearingScalarLike",
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        omega: float = 1.0,
        gamma_m: float = 1.0,
    ):
        r"""
        Determine the synchrotron spectral regime from physical parameters.

        This is the **user-facing interface** for regime determination. It performs
        unit validation and coercion, converts all inputs to logarithmic CGS form,
        and dispatches to the optimized internal routine
        :meth:`_compute_sed_regime`.

        This method is intended for **diagnostic and introspection purposes** and
        does not compute the SED itself.

        Parameters
        ----------
        F_peak : quantity
            Peak flux density :math:`F_{\nu,\mathrm{pk}}`.
        nu_m : quantity
            Injection frequency :math:`\nu_m`.
        nu_c : quantity
            Cooling frequency :math:`\nu_c`.
        nu_max : quantity
            Maximum synchrotron frequency :math:`\nu_{\max}`.
        omega : float
            Effective emission solid angle.
        gamma_m : float
            Minimum electron Lorentz factor.

        Returns
        -------
        int
            Integer-encoded synchrotron spectral regime identifier.

        Notes
        -----
        - The returned regime applies **globally** to the SED.
        - This method does not depend on the frequency array ``nu``.
        """
        # Handle units and then coerce to log space
        F_peak = ensure_in_units(F_peak, "erg / (cm**2 * s * Hz)")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")

        # Coerce to log space
        log_F_peak = np.log(F_peak)
        log_nu_m = np.log(nu_m)
        log_nu_c = np.log(nu_c)
        log_nu_max = np.log(nu_max)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        # Dispatch to the low-level regime computation
        regime, _ = self._compute_sed_regime(
            log_F_peak=log_F_peak,
            log_nu_m=log_nu_m,
            log_nu_c=log_nu_c,
            log_nu_max=log_nu_max,
            log_omega=log_omega,
            log_gamma_m=log_gamma_m,
        )
        return regime

    # ============================================================ #
    # SSA Frequency Computation                                    #
    # ============================================================ #
    # In this spectrum, we need to compute a separate nu_a for each of the
    # relevant regimes permitted by the nu_c, nu_m, and nu_max ordering.
    def _compute_log_ssa_frequencies(
        self,
        log_F_peak: float,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_omega: float,
        log_gamma_m: float,
    ) -> tuple[dict[_SynchrotronSSACoolingSEDFunctions, float], int]:
        r"""
        Compute candidate self-absorption frequencies for all relevant regimes.

        This method evaluates analytic expressions for the self-absorption frequency
        :math:`\nu_a` appropriate to each allowed synchrotron spectral regime, given
        the system’s cooling state.

        The calculation is based on standard synchrotron absorption scalings and
        assumes a power-law electron distribution with isotropic pitch angles.

        Parameters
        ----------
        log_F_peak : float
            Natural logarithm of the peak flux density.
        log_nu_m : float
            Natural logarithm of the injection frequency.
        log_nu_c : float
            Natural logarithm of the cooling frequency.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency.
        log_omega : float
            Natural logarithm of the effective emission solid angle.
        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor.

        Returns
        -------
        tuple
            A tuple ``(log_nu_a_dict, cooling_regime)`` where:

            - ``log_nu_a_dict`` maps each candidate spectral regime to its
              corresponding :math:`\log \nu_a`,
            - ``cooling_regime`` is an integer flag:
              ``0`` = fast cooling,
              ``1`` = slow cooling,
              ``2`` = no cooling.

        Notes
        -----
        - Only regimes compatible with the cooling state are included.
        - The returned frequencies are **candidates**; final regime selection
          occurs in :meth:`_compute_sed_regime`.
        """
        # Pre-compute the common factor so that we do not need to recompute it.
        log_Q = log_F_peak - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m

        # We'll want to quickly check if we are fast, slow, or non-cooling. This will
        # narrow down the scenarios we need to consider and the corresponding construction
        # of the relevant SSA frequencies.
        if log_nu_c < log_nu_m:
            # This is fast cooling. We have access to S5, S6, and S7 only.
            return {
                self.SPECTRUM_FUNCTIONS.SPECTRUM_5: (6 * log_Q / 13) + (3 * log_nu_m / 13) - (2 * log_nu_c / 13),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_6: (2 * log_Q / 5) + (log_nu_m / 5),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_7: (2 * log_Q / 5) + (log_nu_m / 5),
            }, 0
        elif (log_nu_m < log_nu_c) and (log_nu_c < log_nu_max):
            # This is slow cooling with nu_c < nu_max. We have access to S3 and S4 and S7 only.
            return {
                self.SPECTRUM_FUNCTIONS.SPECTRUM_3: (6 * log_Q / 13) + (log_nu_m / 13),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_4: (2 * log_Q / 5) + (log_nu_m / 5),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_7: (2 * log_Q / 5) + (log_nu_m / 5),
            }, 1
        else:
            # This is the NO COOLING case. We have access to S1 and S2 only.
            return {
                self.SPECTRUM_FUNCTIONS.SPECTRUM_1: (6 * log_Q / 13) + (log_nu_m / 13),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_2: (2 * log_Q / 5) + (log_nu_m / 5),
            }, 2

    # ============================================================ #
    # Regime-Specific SED Kernel                                   #
    # ============================================================ #
    # noinspection PyCallingNonCallable
    def _log_opt_sed_from_regime(
        self,
        regime: _SynchrotronSSACoolingSEDFunctions,
        log_nu: "_ArrayLike",
        log_nu_m: float,
        log_nu_a: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_peak: float,
        log_nu_ac: float = None,
        p: float = 3.0,
        s: float = -1.0,
    ):
        r"""
        Evaluate the log-space SED for a fixed spectral regime.

        This method dispatches to the **regime-specific numerical kernel**
        corresponding to the supplied regime identifier and applies the overall
        flux normalization.

        All branching logic on spectral shape is confined to this method.

        Parameters
        ----------
        regime : enum
            Spectral regime identifier from
            :class:`~_SynchrotronSSACoolingSEDFunctions`.
        log_nu : array-like
            Natural logarithm of the evaluation frequencies.
        log_nu_m, log_nu_a, log_nu_c, log_nu_max : float
            Natural logarithms of the characteristic frequencies.
        log_F_peak : float
            Natural logarithm of the peak flux density.
        log_nu_ac : float, optional
            Natural logarithm of the stratified SSA transition frequency.
        p : float
            Electron power-law index.
        s : float
            Smoothing parameter for spectral breaks.

        Returns
        -------
        array-like
            Natural logarithm of the SED evaluated at ``log_nu``.

        Notes
        -----
        - This method assumes the regime is **already validated**.
        - All SED kernels are evaluated in log-space for numerical stability.
        """
        # Given that we have the regime, we can dispatch to the appropriate function.
        if regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_1:
            return regime(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s) + log_F_peak
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_2:
            return regime(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s) + log_F_peak
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_3:
            return regime(log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_max, p, s) + log_F_peak
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_4:
            return regime(log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_max, p, s) + log_F_peak
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_5:
            return regime(log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_ac or log_nu_a, log_nu_max, p, s) + log_F_peak
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_6:
            return regime(log_nu, log_nu_m, log_nu_a, log_nu_ac or log_nu_a, log_nu_max, p, s) + log_F_peak
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_7:
            return regime(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s) + log_F_peak
        else:
            raise RuntimeError("Unrecognized regime in _log_opt_sed_from_regime.")

    def _log_opt_sed(
        self,
        log_nu: "_ArrayLike",
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_peak: float,
        log_omega: float,
        log_gamma_m: float,
        log_nu_ac: float = None,
        p: float = 3.0,
        s: float = -1.0,
    ):
        r"""
        Optimized log-space SED evaluation with regime determination.

        This method orchestrates the full SED evaluation by:

        1. Determining the global spectral regime,
        2. Computing the corresponding self-absorption frequency,
        3. Dispatching to the appropriate regime-specific kernel.

        All inputs are assumed to be in **natural logarithmic CGS units**.

        Returns
        -------
        array-like
            Natural logarithm of the synchrotron SED evaluated at ``log_nu``.
        """
        # Determine the regime
        regime, log_nu_a = self._compute_sed_regime(
            log_F_peak,
            log_nu_m,
            log_nu_c,
            log_nu_max,
            log_omega,
            log_gamma_m,
        )
        # Dispatch to the appropriate regime function
        return self._log_opt_sed_from_regime(
            regime,
            log_nu,
            log_nu_m,
            log_nu_a,
            log_nu_c,
            log_nu_max,
            log_F_peak,
            log_nu_ac,
            p,
            s,
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        F_peak: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        nu_ac: "_UnitBearingScalarLike" = None,
        omega: float = 4 * np.pi,
        gamma_m: float = 1,
        p: float = 3,
        s: float = -1,
    ):
        r"""
        Evaluate the synchrotron spectral energy distribution.

        This is the **primary user-facing interface** for computing the synchrotron
        SED. It performs unit validation, converts inputs to logarithmic CGS form,
        and dispatches to the optimized internal implementation.

        Parameters
        ----------
        nu : quantity or array-like
            Frequencies at which to evaluate the SED.
        nu_m, nu_c, nu_max : quantity
            Injection, cooling, and maximum synchrotron frequencies.
        F_peak : quantity
            Peak flux density normalization.
        nu_ac : quantity, optional
            Stratified SSA transition frequency.
        omega : float
            Effective emission solid angle.
        gamma_m : float
            Minimum electron Lorentz factor.
        p : float
            Electron power-law index.
        s : float
            Spectral smoothing parameter.

        Returns
        -------
        astropy.units.Quantity
            Flux density :math:`F_\nu` evaluated at ``nu``.

        Notes
        -----
        - This is the **only method** that enforces units.
        - All internal calculations are performed in log-space.
        """
        # Enforce units on each of the inputs
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        F_peak = ensure_in_units(F_peak, "erg s^-1 cm^-2 Hz^-1")
        nu_max = ensure_in_units(nu_max, "Hz")
        nu_ac = ensure_in_units(nu_ac, "Hz") if nu_ac is not None else None

        # Dispatch to the optimized log-space SED
        log_sed = self._log_opt_sed(
            np.log(nu),
            np.log(nu_m),
            np.log(nu_c),
            np.log(nu_max),
            np.log(F_peak),
            np.log(omega),
            np.log(gamma_m),
            log_nu_ac=np.log(nu_ac) if nu_ac is not None else None,
            p=p,
            s=s,
        )

        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)


class PowerLaw_Cooling_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Optically thin synchrotron spectral energy distribution with radiative cooling.

    This class implements the canonical synchrotron SED produced by a
    power-law electron energy distribution subject to radiative cooling,
    **without synchrotron self-absorption (SSA)**. It is appropriate for
    optically thin emission regions where the SSA turnover lies below the
    lowest frequency of interest.

    The spectrum is constructed using smoothed broken power laws (SBPLs)
    and supports the full set of cooling regimes encountered in standard
    synchrotron theory:

    .. math::

        \nu_c < \nu_m \quad \text{(fast cooling)}

    .. math::

        \nu_m < \nu_c < \nu_{\max} \quad \text{(slow cooling)}

    .. math::

        \nu_c > \nu_{\max} \quad \text{(effectively non-cooling)}

    Regime selection is automatic and based entirely on the ordering of
    the characteristic break frequencies.

    See :ref:`synchrotron_theory` for a detailed derivation of the spectral
    slopes and normalization conventions.

    Physical assumptions
    ----------------------

    This SED assumes:

    - A power-law electron energy distribution

      .. math::

          \frac{dN}{d\gamma} \propto \gamma^{-p},
          \qquad \gamma \ge \gamma_{\min}

    Model parameters
    ----------------

    .. tab-set::

        .. tab-item:: Free parameters

            .. list-table::
                :header-rows: 1
                :widths: 20 40

                * - Parameter
                  - Description
                * - ``nu_m``
                  - Injection (minimum electron) synchrotron frequency
                * - ``nu_c``
                  - Cooling break frequency
                * - ``F_peak``
                  - Flux density normalization at the spectral peak
                * - ``nu_max``
                  - Maximum synchrotron frequency cutoff

        .. tab-item:: Hyper-parameters

            .. list-table::
                :header-rows: 1
                :widths: 20 40

                * - Parameter
                  - Description
                * - ``p``
                  - Electron energy power-law index
                * - ``s``
                  - SBPL smoothness parameter (``s < 0`` for physical behavior)

        .. tab-item:: Derived quantities

            .. list-table::
                :header-rows: 1
                :widths: 20 40

                * - Quantity
                  - Description
                * - Cooling regime
                  - Determined by ordering of ``nu_m``, ``nu_c``, ``nu_max``

    Cooling regimes and spectral slopes
    ------------------------------------

    **Fast cooling** (:math:`\nu_c < \nu_m`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_c \
            \nu^{-1/2}, & \nu_c < \nu < \nu_m \
            \nu^{-p/2}, & \nu > \nu_m
        \end{cases}

    **Slow cooling** (:math:`\nu_m < \nu_c < \nu_{\max}`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_m \
            \nu^{-(p-1)/2}, & \nu_m < \nu < \nu_c \
            \nu^{-p/2}, & \nu > \nu_c
        \end{cases}

    **Non-cooling limit** (:math:`\nu_c > \nu_{\max}`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_m \
            \nu^{-(p-1)/2}, & \nu > \nu_m
        \end{cases}

    Implementation notes
    ----------------------

    - All internal calculations are performed in log-space for numerical
      stability.
    - Regime selection is global and does not depend on the frequency array.
    - This class does **not** compute or require an SSA frequency.
    - The SED normalization is applied multiplicatively via ``F_peak``.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.synchrotron import (
            PowerLaw_Cooling_SynchrotronSED
        )

        sed = PowerLaw_Cooling_SynchrotronSED()

        nu = np.logspace(8, 20, 1000) * u.Hz

        Fnu = sed.sed(
            nu,
            nu_m = 1e12 * u.Hz,
            nu_c = 1e15 * u.Hz,
            nu_max = 1e19 * u.Hz,
            F_peak = 1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            p = 2.5,
            s = -0.05,
        )


    See Also
    --------
    - :class:`PowerLaw_SSA_SynchrotronSED`
    - :class:`PowerLaw_Cooling_SSA_SynchrotronSED`
    - :ref:`synchrotron_theory`
    """

    SPECTRUM_FUNCTIONS = _SynchrotronCoolingSEDFunctions

    # ------------------------------------------------------------ #
    # Regime determination                                        #
    # ------------------------------------------------------------ #
    def _compute_sed_regime(self, log_nu_m: float, log_nu_c: float, log_nu_max: float):
        r"""
        Determine the optically thin synchrotron cooling regime.

        This low-level method classifies the synchrotron spectrum based on the
        ordering of the characteristic frequencies in log-space:

        .. math::

            \nu_m \quad \text{(injection frequency)}, \qquad
            \nu_c \quad \text{(cooling frequency)}, \qquad
            \nu_{\max} \quad \text{(high-frequency cutoff)}.

        The regime determination is **global** and applies to the entire SED;
        it does not depend on the evaluation frequency array ``nu``.

        Parameters
        ----------
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency :math:`\nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.

        Returns
        -------
        regime : enum-like
            A member of :attr:`SPECTRUM_FUNCTIONS` identifying the cooling regime.
        metadata : dict
            Empty dictionary (present for API consistency with multi-regime SEDs
            that require auxiliary derived parameters).

        Notes
        -----
        The regimes correspond to the following orderings:

        - **Fast cooling**: :math:`\nu_c < \nu_m`
        - **Slow cooling**: :math:`\nu_m < \nu_c < \nu_{\max}`
        - **Non-cooling**: :math:`\nu_c > \nu_{\max}`

        This method assumes that all inputs are already in log-space and does
        not perform unit validation.
        """
        if log_nu_c < log_nu_m:
            return self.SPECTRUM_FUNCTIONS.SPECTRUM_1, {}
        elif log_nu_m < log_nu_c < log_nu_max:
            return self.SPECTRUM_FUNCTIONS.SPECTRUM_2, {}
        elif log_nu_c > log_nu_max:
            return self.SPECTRUM_FUNCTIONS.SPECTRUM_3, {}
        else:
            raise RuntimeError("Could not determine cooling regime.")

    def determine_sed_regime(
        self, nu_m: "_UnitBearingScalarLike", nu_c: "_UnitBearingScalarLike", nu_max: "_UnitBearingScalarLike" = np.inf
    ):
        r"""
        Public interface for determining the synchrotron cooling regime.

        This method provides a user-facing wrapper around
        :meth:`_compute_sed_regime`. It handles unit validation and conversion
        to log-space before performing regime classification.

        Parameters
        ----------
        nu_m : quantity-like
            Injection frequency :math:`\nu_m`.
        nu_c : quantity-like
            Cooling frequency :math:`\nu_c`.
        nu_max : quantity-like, optional
            Maximum synchrotron frequency :math:`\nu_{\max}`.
            Defaults to :math:`\infty`.

        Returns
        -------
        regime : enum-like
            Cooling regime identifier corresponding to one of the
            optically thin synchrotron spectra.

        Notes
        -----
        - This method does **not** depend on the frequency array used to
          evaluate the SED.
        - The returned regime must be consistent with the spectrum produced
          by :meth:`sed` for the same parameters.
        """
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")

        regime, _ = self._compute_sed_regime(
            log_nu_m=np.log(nu_m),
            log_nu_c=np.log(nu_c),
            log_nu_max=np.log(nu_max),
        )
        return regime

    # ------------------------------------------------------------ #
    # Regime-specific kernel                                       #
    # ------------------------------------------------------------ #
    def _log_opt_sed_from_regime(
        self,
        log_nu: float,
        regime: Callable,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_peak: float,
        p: float,
        s: float,
    ):
        r"""
        Evaluate the log-space synchrotron SED for a fixed cooling regime.

        This method implements the **numerical kernel** for computing the
        synchrotron SED once the cooling regime has been determined. All
        branching on the regime identifier occurs here.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the frequencies at which to evaluate the SED.
        regime : enum-like
            Cooling regime identifier returned by
            :meth:`_compute_sed_regime`.
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency :math:`\nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.
        log_F_peak : float
            Natural logarithm of the peak flux density normalization.
        p : float
            Electron energy power-law index.
        s : float
            SBPL smoothness parameter.

        Returns
        -------
        log_sed : array-like
            Natural logarithm of the flux density
            :math:`\log F_\nu` evaluated at ``log_nu``.

        Notes
        -----
        - The returned spectrum is **not normalized** until ``log_F_peak``
          is added.
        - This method assumes an optically thin emitting region and does not
          include synchrotron self-absorption.
        - No unit validation is performed.
        """
        if regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_1:
            log_sed = regime(log_nu, log_nu_m, log_nu_c, log_nu_max, p, s)
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_2:
            log_sed = regime(log_nu, log_nu_m, log_nu_c, log_nu_max, p, s)
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_3:
            log_sed = regime(log_nu, log_nu_m, log_nu_max, p, s)
        else:
            raise RuntimeError("Unrecognized cooling regime.")

        return log_sed + log_F_peak

    # ------------------------------------------------------------ #
    # User-facing API                                             #
    # ------------------------------------------------------------ #
    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        F_peak: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        p: float = 2.5,
        s: float = -1.0,
    ):
        r"""
        Evaluate the optically thin synchrotron spectral energy distribution.

        This is the primary user-facing method for computing the synchrotron
        flux density :math:`F_\nu` from a cooling power-law electron population
        without synchrotron self-absorption.

        Parameters
        ----------
        nu : quantity-like or array-like
            Frequencies at which to evaluate the SED.
        nu_m : quantity-like
            Injection frequency :math:`\nu_m`.
        nu_c : quantity-like
            Cooling frequency :math:`\nu_c`.
        F_peak : quantity-like
            Peak flux density normalization.
        nu_max : quantity-like, optional
            Maximum synchrotron frequency :math:`\nu_{\max}`.
            Defaults to :math:`\infty`.
        p : float, optional
            Electron energy power-law index. Default is ``2.5``.
        s : float, optional
            SBPL smoothness parameter. Must be negative for physical behavior.
            Default is ``-1.0``.

        Returns
        -------
        flux : astropy.units.Quantity
            Flux density :math:`F_\nu` evaluated at ``nu``.

        Notes
        -----
        - Units are validated and coerced internally.
        - Regime selection is automatic and based on the ordering of
          :math:`\nu_m`, :math:`\nu_c`, and :math:`\nu_{\max}`.
        - The returned spectrum is continuous and differentiable due to
          SBPL smoothing.
        """
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")
        F_peak = ensure_in_units(F_peak, "erg s^-1 cm^-2 Hz^-1")

        log_sed = self._log_opt_sed(
            np.log(nu),
            log_nu_m=np.log(nu_m),
            log_nu_c=np.log(nu_c),
            log_nu_max=np.log(nu_max),
            log_F_peak=np.log(F_peak),
            p=p,
            s=s,
        )
        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)


class PowerLaw_SSA_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Synchrotron spectral energy distribution with synchrotron self-absorption (SSA) and **no radiative cooling**.

    This class implements the optically thick–to–optically thin synchrotron
    spectrum produced by a power-law electron population in the absence of
    significant radiative cooling. It is appropriate when the cooling frequency
    lies well above the maximum synchrotron frequency of interest:

    .. math::

        \nu_c \gg \nu_{\max}.

    The spectrum includes:

    - Synchrotron self-absorption (SSA),
    - Optically thick Rayleigh–Jeans and SSA power-law segments,
    - Optically thin synchrotron emission,
    - A high-frequency exponential cutoff.

    The SED is assembled using **log-space smoothed broken power laws (SFBPLs)**,
    following the same numerical and conceptual framework as
    :class:`PowerLaw_Cooling_SSA_SynchrotronSED`, but restricted to the
    non-cooling limit.

    See :ref:`synchrotron_ssa_theory` for theoretical details.

    Spectral regimes
    ----------------

    Two spectral configurations are supported, depending on the ordering of the
    self-absorption and injection frequencies:

    - **SSA below the peak**: :math:`\nu_a < \nu_m`
    - **SSA above the peak**: :math:`\nu_m < \nu_a`

    The appropriate regime is selected automatically and globally.

    Parameters
    ----------

    .. tab-set::

        .. tab-item:: Free parameters

            .. list-table::
                :header-rows: 1
                :widths: 25 60

                * - Parameter
                  - Description
                * - ``nu_m``
                  - Injection (minimum-electron) synchrotron frequency
                * - ``F_peak``
                  - Flux density normalization at the spectral peak
                * - ``nu_max``
                  - Maximum synchrotron frequency cutoff

        .. tab-item:: Hyper-parameters

            .. list-table::
                :header-rows: 1
                :widths: 25 60

                * - Parameter
                  - Description
                * - ``p``
                  - Electron energy power-law index
                * - ``s``
                  - SFBPL smoothness parameter (``s < 0``)
                * - ``omega``
                  - Effective emission solid angle
                * - ``gamma_m``
                  - Minimum electron Lorentz factor

        .. tab-item:: Derived parameters (internal)

            .. list-table::
                :header-rows: 1
                :widths: 25 60

                * - Quantity
                  - Description
                * - ``nu_a``
                  - Self-absorption frequency (computed internally)
                * - Regime identifier
                  - Discrete spectral branch selector

    Implementation notes
    --------------------

    - The SSA frequency :math:`\nu_a` is computed self-consistently using
      analytic synchrotron absorption scalings.
    - All calculations are performed in **logarithmic space**.
    - No unit checking is performed in optimized methods.
    - Pitch-angle distributions are assumed isotropic.

    Example
    --------

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.synchrotron import (
            PowerLaw_SSA_SynchrotronSED,
        )

        sed = PowerLaw_SSA_SynchrotronSED()

        nu = np.logspace(8, 14, 500) * u.Hz

        Fnu = sed.sed(
            nu,
            nu_m=1e11 * u.Hz,
            F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            p=2.5,
            s=-0.05,
        )
    """

    SPECTRUM_FUNCTIONS = _SynchrotronSSASEDFunctions

    # ============================================================ #
    # Regime determination                                        #
    # ============================================================ #
    def _compute_sed_regime(
        self,
        log_nu_m: float,
        log_nu_a: float,
        **_,
    ):
        r"""Determine the SSA spectral regime in the absence of cooling."""
        if log_nu_a < log_nu_m:
            return self.SPECTRUM_FUNCTIONS.SPECTRUM_1, {}
        else:
            return self.SPECTRUM_FUNCTIONS.SPECTRUM_2, {}

    def determine_sed_regime(
        self,
        nu_m: "_UnitBearingScalarLike",
        F_peak: "_UnitBearingScalarLike",
        omega: float = 4 * np.pi,
        gamma_m: float = 1.0,
    ):
        r"""
        Determine the SSA synchrotron regime from physical parameters.

        This method computes the self-absorption frequency internally and
        returns the corresponding spectral regime identifier.
        """
        nu_m = ensure_in_units(nu_m, "Hz")
        F_peak = ensure_in_units(F_peak, "erg s^-1 cm^-2 Hz^-1")

        log_nu_m = np.log(nu_m)
        log_F_peak = np.log(F_peak)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        log_nu_a = self._compute_log_nu_a(log_F_peak, log_nu_m, log_omega, log_gamma_m)

        regime, _ = self._compute_sed_regime(
            log_nu_m=log_nu_m,
            log_nu_a=log_nu_a,
        )
        return regime

    # ============================================================ #
    # SSA frequency computation                                   #
    # ============================================================ #
    def _compute_log_nu_a(
        self,
        log_F_peak: float,
        log_nu_m: float,
        log_omega: float,
        log_gamma_m: float,
    ) -> float:
        r"""
        Compute the self-absorption frequency in the non-cooling limit.

        This uses the standard analytic scaling

        .. math::

            \nu_a \propto
            \left(
                \frac{F_{\nu,\mathrm{pk}}}
                     {m_e c^2 \, \Omega \, \gamma_m}
            \right)^{2/5}
            \nu_m^{1/5}

        expressed entirely in logarithmic form.
        """
        log_Q = log_F_peak - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m

        return (2 * log_Q / 5) + (log_nu_m / 5)

    # ============================================================ #
    # Regime-specific kernel                                      #
    # ============================================================ #
    def _log_opt_sed_from_regime(
        self,
        log_nu,
        regime,
        log_nu_m,
        log_nu_a,
        log_nu_max,
        log_F_peak,
        p,
        s,
    ):
        if regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_1:
            log_sed = _log_powerlaw_sbpl_sed_ssa_1(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_2:
            log_sed = _log_powerlaw_sbpl_sed_ssa_2(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
        else:
            raise RuntimeError("Unrecognized SSA regime.")

        return log_sed + log_F_peak

    # ============================================================ #
    # User-facing API                                             #
    # ============================================================ #
    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_m: "_UnitBearingScalarLike",
        F_peak: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        omega: float = 4 * np.pi,
        gamma_m: float = 1.0,
        p: float = 2.5,
        s: float = -1.0,
    ):
        r"""Evaluate the synchrotron SED with self-absorption and no cooling."""
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")
        F_peak = ensure_in_units(F_peak, "erg s^-1 cm^-2 Hz^-1")

        log_nu = np.log(nu)
        log_nu_m = np.log(nu_m)
        log_nu_max = np.log(nu_max)
        log_F_peak = np.log(F_peak)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        log_nu_a = self._compute_log_nu_a(log_F_peak, log_nu_m, log_omega, log_gamma_m)

        log_sed = self._log_opt_sed_from_regime(
            *self._compute_sed_regime(log_nu_m, log_nu_a),
            log_nu,
            log_nu_m,
            log_nu_a,
            log_nu_max,
            log_F_peak,
            p,
            s,
        )

        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)


class SSA_SED_PowerLaw(SynchrotronSED):
    r"""
    Synchrotron self-absorbed (SSA) broken power-law spectral energy distribution.

    This class implements a phenomenological synchrotron spectral energy
    distribution characterized by a single smooth spectral break arising from
    synchrotron self-absorption (SSA). Below the break frequency, the spectrum is
    optically thick and follows a power-law slope of :math:`+5/2`; above the
    break, the spectrum is optically thin and follows a power-law slope of
    :math:`-(p-1)/2`, where :math:`p` is the power-law index of the electron
    energy distribution.

    The SED is implemented as a **smoothly broken power law**, ensuring numerical
    stability and differentiability across the SSA turnover. This makes the
    class well-suited for use in parameter inference, optimization, and
    likelihood-based modeling.

    In addition to forward evaluation of the SED, this class implements
    **analytic closure relations** that allow inversion between phenomenological
    SED parameters (the SSA break frequency and peak flux) and underlying
    physical properties of the emitting region (magnetic field strength and
    radius). These closure relations follow the formalism presented in
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    This class is intended for modeling radio synchrotron emission from
    approximately homogeneous emitting regions, such as those encountered in:

    - supernova radio afterglows,
    - compact object-powered transients,
    - synchrotron-emitting shells with a single dominant SSA turnover.

    No assumptions are made about time evolution, shock dynamics, or cooling
    regimes beyond those implicit in the broken power-law description.

    Notes
    -----
    - The break frequency :math:`\nu_{\rm brk}` is defined as the frequency at
      which the optically thick and optically thin asymptotic power laws
      intersect.
    - The closure relations implemented here assume a power-law electron energy
      distribution with finite bounds
      :math:`\gamma_{\rm min} \le \gamma \le \gamma_{\rm max}`.
    - The special case :math:`p = 2` is not supported due to logarithmic
      divergences in the electron energy integral and must be handled
      separately.

    See Also
    --------
    SynchrotronSED
        Abstract base class defining the interface for all synchrotron SED
        implementations.

    References
    ----------
    .. footbibliography::
    """

    # ================================================= #
    # Instantiation                                     #
    # ================================================= #
    def __init__(self):
        r"""Instantiate the SSA SED object."""
        # There are no class-wide constants to pre-compute for this SED.
        super().__init__()

    # ================================================ #
    # SED Function Implementation                      #
    # ================================================ #
    def _log_opt_sed(
        self,
        nu: "_ArrayLike",
        nu_brk: float,
        F_nu_brk: float,
        p: float,
        s: float,
    ) -> "_ArrayLike":
        r"""
        Low-level optimized SSA broken power-law SED.

        This method implements the synchrotron self-absorbed (SSA) broken power-law SED
        in a performance-optimized manner. It assumes that all inputs are provided as
        dimensionless scalars or NumPy arrays in CGS units. No unit validation or safety
        checks are performed.

        Parameters
        ----------
        nu: float or array-like
            Frequency at which to evaluate the SED (in Hz-equivalent CGS).
        nu_brk: float
            Break frequency (Hz-equivalent CGS).
        F_nu_brk: float
            Flux density at the break frequency (CGS units).
        p: float
            Power-law index of the electron energy distribution.
        s: float
            Smoothing parameter for the break.

        Returns
        -------
        float or array-like
            The computed SED value at the specified frequency. The output of this SED
            is in CGS units of erg s^-1 cm^-2 Hz^-1.
        """
        return smoothed_BPL(
            nu,
            F_nu_brk,
            nu_brk,
            -(p - 1) / 2,  # optically thin index
            (5 / 2),  # optically thick index
            s,  # smoothing parameter
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_brk: "_UnitBearingArrayLike",
        F_nu_brk: "_UnitBearingArrayLike",
        p: float,
        s: float,
    ):
        r"""
        Compute the synchrotron self-absorbed (SSA) broken power-law SED.

        This is the user-facing interface for the SSA broken power-law spectral
        energy distribution. It validates units, coerces inputs to CGS, and
        dispatches to the optimized low-level backend implementation
        :meth:`_opt_sed`.

        Parameters
        ----------
        nu : float, array-like, or astropy.units.Quantity
            Frequency at which to evaluate the SED. Default units are Hz.
        nu_brk : float or astropy.units.Quantity
            Break frequency of the SED. Default units are Hz.
        F_nu_brk : float or astropy.units.Quantity
            Flux density at the break frequency. Default units are
            erg s^-1 cm^-2 Hz^-1.
        p : float
            Power-law index of the electron energy distribution.
        s : float
            Smoothing parameter controlling the sharpness of the spectral break.

        Returns
        -------
        astropy.units.Quantity
            Flux density evaluated at ``nu`` with units of
            erg s^-1 cm^-2 Hz^-1.
        """
        # --- Unit validation and coercion --- #
        nu = ensure_in_units(nu, u.Hz)
        nu_brk = ensure_in_units(nu_brk, u.Hz)
        F_nu_brk = ensure_in_units(F_nu_brk, u.erg / u.s / u.cm**2 / u.Hz)

        # --- Call optimized backend --- #
        F_nu_cgs = self._log_opt_sed(
            nu=nu,
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            p=p,
            s=s,
        )

        return F_nu_cgs * u.erg / u.s / u.cm**2 / u.Hz

    # ================================================ #
    # Closure Relations.                               #
    # ================================================ #
    # For this SED, we implement the closure relations to go from
    # the phenomenological parameters (nu_brk, F_nu_brk) to the physical
    # parameters (B, R) and vice versa. This is implemented exactly as
    # done in DeMarchi+22.
    def _opt_from_params_to_physics(
        self,
        nu_brk: Union[float, np.ndarray],
        F_nu_brk: Union[float, np.ndarray],
        distance: Union[float, np.ndarray],
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        r"""
        Compute the magnetic field and radius of the emitting region from a broken power-law synchrotron SED.

        Parameters
        ----------
        nu_brk: float or array-like
            The break frequency :math:`\nu_{\rm brk}` where the synchrotron SED transitions from
            optically thick to optically thin. This should be provided in GHz. May be provided either
            as a scalar float (for a single SED) or as a 1-D array (for multiple SEDs).
        F_nu_brk: float or array-like
            The flux density :math:`F_{\nu_{\rm brk}}` at the break frequency. This should be provided
            in Jansky (Jy). May be provided either as a scalar float (for a single SED) or as a 1-D array (for multiple
            SEDs).
        distance: float or array-like
            The distance to the source in Megaparsecs (Mpc). May be provided either as a scalar float (for a single SED)
            or as a 1-D array (for multiple SEDs).
        p: float or array-like
            The power-law index :math:`p` of the electron energy distribution. By default, this is 3.0. If provided as a
            float, the value is used for all SEDs. If provided as an array, its shape must be compatible with that of
            ``nu_brk``.

            .. warning::

                This function does not handle the case where :math:`p = 2` due to singularities in the underlying
                equations.
                Users must ensure that :math:`p` is not equal to 2 when calling this function.

        f: float or array-like
            The filling factor :math:`f` of the emitting region. Default is ``0.5``. If provided as a float, the value
            is
            used for all SEDs. If provided as an array, its shape must be compatible with that of ``nu_brk``.
        theta: float or array-like
            The pitch angle :math:`\theta` between the magnetic field and the line of sight, in radians.
            Default is ``pi/2`` (i.e., perpendicular). If provided as a float, the value is used for all SEDs.
            If provided as an array, its shape must be compatible with that of ``nu_brk``.
        epsilon_B: float or array-like
            The fraction of post-shock energy in magnetic fields, :math:`\epsilon_B`.
        epsilon_E: float or array-like
            The fraction of post-shock energy in relativistic electrons, :math:`\epsilon_E`.
        gamma_min: float or array-like
            The minimum Lorentz factor :math:`\gamma_{\rm min}` of the electron energy.
        gamma_max: float or array-like
            The maximum Lorentz factor :math:`\gamma_{\rm max}` of the electron energy.

        Returns
        -------
        B: float or array-like
            The computed magnetic field strength :math:`B` in Gauss. The shape matches that of the input parameters.
        R: float or array-like
            The computed radius of the emitting region :math:`R` in cm. The shape matches that of the input parameters.

        Notes
        -----
        See notes in the user-facing wrapper function `compute_BR_from_BPL` for details on the
        underlying physics and assumptions.

        Effective 1/19/26: We have modified this to work exclusively in log space due to issues with floating point
        truncation errors. Catastrophic cancellation caused significant loss of accuracy.
        """
        # Validate input parameters. We do NOT check explicitly for shape correctness of the arrays, instead opting
        # to allow an error to rise naturally if the shapes are incompatible during computation. We do want to ensure
        # that all of the constants are cast properly for array operations and masking.
        p, gamma_min, gamma_max = (
            np.asarray(p, dtype="f8"),
            np.asarray(gamma_min, dtype="f8"),
            np.asarray(gamma_max, dtype="f8"),
        )

        # Obtain the synchrotron coefficients relevant for this scenario. We need c_1,c_5, and c_6.
        c_5 = compute_c5_parameter(p)  # Should match shape of p.
        c_6 = compute_c6_parameter(p)  # Should match shape of p.
        c_1 = c_1_cgs  # Scalar constant.

        # Construct the array for delta. See the documentation notes for details on the procedure here / physical
        # motivation. To do this efficiently, we pre-allocate the array and then fill in values based on the conditions.
        # Because we pre-allocate with ones, we only need to fill in values where p < 2. In THIS IMPLEMENTATION, we
        # ignore
        # cases where p == 2 for efficiency; these should be screened for upstream if needed.
        delta = np.ones_like(p)
        _p_lt_2_mask = p < 2
        delta[_p_lt_2_mask] = (gamma_max[_p_lt_2_mask] / gamma_min[_p_lt_2_mask]) ** (
            2 - p[_p_lt_2_mask]
        ) - 1  # Fill in values where p < 2.

        # Construct the "p_norm" term used in the B and R calculations. This allows us to handle the two branches
        # of the solution (p < 2 and p > 2) in a unified way.
        # Note that we take the absolute value here to avoid issues with negative bases and fractional exponents.
        # The p == 2 case is handled upstream.
        p_norm = np.abs(p - 2.0)

        # Compute the electron energy floor using the gamma_min parameter and the standard formula.
        E_l = electron_rest_energy_cgs * gamma_min

        # Normalize unit bearing values and convert them to logarithmic quantities for
        # numerical stability.
        _log_nu_brk = np.log(nu_brk / 5)
        _log_F_nu_brk = np.log(F_nu_brk)
        _log_distance = np.log(distance)
        _log_c1 = np.log(c_1)
        _log_E_l = np.log(E_l)
        _log_delta = np.log(delta)
        _log_sin_theta = np.log(np.sin(theta))
        _log_p_norm = np.log(p_norm)

        # Compute the magnetic field following equation (16) of DeMarchi+22. We break this into
        # components for clarity. The operation should be heavily CPU bound, so this should not have any impact
        # on optimization.
        #
        # Here nu_brk is in GHz, E_l is in erg, distance is in Mpc, F_nu_brk is in Jy, and B will be in Gauss.
        _log_B_coeff = 21.6396 + _log_nu_brk - _log_c1
        _log_B_num = (
            -51.41402455
            + ((4 - 2 * p) * _log_E_l)
            + (2 * _log_delta)
            + (2 * np.log(epsilon_B / epsilon_E))
            + np.log(c_5)
            + ((1 / 2) * (-5 - 2 * p) * _log_sin_theta)
        )
        _log_B_denom = 2 * _log_p_norm + 2 * _log_distance + (2 * (_log_F_nu_brk - np.log(0.5))) + (3 * np.log(c_6))
        _log_B = _log_B_coeff + (2 / (13 + 2 * p)) * (_log_B_num - _log_B_denom)

        # Compute the radius following equation (17) of DeMarchi+22. We break this into parts as well on the same
        # basis as above.
        _log_R_coeff = -21.63956 + _log_c1 + (-1 * _log_nu_brk)
        _log_R_t1 = np.log(12 * epsilon_B) + (-6 - p) * np.log(c_5) + (5 + p) * np.log(c_6)
        _log_R_t2 = (6 + p) * 59.818022 + 2 * _log_sin_theta + (-5 - p) * np.log(np.pi) + (12 + 2 * p) * _log_distance
        _log_R_t3 = (2 - p) * _log_E_l + (6 + p) * _log_F_nu_brk
        _log_R_t4 = -1 * (np.log(epsilon_E) + _log_p_norm + np.log(f / 0.5))
        _log_R = _log_R_coeff + (1 / (13 + 2 * p)) * (_log_R_t1 + _log_R_t2 + _log_R_t3 + _log_R_t4)
        return np.exp(_log_B), np.exp(_log_R)

    def from_params_to_physics(
        self,
        nu_brk: Union[float, np.ndarray, u.Quantity],
        F_nu_brk: Union[float, np.ndarray, u.Quantity],
        distance: Union[float, np.ndarray, u.Quantity],
        *,
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the magnetic field strength and emitting radius from a broken power-law SED.

        This function provides a **user-facing, unit-aware interface** for computing
        the physical properties of a synchrotron-emitting region whose radio spectrum
        exhibits a turnover due to synchrotron self-absorption (SSA). Internally, it
        wraps a low-level optimized routine that implements the analytic inversion
        described in :footcite:t:`demarchiRadioAnalysisSN2004C2022` (DM22).

        The function accepts scalar values, NumPy arrays, or Astropy ``Quantity`` objects
        and performs the necessary unit coercion, validation, and shape checking before
        dispatching to the optimized backend.

        Parameters
        ----------
        nu_brk : float, array-like, or astropy.units.Quantity
            The SSA break (turnover) frequency :math:`\nu_{\rm brk}` separating the
            optically thick and optically thin synchrotron regimes. Default units are GHz, but may
            be overridden by providing ``nu_brk`` as a :class:`astropy.units.Quantity` object. May be
            provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs).

        F_nu_brk : float, array-like, or astropy.units.Quantity
            The flux density :math:`F_{\nu_{\rm brk}}` at the break frequency. Default units
            are Jansky (Jy), but may be overridden by providing ``F_nu_brk`` as a
            :class:`astropy.units.Quantity` object. May be provided as either a scalar
            (for a single SED) or a 1-D array (for multiple SEDs). If provided as an array, shape
            must be compatible with that of ``nu_brk``.

        distance : float, array-like, or astropy.units.Quantity
            The (luminosity) distance to the source. By default, units are Megaparsecs (Mpc),
            but may be overridden by providing ``distance`` as a :class:`astropy.units.Quantity` object.
            May be provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs). If provided
            as an array, shape must be compatible with that of ``nu_brk``.

        p : float or array-like, optional
            The power-law index :math:`p` of the electron Lorentz factor distribution,
            defined by

            .. math::

                N(\Gamma)\, d\Gamma = K_e \Gamma^{-p}\, d\Gamma.

            Default is ``3.0``.

            .. warning::

                This function **does not support** the case :math:`p = 2`, for which
                the electron energy integral is logarithmically divergent and requires
                a separate analytic treatment. If any value of ``p`` is exactly equal
                to 2, a ``ValueError`` is raised.

        f : float or array-like, optional
            Volume filling factor of the synchrotron-emitting region. Default is ``0.5``.

        theta : float or array-like, optional
            Pitch angle :math:`\theta` between the magnetic field and the electron
            velocity, in radians. Default is ``\pi/2`` (isotropic average).

        epsilon_B : float or array-like, optional
            Fraction of post-shock internal energy in magnetic fields,
            :math:`\epsilon_B`. Default is ``0.1``.

        epsilon_E : float or array-like, optional
            Fraction of post-shock internal energy in relativistic electrons,
            :math:`\epsilon_E`. Default is ``0.1``.

        gamma_min : float or array-like, optional
            Minimum electron Lorentz factor :math:`\gamma_{\rm min}`. Default is ``1``.

        gamma_max : float or array-like, optional
            Maximum electron Lorentz factor :math:`\gamma_{\rm max}`. Default is ``1e6``.

            This parameter only affects the calculation when :math:`p < 2`.

        Returns
        -------
        B : astropy.units.Quantity
            The inferred magnetic field strength :math:`B` in **Gauss**.

        R : astropy.units.Quantity
            The inferred radius of the synchrotron-emitting region :math:`R`
            in **cm**.

        Notes
        -----
        This function follows the formalism laid out in :footcite:t:`demarchiRadioAnalysisSN2004C2022` (DM22) to compute
        the magnetic field strength and radius of the synchrotron-emitting region from the observed break frequency and
        flux density of a broken power-law SED. The calculations assume a power-law distribution of electron energies
        characterized by the index :math:`p`.

        Letting :math:`\nu_{\rm brk}` be the break frequency (in GHz) between the SSA-thick and SSA-thin regimes, and
        :math:`F_{\nu_{\rm brk}}` be the flux density (in Jy) at that frequency, the magnetic field strength :math:`B`
        (in Gauss) and radius :math:`R` (in cm) of the emitting region can be computed
        by requiring that the asymptotic behavior of
        the optically thick and thin synchrotron spectra match at :math:`\nu_{\rm brk}`. The equations used are
        equations
        (16) and (17) from DM22 with minor alterations.

        In treating the electron energy distribution, some additional care is taken based on the value of :math:`p`. In
        particular, we assume a power-law distribution of electron Lorentz factors :math:`\Gamma` such that

        .. math::

            N(\Gamma) d\Gamma = K_e \Gamma^{-p} d\Gamma,\;\; \Gamma_{\rm min} \leq \Gamma \leq \Gamma_{\rm max},

        where :math:`K_e` is the normalization constant, :math:`\Gamma_{\rm min}` is the minimum Lorentz factor,
        and :math:`\Gamma_{\rm max}` is the maximum Lorentz factor. For values of :math:`p > 2`, the total energy is
        dominated by electrons near :math:`\Gamma_{\rm min}`, while for :math:`p < 2`, it is dominated by those near
        :math:`\Gamma_{\rm max}`.

        To account for this, when :math:`p > 2`, we enforce :math:`\Gamma_{\rm max} = \infty` in the energy integral,
        while
        for :math:`p < 2`, we enforce the upper limit on the energy integral to be :math:`\Gamma_{\rm max}`. This leads
        to
        a correction factor :math:`\delta` defined as:

        .. math::

            \delta = \begin{cases} 1, & p > 2 \[6pt]
            \left(\frac{\Gamma_{\rm max}}{\Gamma_{\rm min}}\right)^{2 - p} - 1, & p < 2 \end{cases}

        which modifies the expressions for :math:`B` and :math:`R` accordingly.

        References
        ----------

        .. footbibliography::

        """
        # Validate units of all unit bearing quantities and coerce them to the expected
        # units for the optimized backend.
        nu_brk = ensure_in_units(nu_brk, u.GHz)
        F_nu_brk = ensure_in_units(F_nu_brk, u.Jy)
        distance = ensure_in_units(distance, u.Mpc)

        # Check the validity of p values. We need to ensure that ``p`` behaves as an array at
        # this point, so we cast it explicitly.
        p = np.asarray(p)
        if np.any(p == 2):
            raise ValueError(
                "compute_BR_from_BPL does not support p = 2. "
                "Use p slightly above or below 2, or implement a dedicated "
                "logarithmic normalization for this case."
            )

        # Dispatch to the optimized backend.
        B, R = self._opt_from_params_to_physics(
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        return B * u.Gauss, R * u.cm

    def _opt_from_physics_to_params(
        self,
        B: Union[float, np.ndarray],
        R: Union[float, np.ndarray],
        distance: Union[float, np.ndarray],
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ):
        r"""
        Compute the synchrotron self-absorption frequency and peak flux.

        This is a **low-level, performance-optimized routine** that assumes all
        inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        The function analytically eliminates the normalization of the electron
        energy distribution using microphysical energy-partition assumptions,
        introducing a correction factor ``delta`` to account for the convergence
        of the electron energy integral when :math:`p < 2`.

        Parameters
        ----------
        B : float or array-like
            Magnetic field strength in Gauss.

        R : float or array-like
            Radius of the emitting region in cm.

        distance : float or array-like
            Distance to the source in cm.

        p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
            See the user-facing function ``compute_BPL_SED_from_BR``.

        Returns
        -------
        nu_brk : float or array-like
            Synchrotron self-absorption break frequency (Hz-equivalent CGS).

        F_nu_brk : float or array-like
            Peak flux density at the break frequency (CGS-equivalent).

        Notes
        -----
        See notes in the user-facing wrapper function `compute_BR_from_BPL` for details on the
        underlying physics and assumptions.
        """
        # Validate input parameters. We do NOT check explicitly for shape correctness of the arrays, instead opting
        # to allow an error to rise naturally if the shapes are incompatible during computation. We do want to ensure
        # that all of the constants are cast properly for array operations and masking.
        p, gamma_min, gamma_max = (
            np.asarray(p, dtype="f8"),
            np.asarray(gamma_min, dtype="f8"),
            np.asarray(gamma_max, dtype="f8"),
        )

        # Obtain the synchrotron coefficients relevant for this scenario. We need c_1,c_5, and c_6.
        c_5 = compute_c5_parameter(p)  # Should match shape of p.
        c_6 = compute_c6_parameter(p)  # Should match shape of p.
        c_1 = c_1_cgs  # Scalar constant.

        # Construct the array for delta. See the documentation notes for details on the procedure here / physical
        # motivation. To do this efficiently, we pre-allocate the array and then fill in values based on the conditions.
        # Because we pre-allocate with ones, we only need to fill in values where p < 2. In THIS IMPLEMENTATION, we
        # ignore
        # cases where p == 2 for efficiency; these should be screened for upstream if needed.
        delta = np.ones_like(p)
        _p_lt_2_mask = p < 2
        delta[_p_lt_2_mask] = (gamma_max[_p_lt_2_mask] / gamma_min[_p_lt_2_mask]) ** (
            2 - p[_p_lt_2_mask]
        ) - 1  # Fill in values where p < 2.

        # Construct the "p_norm" term used in the B and R calculations. This allows us to handle the two branches
        # of the solution (p < 2 and p > 2) in a unified way.
        # Note that we take the absolute value here to avoid issues with negative bases and fractional exponents.
        # The p == 2 case is handled upstream.
        p_norm = np.abs(p - 2.0)

        # Compute the electron energy floor using the gamma_min parameter and the standard formula.
        E_l = electron_rest_energy_cgs * gamma_min

        # Calculate nu_brk. We break this into parts for clarity. See the notes on this function for an explanation of
        # where this formula comes from.
        _nu_brk_coeff = 2 * c_1
        _nu_brk_t1 = R * f * (epsilon_E / (delta * epsilon_B)) * p_norm / (6 * np.pi)
        _nu_brk_t2 = c_6 ** (2 / (p + 4))
        _nu_brk_t3 = np.sin(theta) ** ((p + 2) / (p + 4))
        _nu_brk_E_l_exp = 2 * (p - 2) / (p + 4)
        _nu_brk_B_exp = (p + 6) / (p + 4)

        nu_brk = (
            _nu_brk_coeff
            * (_nu_brk_t1 ** (2 / (p + 4)))
            * (E_l**_nu_brk_E_l_exp)
            * (B**_nu_brk_B_exp)
            * _nu_brk_t2
            * _nu_brk_t3
        )

        # Calculate F_nu_brk by inserting nu_brk into either the optically thick or thin formula.
        # We choose the optically thick formula here (equation 14 of DeMarchi+22) for consistency.
        _F_nu_brk_coeff = (c_5 / c_6) * np.pi * (R / distance) ** 2
        _F_nu_brk_t1 = (B * np.sin(theta)) ** (-1 / 2)
        _F_nu_brk_t2 = (nu_brk / (2 * c_1)) ** (5 / 2)

        F_nu_brk = _F_nu_brk_coeff * _F_nu_brk_t1 * _F_nu_brk_t2

        return nu_brk, F_nu_brk

    def from_physics_to_params(
        self,
        B: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        distance: Union[float, np.ndarray, u.Quantity],
        *,
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ):
        r"""
        Compute the synchrotron self-absorption break frequency and peak flux.

        This is a **low-level, performance-optimized routine** that assumes all
        inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        The function analytically eliminates the normalization of the electron
        energy distribution using microphysical energy-partition assumptions,
        introducing a correction factor ``delta`` to account for the convergence
        of the electron energy integral when :math:`p < 2`.

        Parameters
        ----------
        B : float or array-like
            Magnetic field strength in Gauss.

        R : float or array-like
            Radius of the emitting region in cm.

        distance : float or array-like
            Distance to the source in Mpc.

        p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
            See the user-facing function ``compute_BPL_SED_from_BR``.

        Returns
        -------
        nu_brk : float or array-like
            Synchrotron self-absorption break frequency (Hz-equivalent CGS).

        F_nu_brk : float or array-like
            Peak flux density at the break frequency (CGS-equivalent).

        Notes
        -----
        In the optically thick regime, the synchrotron SED follows a power law

        .. math::

            F_\nu = \frac{c_5}{c_6} \left(B\sin\theta\right)^{-1/2} \left(\frac{\nu}{2c_1}\right)^{5/2}
            \frac{\pi R^2}{d^2}.

        In the optically thin regime, the SED follows a different power law:

        .. math::

            F_\nu = \frac{4\pi f R^3}{3 d^2} c_5 N_0 \left(B\sin \theta\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

        We define the break frequency :math:`\nu_{\rm brk}` as the frequency where these two power laws intersect and
        the
        corresponding peak flux density :math:`F_{\nu_{\rm brk}}`. By equating the two expressions for :math:`F_\nu` at
        :math:`\nu = \nu_{\rm brk}`, we can solve for :math:`\nu_{\rm brk}` and :math:`F_{\nu_{\rm brk}}` in terms of
        the physical parameters :math:`B`, :math:`R`, and :math:`d`.

        Equation these two equations yields

        .. math::

            \nu_{\rm brk} = 2 c_1 \left[\frac{4}{3} c_6 f N_0\right]^{2/(p+4)} R^{2/(p+4)}
            \left(B\sin\theta\right)^{(p+2)/(p+4)}

        Inserting :math:`\nu_{\rm brk}` back into either expression for :math:`F_\nu` gives the peak flux density
        :math:`F_{\nu_{\rm brk}}`.

        The normalization :math:`N_0` of the electron distribution is eliminated
        analytically by equating the total electron energy density to a fraction
        :math:`\epsilon_E` of the post-shock internal energy density, with magnetic
        energy fraction :math:`\epsilon_B`.

        For :math:`p \neq 2`, this introduces a correction factor

        .. math::

            \delta =
            \begin{cases}
            1, & p > 2 \
            (\gamma_{\max}/\gamma_{\min})^{2-p} - 1, & p < 2
            \end{cases}

        which accounts for the convergence properties of the electron energy integral.
        """
        # Validate units of all unit bearing quantities and coerce them to the expected
        # units for the optimized backend.
        if hasattr(B, "units"):
            B = B.to_value(u.Gauss)
        if hasattr(R, "units"):
            R = R.to_value(u.cm)
        if hasattr(distance, "units"):
            distance = distance.to_value(u.cm)

        # Check the validity of p values. We need to ensure that ``p`` behaves as an array at
        # this point, so we cast it explicitly.
        p = np.asarray(p)
        if np.any(p == 2):
            raise ValueError(
                "compute_BR_from_BPL does not support p = 2. "
                "Use p slightly above or below 2, or implement a dedicated "
                "logarithmic normalization for this case."
            )

        # Dispatch to the optimized backend.
        nu, F_nu = self._opt_from_physics_to_params(
            B=B,
            R=R,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        return nu * u.Hz, F_nu * u.erg / (u.s * u.cm**2 * u.Hz)
