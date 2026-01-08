"""
Module for computing synchrotron frequencies of various sorts.

This module provides functions to compute key synchrotron frequencies used in
modeling synchrotron radiation from astrophysical sources. These include the
synchrotron cooling frequency, characteristic frequency, gyrofrequency,
and synchrotron self-absorption frequency. The functions are optimized for
CGS units for performance-critical applications.
"""

from typing import Union

import numpy as np
from astropy import constants
from astropy import units as u

from triceratops.radiation.constants import c_1_cgs, electron_rest_energy_cgs
from triceratops.radiation.synchrotron.utils import compute_c6_parameter

__all__ = [
    "compute_cooling_gamma",
    "compute_cooling_frequency",
    "compute_IC_cooling_gamma",
    "compute_IC_cooling_frequency",
    "compute_characteristic_frequency",
    "compute_gyrofrequency",
    "compute_nu_critical",
    "compute_nu_ssa",
]

# ==================================================== #
# CGS CONSTANTS FOR SYNCHROTRON FREQUENCY CALCULATIONS #
# ==================================================== #
_cooling_frequency_coefficient_cgs = (
    (18 * np.pi * constants.m_e * constants.c * constants.e.esu) / (constants.sigma_T**2)
).cgs.value
_characteristic_frequency_coefficient_cgs = (0.5 * constants.e.esu / (np.pi * constants.m_e * constants.c)).cgs.value
_gyrofrequency_coefficient_cgs = (constants.e.esu / (constants.m_e * constants.c)).cgs.value


# ========================================== #
# CGS-ONLY OPTIMIZED FREQUENCY CALCULATIONS  #
# ========================================== #
def _optimized_compute_cooling_gamma(
    B: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
):
    r"""
    Compute the electron Lorentz factor corresponding to the synchrotron cooling frequency (CGS, optimized).

    Parameters
    ----------
    B : float or array-like
        Magnetic field strength in Gauss.

    t : float or array-like
        Time since electron acceleration in seconds. This can be understood as
        the relevant dynamical time.

    Returns
    -------
    gamma_c : float or array-like
        Electron Lorentz factor corresponding to the synchrotron cooling frequency (CGS-equivalent).

    Notes
    -----
    This function computes the Lorentz factor of electrons whose radiative cooling time equals
    the age of the system. The expression is derived from equating the synchrotron cooling time
    to the dynamical time.

    The Lorentz factor is given by:

    .. math::

        \gamma_c = \frac{6 \pi m_e c}{\sigma_T B^2 t}

    No unit validation is performed. Inputs must already be in CGS.
    """
    return (6 * np.pi * constants.m_e.cgs.value * constants.c.cgs.value) / (constants.sigma_T.cgs.value * B**2 * t)


def _optimized_compute_cooling_frequency(
    B: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
):
    r"""
    Compute the synchrotron cooling frequency (CGS, optimized).

    Parameters
    ----------
    B : float or array-like
        Magnetic field strength in Gauss.

    t : float or array-like
        Time since electron acceleration in seconds. This can be understood as
        the relevant dynamical time.

    Returns
    -------
    nu_c : float or array-like
        Synchrotron cooling frequency in Hz (CGS-equivalent).

    Notes
    -----
    This function implements equation (3) of DeMarchi et al. (2022):

    .. math::

        \nu_c = \frac{18\pi m_e c e}{\sigma_T^2 B^3 t^2}

    No unit validation is performed. Inputs must already be in CGS.
    """
    return _cooling_frequency_coefficient_cgs / (B**3 * t**2)


def _optimized_compute_IC_cooling_gamma(
    L_bol: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
):
    r"""
    Compute the electron Lorentz factor corresponding to the inverse Compton cooling frequency (CGS, optimized).

    Parameters
    ----------
    L_bol : float or array-like
        Bolometric luminosity of the radiation field in erg/s.

    R : float or array-like
        Distance from the radiation source in cm.

    t : float or array-like
        Time since electron acceleration in seconds. This can be understood as
        the relevant dynamical time.

    Returns
    -------
    gamma_IC : float or array-like
        Electron Lorentz factor corresponding to the inverse Compton cooling frequency (CGS-equivalent).

    Notes
    -----
    This function computes the Lorentz factor of electrons whose inverse Compton cooling time equals
    the age of the system. The expression is derived from equating the inverse Compton cooling time
    to the dynamical time.

    The Lorentz factor is given by:

    .. math::

        \gamma_{IC} = \frac{3 m_e c^3 R^2}{\sigma_T L_{bol} t}

    No unit validation is performed. Inputs must already be in CGS.
    """
    return (3 * np.pi * constants.m_e.cgs.value * constants.c.cgs.value**2 * R**2) / (
        constants.sigma_T.cgs.value * L_bol * t
    )


def _optimized_compute_IC_cooling_frequency(
    L_bol: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
):
    r"""
    Compute the inverse Compton cooling frequency (CGS, optimized).

    Parameters
    ----------
    L_bol : float or array-like
        Bolometric luminosity of the radiation field in erg/s.

    R : float or array-like
        Distance from the radiation source in cm.

    B : float or array-like
        Magnetic field strength in Gauss.

    t : float or array-like
        Time since electron acceleration in seconds. This can be understood as
        the relevant dynamical time.

    Returns
    -------
    nu_IC_c : float or array-like
        Inverse Compton cooling frequency in Hz (CGS-equivalent).

    Notes
    -----
    This function computes the inverse Compton cooling frequency using the provided parameters.

    No unit validation is performed. Inputs must already be in CGS.
    """
    # Calculate the corresponding gamma_IC:
    gamma_IC = _optimized_compute_IC_cooling_gamma(L_bol, R, t)

    # Now compute the characteristic frequency for this gamma_IC in the magnetic field B
    return _optimized_compute_characteristic_frequency(B, gamma_IC)


def _optimized_compute_characteristic_frequency(
    B: Union[float, np.ndarray],
    gamma: Union[float, np.ndarray],
):
    r"""
    Compute the synchrotron characteristic frequency (CGS, optimized).

    Parameters
    ----------
    B : float or array-like
        Magnetic field strength in Gauss.

    gamma : float or array-like
        Electron Lorentz factor.

    Returns
    -------
    nu : float or array-like
        Synchrotron characteristic frequency in Hz (CGS-equivalent).

    Notes
    -----
    Implements equation (4) of :footcite:t:`demarchiRadioAnalysisSN2004C2022`.:

    .. math::

        \nu(\gamma) = \frac{e}{2\pi m_e c} B \gamma^2

    No unit validation is performed.

    References
    ----------
    .. footbibliography::
    """
    return _characteristic_frequency_coefficient_cgs * B * gamma**2


def _optimized_compute_nu_gyro(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
):
    r"""
    Compute the synchrotron gyrofrequency (CGS, optimized).

    Parameters
    ----------
    gamma : float or array-like
        Electron Lorentz factor.

    B : float or array-like
        Magnetic field strength in Gauss.

    Returns
    -------
    nu_g : float or array-like
        Synchrotron gyrofrequency in Hz (CGS-equivalent).

    Notes
    -----
    Implements the gyrofrequency formula:

    .. math::

        \nu_g = \frac{e B}{m_e c \gamma}

    No unit validation is performed.
    """
    return _gyrofrequency_coefficient_cgs * B / gamma


def _optimized_compute_nu_critical(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
):
    r"""
    Compute the synchrotron critical frequency (CGS, optimized).

    Parameters
    ----------
    gamma : float or array-like
        Electron Lorentz factor.

    B : float or array-like
        Magnetic field strength in Gauss.

    Returns
    -------
    nu_critical : float or array-like
        Synchrotron critical frequency in Hz (CGS-equivalent).

    Notes
    -----
    Implements the critical frequency formula:

    .. math::

        \nu_{critical} = \frac{3 e B \gamma^2}{4 \pi m_e c}

    No unit validation is performed.
    """
    return (3 / (4 * np.pi)) * _gyrofrequency_coefficient_cgs * B * gamma**2


def _optimized_compute_nu_ssa(
    B: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    p: Union[float, np.ndarray] = 3.0,
    f: Union[float, np.ndarray] = 0.5,
    theta: Union[float, np.ndarray] = np.pi / 2,
    epsilon_B: Union[float, np.ndarray] = 0.1,
    epsilon_E: Union[float, np.ndarray] = 0.1,
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = 1e6,
):
    r"""
    Optimized CGS-only computation of the synchrotron self-absorption frequency.

    Implements equation (9) of DeMarchi et al. (2022), with analytic elimination
    of the electron distribution normalization using microphysical energy
    partition assumptions.

    All inputs must already be in CGS units.
    """
    # Cast to float64 arrays for vectorized math
    p = np.asarray(p, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    # Synchrotron coefficients
    c_6 = compute_c6_parameter(p)
    c_1 = c_1_cgs  # scalar

    # Delta correction for electron energy integral
    delta = np.ones_like(p)
    mask = p < 2
    delta[mask] = (gamma_max[mask] / gamma_min[mask]) ** (2 - p[mask]) - 1

    # |p - 2| factor for unified normalization
    p_norm = np.abs(p - 2.0)

    # Minimum electron energy
    E_l = electron_rest_energy_cgs * gamma_min

    # Electron distribution normalization N_0
    N_0 = (epsilon_E / (epsilon_B * delta)) * (p_norm / (8 * np.pi)) * (B**2 / E_l ** (2 - p))

    # Synchrotron self-absorption frequency (DM22 Eq. 9)
    prefactor = 2 * c_1
    geom_term = ((4.0 / 3.0) * f * R * c_6) ** (2.0 / (p + 4.0))
    field_term = (B * np.sin(theta)) ** ((p + 2.0) / (p + 4.0))
    norm_term = N_0 ** (2.0 / (p + 4.0))

    nu_ssa = prefactor * geom_term * norm_term * field_term

    return nu_ssa


def compute_cooling_gamma(
    B: Union[float, np.ndarray, u.Quantity],
    t: Union[float, np.ndarray, u.Quantity],
):
    r"""
    Compute the electron Lorentz factor corresponding to the synchrotron cooling frequency.

    The cooling Lorentz factor corresponds to the Lorentz factor of electrons whose radiative
    cooling time equals the age of the system.

    This is given by :footcite:p:`demarchiRadioAnalysisSN2004C2022` as

    .. math::

        \gamma_c = \frac{6 \pi m_e c}{\sigma_T B^2 t},

    where :math:`t` is a relevant measure of the dynamical time since over which
    electrons have been radiating.

    Parameters
    ----------
    B : float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    t : float, array-like, or astropy.units.Quantity
        Time since electron acceleration. Default units are seconds.

    Returns
    -------
    gamma_c : float or array-like
        Electron Lorentz factor corresponding to the synchrotron cooling frequency.

    Notes
    -----
    This function wraps a low-level CGS implementation and performs
    unit coercion only. Shape compatibility is handled implicitly by NumPy.

    The underlying expression is taken from equation (3) of
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    References
    ----------
    .. footbibliography::
    """
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)
    if hasattr(t, "units"):
        t = t.to_value(u.s)

    gamma_c = _optimized_compute_cooling_gamma(B, t)
    return gamma_c


def compute_cooling_frequency(
    B: Union[float, np.ndarray, u.Quantity],
    t: Union[float, np.ndarray, u.Quantity],
):
    r"""
    Compute the synchrotron cooling frequency.

    The cooling frequency corresponds to the characteristic synchrotron
    frequency of electrons whose radiative cooling time equals the age
    of the system.

    This is given by :footcite:p:`demarchiRadioAnalysisSN2004C2022` as

    .. math::

        \nu_c = \frac{18\pi m_e c e}{\sigma_T^2 B^3 t^2},

    where :math:`t` is a relevant measure of the dynamical time since over which
    electrons have been radiating.

    Parameters
    ----------
    B : float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    t : float, array-like, or astropy.units.Quantity
        Time since electron acceleration. Default units are seconds.

    Returns
    -------
    nu_c : astropy.units.Quantity
        Synchrotron cooling frequency in Hz.

    Notes
    -----
    This function wraps a low-level CGS implementation and performs
    unit coercion only. Shape compatibility is handled implicitly by NumPy.

    The underlying expression is taken from equation (3) of
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    It should be noted that the cooling frequency computed by this function does **NOT** consider the
    ambient radiation field. If inverse Compton cooling is significant, the effective cooling frequency
    will be lower than the value computed here. That should be computed using the :func:`compute_IC_cooling_frequency`
    function.

    References
    ----------
    .. footbibliography::
    """
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)
    if hasattr(t, "units"):
        t = t.to_value(u.s)

    nu_c = _optimized_compute_cooling_frequency(B, t)
    return nu_c * u.Hz


def compute_IC_cooling_gamma(
    L_bol: Union[float, np.ndarray, u.Quantity],
    R: Union[float, np.ndarray, u.Quantity],
    t: Union[float, np.ndarray, u.Quantity],
):
    r"""
    Compute the electron Lorentz factor corresponding to the inverse Compton cooling frequency.

    The inverse Compton cooling Lorentz factor corresponds to the Lorentz factor of electrons whose
    inverse Compton cooling time equals the age of the system.

    This is given by

    .. math::

        \gamma_{IC} = \frac{3 m_e c^3 R^2}{\sigma_T L_{bol} t},

    where :math:`L_{bol}` is the bolometric luminosity of the ambient radiation field,
    :math:`R` is the distance from the radiation source, and :math:`t` is a relevant measure
    of the dynamical time since over which electrons have been radiating.

    Parameters
    ----------
    L_bol : float, array-like, or astropy.units.Quantity
        Bolometric luminosity of the radiation field. Default units are erg/s.

    R : float, array-like, or astropy.units.Quantity
        Distance from the radiation source. Default units are cm.

    t : float, array-like, or astropy.units.Quantity
        Time since electron acceleration. Default units are seconds.

    Returns
    -------
    gamma_IC : float or array-like
        Electron Lorentz factor corresponding to the inverse Compton cooling frequency.

    Notes
    -----
    This function wraps a low-level CGS implementation and performs
    unit coercion only. Shape compatibility is handled implicitly by NumPy.
    """
    if hasattr(L_bol, "units"):
        L_bol = L_bol.to_value(u.erg / u.s)
    if hasattr(R, "units"):
        R = R.to_value(u.cm)
    if hasattr(t, "units"):
        t = t.to_value(u.s)

    gamma_IC = _optimized_compute_IC_cooling_gamma(L_bol, R, t)
    return gamma_IC


def compute_IC_cooling_frequency(
    L_bol: Union[float, np.ndarray, u.Quantity],
    R: Union[float, np.ndarray, u.Quantity],
    B: Union[float, np.ndarray, u.Quantity],
    t: Union[float, np.ndarray, u.Quantity],
):
    r"""
    Compute the inverse Compton cooling frequency.

    The inverse Compton cooling frequency corresponds to the characteristic synchrotron
    frequency of electrons whose inverse Compton cooling time equals the age of the system.

    This is given by

    .. math::

        \nu_{IC,c} = \gamma_{IC}^2 \frac{e B}{2 \pi m_e c},

    where :math:`\gamma_{IC} = \frac{3 m_e c^3 R^2}{\sigma_T L_{bol} t}` is the Lorentz factor
    of electrons whose inverse Compton cooling time equals the dynamical time.

    Parameters
    ----------
    L_bol : float, array-like, or astropy.units.Quantity
        Bolometric luminosity of the radiation field. Default units are erg/s.

    R : float, array-like, or astropy.units.Quantity
        Distance from the radiation source. Default units are cm.

    B : float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.


    t : float, array-like, or astropy.units.Quantity
        Time since electron acceleration. Default units are seconds.

    Returns
    -------
    nu_IC_c : astropy.units.Quantity
        Inverse Compton cooling frequency in Hz.

    Notes
    -----
    This function wraps a low-level CGS implementation and performs
    unit coercion only. Shape compatibility is handled implicitly by NumPy.
    """
    if hasattr(L_bol, "units"):
        L_bol = L_bol.to_value(u.erg / u.s)
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)
    if hasattr(R, "units"):
        R = R.to_value(u.cm)
    if hasattr(t, "units"):
        t = t.to_value(u.s)

    nu_IC_c = _optimized_compute_IC_cooling_frequency(L_bol, R, B, t)
    return nu_IC_c * u.Hz


def compute_characteristic_frequency(
    B: Union[float, np.ndarray, u.Quantity],
    gamma: Union[float, np.ndarray],
):
    r"""
    Compute the synchrotron characteristic frequency for relativistic electrons.

    For relativistic electrons in a **non-relativistic flow**, the characteristic frequency
    corresponds roughly to the peak of the single-electron synchrotron emission spectrum. It is
    precisely the frequency at which an electron with the lowest Lorentz factor in the
    distribution emits the bulk of its synchrotron radiation :footcite:p:`demarchiRadioAnalysisSN2004C2022`.

    .. math::

        \nu_m = \gamma_m^2 \frac{e B}{2 \pi m_e c}

    Parameters
    ----------
    B : float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    gamma : float or array-like
        Electron Lorentz factor.

    Returns
    -------
    nu : astropy.units.Quantity
        Synchrotron characteristic frequency in Hz.

    Notes
    -----
    This function computes the characteristic frequency associated with
    synchrotron emission from electrons of Lorentz factor ``gamma`` in
    a magnetic field ``B``.

    The calculation follows equation (4) of
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    References
    ----------
    .. footbibliography::
    """
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)

    nu = _optimized_compute_characteristic_frequency(B, gamma)
    return nu * u.Hz


def compute_gyrofrequency(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray, u.Quantity],
) -> u.Quantity:
    r"""
    Compute the synchrotron gyrofrequency for relativistic electrons.

    The gyrofrequency corresponds to the frequency at which a charged particle
    orbits in a magnetic field. For relativistic electrons, this frequency is
    modified by the Lorentz factor of the electron.

    Parameters
    ----------
    gamma : float or array-like
        Electron Lorentz factor.

    B : float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    Returns
    -------
    nu_g : astropy.units.Quantity
        Synchrotron gyrofrequency in Hz.

    Notes
    -----
    The gyrofrequency for a relativistic electron is given by

    .. math::

        \nu_g = \frac{e B}{m_e c \gamma}

    This function computes the gyrofrequency associated with
    synchrotron emission from electrons of Lorentz factor ``gamma`` in
    a magnetic field ``B``.
    """
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)

    return _optimized_compute_nu_gyro(gamma, B) * u.Hz


def compute_nu_critical(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray, u.Quantity],
) -> u.Quantity:
    r"""
    Compute the synchrotron critical frequency for relativistic electrons.

    The critical frequency corresponds to the frequency at which the synchrotron
    emission from a relativistic electron peaks. This follows the formalism as described in
    :footcite:t:`RybickiLightman`.

    Parameters
    ----------
    gamma : float or array-like
        Electron Lorentz factor.

    B : float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    Returns
    -------
    nu_critical : astropy.units.Quantity
        Synchrotron critical frequency in Hz.

    Notes
    -----
    The critical frequency for a relativistic electron is given by

    .. math::

        \nu_{critical} = \frac{3 e B \gamma^2}{4 \pi m_e c}

    This function computes the critical frequency associated with
    synchrotron emission from electrons of Lorentz factor ``gamma`` in
    a magnetic field ``B``.
    """
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)

    return _optimized_compute_nu_critical(gamma, B) * u.Hz


def compute_nu_ssa(
    B: Union[float, np.ndarray, u.Quantity],
    R: Union[float, np.ndarray, u.Quantity],
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
    Compute the synchrotron self-absorption (SSA) frequency.

    Parameters
    ----------
    B : float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    R : float, array-like, or astropy.units.Quantity
        Radius of the emitting region. Default units are cm.

    p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
        Standard synchrotron and microphysical parameters. See Notes.

    Returns
    -------
    nu_ssa : astropy.units.Quantity
        Synchrotron self-absorption frequency in Hz.

    Notes
    -----
    The SSA frequency is defined as the frequency at which the synchrotron
    optical depth equals unity. This function implements equation (9) of
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`:

    .. math::

        \nu_{\rm ssa} = 2 c_1 \left(\frac{4f R c_6 N_0}{3}\right)^{2/(p+4)} (B \sin\theta)^{(p+2)/(p+4)},

    The normalization of the electron energy distribution is eliminated
    analytically by equating the electron and magnetic energy densities:

    .. math::

        u_e = \epsilon_E u_{\rm int}, \quad
        u_B = \epsilon_B u_{\rm int}.

    For :math:`p \neq 2`, this introduces a correction factor

    .. math::

        \delta =
        \begin{cases}
        1, & p > 2 \\
        (\gamma_{\max}/\gamma_{\min})^{2-p} - 1, & p < 2
        \end{cases}

    which accounts for the convergence properties of the electron energy
    integral.

    References
    ----------
    .. footbibliography::
    """
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)
    if hasattr(R, "units"):
        R = R.to_value(u.cm)

    p = np.asarray(p)
    if np.any(p == 2):
        raise ValueError(
            "compute_nu_ssa does not support p = 2 due to logarithmic divergence in the electron energy integral."
        )

    nu_ssa = _optimized_compute_nu_ssa(
        B=B,
        R=R,
        p=p,
        f=f,
        theta=theta,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    return nu_ssa * u.Hz


def _determine_SYNCH_regime_from_nu(
    nu_m: Union[float, np.ndarray],
    nu_c: Union[float, np.ndarray],
    nu_ssa: Union[float, np.ndarray],
) -> np.ndarray:
    r"""
    Identify the synchrotron spectral regime based on the characteristic frequencies.

    See the higher-level API function :func:`determine_synchrotron_regime_from_frequencies`
    for documentation.

    Parameters
    ----------
    nu_m: array-like
        The characteristic synchrotron frequency determined by the minimum energy electrons in the
        population.
    nu_c: array-like
        The synchrotron cooling frequency. This may either be the standard synchrotron cooling frequency
        or the inverse Compton cooling frequency, depending on context.
    nu_ssa: array-like
        The synchrotron self-absorption frequency.

    Returns
    -------
    np.ndarray
        The synchrotron spectral regime identifiers as an array of uint8 values.
        Each value corresponds to a specific regime as described in the higher-level API function.
    """
    # Coerce all of the inputs to numpy arrays so that we can ensure correct
    # mapping access.
    nu_m, nu_c, nu_ssa = np.asarray(nu_m), np.asarray(nu_c), np.asarray(nu_ssa)

    # Construct the relevant regime masks.
    _regime_1_mask = (nu_ssa <= nu_m) & (nu_m < nu_c)
    _regime_2_mask = (nu_ssa > nu_m) & (nu_ssa < nu_c)
    _regime_3_mask = (nu_c <= nu_ssa) & (nu_m <= nu_ssa)
    _regime_4_mask = (nu_c <= nu_ssa) & (nu_ssa < nu_m)
    _regime_5_mask = (nu_ssa < nu_c) & (nu_c < nu_m)

    # Initialize output array
    regime_ids = np.zeros_like(nu_m, dtype=np.uint8)
    regime_ids[_regime_1_mask] = 1
    regime_ids[_regime_2_mask] = 2
    regime_ids[_regime_3_mask] = 3
    regime_ids[_regime_4_mask] = 4
    regime_ids[_regime_5_mask] = 5

    return regime_ids


def determine_synch_regime_from_frequencies(
    nu_m: Union[float, u.Quantity],
    nu_c: Union[float, u.Quantity],
    nu_ssa: Union[float, u.Quantity],
) -> str:
    r"""
    Determine the synchrotron spectral regime based on characteristic frequencies.

    This function classifies the synchrotron spectral regime by comparing the characteristic
    frequencies: the minimum electron frequency (:math:`nu_m`), the cooling frequency (:math:`nu_c`), and the
    self-absorption frequency (:math:`nu_a`). The classification follows standard synchrotron theory
    and is useful for interpreting synchrotron emission spectra.

    Following the convention of :footcite:t:`GranotSari2002SpectralBreaks`, the regimes are defined as

    .. list-table::
       :widths: auto
       :header-rows: 1

       * - Regime
         - Condition
         - Description
       * - I
         - :math:`\nu_a < \nu_m < \nu_c`
         - Slow cooling, optically thin peak.
       * - II
         - :math:`\nu_m < \nu_a < \nu_c`
         - Slow cooling, optically thick peak.
       * - III
         - :math:`\nu_a > \nu_c, \nu_m`
         - SSA dominated.
       * - IV
         - :math:`\nu_c < \nu_{a} < \nu_m`
         - Fast cooling, optically thick
       * - V
         - :math:`\nu_a < \nu_c < \nu_m`
         - Fast cooling, optically thin.


    Parameters
    ----------
    nu_m: float or numpy.ndarray or astropy.units.Quantity
        The characteristic synchrotron frequency determined by the minimum energy electrons in the
        population. This can be computed using :func:`compute_characteristic_frequency`.
    nu_c: float or numpy.ndarray or astropy.units.Quantity
        The synchrotron cooling frequency. This can be computed using :func:`compute_cooling_frequency`
        or :func:`compute_IC_cooling_frequency`, depending on the cooling mechanism.
    nu_ssa: float or numpy.ndarray or astropy.units.Quantity
        The synchrotron self-absorption frequency. This can be computed using :func:`compute_nu_ssa`.

    Returns
    -------
    numpy.ndarray
        An array of integers (uint8) representing the synchrotron spectral regime for each set of input frequencies.
        Each value corresponds to the numerical designation of the regime as described above (1 through 5).
    """
    # Coerce all unit-bearing inputs to Hz
    if hasattr(nu_m, "units"):
        nu_m = nu_m.to_value(u.Hz)
    if hasattr(nu_c, "units"):
        nu_c = nu_c.to_value(u.Hz)
    if hasattr(nu_ssa, "units"):
        nu_ssa = nu_ssa.to_value(u.Hz)

    # Pass off to the low-level implementation
    regime_ids = _determine_SYNCH_regime_from_nu(nu_m, nu_c, nu_ssa)

    # Check for pass through without assignment.
    if np.any(regime_ids == 0):
        raise RuntimeError("Failed to assign synchrotron regime ID for some input frequencies.")

    # Check if we have a 1D array with a single element and return scalar if so
    if regime_ids.ndim == 1 and regime_ids.size == 1:
        return regime_ids[0]
    return regime_ids


def _determine_SYNCH_regime_from_physics(
    B: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    L_bol: Union[float, np.ndarray],
    p: Union[float, np.ndarray] = 3.0,
    f: Union[float, np.ndarray] = 0.5,
    theta: Union[float, np.ndarray] = np.pi / 2,
    epsilon_B: Union[float, np.ndarray] = 0.1,
    epsilon_E: Union[float, np.ndarray] = 0.1,
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = 1e6,
):
    r"""
    Determine the synchrotron spectral regime based on physical parameters assuming either cooling scheme.

    This function computes the characteristic frequencies (:math:`\nu_m`, :math:`\nu_c`, :math:`\nu_a`)
    from the provided physical parameters and then classifies the synchrotron spectral regime
    by comparing these frequencies.
    """
    # Compute the relevant frequencies. We use CHARACTERISTIC, not CRITICAL following the
    # standard convention for nu_m in deMarchi+.
    nu_m = _optimized_compute_characteristic_frequency(
        B,
        gamma_min,
    )
    nu_c_IC = _optimized_compute_IC_cooling_frequency(L_bol, R, B, t)
    nu_c_synch = _optimized_compute_cooling_frequency(B, t)
    nu_c = np.minimum(nu_c_IC, nu_c_synch)
    nu_ssa = _optimized_compute_nu_ssa(
        B,
        R,
        p,
        f,
        theta,
        epsilon_B,
        epsilon_E,
        gamma_min,
        gamma_max,
    )

    # Now return the regime IDs:
    return _determine_SYNCH_regime_from_nu(nu_m, nu_c, nu_ssa)


def _determine_SYNCH_regime_from_physics_IC_cooling(
    B: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    L_bol: Union[float, np.ndarray],
    p: Union[float, np.ndarray] = 3.0,
    f: Union[float, np.ndarray] = 0.5,
    theta: Union[float, np.ndarray] = np.pi / 2,
    epsilon_B: Union[float, np.ndarray] = 0.1,
    epsilon_E: Union[float, np.ndarray] = 0.1,
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = 1e6,
):
    r"""
    Determine the synchrotron spectral regime based on physical parameters assuming IC cooling.

    This function computes the characteristic frequencies (:math:`\nu_m`, :math:`\nu_c`, :math:`\nu_a`)
    from the provided physical parameters and then classifies the synchrotron spectral regime
    by comparing these frequencies.
    """
    # Compute the relevant frequencies. We use CHARACTERISTIC, not CRITICAL following the
    # standard convention for nu_m in deMarchi+.
    nu_m = _optimized_compute_characteristic_frequency(
        B,
        gamma_min,
    )
    nu_c = _optimized_compute_IC_cooling_frequency(L_bol, R, B, t)
    nu_ssa = _optimized_compute_nu_ssa(
        B,
        R,
        p,
        f,
        theta,
        epsilon_B,
        epsilon_E,
        gamma_min,
        gamma_max,
    )

    # Now return the regime IDs:
    return _determine_SYNCH_regime_from_nu(nu_m, nu_c, nu_ssa)


def _determine_SYNCH_regime_from_physics_synchrotron_cooling(
    B: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    p: Union[float, np.ndarray] = 3.0,
    f: Union[float, np.ndarray] = 0.5,
    theta: Union[float, np.ndarray] = np.pi / 2,
    epsilon_B: Union[float, np.ndarray] = 0.1,
    epsilon_E: Union[float, np.ndarray] = 0.1,
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = 1e6,
):
    r"""
    Determine the synchrotron spectral regime based on physical parameters assuming synchrotron cooling.

    This function computes the characteristic frequencies (:math:`\nu_m`, :math:`\nu_c`, :math:`\nu_a`)
    from the provided physical parameters and then classifies the synchrotron spectral regime
    by comparing these frequencies.

    This is done via calls to the various low-level helper functions that compute the relevant frequencies
    and then a final call to the regime identification function.
    """
    # Compute the relevant frequencies. We use CHARACTERISTIC, not CRITICAL following the
    # standard convention for nu_m in deMarchi+.
    nu_m = _optimized_compute_characteristic_frequency(
        B,
        gamma_min,
    )
    nu_c = _optimized_compute_cooling_frequency(B, t)
    nu_ssa = _optimized_compute_nu_ssa(
        B,
        R,
        p,
        f,
        theta,
        epsilon_B,
        epsilon_E,
        gamma_min,
        gamma_max,
    )

    # Now return the regime IDs:
    return _determine_SYNCH_regime_from_nu(nu_m, nu_c, nu_ssa)


def determine_synch_regime_from_phys(
    B: Union[float, np.ndarray, u.Quantity],
    t: Union[float, np.ndarray, u.Quantity],
    R: Union[float, np.ndarray, u.Quantity],
    L_bol: Union[float, np.ndarray, u.Quantity] = None,
    p: Union[float, np.ndarray] = 3.0,
    f: Union[float, np.ndarray] = 0.5,
    theta: Union[float, np.ndarray] = np.pi / 2,
    epsilon_B: Union[float, np.ndarray] = 0.1,
    epsilon_E: Union[float, np.ndarray] = 0.1,
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = 1e6,
) -> np.ndarray:
    r"""
    Determine the synchrotron spectral regime based on physical parameters.

    This function computes the characteristic frequencies (:math:`\nu_m`, :math:`\nu_c`, :math:`\nu_a`)
    from the provided physical parameters and then classifies the synchrotron spectral regime
    by comparing these frequencies.

    Parameters
    ----------
    B: float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.
    t: float, array-like, or astropy.units.Quantity
        Time since electron acceleration. Default units are seconds.
    R: float, array-like, or astropy.units.Quantity
        Radius of the emitting region. Default units are cm.
    L_bol: float, array-like, or astropy.units.Quantity, optional
        Bolometric luminosity of the radiation field. Default units are erg/s. If not provided,
        only synchrotron cooling is considered.
    p: float or array-like, optional
        Power-law index of the electron energy distribution. Default is 3.0.
    f: float or array-like, optional
        Filling factor of the emitting region. Default is 0.5.
    theta: float or array-like, optional
        Pitch angle between the electron velocity and the magnetic field. Default is pi/2 (90 degrees).
    epsilon_B: float or array-like, optional
        Fraction of the internal energy in the magnetic field. Default is 0.1.
    epsilon_E: float or array-like, optional
        Fraction of the internal energy in the electrons. Default is 0.1.
    gamma_min: float or array-like, optional
        Minimum Lorentz factor of the electron energy distribution. Default is 1.0.
    gamma_max: float or array-like, optional
        Maximum Lorentz factor of the electron energy distribution. Default is 1e6.

    Returns
    -------
    numpy.ndarray or int
        An array of integers (uint8) representing the synchrotron spectral regime for each set of input parameters.
    """
    # Coerce to CGS
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)
    if hasattr(t, "units"):
        t = t.to_value(u.s)
    if hasattr(R, "units"):
        R = R.to_value(u.cm)
    if L_bol is not None and hasattr(L_bol, "units"):
        L_bol = L_bol.to_value(u.erg / u.s)

    # Determine which evaluation function we're going to use.
    if L_bol is None:
        evaluator = _determine_SYNCH_regime_from_physics_synchrotron_cooling
    else:
        evaluator = _determine_SYNCH_regime_from_physics

    # Now compute
    regime_ids = evaluator(
        B=B,
        t=t,
        R=R,
        L_bol=L_bol,
        p=p,
        f=f,
        theta=theta,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    # Check for pass through without assignment.
    if np.any(regime_ids == 0):
        raise RuntimeError("Failed to assign synchrotron regime ID for some input frequencies.")

    # Check if we have a 1D array with a single element and return scalar if so
    if regime_ids.ndim == 1 and regime_ids.size == 1:
        return regime_ids[0]
    return regime_ids
