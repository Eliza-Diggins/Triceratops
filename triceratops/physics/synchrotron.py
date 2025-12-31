"""
Physics routines for synchrotron emission processes.

This module provides functions to compute key coefficients and parameters
related to synchrotron radiation. The bulk of the theory behind these functions is drawn from
the work of :footcite:t:`1970ranp.book.....P` and :footcite:t:`1979rpa..book.....R` with some formulations
drawn from more refined literature as discussed in the docstrings for each function / class.
"""

from typing import Union

import numpy as np
from astropy import constants
from astropy import units as u
from scipy.special import gamma as gamma_func

from .constants import c_1_cgs, electron_rest_energy_cgs

# ============================================================== #
# SYNCHROTRON EMISSION AND ABSORPTION COEFFICIENTS               #
# ============================================================== #
# As described in standard texts, there are two coefficients (dependent on p) which dictate
# the coefficient of the emission / absorption scaling for synchrotron from power-law populations.
# These functions implement these.
#
# Define pre-computed coefficients so that they do not need to be re-computed at call-time.
_c5_coefficient_cgs = np.sqrt(3) / (16 * np.pi) * (constants.e.esu**3 / (constants.m_e * constants.c**2)).cgs.value
"""float: The coefficient for the ``c5`` parameter in CGS units."""
_c6_coefficient_cgs = np.sqrt(3) * np.pi / 72 * (constants.e.esu * constants.m_e**5 * constants.c**10).cgs.value
"""float: the coefficient for the ``c6`` parameter in CGS units."""


def compute_c5_parameter(p: Union[float, np.ndarray] = 3.0) -> float:
    r"""
    Compute the :math:`c_5(p)` coefficient for synchrotron assuming a power-law electron population.

    Parameters
    ----------
    p : float or array-like, optional
        Power-law index of the electron Lorentz factor distribution,
        :math:`N(\Gamma) \propto \Gamma^{-p}`. Default is ``3.0``.

    Returns
    -------
    float
        The synchrotron emissivity coefficient :math:`c_5(p)` in CGS units.

    Notes
    -----
    The radiative power emitted per unit frequency by a single relativistic
    electron with Lorentz factor :math:`\Gamma` spiraling in a magnetic field
    :math:`B` is

    .. math::

        P(\nu, \Gamma)
        =
        \frac{\sqrt{3}\, e^3 B}{m_e c^2}
        \sin\alpha\,
        F\!\left(\frac{\nu}{\nu_c}\right),

    where :math:`\alpha` is the pitch angle, :math:`F(x)` is the
    **synchrotron kernel**

    .. math::

        F(x) = x \int_x^\infty K_{5/3}(z)\, dz,

    and the critical frequency is

    .. math::

        \nu_c
        =
        \frac{3 e B \sin\alpha}{4\pi m_e c}\, \Gamma^2.

    For an isotropic distribution of pitch angles and a power-law distribution
    of electron Lorentz factors,

    .. math::

        N(\Gamma)\, d\Gamma
        =
        K_e\, \Gamma^{-p}\, d\Gamma,

    where :math:`K_e` is the normalization of the electron number density
    (units of :math:`\mathrm{cm^{-3}}`), the synchrotron emissivity
    :math:`j_\nu` (power per unit volume per unit frequency per unit solid angle)
    is obtained by integrating the single-electron power over the distribution:

    .. math::

        j_\nu
        =
        \int P(\nu, \Gamma)\, N(\Gamma)\, d\Gamma.

    Carrying out this integration analytically yields

    .. math::

        j_\nu
        =
        c_5(p)\,
        K_e\,
        B^{(p+1)/2}\,
        \nu^{-(p-1)/2},

    where the coefficient :math:`c_5(p)` encapsulates the full integration
    of the synchrotron kernel over the power-law electron distribution and
    depends only on the spectral index :math:`p`.

    The resulting emissivity has CGS units of

    .. math::

        [j_\nu] = \mathrm{erg\ s^{-1}\ cm^{-3}\ Hz^{-1}\ sr^{-1}}.

    The value of :math:`c_5(p)` is given by (see :footcite:p:`1970ranp.book.....P` and
    :footcite:p:`1979rpa..book.....R`)

    .. math::

        c_5(p) = \frac{\sqrt{3}}{16\pi} \left(\frac{e^3}{m_e c^2}\right) \frac{p + 7/3}{p + 1}
         \Gamma\left(\frac{3p - 1}{12}\right) \Gamma\left(\frac{3p + 7}{12}\right).

    .. rubric:: References

    .. footbibliography::

    """
    dimless_part = (p + 7 / 3) / (p + 1) * gamma_func((3 * p - 1) / 12) * gamma_func((3 * p + 7) / 12)

    # Multiply the p-dependent term by the globally defined _c5_coefficient_cgs coefficient.
    return _c5_coefficient_cgs * dimless_part


def compute_c6_parameter(p: Union[float, np.ndarray] = 3.0) -> float:
    r"""
    Compute the :math:`c_6(p)` coefficient for synchrotron self-absorption from a power-law population of electrons.

    Parameters
    ----------
    p : float or array-like, optional
        Power-law index of the electron Lorentz factor distribution,
        :math:`N(\Gamma) \propto \Gamma^{-p}`. Default is 3.0.

    Returns
    -------
    float
        The synchrotron self-absorption coefficient :math:`c_6(p)` in CGS units.

    Notes
    -----
    For an isotropic distribution of pitch angles and a power-law electron
    population,

    .. math::

        N(\Gamma)\, d\Gamma = K_e\, \Gamma^{-p}\, d\Gamma,

    the synchrotron self-absorption coefficient :math:`\alpha_\nu`
    (with units of :math:`\mathrm{cm^{-1}}`) can be written as

    .. math::

        \alpha_\nu
        =
        c_6(p)\,
        K_e\,
        B^{(p+2)/2}\,
        \nu^{-(p+4)/2}.

    The coefficient :math:`c_6(p)` encapsulates the full analytic integration
    of the synchrotron absorption kernel over the power-law electron
    distribution and depends only on the spectral index :math:`p`.

    In the :footcite:p:`1970ranp.book.....P` and :footcite:p:`1979rpa..book.....R` convention appropriate
    for radio supernova synchrotron self-absorption modeling,

    .. math::

        c_6(p)
        =
        \frac{\sqrt{3}\, e^3}{16\pi m_e}
        \left(\frac{3e}{2\pi m_e^3 c^5}\right)^{p/2}
        (p + 2)\,
        \Gamma\!\left(\frac{3p + 2}{12}\right)
        \Gamma\!\left(\frac{3p + 10}{12}\right).

    This implementation uses an algebraically equivalent form in which all
    dimensional constants are grouped into a single prefactor and the
    remaining dependence on :math:`p` is purely dimensionless. This form
    is numerically stable and commonly used in radio supernova modeling codes.

    .. rubric:: References

    .. footbibliography::

    """
    # Purely dimensionless p-dependent part
    dimensionless_part = (p + 10 / 3) * gamma_func((3 * p + 2) / 12) * gamma_func((3 * p + 10) / 12)

    # Scale by the standard coefficient and return.
    return _c6_coefficient_cgs * dimensionless_part


# ============================================================== #
# SYNCHROTRON FREQUENCIES                                        #
# ============================================================== #
# In this section of the code, we define functions to compute various
# characteristic frequencies associated with synchrotron radiation from
# relativistic electrons. These frequencies are important for understanding
# the spectral properties of synchrotron-emitting sources.
#
# Define constant coefficients so that we do not waste time recomputing them
# every time the function is called. These are lightweight and can be stored
# at module load time.
_cooling_frequency_coefficient_cgs = (
    (18 * np.pi * constants.m_e * constants.c * constants.e.esu) / (constants.sigma_T**2)
).cgs.value
"""float: Coefficient for synchrotron cooling frequency in CGS units."""

_characteristic_frequency_coefficient_cgs = (0.5 * constants.e.esu / (np.pi * constants.m_e * constants.c)).cgs.value
"""float: Coefficient for synchrotron characteristic frequency in CGS units."""


# Now define functions to compute the cooling, characteristic, and SSA frequencies.
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


def compute_cooling_frequency(
    B: Union[float, np.ndarray, u.Quantity],
    t: Union[float, np.ndarray, u.Quantity],
):
    r"""
    Compute the synchrotron cooling frequency.

    The cooling frequency corresponds to the characteristic synchrotron
    frequency of electrons whose radiative cooling time equals the age
    of the system.

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


def compute_characteristic_frequency(
    B: Union[float, np.ndarray, u.Quantity],
    gamma: Union[float, np.ndarray],
):
    r"""
    Compute the synchrotron characteristic frequency for relativistic electrons.

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
    """
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


# ============================================================== #
# MACROPHYSICS HELPERS                                           #
# ============================================================== #
# These functions are used to compute various macroscopic physical
# quantities from microphysical parameters and vice-versa. Notably, these
# functions generally rely on some assumptions about the underlying physics and
# should therefore be used as building-blocks within more complete models and not taken
# as a ground-truth piece of physics without validation.
#
# Some functions in this section of the package are optimized versions of
# more general functions found elsewhere in the codebase. These optimized
# versions are intended to be faster and more efficient, but may rely on static type assumptions
# or other constraints that make them less flexible than the more general versions.
#
# DEVELOPER NOTES:
# TODO: Handling of the power-law normalization in the BR to / from BPL functions could be compartmentalized.
#       This could be achieved by re-deriving eq 16 and 17 in terms of N_0 directly and then having a helper function
#       that computes N_0 from the microphysical parameters. This would reduce code duplication and make it easier
#       to maintain consistency across the functions. We leave this for future due to labor constraints. Those
#       normalizations
#       are however written as separate functions for use elsewhere.
# TODO: The optimized cases should become cython or numba functions for further speed improvements at some point. Again,
#       I haven't done this yet for labor reasons.


def _optimized_compute_powerlaw_energy_moment(
    p: Union[float, np.ndarray],
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = np.inf,
    *,
    order: int = 1,
) -> np.ndarray:
    r"""
    Compute the ``order``-th moment of a power-law electron distribution.

    Evaluates

    .. math::

        \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{\,\mathrm{order} - p}\, d\gamma

    using fully analytic, vectorized expressions.
    """
    # Coerce inputs
    p = np.asarray(p, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    moment = np.zeros_like(p, dtype="f8")

    # Define exponent and critical index
    exponent = order + 1.0 - p

    p_lt = exponent > 0.0  # upper-limit dominated
    p_gt = exponent < 0.0  # lower-limit dominated
    p_eq = exponent == 0.0  # logarithmic

    # p != order+1
    moment[p_lt | p_gt] = (
        gamma_max[p_lt | p_gt] ** exponent[p_lt | p_gt] - gamma_min[p_lt | p_gt] ** exponent[p_lt | p_gt]
    ) / exponent[p_lt | p_gt]

    # p == order+1
    moment[p_eq] = np.log(gamma_max[p_eq] / gamma_min[p_eq])

    return moment


def compute_powerlaw_energy_moment(
    p: Union[float, np.ndarray],
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = np.inf,
    *,
    order: int = 1,
) -> Union[float, np.ndarray]:
    r"""
    Compute the ``order``-th moment of a power-law electron distribution.

    This evaluates

    .. math::

        \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{\,\mathrm{order} - p}\, d\gamma.

    Parameters
    ----------
    p : float or array-like
        Power-law index of the electron Lorentz factor distribution.
    gamma_min : float or array-like
        Minimum Lorentz factor (must be > 0).
    gamma_max : float or array-like
        Maximum Lorentz factor. May be ``inf`` if the integral converges.
    order : int, optional
        Moment order. Default is ``1`` (energy moment).

    Returns
    -------
    float or numpy.ndarray
        Moment value(s).
    """
    __is_scalar = np.isscalar(p)

    # Coerce for validation
    p = np.asarray(p, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    if np.any(gamma_min <= 0):
        raise ValueError("gamma_min must be strictly positive.")

    exponent = order + 1.0 - p

    # Divergent upper-limit case
    if np.any((exponent > 0) & np.isinf(gamma_max)):
        raise ValueError("gamma_max must be finite when p < order + 1 for convergence.")

    result = _optimized_compute_powerlaw_energy_moment(
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        order=order,
    )

    return result.item() if __is_scalar else result


def compute_powerlaw_normalization_from_microphysics(
    B: Union[float, np.ndarray, u.Quantity],
    p: Union[float, np.ndarray],
    epsilon_B: Union[float, np.ndarray],
    epsilon_E: Union[float, np.ndarray],
    *,
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = np.inf,
    mode: str = "gamma",
):
    r"""
    Compute the normalization of a power-law electron distribution from microphysical energy-partition parameters.

    If :math:`\epsilon_B` and :math:`\varepsilon_E` are the fractions of the
    thermal energy which are allocated to magnetic fields and relativistic electrons, then

    .. math::

        \frac{1}{\epsilon_B} \frac{B^2}{8\pi} = u_{\rm int} = \frac{m_e c^2}{\epsilon_E}
        \int_{\gamma_{\min}}^{\gamma_{\max}} N(\gamma) \gamma \, d\gamma.

    For a power-law distribution, this can be computed analytically as is done in this function. We can therefore
    compute the normalization of the power-law distribution.

    Parameters
    ----------
    B : float, array-like, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss, but an :class:`astropy.units.Quantity` may be provided
        to use general units.
    p : float or array-like
        Power-law index of the electron Lorentz factor distribution.
    epsilon_B : float or array-like
        Fraction of post-shock energy in magnetic fields. Default is ``0.1``.
    epsilon_E : float or array-like
        Fraction of post-shock energy in relativistic electrons. Default is ``0.1``.
    gamma_min : float or array-like, optional
        Minimum Lorentz factor. Default is ``1``.
    gamma_max : float or array-like, optional
        Maximum Lorentz factor. Default is ``inf``.
    mode : {'gamma', 'energy'}, optional
        Return normalization for:

        - ``'gamma'``: compute :math:`N_0` in :math:`N(\gamma) = N_0 \gamma^{-p}`.
        - ``'energy'``: compute :math:`K_E` in :math:`N(E) = K_E E^{-p}`.

    Returns
    -------
    astropy.units.Quantity
        Power-law normalization. If ``mode='gamma'``, units are :math:`\mathrm{cm^{-3}}`.
        If ``mode='energy'``, units are :math:`\mathrm{cm^{-3}\ erg^{p-1}}`.
    """
    # Track scalar return
    __is_scalar = np.isscalar(p)

    # Enforce units on the B-field.
    if hasattr(B, "units"):
        B = B.to_value(u.Gauss)

    # Validate inputs before passing off to the low-level callable. This
    # includes checking for convergence of the energy integral.
    p = np.asarray(p, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    if np.any(gamma_min <= 0):
        raise ValueError("gamma_min must be strictly positive.")

    exponent = 2.0 - p
    if np.any((exponent > 0) & np.isinf(gamma_max)):
        raise ValueError("gamma_max must be finite when p < 2 for energy normalization.")

    # Compute N0 in gamma-space
    N0_gamma = _optimized_compute_PL_N0_from_microphysics(
        B=B,
        p=p,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    # Convert normalization if requested
    if mode == "gamma":
        result = N0_gamma * u.cm**-3
    elif mode == "energy":
        result = N0_gamma * electron_rest_energy_cgs ** (p - 1) * u.cm**-3 * u.erg ** (p - 1)
    else:
        raise ValueError("mode must be either 'gamma' or 'energy'.")

    return result.item() if __is_scalar else result


def _optimized_compute_PL_N0_from_microphysics(
    B: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    epsilon_B: Union[float, np.ndarray],
    epsilon_E: Union[float, np.ndarray],
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = np.inf,
) -> np.ndarray:
    """
    Compute the normalization :math:`N_0` of a power-law electron distribution using equipartition arguments.

    Assumes CGS units throughout and returns ``N_0`` in ``cm^{-3}``.
    """
    # Coerce inputs
    B = np.asarray(B, dtype="f8")
    p = np.asarray(p, dtype="f8")
    epsilon_B = np.asarray(epsilon_B, dtype="f8")
    epsilon_E = np.asarray(epsilon_E, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    # Magnetic energy density
    u_B = B**2 / (8.0 * np.pi)

    moment = _optimized_compute_powerlaw_energy_moment(
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        order=1,
    )

    # Normalization
    N0 = (epsilon_E / epsilon_B) * u_B / (electron_rest_energy_cgs * moment)

    return N0


def compute_BR_from_BPL_SED(
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
    the optically thick and thin synchrotron spectra match at :math:`\nu_{\rm brk}`. The equations used are equations
    (16) and (17) from DM22 with minor alterations.

    In treating the electron energy distribution, some additional care is taken based on the value of :math:`p`. In
    particular, we assume a power-law distribution of electron Lorentz factors :math:`\Gamma` such that

    .. math::

        N(\Gamma) d\Gamma = K_e \Gamma^{-p} d\Gamma,\;\; \Gamma_{\rm min} \leq \Gamma \leq \Gamma_{\rm max},

    where :math:`K_e` is the normalization constant, :math:`\Gamma_{\rm min}` is the minimum Lorentz factor,
    and :math:`\Gamma_{\rm max}` is the maximum Lorentz factor. For values of :math:`p > 2`, the total energy is
    dominated by electrons near :math:`\Gamma_{\rm min}`, while for :math:`p < 2`, it is dominated by those near
    :math:`\Gamma_{\rm max}`.

    To account for this, when :math:`p > 2`, we enforce :math:`\Gamma_{\rm max} = \infty` in the energy integral, while
    for :math:`p < 2`, we enforce the upper limit on the energy integral to be :math:`\Gamma_{\rm max}`. This leads to
    a correction factor :math:`\delta` defined as:

    .. math::

        \delta = \begin{cases} 1, & p > 2 \\[6pt]
        \left(\frac{\Gamma_{\rm max}}{\Gamma_{\rm min}}\right)^{2 - p} - 1, & p < 2 \end{cases}

    which modifies the expressions for :math:`B` and :math:`R` accordingly.

    References
    ----------

    .. footbibliography::

    """
    # Validate units of all unit bearing quantities and coerce them to the expected
    # units for the optimized backend.
    if hasattr(nu_brk, "units"):
        nu_brk = nu_brk.to_value(u.GHz)
    if hasattr(F_nu_brk, "units"):
        F_nu_brk = F_nu_brk.to_value(u.Jy)
    if hasattr(distance, "units"):
        distance = distance.to_value(u.Mpc)

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
    B, R = _optimized_compute_BR_from_BPL_SED(
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


def _optimized_compute_BR_from_BPL_SED(
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
):
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

            This function does not handle the case where :math:`p = 2` due to singularities in the underlying equations.
            Users must ensure that :math:`p` is not equal to 2 when calling this function.

    f: float or array-like
        The filling factor :math:`f` of the emitting region. Default is ``0.5``. If provided as a float, the value is
        used for all SEDs. If provided as an array, its shape must be compatible with that of ``nu_brk``.
    theta: float or array-like
        The pitch angle :math:`\theta` between the magnetic field and the line of sight, in radians.
        Default is ``pi/2`` (i.e., perpendicular). If provided as a float, the value is used for all SEDs.
        If provided as an array, its shape must be compatible with that of ``nu_brk``.
    epsilon_B: float or array-like
        The fraction of post-shock energy in magnetic fields, :math:`\\epsilon_B`.
    epsilon_E: float or array-like
        The fraction of post-shock energy in relativistic electrons, :math:`\\epsilon_E`.
    gamma_min: float or array-like
        The minimum Lorentz factor :math:`\\gamma_{\rm min}` of the electron energy.
    gamma_max: float or array-like
        The maximum Lorentz factor :math:`\\gamma_{\rm max}` of the electron energy.

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
    # Because we pre-allocate with ones, we only need to fill in values where p < 2. In THIS IMPLEMENTATION, we ignore
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

    # Compute the magnetic field following equation (16) of DeMarchi+22. We break this into
    # components for clarity. The operation should be heavily CPU bound, so this should not have any impact
    # on optimization.
    #
    # Here nu_brk is in GHz, E_l is in erg, distance is in Mpc, F_nu_brk is in Jy, and B will be in Gauss.
    _B_coeff = 2.50e9 * (nu_brk / 5) * (1 / c_1)
    _B_num = (
        4.69e-23
        * E_l ** (4 - 2 * p)
        * delta**2
        * (epsilon_B / epsilon_E) ** 2
        * c_5
        * np.sin(theta) ** (1 / 2 * (-5 - 2 * p))
    )
    _B_denom = p_norm**2 * distance**2 * (f / 0.5) ** 2 * F_nu_brk * c_6**3

    B = _B_coeff * (_B_num / _B_denom) ** (2 / (13 + 2 * p))

    # Compute the radius following equation (17) of DeMarchi+22. We break this into parts as well on the same
    # basis as above.
    _R_coeff = (2.50e9**-1) * c_1 * (nu_brk / 5) ** -1
    _R_t1 = (12 * epsilon_B) * (c_5 ** (-6 - p)) * (c_6 ** (5 + p))
    _R_t2 = (9.52e25) ** (6 + p) * np.sin(theta) ** 2 * np.pi ** (-5 - p) * distance ** (12 + 2 * p)
    _R_t3 = E_l ** (2 - p) * F_nu_brk ** (6 + p)
    _R_t4 = (epsilon_E * (p_norm) * (f / 0.5)) ** -1

    R = _R_coeff * (_R_t1 * _R_t2 * _R_t3 * _R_t4) ** (1 / (13 + 2 * p))

    return B, R


def compute_BPL_SED_from_BR(
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

    We define the break frequency :math:`\nu_{\rm brk}` as the frequency where these two power laws intersect and the
    corresponding peak flux density :math:`F_{\nu_{\rm brk}}`. By equating the two expressions for :math:`F_\nu` at
    :math:`\nu = \nu_{\rm brk}`, we can solve for :math:`\nu_{\rm brk}` and :math:`F_{\nu_{\rm brk}}` in terms of the
    physical parameters :math:`B`, :math:`R`, and :math:`d`.

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
        1, & p > 2 \\
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
    nu, F_nu = _optimized_compute_BPL_SED_from_BR(
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


def _optimized_compute_BPL_SED_from_BR(
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
    """
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
    # Because we pre-allocate with ones, we only need to fill in values where p < 2. In THIS IMPLEMENTATION, we ignore
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
