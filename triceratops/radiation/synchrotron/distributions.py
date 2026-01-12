"""
Routines for computing properties of electron distributions used in synchrotron radiation modeling.

This module provides functions to compute moments of power-law electron distributions
and to derive the normalization of these distributions based on microphysical parameters
such as magnetic field strength and energy partition fractions. These tools are essential
for modeling synchrotron emission in astrophysical contexts, particularly in transient events
like supernovae and gamma-ray bursts.
"""

from typing import Union

import numpy as np
from astropy import units as u

from triceratops.radiation.constants import electron_rest_energy_cgs

__all__ = [
    "compute_powerlaw_energy_moment",
    "compute_powerlaw_normalization_from_microphysics",
]

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
