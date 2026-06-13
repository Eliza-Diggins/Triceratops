"""
Base utilities for synchrotron microphysics.

Contains the equipartition magnetic field utility shared across all distribution families.
"""

import numpy as np
from astropy import units as u

from trilobite.utils.misc_utils import ensure_in_units


def _opt_equipart_magnetic_field(
    u_therm: "float | np.ndarray",
    epsilon_B: "float | np.ndarray",
) -> np.ndarray:
    """
    Compute the magnetic field strength from the thermal energy density and epsilon_B.

    Assumes CGS units throughout and returns B in Gauss.
    """
    u_therm = np.asarray(u_therm, dtype="f8")
    epsilon_B = np.asarray(epsilon_B, dtype="f8")

    B = np.sqrt(8.0 * np.pi * epsilon_B * u_therm)

    return B.reshape(()) if B.ndim == 0 else B


def compute_equipartition_magnetic_field(
    u_therm,
    epsilon_B,
) -> u.Quantity:
    r"""
    Compute the magnetic field strength from the thermal energy density and epsilon_B.

    If :math:`\varepsilon_B` is the fraction of the thermal energy density allocated to magnetic fields,
    then the magnetic field strength can be computed as

    .. math::

        B = \sqrt{8 \pi \varepsilon_B u_{\rm int}}.

    Parameters
    ----------
    u_therm : float, array-like, or astropy.units.Quantity
        Thermal energy density. Default units are ``erg cm^{-3}``, but an :class:`astropy.units.Quantity` may
        be provided to use general units.
    epsilon_B : float or array-like
        Fraction of post-shock energy in magnetic fields. Default is ``0.1``.

    Returns
    -------
    astropy.units.Quantity
        Magnetic field strength in Gauss.
    """
    # Enforce units on the thermal energy density.
    u_therm = ensure_in_units(u_therm, u.erg / u.cm**3)

    # Compute B-field
    B = _opt_equipart_magnetic_field(
        u_therm=u_therm,
        epsilon_B=epsilon_B,
    )

    return B * u.Gauss
