"""
General utilities for synchrotron radiation calculations.

This module contains some catch-all utility functions for synchrotron radiation
calculations, including functions to compute the :math:`c_5(p)` and :math:`c_6(p)` coefficients
for synchrotron emissivity and absorption from a power-law population of electrons.
"""

from typing import Union

import numpy as np
from astropy import constants
from astropy import constants as const
from astropy import units as u
from scipy.special import gamma as gamma_func

__all__ = [
    "compute_c5_parameter",
    "compute_c6_parameter",
]

# =========================================== #
# CGS CONSTANTS FOR SYNCHROTRON CALCULATIONS  #
# =========================================== #
_c5_coefficient_cgs = (np.sqrt(3) / (16 * np.pi)) * (constants.e.esu**3 / (constants.m_e * constants.c**2)).cgs.value
_c6_coefficient_cgs = np.sqrt(3) * (np.pi / 72) * (constants.e.esu * constants.m_e**5 * constants.c**10).cgs.value

c_1: u.Quantity = (3 / (4 * np.pi)) * (const.e.esu / (const.m_e**3 * const.c**5))
r"""astropy.units.Quantity: Synchrotron radiation constant :math:`c_1`.

The :math:`c_1` constant is the coefficient appearing in the synchrotron frequency :footcite:p:`1970ranp.book.....P`

.. math::

    \nu_c = \frac{3e}{4\pi m_e c} B\sin \alpha \Gamma^2 = c_1 B \sin \alpha E^2.

Thus,

.. math::

    c_1 = \frac{3}{4\pi} \frac{e}{m_e^3 c^5}.

References
----------
.. footbibliography::
"""
c_1_cgs: float = c_1.cgs.value
"""float: Synchrotron radiation constant :math:`c_1` in CGS units."""

# Cooling normalization constants.
chi_c = (
    (3 * np.sqrt(3) / (32 * np.pi**2))
    * (4 * np.pi / 3) ** (1 / 2)
    * (constants.m_e**3 * constants.c**5 / constants.e.esu**3) ** (1 / 2)
    * constants.sigma_T
)
r"""astropy.units.Quantity: Constant :math:`\chi_c` in synchrotron normalization calculations.

For more details on this parameter, see :ref:`synch_sed_theory`.
"""
chi_c_cgs: float = chi_c.cgs.value
r"""float: Constant :math:`\chi_c` in CGS units."""
log_chi_c_cgs = np.log(chi_c_cgs)
r"""float: Natural logarithm of the constant :math:`\chi_c` in CGS units."""


# =========================================== #
# C5 AND C6 PARAMETERS                        #
# =========================================== #
# These are the c5 and c6 coefficients as defined in Pacholczyk (1970) and used in
# deMarchi+22 which are common in radio supernova modeling.
def compute_c5_parameter(p: Union[float, np.ndarray] = 3.0) -> float:
    r"""
    Compute the :math:`c_5(p)` coefficient for synchrotron assuming a power-law electron population.

    Parameters
    ----------
    p : float or array-like, optional
        Power-law index of the electron Lorentz factor (or energy) distribution,
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
        N_0\, \Gamma^{-p}\, d\Gamma,

    where :math:`N_0` is the normalization of the electron number density
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
        c_5(p)\,N_0\,(m_e c^2)^{p-1}
        \left(B\sin\alpha\right)^{(p+1)/2}
        \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

    where the coefficient :math:`c_5(p)` encapsulates the full integration
    of the synchrotron kernel over the power-law electron distribution and
    depends only on the spectral index :math:`p`.

    The resulting emissivity has CGS units of

    .. math::

        [j_\nu] = \mathrm{erg\ s^{-1}\ cm^{-3}\ Hz^{-1}\ sr^{-1}}.

    The value of :math:`c_5(p)` is given by (see :footcite:p:`1970ranp.book.....P` and
    :footcite:p:`RybickiLightman`)

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
        c_6(p)\,N_0\,(m_e c^2)^{p-1}
        (B\sin\alpha)^{(p+2)/2}
        \left(\frac{\nu}{2 c_1}\right)^{-(p+4)/2},


    The coefficient :math:`c_6(p)` encapsulates the full analytic integration
    of the synchrotron absorption kernel over the power-law electron
    distribution and depends only on the spectral index :math:`p`.

    In the :footcite:p:`1970ranp.book.....P` and :footcite:p:`RybickiLightman` convention appropriate
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


def compute_chi_m_parameter(p: Union[float, np.ndarray] = 3.0) -> float:
    r"""
    Compute the :math:`\chi_m(p)` coefficient for synchrotron normalization calculations.

    :math:`\chi_m(p)` is used in the calculation of the minimum electron Lorentz factor
    normalization in synchrotron SEDs. It takes the form

    .. math::

        \chi_m(p) = \left[\left(\frac{3 \sqrt{3}}{32\pi^2}\right)
                    \left(\frac{4\pi}{3}\right)^{(1-p)/2}
                    \frac{m_e^{(3-p)/2} c^{(5-p)/2}}{q^{(3-p)/2}}
                    \sigma_T\right]

    Parameters
    ----------
    p : float or array-like, optional
        Power-law index of the electron Lorentz factor distribution,
        :math:`N(\Gamma) \propto \Gamma^{-p}`. Default is 3.0.

    Returns
    -------
    float
        The synchrotron normalization coefficient :math:`\chi_m(p)` in CGS units.

    Notes
    -----
    See :ref:`synch_sed_theory` for more details on this parameter.
    """
    _coef = (3 * np.sqrt(3) / (32 * np.pi**2)) * (4 * np.pi / 3) ** ((1 - p) / 2) * constants.sigma_T
    dimless_part = (
        constants.m_e ** ((3 - p) / 2) * constants.c ** ((5 - p) / 2) / constants.e.esu ** ((3 - p) / 2)
    ).cgs.value

    return _coef.cgs.value * dimless_part


def compute_log_chi_m_parameter(p: Union[float, np.ndarray] = 3.0) -> float:
    r"""
    Compute the natural logarithm of the :math:`\chi_m(p)` coefficient for synchrotron normalization calculations.

    See :func:`compute_chi_m_parameter` for details on :math:`\chi_m(p)`.

    Parameters
    ----------
    p : float or array-like, optional
        Power-law index of the electron Lorentz factor distribution,
        :math:`N(\Gamma) \propto \Gamma^{-p}`. Default is 3.0.

    Returns
    -------
    float
        The natural logarithm of the synchrotron normalization coefficient :math:`\chi_m(p)` in CGS units.
    """
    return np.log(compute_chi_m_parameter(p))
