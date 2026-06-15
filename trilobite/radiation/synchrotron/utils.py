"""
General utilities for synchrotron radiation calculations.

This module contains some catch-all utility functions for synchrotron radiation
calculations, including functions to compute the :math:`c_5(p)` and :math:`c_6(p)` coefficients
for synchrotron emissivity and absorption from a power-law population of electrons. Most of the
available utilities here are related to constants required for various synchrotron computations.
"""

from typing import Union

import numpy as np
from astropy import constants
from astropy import constants as const
from astropy import units as u
from scipy.special import gamma as gamma_func

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

c_1_gamma: u.Quantity = (3 / (4 * np.pi)) * (const.e.esu / (const.m_e * const.c))
r"""astropy.units.Quantity: Synchrotron constant :math:`c_{1,\gamma}`.

The :math:`c_{1,\gamma}` constant is the coefficient appearing in the synchrotron frequency when expressed
 in terms of the electron Lorentz factor :math:`\Gamma` rather than energy:

 .. math::

    \nu_c = \frac{3e}{4\pi m_e c} B\sin \alpha \Gamma^2 = c_{1,\gamma} B \sin \alpha \Gamma^2.
"""
c_1_gamma_cgs: float = c_1_gamma.cgs.value
r"""float: Synchrotron constant :math:`c_{1,\gamma}` in CGS units."""
_log_c_1_gamma_cgs = np.log(c_1_gamma_cgs)

c_1_gamma_iso: u.Quantity = (3 / (2 * np.pi**2)) * (const.e.esu / (const.m_e * const.c))
r"""astropy.units.Quantity: Synchrotron constant :math:`c_{1,\gamma}^{\mathrm{iso}}` for isotropic distributions.

The :math:`c_{1,\gamma}^{\mathrm{iso}}` constant is the coefficient appearing in the synchrotron frequency when
expressed
in terms of the electron Lorentz factor :math:`\Gamma` and assuming an isotropic distribution of pitch angles.
The isotropic pitch-angle averaging factor :math:`2/\pi` is included in this constant, so that the critical
frequency for an isotropic distribution can be written as

.. math::

    \nu_c^{\mathrm{iso}} = \frac{3e}{2\pi^2 m_e c} B \Gamma^2 = c_{1,\gamma}^{\mathrm{iso}} B \Gamma^2.
"""
c_1_gamma_iso_cgs: float = c_1_gamma_iso.cgs.value
r"""float: Synchrotron constant :math:`c_{1,\gamma}^{\mathrm{iso}}` for isotropic distributions in CGS units."""
_log_c_1_gamma_iso_cgs = np.log(c_1_gamma_iso.cgs.value)
# ------------------------------------------------------------------
# Synchrotron normalization constants (chi)
# ------------------------------------------------------------------
chi_cgs = 4 * _c5_coefficient_cgs
r"""
float
    Numerical synchrotron normalization coefficient
    :math:`\chi \equiv 4 c_5`.

    This constant represents **only the coefficient prefactor** extracted from
    the standard synchrotron emissivity constant :math:`c_5` (in CGS units).
    It is used as a bookkeeping convenience in SED normalization expressions
    and does **not** by itself represent a physical power, emissivity, or flux.

    In particular, :math:`\chi` does *not* include:

    - electron distribution normalization,
    - magnetic-field or frequency dependence,
    - pitch-angle averaging,
    - geometric or distance factors.

    These contributions are applied explicitly elsewhere in the SED
    construction.
"""

_log_chi_cgs = np.log(chi_cgs)

_chi_abs_cgs = chi_cgs / constants.m_e.cgs.value
r"""float: SSA prefactor :math:`\chi/m_e = \sqrt{3}e^3/(4\pi m_e^2 c^2)` in CGS."""
_log_chi_abs_cgs = np.log(_chi_abs_cgs)
r"""float: Natural log of the SSA prefactor :math:`\chi/m_e`."""
r"""
float
    Natural logarithm of the synchrotron normalization coefficient
    :math:`\chi = 4 c_5`.

    Stored separately to support numerically stable, log-space SED
    normalization and inference workflows.
"""

chi_cgs_iso = (2 / np.pi) * chi_cgs
r"""
float
    Synchrotron normalization coefficient for an **isotropic pitch-angle
    distribution**.

    This quantity applies the standard isotropic pitch-angle averaging factor
    :math:`2 / \pi` to the coefficient-only normalization constant
    :math:`\chi = 4 c_5`.

    As with ``chi_cgs``, this constant represents only the numerical prefactor
    and must be combined with additional physical factors to construct a full
    synchrotron emissivity or SED normalization.
"""

_log_chi_cgs_iso = np.log(chi_cgs_iso)
r"""
float
    Natural logarithm of the isotropic-pitch-angle synchrotron normalization
    coefficient.

    Used internally for log-space normalization and inference when
    pitch-angle-averaged synchrotron emissivities are assumed.
"""


# =========================================== #
# C5 AND C6 PARAMETERS                        #
# =========================================== #
# These are the c5 and c6 coefficients as defined in Pacholczyk (1970) and used in
# deMarchi+22 which are common in radio supernova modeling.
def compute_c5_parameter(
    p: Union[float, np.ndarray] = 3.0,
    pitch_average: bool = False,
) -> Union[float, np.ndarray]:
    r"""
    Compute the synchrotron emissivity coefficient :math:`c_5(p)`.

    This coefficient appears in the optically thin synchrotron emissivity of a
    power-law electron population,

    .. math::

        N(\Gamma)\,d\Gamma = N_0 \Gamma^{-p}\,d\Gamma,

    where :math:`\Gamma` is the electron Lorentz factor and :math:`N_0` is the
    number-density normalization. For a fixed pitch angle :math:`\alpha`, the
    emissivity can be written in the Pacholczyk form

    .. math::

        j_\nu
        =
        c_5(p)\,
        N_0\,
        (m_e c^2)^{p-1}
        \left(B\sin\alpha\right)^{(p+1)/2}
        \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2},

    where :math:`j_\nu` is the power per unit volume, frequency, and solid
    angle.

    Parameters
    ----------
    p : float or array-like, optional
        Power-law index of the electron Lorentz-factor distribution,

        .. math::

            N(\Gamma) \propto \Gamma^{-p}.

        Default is ``3.0``.
    pitch_average : bool, optional
        If `False`, return the fixed-pitch-angle coefficient :math:`c_5(p)`,
        leaving the pitch-angle dependence explicitly in the factor
        :math:`(B\sin\alpha)^{(p+1)/2}`.

        If `True`, multiply :math:`c_5(p)` by the isotropic pitch-angle average

        .. math::

            \left\langle
            \sin^{(p+1)/2}\alpha
            \right\rangle
            =
            \frac{\sqrt{\pi}}{2}
            \frac{
                \Gamma\left(\frac{p+5}{4}\right)
            }{
                \Gamma\left(\frac{p+7}{4}\right)
            },

        so that the emissivity should instead be written with
        :math:`B^{(p+1)/2}` rather than
        :math:`(B\sin\alpha)^{(p+1)/2}`.

        Default is `False`.

    Returns
    -------
    float or numpy.ndarray
        The synchrotron emissivity coefficient :math:`c_5(p)` in CGS units.
        If ``p`` is array-like, the returned value has the broadcast shape of
        ``p``.

    Notes
    -----
    The single-electron synchrotron power per unit frequency is

    .. math::

        P(\nu, \Gamma)
        =
        \frac{\sqrt{3}\,e^3 B}{m_e c^2}
        \sin\alpha\,
        F\left(\frac{\nu}{\nu_c}\right),

    where

    .. math::

        F(x) = x\int_x^\infty K_{5/3}(z)\,dz

    is the synchrotron kernel, and

    .. math::

        \nu_c
        =
        \frac{3eB\sin\alpha}{4\pi m_e c}\Gamma^2

    is the critical frequency.

    Integrating this expression over a power-law electron population gives

    .. math::

        c_5(p)
        =
        \frac{\sqrt{3}}{16\pi}
        \frac{e^3}{m_e c^2}
        \frac{p + 7/3}{p + 1}
        \Gamma\left(\frac{3p - 1}{12}\right)
        \Gamma\left(\frac{3p + 7}{12}\right).

    For an isotropic distribution of electron directions, the folded
    pitch-angle probability density on
    :math:`0 \le \alpha \le \pi/2` is

    .. math::

        P(\alpha) = \sin\alpha.

    Therefore,

    .. math::

        \left\langle \sin^k\alpha \right\rangle
        =
        \int_0^{\pi/2}\sin^{k+1}\alpha\,d\alpha
        =
        \frac{\sqrt{\pi}}{2}
        \frac{
            \Gamma\left(\frac{k+2}{2}\right)
        }{
            \Gamma\left(\frac{k+3}{2}\right)
        }.

    Setting :math:`k=(p+1)/2` gives the pitch-angle correction used when
    ``pitch_average=True``.

    References
    ----------
    .. footbibliography::
    """
    p = np.asarray(p)

    # Fixed-pitch-angle, dimensionless part of c_5(p).
    dimless_part = (p + 7.0 / 3.0) / (p + 1.0) * gamma_func((3.0 * p - 1.0) / 12.0) * gamma_func((3.0 * p + 7.0) / 12.0)

    if pitch_average:
        # Isotropic pitch-angle average of sin(alpha)^((p + 1) / 2).
        pitch_angle_factor = 0.5 * np.sqrt(np.pi) * gamma_func((p + 5.0) / 4.0) / gamma_func((p + 7.0) / 4.0)
        dimless_part *= pitch_angle_factor

    c5 = _c5_coefficient_cgs * dimless_part

    # Preserve scalar-in, scalar-out behavior.
    if c5.ndim == 0:
        return float(c5)

    return c5


def compute_c6_parameter(
    p: Union[float, np.ndarray] = 3.0,
    pitch_average: bool = False,
) -> Union[float, np.ndarray]:
    r"""
    Compute the synchrotron self-absorption coefficient :math:`c_6(p)`.

    This coefficient appears in the synchrotron self-absorption coefficient of
    a power-law electron population,

    .. math::

        N(\Gamma)\,d\Gamma = N_0 \Gamma^{-p}\,d\Gamma,

    where :math:`\Gamma` is the electron Lorentz factor and :math:`N_0` is the
    number-density normalization. For a fixed pitch angle :math:`\alpha`, the
    absorption coefficient can be written in the Pacholczyk form

    .. math::

        \alpha_\nu
        =
        c_6(p)\,
        N_0\,
        (m_e c^2)^{p-1}
        \left(B\sin\alpha\right)^{(p+2)/2}
        \left(\frac{\nu}{2c_1}\right)^{-(p+4)/2},

    where :math:`\alpha_\nu` has units of inverse length.

    Parameters
    ----------
    p : float or array-like, optional
        Power-law index of the electron Lorentz-factor distribution,

        .. math::

            N(\Gamma) \propto \Gamma^{-p}.

        Default is ``3.0``.
    pitch_average : bool, optional
        If `False`, return the fixed-pitch-angle coefficient :math:`c_6(p)`,
        leaving the pitch-angle dependence explicitly in the factor
        :math:`(B\sin\alpha)^{(p+2)/2}`.

        If `True`, multiply :math:`c_6(p)` by the isotropic pitch-angle average

        .. math::

            \left\langle
            \sin^{(p+2)/2}\alpha
            \right\rangle
            =
            \frac{\sqrt{\pi}}{2}
            \frac{
                \Gamma\left(\frac{p+6}{4}\right)
            }{
                \Gamma\left(\frac{p+8}{4}\right)
            },

        so that the absorption coefficient should instead be written with
        :math:`B^{(p+2)/2}` rather than
        :math:`(B\sin\alpha)^{(p+2)/2}`.

        Default is `False`.

    Returns
    -------
    float or numpy.ndarray
        The synchrotron self-absorption coefficient :math:`c_6(p)` in CGS
        units. If ``p`` is array-like, the returned value has the broadcast
        shape of ``p``.

    Notes
    -----
    The synchrotron self-absorption coefficient can be written as

    .. math::

        \alpha_\nu
        =
        -\frac{1}{8\pi m_e \nu^2}
        \int d\Gamma\,
        P(\nu,\Gamma)\,
        \Gamma^2
        \frac{\partial}{\partial\Gamma}
        \left[
            \frac{1}{\Gamma^2}
            \frac{dN}{d\Gamma}
        \right].

    For a power-law electron population,

    .. math::

        \frac{dN}{d\Gamma}
        =
        N_0\Gamma^{-p},

    the derivative term becomes

    .. math::

        \Gamma^2
        \frac{\partial}{\partial\Gamma}
        \left[
            \frac{1}{\Gamma^2}
            \frac{dN}{d\Gamma}
        \right]
        =
        -(p+2)N_0\Gamma^{-(p+1)}.

    Therefore,

    .. math::

        \alpha_\nu
        =
        \frac{(p+2)N_0}{8\pi m_e \nu^2}
        \int d\Gamma\,
        P(\nu,\Gamma)\,
        \Gamma^{-(p+1)}.

    Carrying out the synchrotron-kernel integral gives

    .. math::

        c_6(p)
        =
        \frac{\sqrt{3}\, e^3}{16\pi m_e}
        \left(\frac{3e}{2\pi m_e^3 c^5}\right)^{p/2}
        (p + 2)\,
        \Gamma\left(\frac{3p + 2}{12}\right)
        \Gamma\left(\frac{3p + 10}{12}\right).

    This implementation uses an algebraically equivalent form in which all
    dimensional constants are grouped into the global prefactor
    ``_c6_coefficient_cgs`` and the remaining dependence on :math:`p` is
    purely dimensionless:

    .. math::

        \left(p+\frac{10}{3}\right)
        \Gamma\left(\frac{3p+2}{12}\right)
        \Gamma\left(\frac{3p+10}{12}\right).

    For an isotropic distribution of electron directions, the folded
    pitch-angle probability density on
    :math:`0 \le \alpha \le \pi/2` is

    .. math::

        P(\alpha) = \sin\alpha.

    Therefore,

    .. math::

        \left\langle \sin^k\alpha \right\rangle
        =
        \int_0^{\pi/2}\sin^{k+1}\alpha\,d\alpha
        =
        \frac{\sqrt{\pi}}{2}
        \frac{
            \Gamma\left(\frac{k+2}{2}\right)
        }{
            \Gamma\left(\frac{k+3}{2}\right)
        }.

    Setting :math:`k=(p+2)/2` gives the pitch-angle correction used when
    ``pitch_average=True``.

    References
    ----------
    .. footbibliography::
    """
    p = np.asarray(p)

    # Fixed-pitch-angle, dimensionless part of c_6(p).
    dimless_part = (p + 10.0 / 3.0) * gamma_func((3.0 * p + 2.0) / 12.0) * gamma_func((3.0 * p + 10.0) / 12.0)

    if pitch_average:
        # Isotropic pitch-angle average of sin(alpha)^((p + 2) / 2).
        pitch_angle_factor = 0.5 * np.sqrt(np.pi) * gamma_func((p + 6.0) / 4.0) / gamma_func((p + 8.0) / 4.0)
        dimless_part *= pitch_angle_factor

    c6 = _c6_coefficient_cgs * dimless_part

    # Preserve scalar-in, scalar-out behavior.
    if c6.ndim == 0:
        return float(c6)

    return c6
