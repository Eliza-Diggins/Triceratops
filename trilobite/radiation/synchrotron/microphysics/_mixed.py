"""
Mixed MJD + PL electron distribution microphysics.

Functions for normalizing and working with distributions that combine
a Maxwell-Jüttner thermal component and a power-law non-thermal component.
"""

from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units as u

from trilobite.utils.misc_utils import ensure_in_units

from ._mjd import _opt_MJD_norm_from_magnetic_field, _opt_MJD_norm_from_thermal
from ._pl import _opt_PL_norm_from_magnetic_field, _opt_PL_norm_from_thermal

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike


def _opt_mixed_norm_from_magnetic_field(
    B: "_ArrayLike",
    Theta: "_ArrayLike",
    p: "_ArrayLike",
    delta: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> "tuple[np.ndarray, np.ndarray]":
    r"""
    Compute normalizations for a mixed Maxwell-Jüttner plus power-law electron distribution from the magnetic field.

    The total electron energy density is split between a thermal (Maxwell-Jüttner) component and a
    non-thermal (power-law) component according to the thermal energy fraction :math:`\delta`:

    .. math::

        \varepsilon_{E,\rm therm} = \delta\,\varepsilon_E,
        \qquad
        \varepsilon_{E,\rm PL} = (1 - \delta)\,\varepsilon_E.

    Each component's normalization is then computed independently via equipartition:

    - :math:`N_{\rm therm}` from :func:`_opt_MJD_norm_from_magnetic_field` with
      :math:`\varepsilon_E \to \delta\,\varepsilon_E`.
    - :math:`N_0` from :func:`_opt_PL_norm_from_magnetic_field` with
      :math:`\varepsilon_E \to (1-\delta)\,\varepsilon_E`.

    Parameters
    ----------
    B : array-like
        Magnetic field strength in Gauss (CGS).
    Theta : array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)` for the thermal component.
    p : array-like
        Power-law index of the non-thermal electron distribution.
    delta : array-like
        Fraction of the total electron energy density in the thermal component. Must satisfy
        :math:`0 \le \delta \le 1`.
    epsilon_B : array-like
        Fraction of post-shock energy in the magnetic field.
    epsilon_E : array-like
        Total fraction of post-shock energy in relativistic electrons.
    gamma_min : array-like, optional
        Minimum Lorentz factor of the power-law component. Default is ``1``.
    gamma_max : array-like, optional
        Maximum Lorentz factor of the power-law component. Default is ``inf``.

    Returns
    -------
    N_therm : numpy.ndarray
        Total number density of the Maxwell-Jüttner component in :math:`\mathrm{cm^{-3}}`.
    N0_pl : numpy.ndarray
        Normalization constant of the power-law component :math:`N_0` in
        :math:`N(\gamma) = N_0\,\gamma^{-p}`, in :math:`\mathrm{cm^{-3}}`.
    """
    delta = np.asarray(delta, dtype="f8")

    N_therm = _opt_MJD_norm_from_magnetic_field(
        B=B,
        Theta=Theta,
        epsilon_B=epsilon_B,
        epsilon_E=delta * epsilon_E,
    )

    N0_pl = _opt_PL_norm_from_magnetic_field(
        B=B,
        p=p,
        epsilon_B=epsilon_B,
        epsilon_E=(1.0 - delta) * epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    return N_therm, N0_pl


def _opt_mixed_norm_from_thermal(
    u_therm: "_ArrayLike",
    Theta: "_ArrayLike",
    p: "_ArrayLike",
    delta: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> "tuple[np.ndarray, np.ndarray]":
    r"""
    Compute normalizations for a mixed Maxwell-Jüttner plus power-law electron distribution.

    Identical in physics to :func:`_opt_mixed_norm_from_magnetic_field`, but takes the thermal
    energy density :math:`u_{\rm therm}` directly rather than inferring it from the magnetic field. The total
    electron energy density is partitioned as

    .. math::

        \varepsilon_{E,\rm therm} = \delta\,\varepsilon_E,
        \qquad
        \varepsilon_{E,\rm PL} = (1 - \delta)\,\varepsilon_E.

    Parameters
    ----------
    u_therm : array-like
        Thermal energy density in :math:`\mathrm{erg\,cm^{-3}}` (CGS).
    Theta : array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)` for the thermal component.
    p : array-like
        Power-law index of the non-thermal electron distribution.
    delta : array-like
        Fraction of the total electron energy density in the thermal component. Must satisfy
        :math:`0 \le \delta \le 1`.
    epsilon_E : array-like
        Total fraction of post-shock energy in relativistic electrons.
    gamma_min : array-like, optional
        Minimum Lorentz factor of the power-law component. Default is ``1``.
    gamma_max : array-like, optional
        Maximum Lorentz factor of the power-law component. Default is ``inf``.

    Returns
    -------
    N_therm : numpy.ndarray
        Total number density of the Maxwell-Jüttner component in :math:`\mathrm{cm^{-3}}`.
    N0_pl : numpy.ndarray
        Normalization constant of the power-law component :math:`N_0` in
        :math:`N(\gamma) = N_0\,\gamma^{-p}`, in :math:`\mathrm{cm^{-3}}`.
    """
    delta = np.asarray(delta, dtype="f8")

    N_therm = _opt_MJD_norm_from_thermal(
        u_therm=u_therm,
        Theta=Theta,
        epsilon_E=delta * epsilon_E,
    )

    N0_pl = _opt_PL_norm_from_thermal(
        u_therm=u_therm,
        p=p,
        epsilon_E=(1.0 - delta) * epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    return N_therm, N0_pl


# ============================================================= #
# Public wrappers                                               #
# ============================================================= #


def compute_MJD_and_PL_norm_from_magnetic_field(
    B: Union[float, np.ndarray, u.Quantity],
    Theta: "_ArrayLike",
    p: "_ArrayLike",
    delta: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    *,
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> "tuple[u.Quantity, u.Quantity]":
    r"""
    Compute normalizations for a mixed Maxwell-Jüttner plus power-law electron distribution from the magnetic field.

    The total electron energy budget is split by the thermal fraction :math:`\delta`:

    .. math::

        \varepsilon_{E,\rm therm} = \delta\,\varepsilon_E,
        \qquad
        \varepsilon_{E,\rm PL} = (1 - \delta)\,\varepsilon_E.

    Each component's normalization is computed independently via equipartition. See
    :func:`compute_MJD_norm_from_magnetic_field` and :func:`compute_PL_norm_from_magnetic_field`
    for the individual closures.

    Parameters
    ----------
    B : float, array-like, or ~astropy.units.Quantity
        Magnetic field strength. Bare values are interpreted as Gauss.
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)` for the thermal component.
    p : float or array-like
        Power-law index of the non-thermal electron distribution.
    delta : float or array-like
        Fraction of the total electron energy density in the thermal component
        (:math:`0 \le \delta \le 1`).
    epsilon_B : float or array-like
        Fraction of post-shock energy in the magnetic field.
    epsilon_E : float or array-like
        Total fraction of post-shock energy in relativistic electrons.
    gamma_min : float or array-like, optional
        Minimum Lorentz factor of the power-law component. Default is ``1``.
    gamma_max : float or array-like, optional
        Maximum Lorentz factor of the power-law component. Default is ``inf``.

    Returns
    -------
    N_therm : ~astropy.units.Quantity
        Total number density of the Maxwell-Jüttner component in :math:`\mathrm{cm^{-3}}`.
    N0_pl : ~astropy.units.Quantity
        Power-law normalization constant :math:`N_0` in :math:`N(\gamma) = N_0\,\gamma^{-p}`,
        in :math:`\mathrm{cm^{-3}}`.
    """
    B = ensure_in_units(B, u.G)
    delta = np.asarray(delta, dtype="f8")

    if np.any((delta < 0) | (delta > 1)):
        raise ValueError("delta must be in [0, 1].")

    N_therm, N0_pl = _opt_mixed_norm_from_magnetic_field(
        B=B,
        Theta=Theta,
        p=p,
        delta=delta,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    return N_therm * u.cm**-3, N0_pl * u.cm**-3


def compute_MJD_and_PL_norm_from_thermal_energy_density(
    u_therm: Union[float, np.ndarray, u.Quantity],
    Theta: "_ArrayLike",
    p: "_ArrayLike",
    delta: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    *,
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> "tuple[u.Quantity, u.Quantity]":
    r"""
    Compute normalizations for a mixed Maxwell-Jüttner plus power-law electron distribution.

    Identical in physics to :func:`compute_MJD_and_PL_norm_from_magnetic_field` but takes the thermal
    energy density directly. The total electron energy budget is split by the thermal fraction
    :math:`\delta`:

    .. math::

        \varepsilon_{E,\rm therm} = \delta\,\varepsilon_E,
        \qquad
        \varepsilon_{E,\rm PL} = (1 - \delta)\,\varepsilon_E.

    Parameters
    ----------
    u_therm : float, array-like, or ~astropy.units.Quantity
        Thermal energy density. Bare values are interpreted as :math:`\mathrm{erg\,cm^{-3}}`.
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)` for the thermal component.
    p : float or array-like
        Power-law index of the non-thermal electron distribution.
    delta : float or array-like
        Fraction of the total electron energy density in the thermal component
        (:math:`0 \le \delta \le 1`).
    epsilon_E : float or array-like
        Total fraction of post-shock energy in relativistic electrons.
    gamma_min : float or array-like, optional
        Minimum Lorentz factor of the power-law component. Default is ``1``.
    gamma_max : float or array-like, optional
        Maximum Lorentz factor of the power-law component. Default is ``inf``.

    Returns
    -------
    N_therm : ~astropy.units.Quantity
        Total number density of the Maxwell-Jüttner component in :math:`\mathrm{cm^{-3}}`.
    N0_pl : ~astropy.units.Quantity
        Power-law normalization constant :math:`N_0` in :math:`N(\gamma) = N_0\,\gamma^{-p}`,
        in :math:`\mathrm{cm^{-3}}`.
    """
    u_therm = ensure_in_units(u_therm, u.erg / u.cm**3)
    delta = np.asarray(delta, dtype="f8")

    if np.any((delta < 0) | (delta > 1)):
        raise ValueError("delta must be in [0, 1].")

    N_therm, N0_pl = _opt_mixed_norm_from_thermal(
        u_therm=u_therm,
        Theta=Theta,
        p=p,
        delta=delta,
        epsilon_E=epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    return N_therm * u.cm**-3, N0_pl * u.cm**-3


# ============================================================= #
# Backward-compat aliases (old private names -> new names)       #
# ============================================================= #
_opt_normalize_MJD_and_PL_from_magnetic_field = _opt_mixed_norm_from_magnetic_field
_opt_normalize_MJD_and_PL_from_thermal_energy_density = _opt_mixed_norm_from_thermal
