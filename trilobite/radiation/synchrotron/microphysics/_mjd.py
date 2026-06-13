"""
Maxwell-Jüttner distribution (MJD) electron microphysics.

All functions in this module operate on ultra-relativistic Maxwell-Jüttner distributions of the form
    N(gamma) = N_therm / (2 * Theta^3) * gamma^2 * exp(-gamma/Theta)
"""

from math import factorial
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units as u

from trilobite.radiation.constants import c_cgs, electron_rest_energy_cgs, sigma_T_cgs
from trilobite.utils.misc_utils import ensure_in_units

from ._base import _opt_equipart_magnetic_field

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingScalarLike


def _opt_MJD_moment(
    Theta: "_ArrayLike",
    order: int = 1,
) -> np.ndarray:
    r"""
    Compute the shape factor for the ``order``-th moment of the ultra-relativistic Maxwell-Jüttner distribution.

    For the distribution :math:`N(\gamma) = \frac{N_{\rm therm}}{2\Theta^3}\gamma^2 e^{-\gamma/\Theta}`,
    the physical moment is :math:`M^{(\ell)} = N_{\rm therm} \times S^{(\ell)}(\Theta)` where the
    shape factor is

    .. math::

        S^{(\ell)}(\Theta) = \frac{(\ell+2)!}{2}\,\Theta^\ell.

    Parameters
    ----------
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.
    order : int, optional
        Moment order :math:`\ell \ge 0`. Default is ``1``.

    Returns
    -------
    float or ~numpy.ndarray
        Shape factor :math:`S^{(\ell)}(\Theta)`. Multiply by :math:`N_{\rm therm}` to obtain
        the physical moment in :math:`\mathrm{cm^{-3}}`.
    """
    Theta = np.asarray(Theta, dtype="f8")
    shape = (factorial(order + 2) / 2.0) * Theta**order
    return shape.reshape(()) if shape.ndim == 0 else shape


def _opt_MJD_n_eff(
    N_therm: "_ArrayLike",
    Theta: "_ArrayLike",
) -> np.ndarray:
    r"""
    Compute the effective radiating electron number density for a Maxwell-Jüttner distribution.

    Weights electrons by :math:`\gamma^2` reflecting synchrotron power scaling:

    .. math::

        n_{\rm eff} = \int_0^\infty \gamma^2\,N(\gamma)\,d\gamma = 12\,N_{\rm therm}\,\Theta^2.

    Parameters
    ----------
    N_therm : float or array-like
        Total number density in :math:`\mathrm{cm^{-3}}`.
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.

    Returns
    -------
    float or ~numpy.ndarray
        Effective number density in :math:`\mathrm{cm^{-3}}`.
    """
    N_therm = np.asarray(N_therm, dtype="f8")
    n_eff = N_therm * _opt_MJD_moment(Theta, order=2)
    return n_eff.reshape(()) if n_eff.ndim == 0 else n_eff


def _opt_MJD_norm_from_magnetic_field(
    B: "_ArrayLike",
    Theta: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
) -> np.ndarray:
    r"""
    Compute the total number density of a Maxwell-Jüttner electron distribution from the magnetic field.

    The ultra-relativistic Maxwell-Jüttner distribution is parameterized as

    .. math::

        N(\gamma) = \frac{N_{\rm therm}}{2\Theta^3}\,\gamma^2\,e^{-\gamma/\Theta},

    where :math:`\Theta = kT/(m_e c^2)` is the dimensionless temperature and the prefactor ensures
    :math:`\int_0^\infty N(\gamma)\,d\gamma = N_{\rm therm}`.

    The mean energy density of this distribution is approximated by :footcite:p:`1998ApJ...498..313G`

    .. math::

        U_e \approx N_{\rm therm}\,m_e c^2\,\Theta\,\frac{6 + 15\Theta}{4 + 5\Theta},

    which interpolates between the non-relativistic limit (:math:`U_e \approx \tfrac{3}{2}N_{\rm therm}kT`)
    and the ultra-relativistic limit (:math:`U_e \approx 3N_{\rm therm}kT`).

    Equipartition between the magnetic and electron energy densities then gives

    .. math::

        \varepsilon_E\,\frac{B^2}{8\pi\varepsilon_B}
        = N_{\rm therm}\,m_e c^2\,\Theta\,\frac{6 + 15\Theta}{4 + 5\Theta},

    which is solved for :math:`N_{\rm therm}`.

    Parameters
    ----------
    B : array-like
        Magnetic field strength in Gauss (CGS).
    Theta : array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.
    epsilon_B : array-like
        Fraction of post-shock energy in the magnetic field.
    epsilon_E : array-like
        Fraction of post-shock energy in relativistic electrons.

    Returns
    -------
    numpy.ndarray
        Total number density :math:`N_{\rm therm}` of the Maxwell-Jüttner distribution in
        :math:`\mathrm{cm^{-3}}`. Scalar inputs return a scalar.

    References
    ----------
    .. footbibliography::
    """
    log_B = np.log(np.asarray(B, dtype="f8"))
    Theta = np.asarray(Theta, dtype="f8")
    log_Theta = np.log(Theta)
    epsilon_B = np.asarray(epsilon_B, dtype="f8")
    epsilon_E = np.asarray(epsilon_E, dtype="f8")

    a_Theta = (4 + 5 * Theta) / (6 + 15 * Theta)
    log_N = (
        np.log(epsilon_E / (8 * np.pi * epsilon_B))
        + np.log(a_Theta)
        + 2 * log_B
        - log_Theta
        - np.log(electron_rest_energy_cgs)
    )
    N = np.exp(log_N)

    return N.reshape(()) if N.ndim == 0 else N


def _opt_MJD_norm_from_thermal(
    u_therm: "_ArrayLike",
    Theta: "_ArrayLike",
    epsilon_E: "_ArrayLike",
) -> np.ndarray:
    r"""
    Compute the total number density of a Maxwell-Jüttner electron distribution from the thermal energy density.

    Identical in physics to :func:`_opt_MJD_norm_from_magnetic_field`, but takes the thermal energy
    density :math:`u_{\rm therm}` directly rather than inferring it from the magnetic field via
    :math:`\varepsilon_B`. The equipartition condition becomes

    .. math::

        \varepsilon_E\,u_{\rm therm}
        = N_{\rm therm}\,m_e c^2\,\Theta\,\frac{6 + 15\Theta}{4 + 5\Theta},

    which is solved for :math:`N_{\rm therm}`.

    Parameters
    ----------
    u_therm : array-like
        Thermal energy density in :math:`\mathrm{erg\,cm^{-3}}` (CGS).
    Theta : array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.
    epsilon_E : array-like
        Fraction of post-shock energy in relativistic electrons.

    Returns
    -------
    numpy.ndarray
        Total number density :math:`N_{\rm therm}` of the Maxwell-Jüttner distribution in
        :math:`\mathrm{cm^{-3}}`. Scalar inputs return a scalar.

    References
    ----------
    .. footbibliography::
    """
    log_u = np.log(np.asarray(u_therm, dtype="f8"))
    Theta = np.asarray(Theta, dtype="f8")
    log_Theta = np.log(Theta)
    epsilon_E = np.asarray(epsilon_E, dtype="f8")

    a_Theta = (4 + 5 * Theta) / (6 + 15 * Theta)
    log_N = np.log(epsilon_E) + np.log(a_Theta) + log_u - log_Theta - np.log(electron_rest_energy_cgs)
    N = np.exp(log_N)

    return N.reshape(()) if N.ndim == 0 else N


def _opt_MJD_bol_emissivity_from_magnetic_field(
    B: "_ArrayLike",
    N_therm: "_ArrayLike",
    Theta: "_ArrayLike",
) -> np.ndarray:
    """
    Compute the bolometric synchrotron emissivity for a Maxwell-Jüttner electron distribution.

    Assumes CGS units throughout and returns emissivity in erg s^-1 cm^-3.
    """
    B = np.asarray(B, dtype="f8")
    N_therm = np.asarray(N_therm, dtype="f8")

    U_B = B**2 / (8.0 * np.pi)
    n_eff = _opt_MJD_n_eff(N_therm, Theta)
    emiss = (4.0 / 3.0) * sigma_T_cgs * c_cgs * U_B * n_eff

    return emiss.reshape(()) if emiss.ndim == 0 else emiss


def _opt_MJD_bol_emissivity_from_thermal_full(
    u_therm: "_ArrayLike",
    Theta: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
) -> np.ndarray:
    """
    Compute the bolometric synchrotron emissivity for a Maxwell-Jüttner distribution.

    Assumes CGS units throughout and returns emissivity in erg s^-1 cm^-3.
    """
    B = _opt_equipart_magnetic_field(u_therm=u_therm, epsilon_B=epsilon_B)
    N_therm = _opt_MJD_norm_from_thermal(u_therm=u_therm, Theta=Theta, epsilon_E=epsilon_E)
    return _opt_MJD_bol_emissivity_from_magnetic_field(B=B, N_therm=N_therm, Theta=Theta)


# ============================================================= #
# Public wrappers                                               #
# ============================================================= #


def compute_MJD_moment(
    Theta: "_ArrayLike",
    *,
    order: int = 1,
) -> "_ArrayLike":
    r"""
    Compute the shape factor for the ``order``-th moment of the ultra-relativistic Maxwell-Jüttner distribution.

    This evaluates

    .. math::

        S^{(\ell)}(\Theta) = \frac{(\ell+2)!}{2}\,\Theta^\ell,

    so that the physical moment :math:`M^{(\ell)} = N_{\rm therm}\,S^{(\ell)}(\Theta)`.

    Parameters
    ----------
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.
    order : int, optional
        Moment order :math:`\ell \ge 0`. Default is ``1``.

    Returns
    -------
    float or ~numpy.ndarray
        Shape factor :math:`S^{(\ell)}(\Theta)`.
    """
    return _opt_MJD_moment(Theta, order=order)


def compute_MJD_total_number_density(
    N_therm: Union[float, np.ndarray, u.Quantity],
) -> u.Quantity:
    r"""
    Return the total number density of a Maxwell-Jüttner electron distribution.

    By the normalization convention used throughout this module,
    :math:`\int_0^\infty N(\gamma)\,d\gamma = N_{\rm therm}`, so this function
    simply attaches units and returns :math:`N_{\rm therm}`.

    Parameters
    ----------
    N_therm : float, array-like, or ~astropy.units.Quantity
        Total number density. Bare values are interpreted as :math:`\mathrm{cm^{-3}}`.

    Returns
    -------
    ~astropy.units.Quantity
        Total electron number density in :math:`\mathrm{cm^{-3}}`.
    """
    return ensure_in_units(N_therm, u.cm**-3) * u.cm**-3


def compute_MJD_effective_number_density(
    N_therm: Union[float, np.ndarray, u.Quantity],
    Theta: "_ArrayLike",
) -> u.Quantity:
    r"""
    Compute the effective radiating electron number density for a Maxwell-Jüttner distribution.

    Because the single-electron synchrotron power scales as :math:`P_{\rm syn} \propto \gamma^2`,
    the effective number density is

    .. math::

        n_{\rm eff} = \int_0^\infty \gamma^2\,N(\gamma)\,d\gamma = 12\,N_{\rm therm}\,\Theta^2.

    Parameters
    ----------
    N_therm : float, array-like, or ~astropy.units.Quantity
        Total number density. Bare values are interpreted as :math:`\mathrm{cm^{-3}}`.
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.

    Returns
    -------
    ~astropy.units.Quantity
        Effective radiating electron number density in :math:`\mathrm{cm^{-3}}`.
    """
    N_therm_cgs = ensure_in_units(N_therm, u.cm**-3)
    return _opt_MJD_n_eff(N_therm_cgs, Theta) * u.cm**-3


def compute_MJD_mean_gamma(
    Theta: "_ArrayLike",
) -> "_ArrayLike":
    r"""
    Compute the mean Lorentz factor of a Maxwell-Jüttner electron distribution.

    For the ultra-relativistic Maxwell-Jüttner distribution,

    .. math::

        \langle\gamma\rangle = \frac{M^{(1)}}{M^{(0)}} = 3\,\Theta.

    Parameters
    ----------
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.

    Returns
    -------
    float or ~numpy.ndarray
        Mean Lorentz factor.
    """
    return 3.0 * np.asarray(Theta, dtype="f8")


def compute_MJD_mean_energy(
    Theta: "_ArrayLike",
) -> u.Quantity:
    r"""
    Compute the mean electron energy of a Maxwell-Jüttner distribution.

    For the ultra-relativistic Maxwell-Jüttner distribution,

    .. math::

        \langle E \rangle = m_e c^2\,\langle\gamma\rangle = 3\,m_e c^2\,\Theta.

    Parameters
    ----------
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.

    Returns
    -------
    ~astropy.units.Quantity
        Mean electron energy in :math:`\mathrm{erg}`.
    """
    return 3.0 * np.asarray(Theta, dtype="f8") * electron_rest_energy_cgs * u.erg


def compute_MJD_norm_from_magnetic_field(
    B: Union[float, np.ndarray, u.Quantity],
    Theta: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
) -> u.Quantity:
    r"""
    Compute the total number density of a Maxwell-Jüttner electron distribution from the magnetic field.

    Uses the equipartition condition :math:`\varepsilon_E\,B^2/(8\pi\varepsilon_B) = U_e` with the
    mean energy density approximation :footcite:p:`1998ApJ...498..313G`

    .. math::

        U_e \approx N_{\rm therm}\,m_e c^2\,\Theta\,\frac{6 + 15\Theta}{4 + 5\Theta}

    to solve for the total number density :math:`N_{\rm therm}`.

    Parameters
    ----------
    B : float, array-like, or ~astropy.units.Quantity
        Magnetic field strength. Bare values are interpreted as Gauss.
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.
    epsilon_B : float or array-like
        Fraction of post-shock energy in the magnetic field.
    epsilon_E : float or array-like
        Fraction of post-shock energy in relativistic electrons.

    Returns
    -------
    ~astropy.units.Quantity
        Total number density :math:`N_{\rm therm}` in :math:`\mathrm{cm^{-3}}`.

    References
    ----------
    .. footbibliography::
    """
    B = ensure_in_units(B, u.G)
    N = _opt_MJD_norm_from_magnetic_field(
        B=B,
        Theta=Theta,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
    )
    return N * u.cm**-3


def compute_MJD_norm_from_thermal_energy_density(
    u_therm: Union[float, np.ndarray, u.Quantity],
    Theta: "_ArrayLike",
    epsilon_E: "_ArrayLike",
) -> u.Quantity:
    r"""
    Compute the total number density of a Maxwell-Jüttner electron distribution from the thermal energy density.

    Uses the equipartition condition :math:`\varepsilon_E\,u_{\rm therm} = U_e` with the mean energy
    density approximation :footcite:p:`1998ApJ...498..313G`

    .. math::

        U_e \approx N_{\rm therm}\,m_e c^2\,\Theta\,\frac{6 + 15\Theta}{4 + 5\Theta}

    to solve for :math:`N_{\rm therm}`.

    Parameters
    ----------
    u_therm : float, array-like, or ~astropy.units.Quantity
        Thermal energy density. Bare values are interpreted as :math:`\mathrm{erg\,cm^{-3}}`.
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.
    epsilon_E : float or array-like
        Fraction of post-shock energy in relativistic electrons.

    Returns
    -------
    ~astropy.units.Quantity
        Total number density :math:`N_{\rm therm}` in :math:`\mathrm{cm^{-3}}`.

    References
    ----------
    .. footbibliography::
    """
    u_therm = ensure_in_units(u_therm, u.erg / u.cm**3)
    N = _opt_MJD_norm_from_thermal(
        u_therm=u_therm,
        Theta=Theta,
        epsilon_E=epsilon_E,
    )
    return N * u.cm**-3


def compute_MJD_bol_emissivity(
    B: Union[float, np.ndarray, u.Quantity],
    N_therm: Union[float, np.ndarray, u.Quantity],
    Theta: "_ArrayLike",
) -> u.Quantity:
    r"""
    Compute the bolometric synchrotron emissivity for a Maxwell-Jüttner electron distribution.

    Parameters
    ----------
    B : float, array-like, or ~astropy.units.Quantity
        Magnetic field strength. Bare values are interpreted as Gauss.
    N_therm : float, array-like, or ~astropy.units.Quantity
        Total number density of the Maxwell-Jüttner component. Bare values are
        interpreted as :math:`\mathrm{cm^{-3}}`.
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.

    Returns
    -------
    ~astropy.units.Quantity
        Bolometric synchrotron emissivity in :math:`\mathrm{erg\,s^{-1}\,cm^{-3}}`.

    Notes
    -----
    Uses the single-electron power scaling :math:`P_{\rm syn} \propto \gamma^2` and the
    closed-form effective number density :math:`n_{\rm eff} = 12\,N_{\rm therm}\,\Theta^2`:

    .. math::

        j_{\rm bol} = \tfrac{4}{3}\,\sigma_T\,c\,\frac{B^2}{8\pi}\,n_{\rm eff}.
    """
    B_cgs = ensure_in_units(B, u.G)
    N_therm_cgs = ensure_in_units(N_therm, u.cm**-3)
    emiss = _opt_MJD_bol_emissivity_from_magnetic_field(B=B_cgs, N_therm=N_therm_cgs, Theta=Theta)
    return emiss * (u.erg / u.s / u.cm**3)


def compute_MJD_bol_emissivity_from_thermal_energy_density(
    u_therm: Union[float, np.ndarray, u.Quantity],
    Theta: "_ArrayLike",
    *,
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
) -> u.Quantity:
    r"""
    Compute the bolometric synchrotron emissivity for a Maxwell-Jüttner electron distribution assuming equipartition.

    Both the magnetic field strength and the electron normalization are inferred from
    the thermal energy density via ``epsilon_B`` and ``epsilon_E``.

    Parameters
    ----------
    u_therm : float, array-like, or ~astropy.units.Quantity
        Thermal energy density. Bare values are interpreted as
        :math:`\mathrm{erg\,cm^{-3}}`.
    Theta : float or array-like
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.
    epsilon_B : float or array-like
        Fraction of post-shock energy in magnetic fields.
    epsilon_E : float or array-like
        Fraction of post-shock energy in relativistic electrons.

    Returns
    -------
    ~astropy.units.Quantity
        Bolometric synchrotron emissivity in :math:`\mathrm{erg\,s^{-1}\,cm^{-3}}`.
    """
    u_therm_cgs = ensure_in_units(u_therm, u.erg / u.cm**3)
    emiss = _opt_MJD_bol_emissivity_from_thermal_full(
        u_therm=u_therm_cgs, Theta=Theta, epsilon_B=epsilon_B, epsilon_E=epsilon_E
    )
    return emiss * (u.erg / u.s / u.cm**3)


def get_MJD_distribution(Theta: float, norm: "_UnitBearingScalarLike" = 1, *, with_units: bool = True):
    r"""
    Return a callable ultra-relativistic Maxwell-Jüttner electron distribution.

    The distribution is

    .. math::

        N(\gamma) = \frac{N_{\rm therm}}{2\Theta^3}\,\gamma^2\,e^{-\gamma/\Theta},

    which satisfies :math:`\int_0^\infty N(\gamma)\,d\gamma = N_{\rm therm}`.

    Parameters
    ----------
    Theta : float
        Dimensionless electron temperature :math:`\Theta = kT / (m_e c^2)`.
    norm : float or ~astropy.units.Quantity, optional
        Total number density :math:`N_{\rm therm}`. Bare values are interpreted as
        :math:`\mathrm{cm^{-3}}`. Default is ``1``.
    with_units : bool, optional
        If ``True`` (default), ``norm`` is coerced to :math:`\mathrm{cm^{-3}}` via
        :func:`~trilobite.utils.misc_utils.ensure_in_units` and the returned callable
        returns an :class:`~astropy.units.Quantity` in :math:`\mathrm{cm^{-3}}`.
        If ``False``, ``norm`` is treated as a plain CGS float and the returned callable
        returns a plain :class:`numpy.ndarray`.

    Returns
    -------
    callable
        Function ``N(gamma)`` returning the distribution at Lorentz factor ``gamma``.
        Accepts floats or array-like inputs.
    """
    norm_cgs = ensure_in_units(norm, u.cm**-3) if with_units else float(norm)
    prefactor = norm_cgs / (2.0 * Theta**3)

    def _N(gamma):
        gamma = np.asarray(gamma, dtype="f8")
        result = prefactor * gamma**2 * np.exp(-gamma / Theta)
        return result * u.cm**-3 if with_units else result

    return _N


# ============================================================= #
# Backward-compat aliases (old public names -> new names)        #
# ============================================================= #
compute_electron_gamma_MJD_moment = compute_MJD_moment
compute_mean_gamma_MJD = compute_MJD_mean_gamma
compute_mean_energy_MJD = compute_MJD_mean_energy
compute_bol_emissivity_MJD = compute_MJD_bol_emissivity
compute_bol_emissivity_MJD_from_thermal_energy_density = compute_MJD_bol_emissivity_from_thermal_energy_density
get_maxwell_juttner_distribution = get_MJD_distribution
