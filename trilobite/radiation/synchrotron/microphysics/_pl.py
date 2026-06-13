"""
Power-law (PL) electron distribution microphysics.

All functions in this module operate on power-law electron distributions of the form
    dN/dgamma = N0 * gamma^{-p}
"""

from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units as u

from trilobite.radiation.constants import c_cgs, electron_rest_energy_cgs, sigma_T_cgs
from trilobite.utils.misc_utils import ensure_in_units

from ._base import _opt_equipart_magnetic_field

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingArrayLike, _UnitBearingScalarLike


def _opt_PL_moment(
    p: "_ArrayLike",
    x_min: "_ArrayLike" = 1.0,
    x_max: "_ArrayLike" = np.inf,
    *,
    order: int = 1,
) -> np.ndarray:
    r"""
    Compute the ``order``-th moment of a power-law electron distribution.

    This method computes the **un-normalized** moment of the electron Lorentz factor distribution
    defined as

    .. math::

        \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{\,\mathrm{order} - p}\, d\gamma.

    For a normalized distribution, this result must be multiplied by the normalization constant.

    Parameters
    ----------
    p : float or array-like
        Power-law index of the electron Lorentz factor distribution.
    x_min : float or array-like
        Minimum Lorentz factor.
    x_max : float or array-like
        Maximum Lorentz factor.
    order : int, optional
        Moment order. Default is ``1`` (energy moment).

    Notes
    -----
    This implementation utilizes numpy vectorization for efficiency. It handles the cases where
    the integral converges at either limit based on the value of ``p`` relative to ``order + 1``. It
    also correctly handles the logarithmic case when ``p == order + 1``.

    This function is useful not only for gamma-distributed power laws, but also for energy-distributed
    power laws since the moments are equivalent up to a constant factor.
    """
    # Coerce all inputs to arrays to ensure vectorization works properly. We
    # need to use arrays here because we want to mask instead of using if ... then ...
    # statements for performance.
    p = np.asarray(p, dtype="f8")
    x_min = np.asarray(x_min, dtype="f8")
    x_max = np.asarray(x_max, dtype="f8")

    p, x_min, x_max = np.broadcast_arrays(p, x_min, x_max)

    # Allocate the output array.
    moment = np.zeros_like(p, dtype="f8")

    # Define the exponent and construct the relevant masks for the
    # three cases. We also introduce a tolerance close to machine epsilon to
    # catch the scenarios close to logarithmic behavior.
    exponent = order + 1.0 - p
    _tol = 1e-15

    _is_eq = np.abs(exponent) < _tol
    _is_neq = ~_is_eq

    # Catch the non-logarithmic branch and use the standard formula.
    moment[_is_neq] = (x_max[_is_neq] ** exponent[_is_neq] - x_min[_is_neq] ** exponent[_is_neq]) / exponent[_is_neq]

    # Catch the logarithmic branch.
    moment[_is_eq] = np.log(x_max[_is_eq] / x_min[_is_eq])

    return moment.reshape(()) if moment.ndim == 0 else moment


def _opt_PL_n_total(
    N0: "_ArrayLike",
    p: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> "_ArrayLike":
    """
    Compute the total number density of electrons for a power-law distribution.

    Assumes
        dN/dgamma = N0 gamma^{-p}

    Returns n_tot in the same shape as N0.
    """
    N0 = np.asarray(N0, dtype="f8")

    moment_0 = _opt_PL_moment(
        p=p,
        x_min=gamma_min,
        x_max=gamma_max,
        order=0,
    )

    n_tot = N0 * moment_0
    return n_tot.reshape(()) if n_tot.ndim == 0 else n_tot


def _opt_PL_n_eff(
    N0: "_ArrayLike",
    p: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> "_ArrayLike":
    """
    Compute the effective radiating electron number density for a power-law distribution.

    This weights electrons by gamma^2, reflecting synchrotron power emission.

    Assumes
        dN/dgamma = N0 gamma^{-p}

    Returns n_eff in the same shape as N0.
    """
    N0 = np.asarray(N0, dtype="f8")

    moment_2 = _opt_PL_moment(
        p=p,
        x_min=gamma_min,
        x_max=gamma_max,
        order=2,
    )

    n_eff = N0 * moment_2
    return n_eff.reshape(()) if n_eff.ndim == 0 else n_eff


def _opt_PL_norm_gamma_to_energy(
    N0: "_ArrayLike",
    p: "_ArrayLike",
) -> "_ArrayLike":
    """
    Convert the normalization of a power-law electron distribution from gamma-space to energy-space.

    Assumes:
        N(gamma) = N0 gamma^{-p}
        N(E) = K_E E^{-p}
    Returns K_E in the same shape as N0.
    """
    N0 = np.asarray(N0, dtype="f8")

    # Utilize the relation that N_(0,E) = (m_e c^2)**(p-1) * N_(0,gamma)
    K_E = N0 * (electron_rest_energy_cgs ** (p - 1))
    return K_E.reshape(()) if K_E.ndim == 0 else K_E


def _opt_PL_norm_energy_to_gamma(
    K_E: "_ArrayLike",
    p: "_ArrayLike",
) -> "_ArrayLike":
    """
    Convert the normalization of a power-law electron distribution from energy-space to gamma-space.

    Assumes:
        N(gamma) = N0 gamma^{-p}
        N(E) = K_E E^{-p}
    """
    K_E = np.asarray(K_E, dtype="f8")

    # Utilize the relation that N_(0,gamma) = (m_e c^2)**(1-p) * N_(0,E)
    N0 = K_E * (electron_rest_energy_cgs ** (1 - p))
    return N0.reshape(()) if N0.ndim == 0 else N0


def _opt_PL_norm_from_magnetic_field(
    B: "_ArrayLike",
    p: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> np.ndarray:
    """
    Compute the gamma-space normalization of a power-law electron distribution from the magnetic field.

    Assumes CGS units throughout and returns ``N_0`` in ``cm^{-3}``.
    """
    # Ensure everything is forced to arrays.
    B = np.asarray(B, dtype="f8")
    p = np.asarray(p, dtype="f8")
    epsilon_B = np.asarray(epsilon_B, dtype="f8")
    epsilon_E = np.asarray(epsilon_E, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    # Use the magnetic field to compute the magnetic energy density. We will then
    # use equipartition to compute the normalization. See the theory documentation
    # for a description of the theory here.
    u_B = B**2 / (8.0 * np.pi)

    # Compute the moment of the distribution needed for normalization.
    moment = _opt_PL_moment(
        p=p,
        x_min=gamma_min,
        x_max=gamma_max,
        order=1,
    )

    # Normalization
    N0 = (epsilon_E / epsilon_B) * u_B / (electron_rest_energy_cgs * moment)

    return N0.reshape(()) if N0.ndim == 0 else N0


def _opt_PL_norm_from_thermal(
    u_therm: "_ArrayLike",
    p: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> np.ndarray:
    """
    Compute the gamma-space normalization of a power-law electron distribution from the thermal energy.

    Assumes CGS units throughout and returns ``N_0`` in ``cm^{-3}``.
    """
    # Ensure everything is forced to arrays.
    u_therm = np.asarray(u_therm, dtype="f8")
    p = np.asarray(p, dtype="f8")
    epsilon_E = np.asarray(epsilon_E, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    # Compute the moment of the distribution needed for normalization.
    moment = _opt_PL_moment(
        p=p,
        x_min=gamma_min,
        x_max=gamma_max,
        order=1,
    )

    # Normalization
    N0 = (epsilon_E * u_therm) / (electron_rest_energy_cgs * moment)

    return N0.reshape(()) if N0.ndim == 0 else N0


def _opt_PL_energy_norm_from_magnetic_field(
    B: "_ArrayLike",
    p: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    E_min: "_ArrayLike" = electron_rest_energy_cgs,
    E_max: "_ArrayLike" = np.inf,
) -> np.ndarray:
    """
    Compute the energy-space normalization of a power-law electron distribution from the magnetic field.

    Assumes CGS units throughout and returns ``K_E`` in ``cm^{-3} erg^{p-1}``.
    """
    return _opt_PL_norm_from_magnetic_field(
        B=B,
        p=p,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        gamma_min=E_min / electron_rest_energy_cgs,
        gamma_max=E_max / electron_rest_energy_cgs,
    ) * (electron_rest_energy_cgs ** (p - 1))


def _opt_PL_energy_norm_from_thermal(
    u_therm: "_ArrayLike",
    p: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    E_min: "_ArrayLike" = electron_rest_energy_cgs,
    E_max: "_ArrayLike" = np.inf,
) -> np.ndarray:
    """
    Compute the energy-space normalization of a power-law electron distribution from the thermal energy.

    Assumes CGS units throughout and returns ``K_E`` in ``cm^{-3} erg^{p-1}``.
    """
    return _opt_PL_norm_from_thermal(
        u_therm=u_therm,
        p=p,
        epsilon_E=epsilon_E,
        gamma_min=E_min / electron_rest_energy_cgs,
        gamma_max=E_max / electron_rest_energy_cgs,
    ) * (electron_rest_energy_cgs ** (p - 1))


def _opt_PL_bol_emissivity_from_magnetic_field(
    B: "_ArrayLike",
    N0: "_ArrayLike",
    p: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> np.ndarray:
    """
    Compute the bolometric synchrotron emissivity from a magnetic field and a power-law electron distribution.

    Assumes CGS units throughout and returns emissivity in erg s^-1 cm^-3.
    """
    # Coerce inputs
    B = np.asarray(B, dtype="f8")
    N0 = np.asarray(N0, dtype="f8")
    p = np.asarray(p, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    # Magnetic energy density
    U_B = B**2 / (8.0 * np.pi)

    # Radiative weighting moment
    moment_2 = _opt_PL_moment(
        p=p,
        x_min=gamma_min,
        x_max=gamma_max,
        order=2,
    )

    # Bolometric emissivity
    emiss = (4.0 / 3.0) * sigma_T_cgs * c_cgs * U_B * N0 * moment_2

    return emiss.reshape(()) if emiss.ndim == 0 else emiss


def _opt_PL_bol_emissivity_from_thermal(
    u_therm: "_ArrayLike",
    N0: "_ArrayLike",
    p: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> np.ndarray:
    """
    Compute the bolometric synchrotron emissivity assuming equipartition.

    Assumes CGS units throughout and returns emissivity in erg s^-1 cm^-3.
    """
    return _opt_PL_bol_emissivity_from_magnetic_field(
        B=_opt_equipart_magnetic_field(u_therm=u_therm, epsilon_B=epsilon_B),
        N0=N0,
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )


def _opt_PL_bol_emissivity_from_thermal_full(
    u_therm: "_ArrayLike",
    p: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> np.ndarray:
    """
    Compute the bolometric synchrotron emissivity assuming full equipartition.

    This version eliminates the explicit power-law normalization by using
    epsilon_E and epsilon_B to close the problem.

    Assumes CGS units throughout and returns emissivity in erg s^-1 cm^-3.
    """
    # Coerce inputs
    u_therm = np.asarray(u_therm, dtype="f8")
    p = np.asarray(p, dtype="f8")
    epsilon_B = np.asarray(epsilon_B, dtype="f8")
    epsilon_E = np.asarray(epsilon_E, dtype="f8")
    gamma_min = np.asarray(gamma_min, dtype="f8")
    gamma_max = np.asarray(gamma_max, dtype="f8")

    # Moments
    M1 = _opt_PL_moment(
        p=p,
        x_min=gamma_min,
        x_max=gamma_max,
        order=1,
    )

    M2 = _opt_PL_moment(
        p=p,
        x_min=gamma_min,
        x_max=gamma_max,
        order=2,
    )

    # Bolometric synchrotron emissivity
    emiss = (
        (4.0 / 3.0) * sigma_T_cgs * c_cgs * (epsilon_E * epsilon_B * u_therm**2) / electron_rest_energy_cgs * (M2 / M1)
    )

    return emiss.reshape(()) if emiss.ndim == 0 else emiss


# ============================================================= #
# Public wrappers                                               #
# ============================================================= #


def swap_PL_normalization(N0: "_UnitBearingArrayLike", p: "_ArrayLike", mode: str = "energy"):
    r"""
    Convert the normalization of a power-law electron distribution between gamma-space and energy-space.

    Parameters
    ----------
    N0 : float, array-like, or astropy.units.Quantity
        Power-law normalization. If ``mode='gamma'``, this is :math:`N_0` in :math:`N(\gamma) = N_0 \gamma^{-p}`.
        If ``mode='energy'``, this is :math:`K_E` in :math:`N(E) = K_E E^{-p}`.
    p : float or array-like
        Power-law index of the electron distribution.
    mode : {'gamma', 'energy'}, optional
        Indicates the input normalization type:

        - ``'gamma'``: input is :math:`N_0`.
        - ``'energy'``: input is :math:`K_E`.

    Returns
    -------
    astropy.units.Quantity
        Converted power-law normalization. If input was ``mode='gamma'``, output is ``K_E``.
        If input was ``mode='energy'``, output is ``N_0``.
    """
    if mode == "gamma":
        # Convert from gamma-space to energy-space
        N0_cgs = ensure_in_units(N0, u.cm**-3)
        K_E = _opt_PL_norm_gamma_to_energy(N0=N0_cgs, p=p)
        return K_E * u.cm**-3 * u.erg ** (p - 1)
    elif mode == "energy":
        # Convert from energy-space to gamma-space
        K_E_cgs = ensure_in_units(N0, u.cm**-3 * u.erg ** (p - 1))
        N0 = _opt_PL_norm_energy_to_gamma(K_E=K_E_cgs, p=p)
        return N0 * u.cm**-3
    else:
        raise ValueError("mode must be either 'gamma' or 'energy'.")


def compute_PL_moment(
    p: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
    *,
    order: int = 1,
) -> "_ArrayLike":
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

    Notes
    -----
    This function validates the inputs to ensure convergence of the integral before passing
    off to the optimized low-level implementation. It raises a ``ValueError`` if the inputs
    would lead to a divergent integral.
    """
    # Validate that we have a valid gamma_min (cannot be <= 0)
    # and that the integral converges.
    exponent = order + 1.0 - p

    if np.any(gamma_min <= 0):
        raise ValueError("gamma_min must be strictly positive.")
    if np.any((exponent > 0) & np.isinf(gamma_max)):
        raise ValueError("gamma_max must be finite when p < order + 1 for convergence.")

    # Track scalar return
    result = _opt_PL_moment(
        p=p,
        x_min=gamma_min,
        x_max=gamma_max,
        order=order,
    )

    return result


def compute_PL_energy_moment(
    p: "_ArrayLike",
    E_min: "_UnitBearingArrayLike" = electron_rest_energy_cgs * u.erg,
    E_max: "_UnitBearingArrayLike" = np.inf * u.erg,
    *,
    order: int = 1,
) -> "_ArrayLike":
    r"""
    Compute the ``order``-th moment of a power-law electron energy distribution.

    This evaluates

    .. math::

        \int_{E_{\min}}^{E_{\max}} E^{\,\mathrm{order} - p}\, dE.

    Parameters
    ----------
    p : float or array-like
        Power-law index of the electron energy distribution.
    E_min : float, array-like, or astropy.units.Quantity
        Minimum electron energy (must be > 0). Default is the electron rest-mass energy.
    E_max : float, array-like, or astropy.units.Quantity
        Maximum electron energy. May be ``inf`` if the integral converges. Default is ``inf``.
    order : int, optional
        Moment order. Default is ``1`` (energy moment).

    Returns
    -------
    float or numpy.ndarray
        Moment value(s).

    Notes
    -----
    This is a wrapper of the lower-level gamma-moment function. It converts energies to Lorentz factors
    before passing them off to the gamma-moment function.

    To perform this calculation, we use

    .. math::

        M^{(l)}_E = \int_{E_{\min}}^{E_{\max}} E^{l - p} \, dE
        = \int_{\gamma_{\min}}^{\gamma_{\max}} (m_e c^2 \gamma)^{l - p} m_e c^2 \, d\gamma
        = (m_e c^2)^{l + 1 - p} \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{l - p} \, d\gamma.

    and then simply compute the moment in gamma-space, multiplying by the appropriate factor of
    :math:`(m_e c^2)^{\mathrm{order} + 1 - p}` to account for the change of variables.
    """
    # Convert energies to Lorentz factors
    E_min_cgs = ensure_in_units(E_min, u.erg)
    E_max_cgs = ensure_in_units(E_max, u.erg)
    gamma_min = E_min_cgs / electron_rest_energy_cgs
    gamma_max = E_max_cgs / electron_rest_energy_cgs

    # Pass off to the gamma-moment function
    result = compute_PL_moment(
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        order=order,
    )
    return result * (electron_rest_energy_cgs * u.erg) ** (order + 1 - p)


def compute_PL_mean_gamma(
    p: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> "_ArrayLike":
    r"""
    Compute the mean Lorentz factor :math:`\left<\gamma\right>` of a power-law electron distribution.

    This evaluates

    .. math::

        \langle \gamma \rangle =
        \frac{\int \gamma^{1-p}\, d\gamma}
             {\int \gamma^{-p}\, d\gamma}.

    Parameters
    ----------
    p : float or array-like
        Power-law index.
    gamma_min : float or array-like
        Minimum Lorentz factor.
    gamma_max : float or array-like
        Maximum Lorentz factor.

    Returns
    -------
    float or numpy.ndarray
        Mean Lorentz factor.
    """
    m1 = compute_PL_moment(p=p, gamma_min=gamma_min, gamma_max=gamma_max, order=1)
    m0 = compute_PL_moment(p=p, gamma_min=gamma_min, gamma_max=gamma_max, order=0)
    return m1 / m0


def compute_PL_mean_energy(
    p: "_ArrayLike",
    E_min: "_UnitBearingArrayLike" = electron_rest_energy_cgs * u.erg,
    E_max: "_UnitBearingArrayLike" = np.inf * u.erg,
):
    r"""
    Compute the mean electron energy ⟨E⟩ of a power-law energy distribution.

    This evaluates

    .. math::

        \langle E \rangle =
        \frac{\int E^{1-p}\, dE}
             {\int E^{-p}\, dE}.

    Parameters
    ----------
    p : float or array-like
        Power-law index.
    E_min : float, array-like, or Quantity
        Minimum electron energy.
    E_max : float, array-like, or Quantity
        Maximum electron energy.

    Returns
    -------
    astropy.units.Quantity
        Mean electron energy.
    """
    m1 = compute_PL_energy_moment(p=p, E_min=E_min, E_max=E_max, order=1)
    m0 = compute_PL_energy_moment(p=p, E_min=E_min, E_max=E_max, order=0)
    return m1 / m0


def compute_PL_total_number_density(
    N0: Union[float, np.ndarray, u.Quantity],
    p: "_ArrayLike",
    *,
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> u.Quantity:
    r"""
    Compute the total number density of electrons in a power-law distribution.

    Assumes a distribution of the form

    .. math::

        \frac{dN}{d\gamma} = N_0 \gamma^{-p},
        \qquad \gamma_{\min} \le \gamma \le \gamma_{\max}.

    Parameters
    ----------
    N0 : float, array-like, or Quantity
        Power-law normalization. Units must be ``cm^{-3}`` if provided as a Quantity.
    p : float or array-like
        Power-law index.
    gamma_min : float or array-like, optional
        Minimum Lorentz factor.
    gamma_max : float or array-like, optional
        Maximum Lorentz factor.

    Returns
    -------
    astropy.units.Quantity
        Total electron number density in ``cm^{-3}``.
    """
    # Enforce units on N0
    N0 = ensure_in_units(N0, u.cm**-3)

    n_tot = _opt_PL_n_total(
        N0=N0,
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    return n_tot * u.cm**-3


def compute_PL_effective_number_density(
    N0: Union[float, np.ndarray, u.Quantity],
    p: "_ArrayLike",
    *,
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> u.Quantity:
    r"""
    Compute the effective radiating electron number density for a power-law distribution.

    Because the single-electron synchrotron power scales as :math:`P_{\rm syn} \propto \gamma^2`,
    the effective number density of electrons contributing to the synchrotron emission is given by

    .. math::

        n_{\rm eff} = \int_{\gamma_{\min}}^{\gamma_{\max}} N(\gamma) \gamma^2 \, d\gamma.

    Parameters
    ----------
    N0 : float, array-like, or Quantity
        Power-law normalization. Units must be ``cm^{-3}`` if provided as a Quantity.
    p : float or array-like
        Power-law index.
    gamma_min : float or array-like, optional
        Minimum Lorentz factor.
    gamma_max : float or array-like, optional
        Maximum Lorentz factor.

    Returns
    -------
    astropy.units.Quantity
        Effective radiating electron number density in ``cm^{-3}``.
    """
    # Enforce units on N0
    N0 = ensure_in_units(N0, u.cm**-3)

    n_eff = _opt_PL_n_eff(
        N0=N0,
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    return n_eff * u.cm**-3


def compute_PL_norm_from_magnetic_field(
    B: Union[float, np.ndarray, u.Quantity],
    p: "_ArrayLike",
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    *,
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
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
    # Enforce units on the B-field.
    B = ensure_in_units(B, u.G)

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
    N0_gamma = _opt_PL_norm_from_magnetic_field(
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

    return result


def compute_PL_norm_from_thermal_energy_density(
    u_therm: Union[float, np.ndarray, u.Quantity],
    p: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    *,
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
    mode: str = "gamma",
):
    r"""
    Compute the normalization of a power-law electron distribution from microphysical energy-partition parameters.

    If :math:`\varepsilon_E` is the fraction of the
    thermal energy which is allocated to relativistic electrons, then

    .. math::

        u_{\rm int} = \frac{m_e c^2}{\epsilon_E}
        \int_{\gamma_{\min}}^{\gamma_{\max}} N(\gamma) \gamma \, d\gamma.

    For a power-law distribution, this can be computed analytically as is done in this function. We can therefore
    compute the normalization of the power-law distribution.

    Parameters
    ----------
    u_therm : float, array-like, or astropy.units.Quantity
        Thermal energy density. Default units are ``erg cm^{-3}``, but an :class:`astropy.units.Quantity` may
        be provided to use general units.
    p : float or array-like
        Power-law index of the electron Lorentz factor distribution.
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
    # Enforce units on the thermal energy density.
    u_therm = ensure_in_units(u_therm, u.erg / u.cm**3)

    # Validate inputs before passing off to the low-level callable. This
    # includes checking for convergence of the energy integral.
    exponent = 2.0 - p
    if np.any(gamma_min <= 0):
        raise ValueError("gamma_min must be strictly positive.")
    if np.any((exponent > 0) & np.isinf(gamma_max)):
        raise ValueError("gamma_max must be finite when p < 2 for energy normalization.")

    # Compute N0 in gamma-space
    N0_gamma = _opt_PL_norm_from_thermal(
        u_therm=u_therm,
        p=p,
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

    return result


def compute_PL_bol_emissivity(
    B: Union[float, np.ndarray, u.Quantity],
    N0: Union[float, np.ndarray, u.Quantity],
    p: "_ArrayLike",
    *,
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> u.Quantity:
    r"""
    Compute the bolometric synchrotron emissivity from an explicit magnetic field and power-law electron distribution.

    Parameters
    ----------
    B : float, array-like, or Quantity
        Magnetic field strength. Default units are Gauss.
    N0 : float, array-like, or Quantity
        Power-law normalization of the electron distribution.
        Must have units of ``cm^{-3}``.
    p : float or array-like
        Power-law index of the electron distribution.
    gamma_min : float or array-like, optional
        Minimum Lorentz factor.
    gamma_max : float or array-like, optional
        Maximum Lorentz factor.

    Returns
    -------
    astropy.units.Quantity
        Bolometric synchrotron emissivity in ``erg s^{-1} cm^{-3}``.
    """
    # Enforce units
    B = ensure_in_units(B, u.Gauss)
    N0 = ensure_in_units(N0, u.cm**-3)

    emiss = _opt_PL_bol_emissivity_from_magnetic_field(
        B=B,
        N0=N0,
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    return emiss * (u.erg / u.s / u.cm**3)


def compute_PL_bol_emissivity_from_thermal_energy_density(
    u_therm: Union[float, np.ndarray, u.Quantity],
    p: "_ArrayLike",
    *,
    epsilon_B: "_ArrayLike",
    epsilon_E: "_ArrayLike",
    gamma_min: "_ArrayLike" = 1.0,
    gamma_max: "_ArrayLike" = np.inf,
) -> u.Quantity:
    r"""
    Compute the bolometric synchrotron emissivity assuming full equipartition.

    The magnetic field strength and electron normalization are both inferred
    from the thermal energy density using ``epsilon_B`` and ``epsilon_E``.

    Parameters
    ----------
    u_therm : float, array-like, or Quantity
        Thermal energy density. Default units are ``erg cm^{-3}``.
    p : float or array-like
        Power-law index of the electron distribution.
    epsilon_B : float or array-like
        Fraction of thermal energy in magnetic fields.
    epsilon_E : float or array-like
        Fraction of thermal energy in relativistic electrons.
    gamma_min : float or array-like, optional
        Minimum Lorentz factor.
    gamma_max : float or array-like, optional
        Maximum Lorentz factor.

    Returns
    -------
    astropy.units.Quantity
        Bolometric synchrotron emissivity in ``erg s^{-1} cm^{-3}``.
    """
    # Enforce units
    u_therm = ensure_in_units(u_therm, u.erg / u.cm**3)

    emiss = _opt_PL_bol_emissivity_from_thermal_full(
        u_therm=u_therm,
        p=p,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    return emiss * (u.erg / u.s / u.cm**3)


def get_PL_distribution(
    p: float,
    norm: "_UnitBearingScalarLike" = 1,
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    *,
    with_units: bool = True,
):
    r"""
    Return a callable power-law electron distribution.

    The distribution is

    .. math::

        N(\gamma) = N_0\,\gamma^{-p}, \quad \gamma_{\min} \le \gamma \le \gamma_{\max},

    and zero outside that range.

    Parameters
    ----------
    p : float
        Power-law spectral index.
    norm : float or ~astropy.units.Quantity, optional
        Amplitude :math:`N_0`. Bare values are interpreted as :math:`\mathrm{cm^{-3}}`.
        Default is ``1``.
    gamma_min : float, optional
        Minimum Lorentz factor. Default is ``1.0``.
    gamma_max : float, optional
        Maximum Lorentz factor. Default is ``np.inf``.
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

    def _N(gamma):
        gamma = np.asarray(gamma, dtype="f8")
        result = np.where((gamma >= gamma_min) & (gamma <= gamma_max), norm_cgs * gamma ** (-p), 0.0)
        return result * u.cm**-3 if with_units else result

    return _N


# ============================================================= #
# Backward-compat aliases (old public names -> new names)        #
# ============================================================= #
swap_electron_PL_normalization = swap_PL_normalization
compute_electron_gamma_PL_moment = compute_PL_moment
compute_electron_energy_PL_moment = compute_PL_energy_moment
compute_mean_gamma_PL = compute_PL_mean_gamma
compute_mean_energy_PL = compute_PL_mean_energy
compute_bol_emissivity = compute_PL_bol_emissivity
compute_bol_emissivity_from_thermal_energy_density = compute_PL_bol_emissivity_from_thermal_energy_density
get_power_law_distribution = get_PL_distribution
