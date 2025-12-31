"""
Physics utilities for computing free-free absorption of synchrotron radiation from shock interfaces.

This module's core objective is to allow the calculation of the free-free attenuation of synchrotron radiation
as it passes through the ionized medium surrounding the transient shocks studied in the Triceratops framework.
"""

from typing import Union

import numpy as np
from astropy import units as u

# noinspection PyUnresolvedReferences
from astropy.constants import m_p


# ================================================ #
# Free-Free Ratiative Transfer Parameters          #
# ================================================ #
# This section contains functions to compute the free-free absorption coefficient and other
# relevant radiative transfer properties in different scenarios. For the most part, these
# are focused on scenarios relevant for radio transients, meaning that the RJ tail is assumed.
def compute_RJ_FF_absorption_coefficient(
    frequency: Union[float, np.ndarray, u.Quantity],
    temperature: Union[float, u.Quantity],
    n_e: Union[float, u.Quantity],
    n_i: Union[float, u.Quantity],
    Z: float = 1,
    g_ff: float = 5.0,
) -> u.Quantity:
    r"""
    Compute the free–free absorption coefficient in the Rayleigh–Jeans limit.

    This function evaluates the classical thermal bremsstrahlung (free–free)
    absorption coefficient assuming the Rayleigh–Jeans approximation
    :math:`h\nu \ll k_B T`, appropriate for radio frequencies.

    The absorption coefficient is given by

    .. math::

        \alpha_\nu^{\rm ff}
        =
        0.018\;
        T^{-3/2}
        Z^2
        n_e n_i
        \nu^{-2}
        g_{\rm ff},

    where all quantities are expressed in CGS units.

    Parameters
    ----------
    frequency : float, array-like, or astropy.units.Quantity
        Observing frequency. If a Quantity is provided, it must be convertible to ``Hz``. If `frequency` is provided
        as an array of shape ``(N,...)``, the calculation will be performed for all frequencies simultaneously.
    temperature : float, or astropy.units.Quantity
        Electron temperature. If a Quantity is provided, it must be convertible to ``Kelvin``.
    n_e : float, or astropy.units.Quantity
        Electron number density. If a Quantity is provided, it must be convertible to ``cm^-3``.
    n_i : float,  or astropy.units.Quantity
        Ion number density. If a Quantity is provided, it must be convertible to ``cm^-3``.
    Z : float, optional
        Ionic charge. Default is 1.
    g_ff : float, optional
        Velocity-averaged Gaunt factor. Default is 5.

    Returns
    -------
    astropy.units.Quantity
        Free–free absorption coefficient with units of ``cm^-1``. If `frequency` and other parameters are arrays,
        the returned array will have shape broadcasted to match the input arrays.

    Notes
    -----
    The free-free absorption coefficient in the Rayleigh–Jeans limit is given by (See equation 5.19b in
    :footcite:p:`1979rpa..book.....R`):

    .. math::

        \alpha_\nu = 0.018 \, T^{-3/2} \, \nu^{-2} \, n_e \, n_i \, Z^2 \, g_{ff},

    where :math:`T` is the electron temperature in Kelvin, :math:`\nu` is the frequency in Hz,
    :math:`n_e` and :math:`n_i` are the electron and ion number densities in :math:`{\rm cm^{-3}}`,
    :math:`Z` is the ionic charge, and :math:`g_{ff}` is the velocity-averaged Gaunt factor.

    For this expression to be valid, the following conditions must be met:

    - This function assumes a fully thermal, non-relativistic plasma.
    - The Rayleigh–Jeans approximation is extremely well satisfied for
      radio supernovae and CSM environments.
    - If relativistic temperatures or infrared/optical frequencies are
      required, this expression is no longer valid.

    **Broadcasting**: This function fully supports NumPy broadcasting. In particular:

    - ``frequency`` may be an array of shape ``(M, 1)``
    - ``n_e`` and ``n_i`` may be arrays of shape ``(1, N)``
    - scalar parameters (``temperature``, ``Z``, ``g_ff``) broadcast over all axes

    In this case, the returned absorption coefficient will have shape ``(M, N)``.

    References
    ----------
    .. footbibliography::
    """
    # Unit coercion
    if hasattr(temperature, "unit"):
        temperature = temperature.to_value(u.K)
    if hasattr(frequency, "unit"):
        frequency = frequency.to_value(u.Hz)
    if hasattr(n_e, "unit"):
        n_e = n_e.to_value(u.cm**-3)
    if hasattr(n_i, "unit"):
        n_i = n_i.to_value(u.cm**-3)

    alpha_cgs = compute_RJ_FF_absorption_coefficient_CGS(
        frequency,
        temperature,
        n_e=n_e,
        n_i=n_i,
        Z=Z,
        g_ff=g_ff,
    )

    return alpha_cgs * u.cm**-1


def compute_RJ_FF_absorption_coefficient_CGS(
    frequency: Union[float, np.ndarray],
    temperature: Union[float, np.ndarray],
    n_e: Union[float, np.ndarray],
    n_i: Union[float, np.ndarray],
    Z: Union[float, np.ndarray] = 1,
    g_ff: Union[float, np.ndarray] = 5.0,
) -> Union[float, np.ndarray]:
    r"""
    Compute the thermal free–free absorption coefficient in the Rayleigh–Jeans limit (CGS).

    See :func:`compute_RJ_FF_absorption_coefficient` for details.

    Parameters
    ----------
    frequency : float or array-like
        Frequency in ``Hz``.
    temperature : float or array-like
        Electron temperature in ``Kelvin``.
    n_e : float or array-like
        Electron number density in ``cm^-3``.
    n_i : float or array-like
        Ion number density in ``cm^-3``.
    Z : float or array-like, optional
        Ionic charge. Default is 1.
    g_ff : float or array-like, optional
        Velocity-averaged Gaunt factor. Default is 5.

    Returns
    -------
    numpy.ndarray
        The free-free absorption coefficient :math:`\alpha_\nu` in :math:`{\rm cm^{-1}}`.

    """
    return 0.018 * temperature ** (-1.5) * frequency ** (-2.0) * n_e * n_i * Z**2 * g_ff


# ================================================ #
# Optical Depth Calculators                        #
# ================================================ #
# Various different physical scenarios require different optical depth calculations. This section
# contains functions to compute the optical depth due to free-free absorption in a variety
# of different scenarios relevant to radio transients. These can be used as components in models
# to achieve the desired optical depth calculation.
def compute_RJ_FF_optical_depth_wind_CGS(
    frequency: Union[float, np.ndarray],
    r: float,
    mdot: float,
    wind_velocity: float,
    *,
    r_max: float = np.inf,
    temperature: float = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> Union[float, np.ndarray]:
    r"""
    Compute the free–free optical depth through a steady wind (Rayleigh–Jeans, CGS).

    This function assumes a stellar wind with **constant velocity** and **mass-loss rate**. All quantities
    must be provided in CGS units. For a more detailed description of the parameters and usage, see
    :func:`compute_RJ_FF_optical_depth_wind`.

    Parameters
    ----------
    frequency : float or numpy.ndarray
        Observing frequency in Hz. This is the only parameter allowed to be array-like.
        Output shape matches ``frequency``.
    r : float
        Inner radius in cm (lower integration bound).
    mdot : float
        Mass-loss rate in g/s.
    wind_velocity : float
        Wind velocity in cm/s.
    r_max : float, optional
        Outer radius in cm. Default is ``np.inf`` (analytic infinite-wind limit).
    temperature : float, optional
        Electron temperature in K. Default is ``1e4``.
    mu_e : float, optional
        Mean molecular weight per free electron. Default is 1.2.
    mu_i : float, optional
        Mean molecular weight per ion. Default is 1.3.
    Z : float, optional
        Ionic charge. Default is 1.
    g_ff : float, optional
        Velocity-averaged Gaunt factor. Default is 5.

    Returns
    -------
    float or numpy.ndarray
        Free–free optical depth (dimensionless), with the same shape as ``frequency``.

    Notes
    -----
    **Vectorization rule:** only ``frequency`` may be array-like; all other inputs must be scalar-like.
    """
    # Proton mass in CGS
    mp = m_p.cgs.value

    # Handle the radial integral term
    if np.isinf(r_max):
        integral = 1.0 / (3.0 * r**3)
    else:
        integral = (1.0 / (3.0 * r**3)) - (1.0 / (3.0 * r_max**3))

    # Assemble optical depth
    tau = (
        0.018
        * temperature ** (-1.5)
        * frequency ** (-2.0)
        * Z**2
        * g_ff
        * mdot**2
        / (16.0 * np.pi**2 * wind_velocity**2 * mp**2 * mu_e * mu_i)
        * integral
    )

    return tau


def compute_RJ_FF_optical_depth_wind(
    frequency: Union[float, np.ndarray, u.Quantity],
    r: Union[float, u.Quantity],
    mdot: Union[float, u.Quantity],
    wind_velocity: Union[float, u.Quantity],
    *,
    r_max: Union[float, u.Quantity] = np.inf,
    temperature: Union[float, u.Quantity] = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> Union[float, np.ndarray]:
    r"""
    Compute the free–free optical depth through a stellar wind in the Rayleigh–Jeans limit (CGS).

    This function computes the optical depth due to free–free absorption (:math:`\tau_{\rm ff}`) due
    to a stellar wind with **constant velocity** and **mass-loss rate**. The density profile of the wind
    is

    .. math::

        \rho(r,t) = \frac{\\dot{M}}{4 \\pi r^2 v_w},

    where :math:`\\dot{M}` is the mass-loss rate and :math:`v_w` is the wind velocity.

    Parameters
    ----------
    frequency: float, array-like, or astropy.units.Quantity
        The observing frequency. If a Quantity is provided, it must be convertible to ``Hz``. `frequency` may be
        provided as an array with shape ``(N,...)`` to compute the optical depth at multiple frequencies simultaneously.
        In this case, all other array inputs must be broadcastable to the shape ``(N,...)``.
    r: float, array-like, or astropy.units.Quantity
        The current radius from the progenitor center. If a Quantity is provided, it must be convertible to ``cm``. This
        sets the inner boundary of the optical depth integral. Must be broadcastable to the shape of `frequency` if that
        is provided as an array.
    mdot: float, array-like, or astropy.units.Quantity
        The mass-loss rate of the wind. If a Quantity is provided, it must be convertible to ``g/s``. Must be
        broadcastable to the shape of `frequency` if that is provided as an array.
    wind_velocity: float, array-like, or astropy.units.Quantity
        The velocity of the wind. If a Quantity is provided, it must be convertible to ``cm/s``. Must be broadcastable
        to the shape of `frequency` if that is provided as an array.
    r_max: float, or astropy.units.Quantity
        The outer radius of the wind region. If a Quantity is provided, it must be convertible to ``cm``. This
        sets the outer boundary of the optical depth integral. By default, this in set to infinity and the integration
        is performed analytically.
    temperature: float, array-like, or astropy.units.Quantity
        The electron temperature of the wind material. If a Quantity is provided, it must be convertible to ``Kelvin``.
        If provided as an array, the shape must match that of `r`. By default, this is set to :math:`10^4 {\rm K}`.
    mu_e: float, array-like
        The mean molecular weight per free electron. Default is 1.2, appropriate for fully ionized solar composition
        material. If provided as an array, the shape must match that of `r`.
    mu_i: float, array-like
        The mean molecular weight per ion. Default is 1.3, appropriate for fully ionized solar composition
        material. If provided as an array, the shape must match that of `r`.
    Z: float, array-like
        The ionic charge. Default is 1. If provided as an array, the shape must match that of `r`.
    g_ff: float, array-like
        The velocity-averaged Gaunt factor. Default is 5. If provided as an array, the shape must match that of `r`.

    Returns
    -------
    float or numpy.ndarray
        Free–free optical depth :math:`\tau_{\rm ff}` with the same shape as ``frequency``.

    Notes
    -----
    The free–free optical depth through a stellar wind with constant velocity and mass-loss rate is given by

    .. math::

        \tau_{\rm ff}
        =
        \\int_{r}^{r_{\rm max}}
        \alpha_\nu^{\rm ff}(r')
        \\, dr',

    where :math:`\alpha_\nu^{\rm ff}(r')` is the free–free absorption coefficient at radius :math:`r'`. Upon
    substituting the density profile of the wind and evaluating the integral, we find

    .. math::

        \tau_{\rm ff}(\nu) = 0.018 \frac{\\dot{M}^2 r_0^{-3}}{48 \\pi^2 v^2 m_p^2 \\mu_e \\mu_i} T^{-3/2} \nu^{-2} Z^2
        g_{\rm ff}
        \\left(1- \\left(\frac{r_0}{r_{\rm max}}\right)^3\right),

    where :math:`m_p` is the proton mass, :math:`\\mu_e` is the mean molecular weight per free electron, and
    :math:`\\mu_i` is the mean molecular weight per ion. This is the formulation used in this function.
    """
    # Unit management: coerce everything to CGS floats.
    if hasattr(frequency, "unit"):
        frequency = frequency.to_value(u.Hz)
    if hasattr(r, "unit"):
        r = r.to_value(u.cm)
    if hasattr(mdot, "unit"):
        mdot = mdot.to_value(u.g / u.s)
    if hasattr(wind_velocity, "unit"):
        wind_velocity = wind_velocity.to_value(u.cm / u.s)
    if hasattr(r_max, "unit"):
        r_max = r_max.to_value(u.cm)
    if hasattr(temperature, "unit"):
        temperature = temperature.to_value(u.K)

    # Call the CGS implementation
    tau_cgs = compute_RJ_FF_optical_depth_wind_CGS(
        frequency,
        r,
        mdot,
        wind_velocity,
        r_max=r_max,
        temperature=temperature,
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )
    return tau_cgs


def compute_RJ_FF_optical_depth_shell_CGS(
    frequency: Union[float, np.ndarray],
    rho: float,
    thickness: float,
    *,
    temperature: float = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> Union[float, np.ndarray]:
    """Low-level CGS version of :func:`compute_RJ_FF_optical_depth_shell`."""
    mp = m_p.cgs.value

    tau = 0.018 * temperature ** (-1.5) * frequency ** (-2.0) * Z**2 * g_ff * rho**2 / (mu_e * mu_i * mp**2) * thickness

    return tau


def compute_RJ_FF_optical_depth_shell(
    frequency: Union[float, np.ndarray, u.Quantity],
    rho: Union[float, u.Quantity],
    thickness: Union[float, u.Quantity],
    *,
    temperature: Union[float, u.Quantity] = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> Union[float, np.ndarray]:
    r"""
    Compute the free–free optical depth through a uniform-density shell.

    This function computes the free–free optical depth :math:`\tau_{\rm ff}` through
    a shell of ionized material with **constant density** and finite radial thickness.
    The shell is assumed to be homogeneous and is characterized by a mass density
    :math:`\rho` and thickness :math:`\Delta r`.

    Under the Rayleigh–Jeans approximation, the optical depth reduces to

    .. math::

        \tau_{\rm ff}(\nu)
        =
        \alpha_\nu^{\rm ff} \, \Delta r,

    where :math:`\alpha_\nu^{\rm ff}` is the thermal free–free absorption coefficient
    evaluated at the shell density.

    Parameters
    ----------
    frequency : float, array-like, or astropy.units.Quantity
        Observing frequency. Must be convertible to ``Hz``.
        This is the only parameter that may be array-like.
    rho : float or astropy.units.Quantity
        Mass density of the shell. Must be convertible to ``g cm^-3``.
    thickness : float or astropy.units.Quantity
        Radial thickness of the shell. Must be convertible to ``cm``.
    temperature : float or astropy.units.Quantity, optional
        Electron temperature of the shell. Must be convertible to ``K``.
        Default is ``1e4``.
    mu_e : float, optional
        Mean molecular weight per free electron. Default is 1.2.
    mu_i : float, optional
        Mean molecular weight per ion. Default is 1.3.
    Z : float, optional
        Ionic charge. Default is 1.
    g_ff : float, optional
        Velocity-averaged Gaunt factor. Default is 5.

    Returns
    -------
    float or numpy.ndarray
        Free–free optical depth :math:`\tau_{\rm ff}` with the same shape as ``frequency``.

    Notes
    -----
    - This model is appropriate for shocked shells, dense CSM walls, or swept-up
      material with approximately uniform density.
    - Geometric effects such as curvature or obliquity are ignored.
    """
    if hasattr(frequency, "unit"):
        frequency = frequency.to_value(u.Hz)
    if hasattr(rho, "unit"):
        rho = rho.to_value(u.g / u.cm**3)
    if hasattr(thickness, "unit"):
        thickness = thickness.to_value(u.cm)
    if hasattr(temperature, "unit"):
        temperature = temperature.to_value(u.K)

    return compute_RJ_FF_optical_depth_shell_CGS(
        frequency,
        rho,
        thickness,
        temperature=temperature,
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_RJ_FF_optical_depth_powerlaw_CGS(
    frequency: Union[float, np.ndarray],
    r: float,
    rho0: float,
    r0: float,
    k: float,
    *,
    r_max: float = np.inf,
    temperature: float = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> Union[float, np.ndarray]:
    r"""Low-level CGS version of :func:`compute_RJ_FF_optical_depth_powerlaw`."""
    mp = m_p.cgs.value

    if k <= 0.5 and np.isinf(r_max):
        raise ValueError("Power-law does not converge for k <= 1/2 with r_max = inf.")

    prefactor = (
        0.018
        * temperature ** (-1.5)
        * frequency ** (-2.0)
        * Z**2
        * g_ff
        * rho0**2
        / (mu_e * mu_i * mp**2)
        * r0 ** (2 * k)
    )

    if np.isinf(r_max):
        integral = r ** (1 - 2 * k) / (2 * k - 1)
    else:
        integral = (r ** (1 - 2 * k) - r_max ** (1 - 2 * k)) / (2 * k - 1)

    return prefactor * integral


def compute_RJ_FF_optical_depth_powerlaw(
    frequency: Union[float, np.ndarray, u.Quantity],
    r: Union[float, u.Quantity],
    rho0: Union[float, u.Quantity],
    r0: Union[float, u.Quantity],
    k: float,
    *,
    r_max: Union[float, u.Quantity] = np.inf,
    temperature: Union[float, u.Quantity] = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> Union[float, np.ndarray]:
    r"""
    Compute the free–free optical depth for a power-law density profile.

    This function computes the free–free optical depth :math:`\tau_{\rm ff}` for
    radiation propagating through a medium with a spherically symmetric power-law
    density profile

    .. math::

        \rho(r) = \rho_0 \left(\frac{r}{r_0}\right)^{-k},

    where :math:`\rho_0` is the density at reference radius :math:`r_0` and
    :math:`k` is the power-law index.

    The optical depth is obtained by integrating the free–free absorption coefficient
    from radius :math:`r` to :math:`r_{\max}`. In the Rayleigh–Jeans limit, this integral
    can be evaluated analytically, yielding

    .. math::

        \tau_{\rm ff}(\nu)
        \propto
        \int_r^{r_{\max}} r'^{-2k} \, dr'.

    For :math:`k > 1/2`, the integral converges in the limit
    :math:`r_{\max} \rightarrow \infty`.

    Parameters
    ----------
    frequency : float, array-like, or astropy.units.Quantity
        Observing frequency. Must be convertible to ``Hz``.
        This is the only parameter that may be array-like.
    r : float or astropy.units.Quantity
        Inner radius of the absorbing region. Must be convertible to ``cm``.
    rho0 : float or astropy.units.Quantity
        Density normalization at radius ``r0``. Must be convertible to ``g cm^-3``.
    r0 : float or astropy.units.Quantity
        Reference radius for the density normalization. Must be convertible to ``cm``.
    k : float
        Power-law index of the density profile.
    r_max : float or astropy.units.Quantity, optional
        Outer radius of the absorbing region. Default is infinity.
    temperature : float or astropy.units.Quantity, optional
        Electron temperature. Must be convertible to ``K``. Default is ``1e4``.
    mu_e : float, optional
        Mean molecular weight per free electron. Default is 1.2.
    mu_i : float, optional
        Mean molecular weight per ion. Default is 1.3.
    Z : float, optional
        Ionic charge. Default is 1.
    g_ff : float, optional
        Velocity-averaged Gaunt factor. Default is 5.

    Returns
    -------
    float or numpy.ndarray
        Free–free optical depth :math:`\tau_{\rm ff}`.

    Notes
    -----
    - For ``k <= 1/2``, the optical depth diverges when ``r_max`` is infinite.
    - This model is suitable for generalized CSM profiles beyond steady winds,
      including stratified or relic mass-loss environments.
    """
    if hasattr(frequency, "unit"):
        frequency = frequency.to_value(u.Hz)
    if hasattr(r, "unit"):
        r = r.to_value(u.cm)
    if hasattr(rho0, "unit"):
        rho0 = rho0.to_value(u.g / u.cm**3)
    if hasattr(r0, "unit"):
        r0 = r0.to_value(u.cm)
    if hasattr(r_max, "unit"):
        r_max = r_max.to_value(u.cm)
    if hasattr(temperature, "unit"):
        temperature = temperature.to_value(u.K)

    return compute_RJ_FF_optical_depth_powerlaw_CGS(
        frequency,
        r,
        rho0,
        r0,
        k,
        r_max=r_max,
        temperature=temperature,
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_RJ_FF_optical_depth_arrays(
    frequency: Union[np.ndarray, u.Quantity],
    radius: Union[np.ndarray, u.Quantity],
    n_e: Union[np.ndarray, u.Quantity],
    n_i: Union[np.ndarray, u.Quantity],
    temperature: Union[np.ndarray, u.Quantity],
    *,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> u.Quantity:
    r"""
    Compute free–free optical depth from tabulated radial profiles.

    This function computes the frequency-dependent free–free optical depth
    by numerically integrating the Rayleigh–Jeans absorption coefficient
    over a supplied radial grid.

    Parameters
    ----------
    frequency : array-like or astropy.units.Quantity
        Observing frequencies. Must be convertible to ``Hz``.
    radius : array-like or astropy.units.Quantity
        Radial coordinates. Must be convertible to ``cm``.
    n_e : array-like or astropy.units.Quantity
        Electron number density. Must be convertible to ``cm^-3``.
    n_i : array-like or astropy.units.Quantity
        Ion number density. Must be convertible to ``cm^-3``.
    temperature : array-like or astropy.units.Quantity
        Electron temperature. Must be convertible to ``K``.
    Z : float, optional
        Ionic charge. Default is 1.
    g_ff : float, optional
        Velocity-averaged Gaunt factor. Default is 5.

    Returns
    -------
    astropy.units.Quantity
        Free–free optical depth (dimensionless), with shape ``(N_freq,)``.

    Notes
    -----
    - This function is fully numerical and makes **no assumptions**
      about the density profile.
    - The radial grid must be monotonic.
    - This is the most general optical depth calculator in the module
      and should be preferred when analytic assumptions are invalid.
    """
    # Unit coercion
    if hasattr(frequency, "unit"):
        frequency = frequency.to_value(u.Hz)
    if hasattr(radius, "unit"):
        radius = radius.to_value(u.cm)
    if hasattr(n_e, "unit"):
        n_e = n_e.to_value(u.cm**-3)
    if hasattr(n_i, "unit"):
        n_i = n_i.to_value(u.cm**-3)
    if hasattr(temperature, "unit"):
        temperature = temperature.to_value(u.K)

    tau = compute_RJ_FF_optical_depth_arrays_CGS(
        frequency=frequency,
        radius=radius,
        n_e=n_e,
        n_i=n_i,
        temperature=temperature,
        Z=Z,
        g_ff=g_ff,
    )

    return tau * u.dimensionless_unscaled


def compute_RJ_FF_optical_depth_arrays_CGS(
    frequency: np.ndarray,
    radius: np.ndarray,
    n_e: np.ndarray,
    n_i: np.ndarray,
    temperature: np.ndarray,
    *,
    Z: float = 1.0,
    g_ff: float = 5.0,
    axis: int = -1,
) -> np.ndarray:
    r"""
    Compute free–free optical depth from tabulated radial profiles (CGS).

    This function numerically integrates the Rayleigh–Jeans free–free absorption
    coefficient over a supplied radial grid. All inputs must be provided as
    CGS floats.

    Parameters
    ----------
    frequency : numpy.ndarray
        Frequencies in Hz. Shape ``(N_freq,)``.
    radius : numpy.ndarray
        Radial coordinates in cm. Must be monotonic. Shape ``(N_r,)``.
    n_e : numpy.ndarray
        Electron number density in ``cm^-3``. Shape ``(N_r,)``.
    n_i : numpy.ndarray
        Ion number density in ``cm^-3``. Shape ``(N_r,)``.
    temperature : numpy.ndarray
        Electron temperature in K. Shape ``(N_r,)``.
    Z : float, optional
        Ionic charge. Default is 1.
    g_ff : float, optional
        Velocity-averaged Gaunt factor. Default is 5.
    axis : int, optional
        Axis along which the radial integration is performed. Default is ``-1``.

    Returns
    -------
    numpy.ndarray
        Optical depth array with shape ``(N_freq,)``.

    Notes
    -----
    - This function performs **no unit validation**.
    - The Rayleigh–Jeans approximation is assumed.
    - This function is intended for arbitrary, tabulated density profiles.
    """
    # Ensure arrays
    frequency = np.atleast_1d(frequency)
    radius = np.asarray(radius)
    n_e = np.asarray(n_e)
    n_i = np.asarray(n_i)
    temperature = np.asarray(temperature)

    # Build alpha_nu grid: shape (N_freq, N_r)
    alpha = (
        0.018
        * temperature[None, :] ** (-1.5)
        * frequency[:, None] ** (-2.0)
        * n_e[None, :]
        * n_i[None, :]
        * Z**2
        * g_ff
    )

    # Integrate over radius
    tau = np.trapezoid(alpha, radius, axis=axis)

    return tau
