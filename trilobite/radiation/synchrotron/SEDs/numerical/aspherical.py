"""
Aspherical synchrotron radiation SEDs.

This module contains functions for computing the spectral energy distribution (SED) of synchrotron radiation
from aspherical sources. The SED is computed by integrating the synchrotron emission over the volume of the source,
taking into account the geometry and orientation of the source.
"""

from collections.abc import Callable
from typing import Union

import numpy as np
from astropy import constants as consts
from astropy import units as u
from scipy.special import logsumexp

from trilobite.physics_utils import resolve_cosmological_distances
from trilobite.radiation.synchrotron.SEDs.numerical.core import NumericalSynchrotronEngine
from trilobite.radiation.synchrotron.utils import _log_c_1_gamma_cgs, _log_chi_abs_cgs, _log_chi_cgs
from trilobite.utils.misc_utils import ensure_in_units


class OnAxisAsymmetricSynchrotronEngine(NumericalSynchrotronEngine):
    r"""
    On-axis synchrotron SED engine for axisymmetric multi-zone outflows.

    This class extends the functionality of :class:`NumericalSynchrotronEngine` to handle axisymmetric outflows
    with angular structure. This class allows users to model synchrotron emission detected by an on-axis observer
    from outflows where physical properties vary with polar angle, such as structured jets or spherical outflows
    with angular gradients.


    Notes
    -----
    The observed flux density is assembled as

    .. math::

        F_\nu
        =
        \frac{2\pi}{D_A^2}
        \int_0^1
        f_A(\xi)\,R(\xi)^2
        \left(\frac{\mathcal{D}(\xi)}{1+z}\right)^{\!3}
        I'_{\nu'(\xi)}(\xi)\,
        \xi\,d\xi,

    where :math:`\mathcal{D}(\xi) = [\Gamma(1-\beta\xi)]^{-1}` is the Doppler factor
    (:math:`\xi = 1` corresponds to on-axis approaching material) and
    :math:`I'_{\nu'(\xi)}` is the comoving-frame specific intensity at
    :math:`\nu'(\xi) = \nu(1+z)/\mathcal{D}(\xi)`.

    The angular integral is evaluated by Gauss-Legendre quadrature\ :footcite:p:`2007nras.book.....P`
    with :math:`n_\theta`
    nodes mapped onto :math:`[0,1]`. All radiative quantities are fully vectorized over
    the ``(n_\nu, n_\theta, n_\gamma)`` array without any Python loop over sightlines.

    Kernel tables are **not** pre-loaded at construction; call
    :meth:`~NumericalSynchrotronEngine.load_avg_first_kernel` and/or
    :meth:`~NumericalSynchrotronEngine.load_first_kernel` before evaluating SEDs.
    The Lorentz-factor grid is built per-call from the ``gamma`` / ``gamma_min`` /
    ``gamma_max`` / ``n_gamma`` arguments, exactly as in the parent class.

    The Doppler convention used here is :math:`\mathcal{D} = [\Gamma(1-\beta\cos\theta)]^{-1}`,
    where :math:`\beta > 0` and :math:`\cos\theta = 1` (on-axis) gives maximum
    blueshift. This convention matches :class:`OnAxisMultizoneSynchrotronSED` and
    standard astrophysical usage for approaching ejecta.

    The electron distribution :math:`N(\gamma)` is passed as a call-time argument of
    shape ``(n_theta, n_gamma)`` (or a callable returning that shape), consistent with
    the parent class interface.

    See Also
    --------
    :class:`NumericalSynchrotronEngine` :
        Base engine providing kernel tables, gamma grids, and single-zone radiative
        transfer.
    """

    # ------------------------------------------ #
    # Initialization                             #
    # ------------------------------------------ #

    def __init__(self, n_theta: int = 10):
        r"""
        Initialize the on-axis axisymmetric synchrotron engine.

        The constructor initializes the parent numerical synchrotron engine and
        precomputes the angular quadrature used to integrate emission over an
        axisymmetric outflow viewed on-axis. Gauss-Legendre nodes are mapped from
        ``[-1, 1]`` to ``[0, 1]`` in

        .. math::

            \xi = \cos\theta,

        where ``xi = 1`` corresponds to material moving directly along the observer
        line of sight and ``xi = 0`` corresponds to the transverse edge of the
        visible hemisphere.

        The stored quadrature arrays are used to approximate integrals of the form

        .. math::

            \int_0^1 f(\xi)\,\xi\,d\xi
            \approx
            \sum_i w_i\,\xi_i\,f(\xi_i),

        which appears in the projected-area weighting for an on-axis axisymmetric
        source.

        Parameters
        ----------
        n_theta : int, optional
            Number of Gauss-Legendre angular quadrature nodes used to sample the
            polar-angle structure of the outflow. Larger values resolve sharper
            angular gradients in quantities such as radius, magnetic field,
            velocity, slab depth, filling factor, and electron distribution, but
            increase the cost of each spectral evaluation. Default is ``10``.

        Notes
        -----
        This constructor does not load synchrotron kernel interpolation tables.
        Before evaluating spectra, call
        :meth:`~NumericalSynchrotronEngine.load_avg_first_kernel` for the
        pitch-angle-averaged formalism and/or
        :meth:`~NumericalSynchrotronEngine.load_first_kernel` for fixed pitch-angle
        calculations.

        The following read-only quadrature arrays are precomputed:

        - ``cos_theta``: mapped Gauss-Legendre nodes :math:`\xi_i = \cos\theta_i`
          on ``[0, 1]``.
        - ``theta``: corresponding polar angles :math:`\theta_i` in radians.
        - ``gl_weights``: Gauss-Legendre weights for :math:`\int_0^1 f(\xi)\,d\xi`.
        - ``quad_weights``: composite weights ``gl_weights * cos_theta`` for
          :math:`\int_0^1 f(\xi)\xi\,d\xi`.

        Raises
        ------
        ValueError
            Recommended if ``n_theta`` is not a positive integer.
        """
        # Pass off to the super-class constructor.
        super().__init__()

        # Generate the gauss-legendre quadrature mappings.
        self._n_theta = n_theta

        # Map Gauss-Legendre nodes from [-1, 1] to [0, 1]: xi = (x+1)/2, w -> w/2
        xi, wi = np.polynomial.legendre.leggauss(n_theta)

        self._cos_theta = (xi + 1.0) / 2.0  # (n_theta,), in [0,1]
        self._gl_weights = wi * 0.5  # (n_theta,) GL weights for [0,1]
        self._theta = np.arccos(self._cos_theta)  # (n_theta,) polar angles [rad]

        # Composite weight for integral(g(xi)*xi dxi, 0, 1) ~ sum(gl_weights * cos_theta * g)
        self._quad_weights = self._gl_weights * self._cos_theta  # (n_theta,)

    # ------------------------------------------ #
    # Dunder Methods                             #
    # ------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_theta={self._n_theta}, "
            f"first_kernel_loaded={self.is_first_kernel_loaded}, "
            f"avg_first_kernel_loaded={self.is_avg_first_kernel_loaded})"
        )

    def __str__(self) -> str:
        kernels = []
        if self.is_first_kernel_loaded:
            kernels.append("first kernel")
        if self.is_avg_first_kernel_loaded:
            kernels.append("pitch-angle averaged kernel")
        kernel_str = ", ".join(kernels) if kernels else "no kernels loaded"
        return f"{self.__class__.__name__} | n_theta={self._n_theta} | {kernel_str}"

    # ------------------------------------------ #
    # Properties                                 #
    # ------------------------------------------ #
    @property
    def n_theta(self) -> int:
        """Number of Gauss-Legendre angular quadrature nodes."""
        return self._n_theta

    @property
    def theta(self) -> np.ndarray:
        """Polar-angle quadrature nodes [rad], shape ``(n_theta,)``."""
        return self._theta.copy()

    @property
    def cos_theta(self) -> np.ndarray:
        r"""Cosine of the quadrature nodes :math:`\xi \in [0, 1]`, shape ``(n_theta,)``."""
        return self._cos_theta.copy()

    @property
    def gl_weights(self) -> np.ndarray:
        r"""Gauss-Legendre weights for :math:`\int_0^1 f(\xi)\,d\xi`, shape ``(n_theta,)``."""
        return self._gl_weights.copy()

    @property
    def quad_weights(self) -> np.ndarray:
        r"""
        Composite weights for :math:`\int_0^1 f(\xi)\,\xi\,d\xi`, shape ``(n_theta,)``.

        Equal to ``gl_weights * cos_theta``.
        """
        return self._quad_weights.copy()

    # ------------------------------------------ #
    # Private Helpers                            #
    # ------------------------------------------ #
    # These are scap methods used to perform various repeated tasks during
    # the calculation process. They are not intended to be called directly by users.
    def _compute_log_doppler(self, beta: np.ndarray) -> np.ndarray:
        r"""
        Log of the Doppler factor per sightline.

        :math:`\ln\mathcal{D}(\xi) = -\ln\Gamma - \ln(1 - \beta\xi)`, where
        :math:`\beta > 0` and :math:`\xi = \cos\theta = 1` gives maximum blueshift.

        Parameters
        ----------
        beta : ~numpy.ndarray, shape ``(n_theta,)``
            Bulk velocity per sightline.

        Returns
        -------
        log_D : ~numpy.ndarray, shape ``(n_theta,)``
        """
        beta = np.broadcast_to(np.asarray(beta, dtype="f8"), (self._n_theta,))
        log_gamma_bulk = -0.5 * np.log1p(-(beta**2))
        return -(log_gamma_bulk + np.log1p(-beta * self._cos_theta))

    def _coerce_on_axis_inputs(
        self,
        nu,
        B,
        N,
        slab_depth,
        beta,
        alpha,
        gamma,
        gamma_min: float,
        gamma_max: float,
        n_gamma: int,
    ):
        """Convert and broadcast all per-call inputs to internal log-CGS arrays."""
        log_nu = np.atleast_1d(np.asarray(np.log(ensure_in_units(nu, u.Hz)), dtype="f8"))
        log_B = np.log(np.broadcast_to(np.asarray(ensure_in_units(B, u.G), dtype="f8"), (self._n_theta,)))
        log_slab = np.log(np.broadcast_to(np.asarray(ensure_in_units(slab_depth, u.cm), dtype="f8"), (self._n_theta,)))
        beta_arr = np.broadcast_to(np.asarray(beta, dtype="f8"), (self._n_theta,))
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)
        sin_alpha_arr = None
        if alpha is not None:
            sin_alpha_arr = np.sin(
                np.broadcast_to(np.asarray(ensure_in_units(alpha, u.rad), dtype="f8"), (self._n_theta,))
            )
        return log_nu, log_B, log_N, log_slab, beta_arr, log_gamma, log_weights, sin_alpha_arr

    def _resolve_flux_geometry(
        self,
        R,
        f_A,
        angular_diameter_distance,
        luminosity_distance,
        proper_distance,
        z,
        cosmology,
    ):
        """Resolve distance and per-sightline geometry to log-CGS arrays."""
        dist = resolve_cosmological_distances(
            redshift=z if z != 0 else None,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        log_D_A = np.log(dist["angular_diameter_distance"].to(u.cm).value)
        log_D_L = np.log(dist["luminosity_distance"].to(u.cm).value)
        log_R = np.log(np.broadcast_to(np.asarray(ensure_in_units(R, u.cm), dtype="f8"), (self._n_theta,)))
        log_f_A = np.log(np.broadcast_to(np.asarray(f_A, dtype="f8"), (self._n_theta,)))
        return log_D_A, log_D_L, log_R, log_f_A

    # ------------------------------------------ #
    # Radiative Quantities API (Private)         #
    # ------------------------------------------ #
    def _compute_log_on_axis_rf_intensity(
        self,
        log_nu_obs: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_slab_depth: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        log_correction: np.ndarray,
        *,
        sin_alpha: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        r"""
        Vectorized comoving-frame specific intensity per sightline.

        Performs a single fused broadcast over ``(n_nu, n_theta, n_gamma)`` — no
        Python loop over sightlines.

        Parameters
        ----------
        log_nu_obs : ~numpy.ndarray, shape ``(n_nu,)``
            Natural log of observer-frame frequencies [Hz].
        log_B : ~numpy.ndarray, shape ``(n_theta,)``
            Natural log of comoving-frame magnetic field [G] per sightline.
        log_N : ~numpy.ndarray, shape ``(n_theta, n_gamma)``
            Natural log of comoving-frame electron distribution per sightline.
        log_slab_depth : ~numpy.ndarray, shape ``(n_theta,)``
            Natural log of comoving-frame LOS transfer depth [cm] per sightline.
        log_gamma : ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz-factor grid.
        log_weights : ~numpy.ndarray, shape ``(n_gamma,)``
            Log of quadrature weights :math:`\gamma_i\,\Delta\!\log\gamma_i`.
        log_correction : ~numpy.ndarray, shape ``(n_theta,)``
            :math:`\ln(\mathcal{D}/(1+z))` per sightline; used to shift observer
            frequencies to the comoving frame via :math:`\nu' = \nu / e^{\text{correction}}`.
        sin_alpha : ~numpy.ndarray or None, shape ``(n_theta,)``
            Sine of pitch angle per sightline. ``None`` selects the PA-averaged kernel.

        Returns
        -------
        log_I_rf : ~numpy.ndarray, shape ``(n_nu, n_theta)``
            Natural log of comoving-frame specific intensity
            [:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`].
        """
        n_nu = log_nu_obs.shape[0]
        n_theta = self._n_theta
        n_gamma = log_gamma.size

        # Doppler-shift observer frequencies to comoving frame per sightline: (n_nu, n_theta)
        log_nu_rf = log_nu_obs[:, np.newaxis] - log_correction[np.newaxis, :]

        # Broadcast views for (n_nu, n_theta, n_gamma) kernel evaluation
        log_nu_rf_v = log_nu_rf[:, :, np.newaxis]  # (n_nu, n_theta, 1)
        log_B_v = log_B[np.newaxis, :, np.newaxis]  # (1,    n_theta, 1)
        log_g_v = log_gamma[np.newaxis, np.newaxis, :]  # (1,    1,       n_gamma)
        log_N_v = log_N[np.newaxis, :, :]  # (1,    n_theta, n_gamma)
        log_w_v = log_weights[np.newaxis, np.newaxis, :]  # (1,    1,       n_gamma)
        log_dlg_v = (log_weights - log_gamma)[np.newaxis, np.newaxis, :]  # (1,    1,       n_gamma)
        log_B_bc = log_B[np.newaxis, :]  # (1,    n_theta)

        if sin_alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_x = log_nu_rf_v - 2.0 * log_g_v - log_B_v - _log_c_1_gamma_cgs
            log_F = np.interp(log_x.ravel(), self._log_x_avg_first_kernel, self._log_avg_first_kernel).reshape(
                n_nu, n_theta, n_gamma
            )
            d_log_F = np.interp(
                log_x.ravel(), self._log_x_avg_first_kernel, self._log_davg_first_kernel_dlog_x
            ).reshape(n_nu, n_theta, n_gamma)
            log_j = logsumexp(log_F + log_N_v + log_w_v, axis=-1) + _log_chi_cgs + log_B_bc
            log_kernel = log_F + np.log(np.maximum(1.0 - d_log_F, np.finfo("f8").tiny))
            log_alpha = (
                _log_chi_abs_cgs + log_B_bc - 2.0 * log_nu_rf + logsumexp(log_kernel + log_N_v + log_dlg_v, axis=-1)
            )
        else:
            self.ensure_first_kernel_loaded()
            log_sa = np.log(np.asarray(sin_alpha, dtype="f8"))
            log_sa_v = log_sa[np.newaxis, :, np.newaxis]  # (1, n_theta, 1)
            log_sa_bc = log_sa[np.newaxis, :]  # (1, n_theta)
            log_x = log_nu_rf_v - 2.0 * log_g_v - log_B_v - _log_c_1_gamma_cgs - log_sa_v
            log_F = np.interp(log_x.ravel(), self._log_x_first_kernel, self._log_first_kernel).reshape(
                n_nu, n_theta, n_gamma
            )
            d_log_F = np.interp(log_x.ravel(), self._log_x_first_kernel, self._log_dfirst_kernel_dlog_x).reshape(
                n_nu, n_theta, n_gamma
            )
            log_j = logsumexp(log_F + log_N_v + log_w_v, axis=-1) + _log_chi_cgs + log_B_bc + log_sa_bc
            log_kernel = log_F + np.log(np.maximum(1.0 - d_log_F, np.finfo("f8").tiny))
            log_alpha = (
                _log_chi_abs_cgs
                + log_B_bc
                + log_sa_bc
                - 2.0 * log_nu_rf
                + logsumexp(log_kernel + log_N_v + log_dlg_v, axis=-1)
            )

        tau = np.exp(np.clip(log_alpha + log_slab_depth[np.newaxis, :], -np.inf, 500.0))
        return log_j - log_alpha + np.log(-np.expm1(-tau))

    # ------------------------------------------ #
    # Public API — Sightline-only quantities     #
    # ------------------------------------------ #
    def compute_emissivity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame synchrotron emissivity per angular sightline.

        This method evaluates the synchrotron emissivity independently along each
        angular quadrature node of the on-axis axisymmetric outflow. The result is
        not integrated over polar angle.

        If ``alpha`` is ``None``, the pitch-angle-averaged synchrotron kernel
        :math:`\bar{F}(x)` is used. Otherwise, the fixed-pitch-angle kernel
        :math:`F(x)` is evaluated using the supplied comoving-frame pitch angle.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are interpreted as Hz.
            The returned spectral axis has shape ``(n_nu,)`` after applying
            :func:`numpy.atleast_1d`.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength for each angular sightline.
            Bare values are interpreted as Gauss. Must be scalar or broadcastable
            to shape ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.

            If array-like, the final axis must correspond to the Lorentz-factor
            grid and the full shape should be ``(n_theta, n_gamma)`` or
            broadcast-compatible with that convention.

            If callable, it is evaluated as ``N(gamma)``, where ``gamma`` is the
            Lorentz-factor grid in linear units, and must return values with final
            axis length ``n_gamma``.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If provided, ``gamma_min``,
            ``gamma_max``, and ``n_gamma`` are ignored. If ``None``, a logarithmic
            grid is constructed from those arguments.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame electron pitch angle for each sightline. Bare values are
            interpreted as radians. Must be scalar or broadcastable to
            ``(n_theta,)``. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        j_nu : ~astropy.units.Quantity, shape ``(n_nu, n_theta)``
            Comoving-frame synchrotron emissivity per sightline in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,Hz^{-1}\,sr^{-1}}`.

        Notes
        -----
        This method requires a loaded synchrotron kernel table. Call
        :meth:`~NumericalSynchrotronEngine.load_avg_first_kernel` before using the
        pitch-angle-averaged branch, or
        :meth:`~NumericalSynchrotronEngine.load_first_kernel` before using the
        fixed-pitch-angle branch.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        log_B = np.log(np.broadcast_to(ensure_in_units(B, u.G), (self._n_theta,)))
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)
        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_j = self._compute_log_pa_emissivity(np.log(nu_cgs), log_B, log_N, log_gamma, log_weights)
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(np.broadcast_to(ensure_in_units(alpha, u.rad), (self._n_theta,)))
            log_j = self._compute_log_emissivity(np.log(nu_cgs), log_B, log_N, log_gamma, log_weights, sin_alpha)
        return np.exp(log_j) * (u.erg / (u.s * u.cm**3 * u.Hz * u.sr))

    def compute_absorption(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame synchrotron self-absorption coefficient per sightline.

        This method evaluates :math:`|\alpha_\nu|` independently at each angular
        quadrature node. The absorption coefficient is computed using the same
        integration-by-parts formulation as the parent numerical engine, avoiding
        explicit differentiation of the electron distribution.

        If ``alpha`` is ``None``, the pitch-angle-averaged synchrotron kernel is
        used. Otherwise, the fixed-pitch-angle kernel is evaluated using the
        supplied comoving-frame pitch angle.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are interpreted as Hz.
            The returned spectral axis has shape ``(n_nu,)`` after applying
            :func:`numpy.atleast_1d`.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength for each angular sightline.
            Bare values are interpreted as Gauss. Must be scalar or broadcastable
            to shape ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.

            If array-like, the expected shape is ``(n_theta, n_gamma)`` with the
            Lorentz-factor axis last. If callable, it is evaluated as ``N(gamma)``
            and must return values compatible with that shape.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If ``None``, a logarithmic grid is built
            from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle for each sightline. Bare values are
            interpreted as radians. Must be scalar or broadcastable to
            ``(n_theta,)``. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        alpha_nu : ~astropy.units.Quantity, shape ``(n_nu, n_theta)``
            Comoving-frame synchrotron self-absorption coefficient per sightline in
            :math:`\mathrm{cm^{-1}}`.

        Notes
        -----
        This method returns the magnitude of the absorption coefficient used in the
        optical depth :math:`\tau_\nu = \alpha_\nu \ell`. It does not perform any
        angular integration.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        log_B = np.log(np.broadcast_to(ensure_in_units(B, u.G), (self._n_theta,)))
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)
        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_a = self._compute_log_pa_absorption_coefficient(np.log(nu_cgs), log_B, log_N, log_gamma, log_weights)
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(np.broadcast_to(ensure_in_units(alpha, u.rad), (self._n_theta,)))
            log_a = self._compute_log_absorption_coefficient(
                np.log(nu_cgs), log_B, log_N, log_gamma, log_weights, sin_alpha
            )
        return np.exp(log_a) * u.cm**-1

    def compute_rest_frame_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        beta: Union[float, np.ndarray],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        z: float = 0.0,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame synchrotron specific intensity per sightline.

        Observer-frame frequencies are shifted into the comoving frame of each
        angular sightline using the local Doppler factor and source redshift. The
        radiative transfer equation is then solved in the plasma rest frame,

        .. math::

            I'_{\nu'}(\xi)
            =
            S'_{\nu'}(\xi)
            \left[1 - \exp\left(-\tau'_{\nu'}(\xi)\right)\right],

        where :math:`\tau'_{\nu'} = \alpha'_{\nu'} \ell'`. The returned intensity is
        the comoving-frame intensity evaluated at the corresponding
        sightline-dependent frequency :math:`\nu'(\xi)`; no final Doppler boost is
        applied to the intensity.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are interpreted as Hz.
            Frequencies are shifted to the comoving frame internally using
            :math:`\nu' = \nu(1+z)/\mathcal{D}(\xi)`.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength for each sightline. Bare values
            are interpreted as Gauss. Must be scalar or broadcastable to
            ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`, with expected shape ``(n_theta, n_gamma)``
            or callable output compatible with that shape.
        slab_depth : float, array-like, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'` for each
            sightline. Bare values are interpreted as cm. Must be scalar or
            broadcastable to ``(n_theta,)``.
        beta : float or array-like
            Bulk velocity :math:`\beta = v/c` for each sightline. Must be scalar or
            broadcastable to ``(n_theta,)``. Positive values correspond to material
            moving toward the on-axis observer under the convention
            :math:`\mathcal{D} = [\Gamma(1-\beta\cos\theta)]^{-1}`.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If ``None``, a logarithmic grid is built
            from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle for each sightline. Bare values are
            interpreted as radians. If ``None``, the pitch-angle-averaged kernel is
            used.
        z : float, optional
            Cosmological redshift. Default is ``0.0``.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        I_rf : ~astropy.units.Quantity, shape ``(n_nu, n_theta)``
            Comoving-frame specific intensity per sightline in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.

        Notes
        -----
        This method is useful for inspecting the local radiative-transfer solution
        before applying the final observer-frame intensity transformation
        :math:`I_\nu = [\mathcal{D}/(1+z)]^3 I'_{\nu'}`.
        """
        log_nu, log_B, log_N, log_slab, beta_arr, log_gamma, log_weights, sin_alpha = self._coerce_on_axis_inputs(
            nu, B, N, slab_depth, beta, alpha, gamma, gamma_min, gamma_max, n_gamma
        )
        log_correction = self._compute_log_doppler(beta_arr) - np.log1p(z)
        log_I = self._compute_log_on_axis_rf_intensity(
            log_nu,
            log_B,
            log_N,
            log_slab,
            log_gamma,
            log_weights,
            log_correction,
            sin_alpha=sin_alpha,
        )
        return np.exp(log_I) * (u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

    def compute_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        beta: Union[float, np.ndarray],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        z: float = 0.0,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame synchrotron specific intensity per sightline.

        This method first evaluates the comoving-frame radiative-transfer solution
        at the Doppler- and redshift-shifted frequency for each sightline, then
        transforms the intensity to the observer frame using the Lorentz invariant
        :math:`I_\nu/\nu^3`:

        .. math::

            I_\nu(\xi)
            =
            \left(\frac{\mathcal{D}(\xi)}{1+z}\right)^3
            I'_{\nu'(\xi)}(\xi),

        with

        .. math::

            \nu'(\xi) = \frac{(1+z)\nu}{\mathcal{D}(\xi)}.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are interpreted as Hz.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength per sightline. Bare values are
            interpreted as Gauss. Must be scalar or broadcastable to
            ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`, with expected shape ``(n_theta, n_gamma)``
            or callable output compatible with that shape.
        slab_depth : float, array-like, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'` per
            sightline. Bare values are interpreted as cm.
        beta : float or array-like
            Bulk velocity :math:`\beta = v/c` per sightline. Must be scalar or
            broadcastable to ``(n_theta,)``.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If ``None``, a logarithmic grid is built
            from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle per sightline. Bare values are interpreted
            as radians. If ``None``, the pitch-angle-averaged kernel is used.
        z : float, optional
            Cosmological redshift. Default is ``0.0``.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        I_nu : ~astropy.units.Quantity, shape ``(n_nu, n_theta)``
            Observer-frame specific intensity per sightline in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.

        Notes
        -----
        The result is not integrated over angle. Use :meth:`compute_flux_density`
        for the projected, angle-integrated observed spectral flux density.
        """
        log_nu, log_B, log_N, log_slab, beta_arr, log_gamma, log_weights, sin_alpha = self._coerce_on_axis_inputs(
            nu, B, N, slab_depth, beta, alpha, gamma, gamma_min, gamma_max, n_gamma
        )
        log_correction = self._compute_log_doppler(beta_arr) - np.log1p(z)
        log_I_rf = self._compute_log_on_axis_rf_intensity(
            log_nu,
            log_B,
            log_N,
            log_slab,
            log_gamma,
            log_weights,
            log_correction,
            sin_alpha=sin_alpha,
        )
        return np.exp(log_I_rf + 3.0 * log_correction[np.newaxis, :]) * (u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

    def compute_rest_frame_brightness_temperature(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        beta: Union[float, np.ndarray],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        z: float = 0.0,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame brightness temperature per sightline.

        The brightness temperature is evaluated from the comoving-frame intensity
        returned by :meth:`compute_rest_frame_specific_intensity` using the
        Rayleigh--Jeans relation

        .. math::

            T'_B(\xi)
            =
            \frac{c^2 I'_{\nu'(\xi)}(\xi)}
                 {2 k_B \nu'(\xi)^2}.

        The input frequency ``nu`` is interpreted as an observer-frame frequency and
        is shifted into the comoving frame separately for each sightline before
        evaluating :math:`T'_B`.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are interpreted as Hz.
            The comoving-frame frequency is computed internally as
            :math:`\nu' = \nu(1+z)/\mathcal{D}(\xi)`.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength per sightline. Bare values are
            interpreted as Gauss. Must be scalar or broadcastable to
            ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`, with expected shape ``(n_theta, n_gamma)``
            or callable output compatible with that shape.
        slab_depth : float, array-like, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'` per
            sightline. Bare values are interpreted as cm.
        beta : float or array-like
            Bulk velocity :math:`\beta = v/c` per sightline. Must be scalar or
            broadcastable to ``(n_theta,)``.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If ``None``, a logarithmic grid is built
            from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle per sightline. Bare values are interpreted
            as radians. If ``None``, the pitch-angle-averaged kernel is used.
        z : float, optional
            Cosmological redshift. Default is ``0.0``.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        T_B_rf : ~astropy.units.Quantity, shape ``(n_nu, n_theta)``
            Comoving-frame brightness temperature per sightline in Kelvin.

        Notes
        -----
        The returned brightness temperature is evaluated in the plasma rest frame.
        Use :meth:`compute_brightness_temperature` for the observer-frame
        brightness temperature.
        """
        log_nu, log_B, log_N, log_slab, beta_arr, log_gamma, log_weights, sin_alpha = self._coerce_on_axis_inputs(
            nu, B, N, slab_depth, beta, alpha, gamma, gamma_min, gamma_max, n_gamma
        )
        log_correction = self._compute_log_doppler(beta_arr) - np.log1p(z)
        log_nu_rf = log_nu[:, np.newaxis] - log_correction[np.newaxis, :]
        log_I_rf = self._compute_log_on_axis_rf_intensity(
            log_nu,
            log_B,
            log_N,
            log_slab,
            log_gamma,
            log_weights,
            log_correction,
            sin_alpha=sin_alpha,
        )
        _log_c = np.log(consts.c.cgs.value)
        _log_kb = np.log(consts.k_B.cgs.value)
        log_T = 2.0 * _log_c + log_I_rf - np.log(2.0) - _log_kb - 2.0 * log_nu_rf
        return np.exp(log_T) * u.K

    def compute_brightness_temperature(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        beta: Union[float, np.ndarray],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        z: float = 0.0,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame brightness temperature per sightline.

        The observer-frame brightness temperature is computed from the
        observer-frame specific intensity using the Rayleigh--Jeans relation

        .. math::

            T_B(\xi)
            =
            \frac{c^2 I_\nu(\xi)}
                 {2 k_B \nu^2}.

        The specific intensity includes the sightline-dependent Doppler boost and
        cosmological redshift transformation.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are interpreted as Hz.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength per sightline. Bare values are
            interpreted as Gauss. Must be scalar or broadcastable to
            ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`, with expected shape ``(n_theta, n_gamma)``
            or callable output compatible with that shape.
        slab_depth : float, array-like, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'` per
            sightline. Bare values are interpreted as cm.
        beta : float or array-like
            Bulk velocity :math:`\beta = v/c` per sightline. Must be scalar or
            broadcastable to ``(n_theta,)``.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If ``None``, a logarithmic grid is built
            from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle per sightline. Bare values are interpreted
            as radians. If ``None``, the pitch-angle-averaged kernel is used.
        z : float, optional
            Cosmological redshift. Default is ``0.0``.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        T_B : ~astropy.units.Quantity, shape ``(n_nu, n_theta)``
            Observer-frame brightness temperature per sightline in Kelvin.

        Notes
        -----
        This method does not integrate over angular sightlines. It returns the
        brightness temperature associated with each quadrature node.
        """
        log_nu, log_B, log_N, log_slab, beta_arr, log_gamma, log_weights, sin_alpha = self._coerce_on_axis_inputs(
            nu, B, N, slab_depth, beta, alpha, gamma, gamma_min, gamma_max, n_gamma
        )
        log_correction = self._compute_log_doppler(beta_arr) - np.log1p(z)
        log_I_rf = self._compute_log_on_axis_rf_intensity(
            log_nu,
            log_B,
            log_N,
            log_slab,
            log_gamma,
            log_weights,
            log_correction,
            sin_alpha=sin_alpha,
        )
        log_I_obs = log_I_rf + 3.0 * log_correction[np.newaxis, :]
        _log_c = np.log(consts.c.cgs.value)
        _log_kb = np.log(consts.k_B.cgs.value)
        log_T = 2.0 * _log_c + log_I_obs - np.log(2.0) - _log_kb - 2.0 * log_nu[:, np.newaxis]
        return np.exp(log_T) * u.K

    # ------------------------------------------ #
    # Public API — Flux density                  #
    # ------------------------------------------ #
    def compute_sightline_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        beta: Union[float, np.ndarray],
        angular_diameter_distance: Union[u.Quantity, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0.0,
        cosmology=None,
        f_A: Union[float, np.ndarray] = 1.0,
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the quadrature-weighted flux-density contribution from each sightline.

        This method evaluates the observer-frame flux-density contribution associated
        with each angular quadrature node of an on-axis axisymmetric outflow. The
        returned array includes the Gauss-Legendre projected-area weight
        ``quad_weights = gl_weights * cos_theta``. Therefore, summing the result over
        the sightline axis gives the same angle-integrated flux density returned by
        :meth:`compute_flux_density`.

        The discretized contribution is

        .. math::

            \Delta F_{\nu,i}
            =
            \frac{2\pi}{D_A^2}
            w_i \xi_i\,
            f_A(\xi_i) R(\xi_i)^2
            \left(\frac{\mathcal{D}(\xi_i)}{1+z}\right)^3
            I'_{\nu'(\xi_i)}(\xi_i),

        where :math:`\xi_i = \cos\theta_i`, ``w_i`` is the Gauss-Legendre weight on
        ``[0, 1]``, and

        .. math::

            \nu'(\xi_i)
            =
            \frac{(1+z)\nu}{\mathcal{D}(\xi_i)}.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are interpreted as Hz.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength per sightline. Bare values are
            interpreted as Gauss. Must be scalar or broadcastable to
            ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`, with expected shape ``(n_theta, n_gamma)``
            or callable output compatible with that shape.
        slab_depth : float, array-like, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'` per
            sightline. Bare values are interpreted as cm.
        R : float, array-like, or ~astropy.units.Quantity
            Emission radius for each sightline. Bare values are interpreted as cm.
            Must be scalar or broadcastable to ``(n_theta,)``.
        beta : float or array-like
            Bulk velocity :math:`\beta = v/c` for each sightline. Must be scalar or
            broadcastable to ``(n_theta,)``.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Angular-diameter distance to the source. One distance specification, or
            a non-zero ``z`` with a usable cosmology, must be provided.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Luminosity distance to the source. Converted internally to the
            angular-diameter distance as needed.
        proper_distance : ~astropy.units.Quantity or None, optional
            Proper or comoving line-of-sight distance, depending on the convention
            used by :func:`resolve_cosmological_distances`.
        z : float, optional
            Cosmological redshift. Used both for the Doppler-redshift correction and
            for distance resolution when no explicit distance is supplied. Default
            is ``0.0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to resolve distances when needed.
        f_A : float or array-like, optional
            Projected-area filling factor for each sightline. Must be scalar or
            broadcastable to ``(n_theta,)``. Default is ``1.0``.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If ``None``, a logarithmic grid is built
            from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle per sightline. Bare values are interpreted
            as radians. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        F_ring : ~astropy.units.Quantity, shape ``(n_nu, n_theta)``
            Quadrature-weighted flux-density contribution from each angular
            sightline in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.

        Notes
        -----
        Since the returned values already include the quadrature weights, the
        angle-integrated flux density is obtained by

        ``F_ring.sum(axis=-1)``.

        Do not multiply by ``quad_weights`` again.
        """
        log_nu, log_B, log_N, log_slab, beta_arr, log_gamma, log_weights, sin_alpha = self._coerce_on_axis_inputs(
            nu, B, N, slab_depth, beta, alpha, gamma, gamma_min, gamma_max, n_gamma
        )
        log_D_A, _, log_R, log_f_A = self._resolve_flux_geometry(
            R, f_A, angular_diameter_distance, luminosity_distance, proper_distance, z, cosmology
        )
        log_correction = self._compute_log_doppler(beta_arr) - np.log1p(z)
        log_I_rf = self._compute_log_on_axis_rf_intensity(
            log_nu,
            log_B,
            log_N,
            log_slab,
            log_gamma,
            log_weights,
            log_correction,
            sin_alpha=sin_alpha,
        )
        log_sightline = (
            np.log(2.0 * np.pi)
            - 2.0 * log_D_A
            + np.log(self._quad_weights)[np.newaxis, :]
            + log_f_A[np.newaxis, :]
            + 2.0 * log_R[np.newaxis, :]
            + 3.0 * log_correction[np.newaxis, :]
            + log_I_rf
        )
        return np.exp(log_sightline) * (u.erg / (u.s * u.cm**2 * u.Hz))

    def compute_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        beta: Union[float, np.ndarray],
        angular_diameter_distance: Union[u.Quantity, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0.0,
        cosmology=None,
        f_A: Union[float, np.ndarray] = 1.0,
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the angle-integrated observer-frame spectral flux density.

        This method integrates the observer-frame synchrotron specific intensity over
        the projected area of an on-axis axisymmetric outflow:

        .. math::

            F_\nu
            =
            \frac{2\pi}{D_A^2}
            \int_0^1
            f_A(\xi) R(\xi)^2
            \left(\frac{\mathcal{D}(\xi)}{1+z}\right)^3
            I'_{\nu'(\xi)}(\xi)\,
            \xi\,d\xi.

        The integral is evaluated using the Gauss-Legendre quadrature nodes and
        weights precomputed at initialization.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are interpreted as Hz.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength per sightline. Bare values are
            interpreted as Gauss. Must be scalar or broadcastable to
            ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`, with expected shape ``(n_theta, n_gamma)``
            or callable output compatible with that shape.
        slab_depth : float, array-like, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'` per
            sightline. Bare values are interpreted as cm.
        R : float, array-like, or ~astropy.units.Quantity
            Emission radius for each sightline. Bare values are interpreted as cm.
            Must be scalar or broadcastable to ``(n_theta,)``.
        beta : float or array-like
            Bulk velocity :math:`\beta = v/c` for each sightline. Must be scalar or
            broadcastable to ``(n_theta,)``.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Angular-diameter distance to the source. One distance specification, or
            a non-zero ``z`` with a usable cosmology, must be provided.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Luminosity distance to the source. Converted internally to the
            angular-diameter distance as needed.
        proper_distance : ~astropy.units.Quantity or None, optional
            Proper or comoving line-of-sight distance, depending on the convention
            used by :func:`resolve_cosmological_distances`.
        z : float, optional
            Cosmological redshift. Used both for the Doppler-redshift correction and
            for distance resolution when no explicit distance is supplied. Default
            is ``0.0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to resolve distances when needed.
        f_A : float or array-like, optional
            Projected-area filling factor for each sightline. Must be scalar or
            broadcastable to ``(n_theta,)``. Default is ``1.0``.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If ``None``, a logarithmic grid is built
            from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle per sightline. Bare values are interpreted
            as radians. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        F_nu : ~astropy.units.Quantity, shape ``(n_nu,)``
            Angle-integrated observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.

        Notes
        -----
        Use :meth:`compute_sightline_flux_density` to inspect the contribution from
        each angular quadrature node before summation.
        """
        log_nu, log_B, log_N, log_slab, beta_arr, log_gamma, log_weights, sin_alpha = self._coerce_on_axis_inputs(
            nu, B, N, slab_depth, beta, alpha, gamma, gamma_min, gamma_max, n_gamma
        )
        log_D_A, _, log_R, log_f_A = self._resolve_flux_geometry(
            R, f_A, angular_diameter_distance, luminosity_distance, proper_distance, z, cosmology
        )
        log_correction = self._compute_log_doppler(beta_arr) - np.log1p(z)
        log_I_rf = self._compute_log_on_axis_rf_intensity(
            log_nu,
            log_B,
            log_N,
            log_slab,
            log_gamma,
            log_weights,
            log_correction,
            sin_alpha=sin_alpha,
        )
        log_integrand = (
            np.log(self._quad_weights)[np.newaxis, :]
            + log_f_A[np.newaxis, :]
            + 2.0 * log_R[np.newaxis, :]
            + 3.0 * log_correction[np.newaxis, :]
            + log_I_rf
        )
        log_F = np.log(2.0 * np.pi) - 2.0 * log_D_A + logsumexp(log_integrand, axis=-1)
        return np.exp(log_F) * (u.erg / (u.s * u.cm**2 * u.Hz))

    # ------------------------------------------ #
    # Public API — Luminosity                    #
    # ------------------------------------------ #
    def compute_isotropic_luminosity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        beta: Union[float, np.ndarray],
        angular_diameter_distance: Union[u.Quantity, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0.0,
        cosmology=None,
        f_A: Union[float, np.ndarray] = 1.0,
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the angle-integrated isotropic-equivalent spectral luminosity.

        This method computes the observed, angle-integrated flux density of the
        on-axis axisymmetric outflow and converts it to an isotropic-equivalent
        spectral luminosity using

        .. math::

            L_{\nu,\mathrm{iso}}
            =
            4\pi D_L^2 F_\nu.

        The flux density :math:`F_\nu` is computed from the angular integral

        .. math::

            F_\nu
            =
            \frac{2\pi}{D_A^2}
            \int_0^1
            f_A(\xi) R(\xi)^2
            \left(\frac{\mathcal{D}(\xi)}{1+z}\right)^3
            I'_{\nu'(\xi)}(\xi)\,
            \xi\,d\xi.

        Parameters
        ----------
        nu : float, array-like, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are interpreted as Hz.
        B : float, array-like, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength per sightline. Bare values are
            interpreted as Gauss. Must be scalar or broadcastable to
            ``(n_theta,)``.
        N : array-like or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`, with expected shape ``(n_theta, n_gamma)``
            or callable output compatible with that shape.
        slab_depth : float, array-like, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'` per
            sightline. Bare values are interpreted as cm.
        R : float, array-like, or ~astropy.units.Quantity
            Emission radius for each sightline. Bare values are interpreted as cm.
            Must be scalar or broadcastable to ``(n_theta,)``.
        beta : float or array-like
            Bulk velocity :math:`\beta = v/c` for each sightline. Must be scalar or
            broadcastable to ``(n_theta,)``.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Angular-diameter distance to the source. One distance specification, or
            a non-zero ``z`` with a usable cosmology, must be provided.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Luminosity distance to the source. Used in
            :math:`4\pi D_L^2 F_\nu`.
        proper_distance : ~astropy.units.Quantity or None, optional
            Proper or comoving line-of-sight distance, depending on the convention
            used by :func:`resolve_cosmological_distances`.
        z : float, optional
            Cosmological redshift. Used both for the Doppler-redshift correction and
            for distance resolution when no explicit distance is supplied. Default
            is ``0.0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to resolve distances when needed.
        f_A : float or array-like, optional
            Projected-area filling factor for each sightline. Must be scalar or
            broadcastable to ``(n_theta,)``. Default is ``1.0``.
        gamma : array-like or None, optional
            Explicit Lorentz-factor grid. If ``None``, a logarithmic grid is built
            from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        alpha : float, array-like, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle per sightline. Bare values are interpreted
            as radians. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Lower bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1.0``.
        gamma_max : float, optional
            Upper bound of the internally generated Lorentz-factor grid. Ignored
            when ``gamma`` is supplied. Default is ``1e8``.
        n_gamma : int, optional
            Number of Lorentz-factor grid points. Ignored when ``gamma`` is
            supplied. Default is ``200``.

        Returns
        -------
        L_iso : ~astropy.units.Quantity, shape ``(n_nu,)``
            Angle-integrated isotropic-equivalent spectral luminosity in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.

        Notes
        -----
        This quantity is isotropic-equivalent, not the true emitted luminosity of an
        anisotropic or relativistically beamed outflow.
        """
        log_nu, log_B, log_N, log_slab, beta_arr, log_gamma, log_weights, sin_alpha = self._coerce_on_axis_inputs(
            nu, B, N, slab_depth, beta, alpha, gamma, gamma_min, gamma_max, n_gamma
        )
        log_D_A, log_D_L, log_R, log_f_A = self._resolve_flux_geometry(
            R, f_A, angular_diameter_distance, luminosity_distance, proper_distance, z, cosmology
        )
        log_correction = self._compute_log_doppler(beta_arr) - np.log1p(z)
        log_I_rf = self._compute_log_on_axis_rf_intensity(
            log_nu,
            log_B,
            log_N,
            log_slab,
            log_gamma,
            log_weights,
            log_correction,
            sin_alpha=sin_alpha,
        )
        log_integrand = (
            np.log(self._quad_weights)[np.newaxis, :]
            + log_f_A[np.newaxis, :]
            + 2.0 * log_R[np.newaxis, :]
            + 3.0 * log_correction[np.newaxis, :]
            + log_I_rf
        )
        log_F = np.log(2.0 * np.pi) - 2.0 * log_D_A + logsumexp(log_integrand, axis=-1)
        return np.exp(np.log(4.0 * np.pi) + 2.0 * log_D_L + log_F) * (u.erg / (u.s * u.Hz))
