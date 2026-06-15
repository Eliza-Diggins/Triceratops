"""
Generic 1-zone SED model, with numerical integration of the electron distribution.

This is the most flexible SED model, but also the slowest. It is used as a reference for testing the other models,
and for cases where the electron distribution is not a simple power law or broken power law.
"""

from collections.abc import Callable
from typing import Union

import numpy as np
from astropy import cosmology as cosmo
from astropy import units as u

from trilobite._typing import _ArrayLike, _UnitBearingArrayLike, _UnitBearingScalarLike
from trilobite.physics_utils import get_cosmology, resolve_cosmological_distances
from trilobite.radiation.synchrotron.electron_distributions import MaxwellJuettner, PowerLaw
from trilobite.radiation.synchrotron.SEDs.numerical.core import NumericalSynchrotronEngine
from trilobite.radiation.synchrotron.SEDs.one_zone.seds import SynchrotronSED
from trilobite.utils.misc_utils import ensure_in_units


class Numerical_PL_SSA_SED(SynchrotronSED):
    r"""
    Numerical one-zone power-law synchrotron SED with self-absorption.

    Evaluates the synchrotron flux density by numerically integrating the
    synchrotron kernel over a power-law electron distribution with full
    radiative transfer (SSA included). Unlike the analytic
    :class:`PowerLaw_SSA_SynchrotronSED`, no assumption is made about the
    spectral regime.

    The electron number density is normalized via equipartition:

    .. math::

        N(\gamma) = N_0\,\gamma^{-p}, \quad \gamma_{\min} \le \gamma \le \gamma_{\max}

    where :math:`N_0` is fixed by :math:`\epsilon_E / \epsilon_B` and the
    magnetic energy density
    (see :meth:`~trilobite.radiation.synchrotron.electron_distributions.PowerLaw._normalize_from_magnetic_field`).

    Parameters
    ----------
    gamma : ~numpy.ndarray or None, optional
        Explicit Lorentz factor grid. If ``None``, a log-uniform grid is
        built from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
    gamma_min : float, optional
        Lower bound of the Lorentz factor grid. Default ``1.0``.
    gamma_max : float, optional
        Upper bound of the Lorentz factor grid. Default ``1e8``.
    n_gamma : int, optional
        Number of quadrature points on the Lorentz factor grid. Default ``200``.
    x_min : float, optional
        Lower bound of the synchrotron kernel :math:`x` grid. Default ``1e-5``.
    x_max : float, optional
        Upper bound of the synchrotron kernel :math:`x` grid. Default ``1e2``.
    num_points : int, optional
        Number of points on the kernel grid. Default ``1000``.
    spacing : {"log", "linear"}, optional
        Kernel grid spacing. Default ``"log"``.
    method : {"exact", "lu"}, optional
        Kernel evaluation method. Default ``"exact"``.
    """

    def __init__(
        self,
        gamma: Union[np.ndarray, None] = None,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
        x_min: float = 1e-5,
        x_max: float = 1e2,
        num_points: int = 1000,
        spacing: str = "log",
        method: str = "exact",
    ):
        self._engine = NumericalSynchrotronEngine()

        self._log_gamma, self._log_weights = self._engine._build_gamma_grid(
            gamma=gamma,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

        kernel_kwargs = {"x_min": x_min, "x_max": x_max, "num_points": num_points, "spacing": spacing, "method": method}
        self._engine.load_avg_first_kernel(**kernel_kwargs)
        self._engine.load_first_kernel(**kernel_kwargs)

    def _log_opt_sed(
        self,
        nu: "_ArrayLike",
        B: float,
        R: float,
        D_A: float,
        *,
        p: float = 3.0,
        f_V: float = 1.0,
        f_A: float = 1.0,
        beta: float = 0.0,
        cos_theta: float = 1.0,
        sin_alpha: Union[float, None] = None,
        z: float = 0.0,
        epsilon_B: float = 0.1,
        epsilon_E: float = 0.1,
        gamma_min: float = 1.0,
        gamma_max: float = 1e6,
    ) -> "_ArrayLike":
        r"""
        Low-level numerical synchrotron SED evaluation.

        All inputs are plain CGS scalars; no unit handling is performed.

        Parameters
        ----------
        nu : array-like
            Observed frequency in Hz.
        B : float
            Magnetic field strength in Gauss.
        R : float
            Outer radius of the emitting region in cm.
        D_A : float
            Angular diameter distance in cm.
        p : float, optional
            Electron power-law index. Default ``3.0``.
        f_V : float, optional
            Volume filling factor. The effective LOS path length passed to
            the radiative transfer is :math:`f_V R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor. The effective projected area is
            :math:`f_A \pi R^2`. Default ``1.0``.
        beta : float, optional
            Bulk velocity :math:`v/c`. Default ``0.0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and the line of
            sight (relativistic correction). Default ``1.0``.
        sin_alpha : float or None, optional
            Sine of the electron pitch angle. ``None`` selects the
            pitch-angle-averaged kernel. Default ``None``.
        z : float, optional
            Source redshift. Default ``0.0``.
        epsilon_B : float, optional
            Magnetic equipartition fraction. Default ``0.1``.
        epsilon_E : float, optional
            Electron equipartition fraction. Default ``0.1``.
        gamma_min : float, optional
            Minimum electron Lorentz factor active in the power law.
            Default ``1.0``.
        gamma_max : float, optional
            Maximum electron Lorentz factor active in the power law.
            Default ``1e6``.

        Returns
        -------
        log_flux : array-like
            :math:`\ln F_\nu` in CGS
            (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`).
        """
        # Electron distribution normalization via equipartition.
        N0 = np.asarray(
            PowerLaw._normalize_from_magnetic_field(
                B,
                epsilon_B,
                epsilon_E,
                p=p,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
            ),
            dtype="f8",
        )

        # Build log_N on the pre-built gamma grid, zeroing outside [gamma_min, gamma_max].
        # Leading batch dims come from N0's shape (scalar → ()).
        gamma_arr = np.exp(self._log_gamma)
        mask = (gamma_arr >= gamma_min) & (gamma_arr <= gamma_max)
        log_N = np.full(N0.shape + (len(self._log_gamma),), -np.inf)
        log_N[..., mask] = np.log(N0)[..., np.newaxis] - p * self._log_gamma[mask]

        log_nu = np.log(np.asarray(nu, dtype="f8"))
        log_B = np.log(float(B))
        log_R_eff = np.log(float(f_V) * float(R))
        log_A_eff = np.log(float(f_A) * np.pi * float(R) ** 2)
        log_D_A = np.log(float(D_A))

        if sin_alpha is not None:
            return self._engine._compute_log_flux_density(
                log_nu=log_nu,
                log_R=log_R_eff,
                log_B=log_B,
                log_N=log_N,
                log_gamma=self._log_gamma,
                log_weights=self._log_weights,
                log_A=log_A_eff,
                log_D_A=log_D_A,
                z=float(z),
                beta=float(beta),
                cos_theta=float(cos_theta),
                sin_alpha=float(sin_alpha),
            )
        else:
            return self._engine._compute_log_pa_flux_density(
                log_nu=log_nu,
                log_R=log_R_eff,
                log_B=log_B,
                log_N=log_N,
                log_gamma=self._log_gamma,
                log_weights=self._log_weights,
                log_A_eff=log_A_eff,
                log_D_A=log_D_A,
                z=float(z),
                beta=float(beta),
                cos_theta=float(cos_theta),
            )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        B: "_UnitBearingScalarLike",
        R: "_UnitBearingScalarLike",
        *,
        p: float = 3.0,
        f_V: float = 1.0,
        f_A: float = 1.0,
        beta: float = 0.0,
        cos_theta: float = 1.0,
        alpha: "_UnitBearingScalarLike" = None,
        epsilon_B: float = 0.1,
        epsilon_E: float = 0.1,
        gamma_min: float = 1.0,
        gamma_max: float = 1e6,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
    ) -> "_UnitBearingArrayLike":
        r"""
        Evaluate the numerical power-law synchrotron SED with self-absorption.

        Parameters
        ----------
        nu : array-like or ~astropy.units.Quantity
            Observed frequency. Bare values are treated as Hz.
        B : float or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        R : float or ~astropy.units.Quantity
            Outer radius of the emitting region. Bare values are treated as cm.
        p : float, optional
            Electron power-law index. Default ``3.0``.
        f_V : float, optional
            Volume filling factor; the effective LOS path length is
            :math:`f_V R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; the effective projected area is
            :math:`f_A \pi R^2`. Default ``1.0``.
        beta : float, optional
            Bulk velocity :math:`v/c`. Default ``0.0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and the line of
            sight. Default ``1.0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Electron pitch angle. Bare values are treated as radians.
            ``None`` selects the pitch-angle-averaged kernel. Default ``None``.
        epsilon_B : float, optional
            Fraction of post-shock energy in magnetic fields. Default ``0.1``.
        epsilon_E : float, optional
            Fraction of post-shock energy in relativistic electrons.
            Default ``0.1``.
        gamma_min : float, optional
            Minimum electron Lorentz factor active in the power law.
            Default ``1.0``.
        gamma_max : float, optional
            Maximum electron Lorentz factor active in the power law.
            Default ``1e6``.
        redshift : float, optional
            Source redshift.
        luminosity_distance : float or ~astropy.units.Quantity, optional
            Luminosity distance. Bare values are treated as cm.
        angular_diameter_distance : float or ~astropy.units.Quantity, optional
            Angular diameter distance. Bare values are treated as cm.
        proper_distance : float or ~astropy.units.Quantity, optional
            Proper (comoving) distance. Bare values are treated as cm.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to derive missing distance measures. Defaults to
            the package-configured cosmology.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        # Resolve cosmological distances → angular diameter distance + redshift.
        _cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=_cosmology,
        )
        D_A = ensure_in_units(distances["angular_diameter_distance"], "cm")
        z = float(distances["redshift"])

        # Convert physical inputs to raw CGS.
        nu_cgs = ensure_in_units(nu, "Hz")
        B_cgs = float(ensure_in_units(B, "G"))
        R_cgs = float(ensure_in_units(R, "cm"))
        D_A_cgs = float(D_A)

        # Resolve pitch angle: None → pitch-averaged.
        sin_alpha = float(np.sin(ensure_in_units(alpha, "rad"))) if alpha is not None else None

        log_flux = self._log_opt_sed(
            nu=nu_cgs,
            B=B_cgs,
            R=R_cgs,
            D_A=D_A_cgs,
            p=p,
            f_V=f_V,
            f_A=f_A,
            beta=beta,
            cos_theta=cos_theta,
            sin_alpha=sin_alpha,
            z=z,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )

        return np.exp(log_flux) * u.erg / (u.cm**2 * u.s * u.Hz)


class Numerical_Thermal_SSA_SED(SynchrotronSED):
    r"""
    Numerical one-zone thermal (Maxwell-Jüttner) synchrotron SED with self-absorption.

    Evaluates the synchrotron flux density by numerically integrating the
    synchrotron kernel over a Maxwell-Jüttner electron distribution with
    full radiative transfer (SSA included).

    The electron distribution is

    .. math::

        N(\gamma) = \frac{N_{\rm therm}}{2\Theta^3}\,\gamma^2\,e^{-\gamma/\Theta},

    where :math:`\Theta = kT/(m_e c^2)` is the dimensionless electron
    temperature and :math:`N_{\rm therm}` is fixed by equipartition (see
    :meth:`~trilobite.radiation.synchrotron.electron_distributions.MaxwellJuettner._normalize_from_magnetic_field`).

    Parameters
    ----------
    gamma : ~numpy.ndarray or None, optional
        Explicit Lorentz factor grid. If ``None``, a log-uniform grid is
        built from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
    gamma_min : float, optional
        Lower bound of the Lorentz factor grid. Default ``1.0``.
    gamma_max : float, optional
        Upper bound of the Lorentz factor grid. Default ``1e8``.
    n_gamma : int, optional
        Number of quadrature points. Default ``200``.
    x_min : float, optional
        Lower bound of the synchrotron kernel :math:`x` grid. Default ``1e-5``.
    x_max : float, optional
        Upper bound of the synchrotron kernel :math:`x` grid. Default ``1e2``.
    num_points : int, optional
        Number of points on the kernel grid. Default ``1000``.
    spacing : {"log", "linear"}, optional
        Kernel grid spacing. Default ``"log"``.
    method : {"exact", "lu"}, optional
        Kernel evaluation method. Default ``"exact"``.
    """

    def __init__(
        self,
        gamma: Union[np.ndarray, None] = None,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
        x_min: float = 1e-5,
        x_max: float = 1e2,
        num_points: int = 1000,
        spacing: str = "log",
        method: str = "exact",
    ):
        self._engine = NumericalSynchrotronEngine()

        self._log_gamma, self._log_weights = self._engine._build_gamma_grid(
            gamma=gamma,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

        kernel_kwargs = {"x_min": x_min, "x_max": x_max, "num_points": num_points, "spacing": spacing, "method": method}
        self._engine.load_avg_first_kernel(**kernel_kwargs)
        self._engine.load_first_kernel(**kernel_kwargs)

    def _log_opt_sed(
        self,
        nu: "_ArrayLike",
        B: float,
        R: float,
        D_A: float,
        Theta: float,
        *,
        f_V: float = 1.0,
        f_A: float = 1.0,
        beta: float = 0.0,
        cos_theta: float = 1.0,
        sin_alpha: Union[float, None] = None,
        z: float = 0.0,
        epsilon_B: float = 0.1,
        epsilon_E: float = 0.1,
    ) -> "_ArrayLike":
        r"""
        Low-level numerical thermal synchrotron SED evaluation.

        All inputs are plain CGS scalars; no unit handling is performed.

        Parameters
        ----------
        nu : array-like
            Observed frequency in Hz.
        B : float
            Magnetic field strength in Gauss.
        R : float
            Outer radius of the emitting region in cm.
        D_A : float
            Angular diameter distance in cm.
        Theta : float
            Dimensionless electron temperature :math:`\Theta = kT/(m_e c^2)`.
        f_V : float, optional
            Volume filling factor. Effective LOS path length is :math:`f_V R`.
            Default ``1.0``.
        f_A : float, optional
            Area filling factor. Effective projected area is
            :math:`f_A \pi R^2`. Default ``1.0``.
        beta : float, optional
            Bulk velocity :math:`v/c`. Default ``0.0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and the line of
            sight (relativistic correction). Default ``1.0``.
        sin_alpha : float or None, optional
            Sine of the electron pitch angle. ``None`` selects the
            pitch-angle-averaged kernel. Default ``None``.
        z : float, optional
            Source redshift. Default ``0.0``.
        epsilon_B : float, optional
            Magnetic equipartition fraction. Default ``0.1``.
        epsilon_E : float, optional
            Electron equipartition fraction. Default ``0.1``.

        Returns
        -------
        log_flux : array-like
            :math:`\ln F_\nu` in CGS
            (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`).
        """
        N_therm = np.asarray(
            MaxwellJuettner._normalize_from_magnetic_field(
                B,
                epsilon_B,
                epsilon_E,
                Theta=Theta,
            ),
            dtype="f8",
        )

        # Maxwell-Jüttner: N(gamma) = N_therm/(2 Theta^3) * gamma^2 * exp(-gamma/Theta)
        log_N = (
            np.log(N_therm)[..., np.newaxis]
            - np.log(2.0)
            - 3.0 * np.log(float(Theta))
            + 2.0 * self._log_gamma
            - np.exp(self._log_gamma) / float(Theta)
        )

        log_nu = np.log(np.asarray(nu, dtype="f8"))
        log_B = np.log(float(B))
        log_R_eff = np.log(float(f_V) * float(R))
        log_A_eff = np.log(float(f_A) * np.pi * float(R) ** 2)
        log_D_A = np.log(float(D_A))

        if sin_alpha is not None:
            return self._engine._compute_log_flux_density(
                log_nu=log_nu,
                log_R=log_R_eff,
                log_B=log_B,
                log_N=log_N,
                log_gamma=self._log_gamma,
                log_weights=self._log_weights,
                log_A=log_A_eff,
                log_D_A=log_D_A,
                z=float(z),
                beta=float(beta),
                cos_theta=float(cos_theta),
                sin_alpha=float(sin_alpha),
            )
        else:
            return self._engine._compute_log_pa_flux_density(
                log_nu=log_nu,
                log_R=log_R_eff,
                log_B=log_B,
                log_N=log_N,
                log_gamma=self._log_gamma,
                log_weights=self._log_weights,
                log_A_eff=log_A_eff,
                log_D_A=log_D_A,
                z=float(z),
                beta=float(beta),
                cos_theta=float(cos_theta),
            )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        B: "_UnitBearingScalarLike",
        R: "_UnitBearingScalarLike",
        Theta: float,
        *,
        f_V: float = 1.0,
        f_A: float = 1.0,
        beta: float = 0.0,
        cos_theta: float = 1.0,
        alpha: "_UnitBearingScalarLike" = None,
        epsilon_B: float = 0.1,
        epsilon_E: float = 0.1,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
    ) -> "_UnitBearingArrayLike":
        r"""
        Evaluate the numerical thermal synchrotron SED with self-absorption.

        Parameters
        ----------
        nu : array-like or ~astropy.units.Quantity
            Observed frequency. Bare values are treated as Hz.
        B : float or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        R : float or ~astropy.units.Quantity
            Outer radius of the emitting region. Bare values are treated as cm.
        Theta : float
            Dimensionless electron temperature :math:`\Theta = kT/(m_e c^2)`.
        f_V : float, optional
            Volume filling factor; the effective LOS path length is
            :math:`f_V R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; the effective projected area is
            :math:`f_A \pi R^2`. Default ``1.0``.
        beta : float, optional
            Bulk velocity :math:`v/c`. Default ``0.0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and the line of
            sight. Default ``1.0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Electron pitch angle. Bare values are treated as radians.
            ``None`` selects the pitch-angle-averaged kernel. Default ``None``.
        epsilon_B : float, optional
            Fraction of post-shock energy in magnetic fields. Default ``0.1``.
        epsilon_E : float, optional
            Fraction of post-shock energy in relativistic electrons.
            Default ``0.1``.
        redshift : float, optional
            Source redshift.
        luminosity_distance : float or ~astropy.units.Quantity, optional
            Luminosity distance. Bare values are treated as cm.
        angular_diameter_distance : float or ~astropy.units.Quantity, optional
            Angular diameter distance. Bare values are treated as cm.
        proper_distance : float or ~astropy.units.Quantity, optional
            Proper (comoving) distance. Bare values are treated as cm.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to derive missing distance measures.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        _cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=_cosmology,
        )
        D_A = ensure_in_units(distances["angular_diameter_distance"], "cm")
        z = float(distances["redshift"])

        nu_cgs = ensure_in_units(nu, "Hz")
        B_cgs = float(ensure_in_units(B, "G"))
        R_cgs = float(ensure_in_units(R, "cm"))
        D_A_cgs = float(D_A)

        sin_alpha = float(np.sin(ensure_in_units(alpha, "rad"))) if alpha is not None else None

        log_flux = self._log_opt_sed(
            nu=nu_cgs,
            B=B_cgs,
            R=R_cgs,
            D_A=D_A_cgs,
            Theta=float(Theta),
            f_V=f_V,
            f_A=f_A,
            beta=beta,
            cos_theta=cos_theta,
            sin_alpha=sin_alpha,
            z=z,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
        )

        return np.exp(log_flux) * u.erg / (u.cm**2 * u.s * u.Hz)


class Numerical_Thermal_PL_SSA_SED(SynchrotronSED):
    r"""
    Numerical one-zone mixed thermal + power-law synchrotron SED with self-absorption.

    Evaluates the synchrotron flux density for a **composite electron
    distribution** consisting of a Maxwell-Jüttner thermal component and a
    non-thermal power-law component, with full radiative transfer (SSA included).

    The total distribution is

    .. math::

        N(\gamma) =
            \underbrace{\frac{N_{\rm therm}}{2\Theta^3}\,\gamma^2\,e^{-\gamma/\Theta}}_{\text{thermal}}
            +
            \underbrace{N_0\,\gamma^{-p}\,\mathbf{1}_{[\gamma_{\min},\gamma_{\max}]}(\gamma)}_{\text{non-thermal}},

    where the thermal fraction :math:`\delta` splits the total electron energy
    budget: :math:`\epsilon_{E,\rm therm} = \delta\,\epsilon_E` and
    :math:`\epsilon_{E,\rm PL} = (1-\delta)\,\epsilon_E` (see
    :meth:`~trilobite.radiation.synchrotron.electron_distributions.MixedDistribution._normalize_from_magnetic_field`).

    Parameters
    ----------
    gamma : ~numpy.ndarray or None, optional
        Explicit Lorentz factor grid. If ``None``, a log-uniform grid is
        built from ``gamma_min``, ``gamma_max``, and ``n_gamma``.
    gamma_min : float, optional
        Lower bound of the Lorentz factor grid. Default ``1.0``.
    gamma_max : float, optional
        Upper bound of the Lorentz factor grid. Default ``1e8``.
    n_gamma : int, optional
        Number of quadrature points. Default ``200``.
    x_min : float, optional
        Lower bound of the synchrotron kernel :math:`x` grid. Default ``1e-5``.
    x_max : float, optional
        Upper bound of the synchrotron kernel :math:`x` grid. Default ``1e2``.
    num_points : int, optional
        Number of points on the kernel grid. Default ``1000``.
    spacing : {"log", "linear"}, optional
        Kernel grid spacing. Default ``"log"``.
    method : {"exact", "lu"}, optional
        Kernel evaluation method. Default ``"exact"``.
    """

    def __init__(
        self,
        gamma: Union[np.ndarray, None] = None,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
        x_min: float = 1e-5,
        x_max: float = 1e2,
        num_points: int = 1000,
        spacing: str = "log",
        method: str = "exact",
    ):
        self._engine = NumericalSynchrotronEngine()

        self._log_gamma, self._log_weights = self._engine._build_gamma_grid(
            gamma=gamma,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

        kernel_kwargs = {"x_min": x_min, "x_max": x_max, "num_points": num_points, "spacing": spacing, "method": method}
        self._engine.load_avg_first_kernel(**kernel_kwargs)
        self._engine.load_first_kernel(**kernel_kwargs)

    def _log_opt_sed(
        self,
        nu: "_ArrayLike",
        B: float,
        R: float,
        D_A: float,
        Theta: float,
        *,
        p: float = 3.0,
        delta: float = 0.5,
        f_V: float = 1.0,
        f_A: float = 1.0,
        beta: float = 0.0,
        cos_theta: float = 1.0,
        sin_alpha: Union[float, None] = None,
        z: float = 0.0,
        epsilon_B: float = 0.1,
        epsilon_E: float = 0.1,
        gamma_min: float = 1.0,
        gamma_max: float = 1e6,
    ) -> "_ArrayLike":
        r"""
        Low-level numerical mixed thermal + power-law synchrotron SED evaluation.

        All inputs are plain CGS scalars; no unit handling is performed.

        Parameters
        ----------
        nu : array-like
            Observed frequency in Hz.
        B : float
            Magnetic field strength in Gauss.
        R : float
            Outer radius of the emitting region in cm.
        D_A : float
            Angular diameter distance in cm.
        Theta : float
            Dimensionless electron temperature :math:`\Theta = kT/(m_e c^2)`.
        p : float, optional
            Power-law index of the non-thermal component. Default ``3.0``.
        delta : float, optional
            Fraction of total electron energy in the thermal component.
            Must satisfy :math:`0 \le \delta \le 1`. Default ``0.5``.
        f_V : float, optional
            Volume filling factor. Effective LOS path length is :math:`f_V R`.
            Default ``1.0``.
        f_A : float, optional
            Area filling factor. Effective projected area is
            :math:`f_A \pi R^2`. Default ``1.0``.
        beta : float, optional
            Bulk velocity :math:`v/c`. Default ``0.0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and the line of
            sight (relativistic correction). Default ``1.0``.
        sin_alpha : float or None, optional
            Sine of the electron pitch angle. ``None`` selects the
            pitch-angle-averaged kernel. Default ``None``.
        z : float, optional
            Source redshift. Default ``0.0``.
        epsilon_B : float, optional
            Magnetic equipartition fraction. Default ``0.1``.
        epsilon_E : float, optional
            Total electron equipartition fraction. Default ``0.1``.
        gamma_min : float, optional
            Minimum Lorentz factor of the non-thermal power-law component.
            Default ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor of the non-thermal power-law component.
            Default ``1e6``.

        Returns
        -------
        log_flux : array-like
            :math:`\ln F_\nu` in CGS
            (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`).
        """
        delta_arr = np.asarray(delta, dtype="f8")
        N_therm = np.asarray(
            MaxwellJuettner._normalize_from_magnetic_field(
                B,
                epsilon_B,
                delta_arr * epsilon_E,
                Theta=Theta,
            ),
            dtype="f8",
        )
        N0_pl = np.asarray(
            PowerLaw._normalize_from_magnetic_field(
                B,
                epsilon_B,
                (1.0 - delta_arr) * epsilon_E,
                p=p,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
            ),
            dtype="f8",
        )

        # Thermal component: Maxwell-Jüttner evaluated over the full grid.
        log_N_mjd = (
            np.log(N_therm)[..., np.newaxis]
            - np.log(2.0)
            - 3.0 * np.log(float(Theta))
            + 2.0 * self._log_gamma
            - np.exp(self._log_gamma) / float(Theta)
        )

        # Non-thermal component: power law gated to [gamma_min, gamma_max].
        gamma_arr = np.exp(self._log_gamma)
        mask = (gamma_arr >= gamma_min) & (gamma_arr <= gamma_max)
        log_N_pl = np.full(N0_pl.shape + (len(self._log_gamma),), -np.inf)
        log_N_pl[..., mask] = np.log(N0_pl)[..., np.newaxis] - p * self._log_gamma[mask]

        log_N = np.logaddexp(log_N_mjd, log_N_pl)

        log_nu = np.log(np.asarray(nu, dtype="f8"))
        log_B = np.log(float(B))
        log_R_eff = np.log(float(f_V) * float(R))
        log_A_eff = np.log(float(f_A) * np.pi * float(R) ** 2)
        log_D_A = np.log(float(D_A))

        if sin_alpha is not None:
            return self._engine._compute_log_flux_density(
                log_nu=log_nu,
                log_R=log_R_eff,
                log_B=log_B,
                log_N=log_N,
                log_gamma=self._log_gamma,
                log_weights=self._log_weights,
                log_A=log_A_eff,
                log_D_A=log_D_A,
                z=float(z),
                beta=float(beta),
                cos_theta=float(cos_theta),
                sin_alpha=float(sin_alpha),
            )
        else:
            return self._engine._compute_log_pa_flux_density(
                log_nu=log_nu,
                log_R=log_R_eff,
                log_B=log_B,
                log_N=log_N,
                log_gamma=self._log_gamma,
                log_weights=self._log_weights,
                log_A_eff=log_A_eff,
                log_D_A=log_D_A,
                z=float(z),
                beta=float(beta),
                cos_theta=float(cos_theta),
            )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        B: "_UnitBearingScalarLike",
        R: "_UnitBearingScalarLike",
        Theta: float,
        *,
        p: float = 3.0,
        delta: float = 0.5,
        f_V: float = 1.0,
        f_A: float = 1.0,
        beta: float = 0.0,
        cos_theta: float = 1.0,
        alpha: "_UnitBearingScalarLike" = None,
        epsilon_B: float = 0.1,
        epsilon_E: float = 0.1,
        gamma_min: float = 1.0,
        gamma_max: float = 1e6,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
    ) -> "_UnitBearingArrayLike":
        r"""
        Evaluate the numerical mixed thermal + power-law synchrotron SED with self-absorption.

        Parameters
        ----------
        nu : array-like or ~astropy.units.Quantity
            Observed frequency. Bare values are treated as Hz.
        B : float or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        R : float or ~astropy.units.Quantity
            Outer radius of the emitting region. Bare values are treated as cm.
        Theta : float
            Dimensionless electron temperature :math:`\Theta = kT/(m_e c^2)`.
        p : float, optional
            Power-law index of the non-thermal electron component. Default ``3.0``.
        delta : float, optional
            Fraction of total electron energy in the thermal component
            (:math:`0 \le \delta \le 1`). Default ``0.5``.
        f_V : float, optional
            Volume filling factor; the effective LOS path length is
            :math:`f_V R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; the effective projected area is
            :math:`f_A \pi R^2`. Default ``1.0``.
        beta : float, optional
            Bulk velocity :math:`v/c`. Default ``0.0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and the line of
            sight. Default ``1.0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Electron pitch angle. Bare values are treated as radians.
            ``None`` selects the pitch-angle-averaged kernel. Default ``None``.
        epsilon_B : float, optional
            Fraction of post-shock energy in magnetic fields. Default ``0.1``.
        epsilon_E : float, optional
            Total fraction of post-shock energy in relativistic electrons.
            Default ``0.1``.
        gamma_min : float, optional
            Minimum Lorentz factor of the non-thermal power-law component.
            Default ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor of the non-thermal power-law component.
            Default ``1e6``.
        redshift : float, optional
            Source redshift.
        luminosity_distance : float or ~astropy.units.Quantity, optional
            Luminosity distance. Bare values are treated as cm.
        angular_diameter_distance : float or ~astropy.units.Quantity, optional
            Angular diameter distance. Bare values are treated as cm.
        proper_distance : float or ~astropy.units.Quantity, optional
            Proper (comoving) distance. Bare values are treated as cm.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to derive missing distance measures.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        _cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=_cosmology,
        )
        D_A = ensure_in_units(distances["angular_diameter_distance"], "cm")
        z = float(distances["redshift"])

        nu_cgs = ensure_in_units(nu, "Hz")
        B_cgs = float(ensure_in_units(B, "G"))
        R_cgs = float(ensure_in_units(R, "cm"))
        D_A_cgs = float(D_A)

        sin_alpha = float(np.sin(ensure_in_units(alpha, "rad"))) if alpha is not None else None

        log_flux = self._log_opt_sed(
            nu=nu_cgs,
            B=B_cgs,
            R=R_cgs,
            D_A=D_A_cgs,
            Theta=float(Theta),
            p=p,
            delta=delta,
            f_V=f_V,
            f_A=f_A,
            beta=beta,
            cos_theta=cos_theta,
            sin_alpha=sin_alpha,
            z=z,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )

        return np.exp(log_flux) * u.erg / (u.cm**2 * u.s * u.Hz)


class NonRelativisticSphericalSynchrotronEngine(NumericalSynchrotronEngine):
    r"""
    Non-relativistic spherical synchrotron engine.

    A thin geometry-specializing wrapper around :class:`NumericalSynchrotronEngine`
    for non-relativistic, spherically symmetric emission regions. Users supply a
    physical radius ``R`` and dimensionless volume and area filling factors ``f_V``
    and ``f_A`` in place of raw ``slab_depth`` and ``A_eff`` arguments. Bulk-motion
    parameters are fixed at zero and removed from every call signature.

    The effective radiative-transfer geometry is:

    .. math::

        A_\mathrm{eff} = \pi R^2 f_A, \qquad
        \ell_\mathrm{eff} = f_V^{1/3}\,R,

    where :math:`f_V` controls the emitting volume
    :math:`V = \tfrac{4}{3}\pi R^3 f_V` and :math:`f_A` modulates the
    projected emitting area.

    See Also
    --------
    :class:`NumericalSynchrotronEngine`
        Base class providing all low-level numerical synchrotron machinery.
    """

    @staticmethod
    def _spherical_geometry(
        R: Union[float, u.Quantity],
        f_V: float,
        f_A: float,
    ):
        """Return ``(slab_depth_cgs, A_eff_cgs)`` from sphere parameters."""
        R_cgs = ensure_in_units(R, u.cm)
        slab_depth_cgs = f_V ** (1.0 / 3.0) * R_cgs
        A_eff_cgs = np.pi * R_cgs**2 * f_A
        return slab_depth_cgs, A_eff_cgs

    # ------------------------------------------------------------------ #
    # Public compute methods                                               #
    # ------------------------------------------------------------------ #

    def compute_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame specific intensity :math:`I_\nu` for a non-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid, shape ``(*nu_shape)``. Bare values
            are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable, called as ``N(gamma)``.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from
            ``gamma_min``, ``gamma_max``, ``n_gamma``.
        f_V : float, optional
            Volume filling factor. Sets the effective slab depth
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        I_nu : ~astropy.units.Quantity
            Observer-frame specific intensity in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        slab_depth, _ = self._spherical_geometry(R, f_V, 1.0)
        return super().compute_specific_intensity(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            z=z,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral flux density :math:`F_\nu` for a non-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid, shape ``(*nu_shape)``. Bare values
            are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable, called as ``N(gamma)``.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from
            ``gamma_min``, ``gamma_max``, ``n_gamma``.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A`. Default ``1.0``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to convert distances.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        slab_depth, A_eff = self._spherical_geometry(R, f_V, f_A)
        return super().compute_flux_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            z=z,
            cosmology=cosmology,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_brightness_temperature(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame brightness temperature :math:`T_B` for a non-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        T_B : ~astropy.units.Quantity
            Observer-frame brightness temperature in Kelvin.
        """
        slab_depth, _ = self._spherical_geometry(R, f_V, 1.0)
        return super().compute_brightness_temperature(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            z=z,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_rest_frame_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame spectral luminosity density :math:`L'_{\nu'}` for a NR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A`. Default ``1.0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_nu : ~astropy.units.Quantity
            Comoving-frame spectral luminosity density in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff = self._spherical_geometry(R, f_V, f_A)
        return super().compute_rest_frame_luminosity_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral luminosity density :math:`L_\nu` for a non-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_nu : ~astropy.units.Quantity
            Observer-frame spectral luminosity density in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff = self._spherical_geometry(R, f_V, f_A)
        return super().compute_luminosity_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            z=z,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_isotropic_luminosity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the isotropic-equivalent spectral luminosity :math:`L_{\nu,\mathrm{iso}}` for a NR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A`. Default ``1.0``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to convert distances.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_iso : ~astropy.units.Quantity
            Isotropic-equivalent spectral luminosity in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff = self._spherical_geometry(R, f_V, f_A)
        return super().compute_isotropic_luminosity(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            z=z,
            cosmology=cosmology,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )


class UltraRelativisticSphericalSynchrotronEngine(NumericalSynchrotronEngine):
    r"""
    Ultra-relativistic spherical synchrotron engine (beamed-cap approximation).

    A thin geometry-specializing wrapper around :class:`NumericalSynchrotronEngine`
    for ultra-relativistic, spherically expanding outflows observed on-axis.
    Equal-arrival-time-surface (EATS) corrections are neglected; only the
    relativistically beamed cap contributes.

    The bulk Lorentz factor :math:`\Gamma` sets both the apparent emitting geometry
    and the comoving slab depth via

    .. math::

        A_\mathrm{eff} = \frac{\pi R^2 f_A}{\Gamma^2}, \qquad
        \ell' = \frac{f_V\,R}{\Gamma},

    and the bulk velocity is derived self-consistently as
    :math:`\beta = \sqrt{1 - \Gamma^{-2}}`. The line-of-sight angle is fixed at
    :math:`\theta = 0` (on-axis observer).

    See Also
    --------
    :class:`NumericalSynchrotronEngine`
        Base class providing all low-level numerical synchrotron machinery.
    """

    @staticmethod
    def _ultra_rel_geometry(
        R: Union[float, u.Quantity],
        Gamma: float,
        f_V: float,
        f_A: float,
    ):
        """Return ``(slab_depth_cgs, A_eff_cgs, beta)`` from blast-wave parameters."""
        R_cgs = ensure_in_units(R, u.cm)
        slab_depth_cgs = f_V * R_cgs / Gamma
        A_eff_cgs = np.pi * R_cgs**2 * f_A / Gamma**2
        beta = float(np.sqrt(1.0 - 1.0 / Gamma**2))
        return slab_depth_cgs, A_eff_cgs, beta

    # ------------------------------------------------------------------ #
    # Public compute methods                                               #
    # ------------------------------------------------------------------ #

    def compute_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame specific intensity :math:`I_\nu` for an UR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        I_nu : ~astropy.units.Quantity
            Observer-frame specific intensity in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        slab_depth, _, beta = self._ultra_rel_geometry(R, Gamma, f_V, 1.0)
        return super().compute_specific_intensity(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            z=z,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral flux density :math:`F_\nu` for an UR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A / \Gamma^2`. Default ``1.0``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to convert distances.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        slab_depth, A_eff, beta = self._ultra_rel_geometry(R, Gamma, f_V, f_A)
        return super().compute_flux_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            z=z,
            cosmology=cosmology,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_brightness_temperature(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame brightness temperature :math:`T_B` for an UR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        T_B : ~astropy.units.Quantity
            Observer-frame brightness temperature in Kelvin.
        """
        slab_depth, _, beta = self._ultra_rel_geometry(R, Gamma, f_V, 1.0)
        return super().compute_brightness_temperature(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            z=z,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_rest_frame_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame spectral luminosity density :math:`L'_{\nu'}` for an UR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A / \Gamma^2`. Default ``1.0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_nu : ~astropy.units.Quantity
            Comoving-frame spectral luminosity density in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff, _ = self._ultra_rel_geometry(R, Gamma, f_V, f_A)
        return super().compute_rest_frame_luminosity_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral luminosity density :math:`L_\nu` for an ultra-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A / \Gamma^2`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_nu : ~astropy.units.Quantity
            Observer-frame spectral luminosity density in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff, beta = self._ultra_rel_geometry(R, Gamma, f_V, f_A)
        return super().compute_luminosity_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            z=z,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_isotropic_luminosity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the spectral luminosity :math:`L_{\nu,\mathrm{iso}}` for an ultra-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A / \Gamma^2`. Default ``1.0``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to convert distances.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_iso : ~astropy.units.Quantity
            Isotropic-equivalent spectral luminosity in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff, beta = self._ultra_rel_geometry(R, Gamma, f_V, f_A)
        return super().compute_isotropic_luminosity(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            z=z,
            cosmology=cosmology,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )
