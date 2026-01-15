"""
Standard SEDs for synchrotron radiation.

This module provides functions to compute standard spectral energy distributions (SEDs) for synchrotron radiation
from power-law electron populations. It includes functions to calculate the characteristic SED shapes based on the
electron energy distribution index.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units as u

from triceratops.profiles import smoothed_BPL
from triceratops.radiation.constants import electron_rest_energy_cgs
from triceratops.radiation.synchrotron.utils import (
    c_1_cgs,
    compute_c5_parameter,
    compute_c6_parameter,
)
from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _ArrayLike, _UnitBearingArrayLike


# =============================================================
# SED Base Class
# =============================================================
# To help compartmentalize the name space and prevent any issues
# with clarity, we provide a small base class for SEDs to serve as a
# guide both for SED implementation and for documentation.
class SynchrotronSED(ABC):
    """
    Base class for synchrotron SED implementation.

    The :class:`SynchrotronSED` is a simple compartment for defining the
    structure of a specific spectral energy distribution. For each SED,
    one needs to provide

    1. The :meth:`sed` method (vis-a-vis the low-level ``_opt_sed`` method), which
       simply provides the phenomenological SED shape as a function of frequency and parameters. For example
       a broken power-law SED would implement the appropriate power-law segments and breaks in this method.

    2. The :meth:`from_params_to_physics` (vis-a-vis the low-level ``_opt_from_params_to_physics`` method), which
       provides the mapping from phenomenological SED parameters (e.g., break frequencies, flux normalizations, etc.)
       to physical parameters (e.g., magnetic field strength, radius, etc.) based on closure relations.
    3. The :meth:`from_physics_to_params` (vis-a-vis the low-level ``_opt_from_physics_to_params`` method), which
       provides the mapping from physical parameters (e.g., magnetic field strength, radius, etc.) to phenomenological
       SED parameters (e.g., break frequencies, flux normalizations, etc.) based on closure relations.

    Components (2) and (3) are **NOT REQUIRED** for an SED to be functional; however, they are generally necessary
    for inference. The necessary extent of the infrastructure is left to the implementer.

    .. hint::

        All of the existing SEDs are implemented using this base class as a guide. If you're
        in need of check out the other classes in this module.

    Instantiation
    --------------
    Instantiation of the SED class may be used to pre-compute class-wide constants, evaluate or construct kernels,
    etc; however, it should **NOT** be used to store SED-specific parameters. All SED-specific parameters
    should be passed directly to the relevant methods. This ensures that, during high-load inference tasks,
    no unnecessary re-instantiation of SED objects is required.
    """

    # ============================================================ #
    # Instantiation and basic structure                            #
    # ============================================================ #
    def __init__(self, *args, **kwargs):
        """
        Instantiate the SED object.

        Parameters
        ----------
        args:
            Positional arguments for SED instantiation.
        kwargs:
            Keyword arguments for SED instantiation.

        Notes
        -----
        The SED instantiation should NOT be used to store SED-specific parameters. All SED-specific parameters
        should be passed directly to the relevant methods. This ensures that, during high-load inference tasks,
        no unnecessary re-instantiation of SED objects is required.
        """
        pass

    # ============================================================ #
    # SED Function Implementation                                  #
    # ============================================================ #
    # Here should be the implementation of the SED function itself,
    # which is a function of nu and some set of additional parameters.
    @abstractmethod
    def _opt_sed(self, nu: "_ArrayLike", **parameters):
        """
        Low-level optimized SED function.

        This method should implement the core SED calculation in a performance-optimized manner.
        It assumes that all inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        Parameters
        ----------
        nu : float or array-like
            Frequency at which to evaluate the SED (in Hz-equivalent CGS).
        **parameters:
            Additional parameters required for the SED calculation.

        Returns
        -------
        float or array-like
            The computed SED value at the specified frequency.
        """
        raise NotImplementedError

    @abstractmethod
    def sed(self, nu: "_UnitBearingArrayLike", **parameters):
        """
        User-facing SED function.

        This method should provide a user-friendly interface for computing the SED.
        It should handle unit validation, coercion, and shape checking before
        dispatching to the low-level optimized backend.

        Parameters
        ----------
        nu : float, array-like, or astropy.units.Quantity
            Frequency at which to evaluate the SED. Default units are Hz, but may
            be overridden by providing ``nu`` as a :class:`astropy.units.Quantity` object.
            May be provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs).

        Parameters
        ----------
            Additional parameters required for the SED calculation.

        Returns
        -------
        float, array-like, or astropy.units.Quantity
            The computed SED value at the specified frequency, with appropriate units.
        """
        raise NotImplementedError

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def from_params_to_physics(self, **parameters):
        """
        Convert phenomenological SED parameters into physical parameters.

        This method provides the user-facing interface for mapping phenomenological
        SED parameters—such as break frequencies, peak fluxes, or normalization
        constants—into underlying physical quantities like magnetic field strength,
        emitting radius, particle energy scales, or energy densities.

        Implementations of this method may apply analytic closure relations derived
        from synchrotron theory or empirical relations commonly used in the
        literature.

        This method is optional: an SED implementation is fully functional without
        it, but closure relations are generally required for inference workflows and
        for coupling SEDs to dynamical or microphysical models.

        Parameters
        ----------
        parameters:
            Keyword arguments specifying phenomenological SED parameters. The exact
            set of required parameters is model-dependent and determined by the
            implementing subclass.

        Returns
        -------
        dict
            A dictionary containing the inferred physical parameters. The contents
            and naming of this dictionary are implementation-specific.

        Notes
        -----
        This method may perform unit validation, coercion, or shape checking before
        dispatching to the low-level optimized implementation
        :meth:`_opt_from_params_to_physics`.

        Subclasses that do not support closure relations may leave this method
        unimplemented.
        """
        raise NotImplementedError

    def _opt_from_params_to_physics(self, **parameters):
        """
        Low-level optimized conversion from SED parameters to physical parameters.

        This method implements the same mapping as
        :meth:`from_params_to_physics`, but assumes that all inputs are provided as
        dimensionless scalars or NumPy arrays in consistent CGS units.

        No unit validation, type checking, or safety checks are performed. This
        method is intended for internal use in performance-critical contexts.

        Parameters
        ----------
        parameters:
            Keyword arguments specifying phenomenological SED parameters in CGS or
            dimensionless form. The exact set of required parameters is
            implementation-specific.

        Returns
        -------
        dict
            A dictionary containing physical parameters in CGS units.

        Notes
        -----
        This method is intended to be called by
        :meth:`from_params_to_physics` after any necessary unit handling has been
        performed.
        """
        raise NotImplementedError

    def _opt_from_physics_to_params(self, **parameters):
        """
        Low-level optimized conversion from physical parameters to SED parameters.

        This method implements the inverse of
        :meth:`_opt_from_params_to_physics`, mapping physical quantities—such as
        magnetic field strength, energy densities, or system size—into
        phenomenological SED parameters like break frequencies or peak fluxes.

        All inputs are assumed to be provided in CGS units or as dimensionless
        scalars. No unit validation or safety checks are performed.

        Parameters
        ----------
        parameters:
            Keyword arguments specifying physical parameters in CGS units. The exact
            set of required parameters is implementation-specific.

        Returns
        -------
        dict
            A dictionary containing phenomenological SED parameters in CGS or
            dimensionless form.

        Notes
        -----
        This method is intended for internal use and should generally be wrapped by
        :meth:`from_physics_to_params` to provide a user-facing interface.
        """
        raise NotImplementedError

    def from_physics_to_params(self, **parameters):
        """
        Convert physical parameters into phenomenological SED parameters.

        This method provides the user-facing interface for mapping physical
        quantities—such as magnetic field strength, emitting radius, or energy
        densities—into phenomenological SED parameters like break frequencies,
        normalization constants, or spectral amplitudes.

        This functionality is primarily used in inference workflows, where physical
        model parameters are sampled and must be converted into observable SED
        quantities.

        Parameters
        ----------
        parameters:
            Keyword arguments specifying physical parameters. The exact set of
            required parameters is model-dependent and determined by the
            implementing subclass.

        Returns
        -------
        dict
            A dictionary containing phenomenological SED parameters.

        Notes
        -----
        This method may perform unit validation, coercion, or shape checking before
        dispatching to the low-level optimized implementation
        :meth:`_opt_from_physics_to_params`.

        Subclasses that do not support inversion of closure relations may leave this
        method unimplemented.
        """
        raise NotImplementedError


# ============================================================ #
# SED Implementations                                          #
# ============================================================ #
# Now we can include concrete implementations of various SEDs. Not
# all of the SEDs we plan to implement are currently implemented in the
# codebase, but we provide a few examples here to illustrate the structure.
class SSA_SED_PowerLaw(SynchrotronSED):
    r"""
    Synchrotron self-absorbed (SSA) broken power-law spectral energy distribution.

    This class implements a phenomenological synchrotron spectral energy
    distribution characterized by a single smooth spectral break arising from
    synchrotron self-absorption (SSA). Below the break frequency, the spectrum is
    optically thick and follows a power-law slope of :math:`+5/2`; above the
    break, the spectrum is optically thin and follows a power-law slope of
    :math:`-(p-1)/2`, where :math:`p` is the power-law index of the electron
    energy distribution.

    The SED is implemented as a **smoothly broken power law**, ensuring numerical
    stability and differentiability across the SSA turnover. This makes the
    class well-suited for use in parameter inference, optimization, and
    likelihood-based modeling.

    In addition to forward evaluation of the SED, this class implements
    **analytic closure relations** that allow inversion between phenomenological
    SED parameters (the SSA break frequency and peak flux) and underlying
    physical properties of the emitting region (magnetic field strength and
    radius). These closure relations follow the formalism presented in
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    This class is intended for modeling radio synchrotron emission from
    approximately homogeneous emitting regions, such as those encountered in:

    - supernova radio afterglows,
    - compact object-powered transients,
    - synchrotron-emitting shells with a single dominant SSA turnover.

    No assumptions are made about time evolution, shock dynamics, or cooling
    regimes beyond those implicit in the broken power-law description.

    Notes
    -----
    - The break frequency :math:`\nu_{\rm brk}` is defined as the frequency at
      which the optically thick and optically thin asymptotic power laws
      intersect.
    - The closure relations implemented here assume a power-law electron energy
      distribution with finite bounds
      :math:`\gamma_{\rm min} \le \gamma \le \gamma_{\rm max}`.
    - The special case :math:`p = 2` is not supported due to logarithmic
      divergences in the electron energy integral and must be handled
      separately.

    See Also
    --------
    SynchrotronSED
        Abstract base class defining the interface for all synchrotron SED
        implementations.

    References
    ----------
    .. footbibliography::
    """

    # ================================================= #
    # Instantiation                                     #
    # ================================================= #
    def __init__(self):
        """Instantiate the SSA SED object."""
        # There are no class-wide constants to pre-compute for this SED.
        super().__init__()

    # ================================================ #
    # SED Function Implementation                      #
    # ================================================ #
    def _opt_sed(
        self,
        nu: "_ArrayLike",
        nu_brk: float,
        F_nu_brk: float,
        p: float,
        s: float,
    ) -> "_ArrayLike":
        r"""
        Low-level optimized SSA broken power-law SED.

        This method implements the synchrotron self-absorbed (SSA) broken power-law SED
        in a performance-optimized manner. It assumes that all inputs are provided as
        dimensionless scalars or NumPy arrays in CGS units. No unit validation or safety
        checks are performed.

        Parameters
        ----------
        nu: float or array-like
            Frequency at which to evaluate the SED (in Hz-equivalent CGS).
        nu_brk: float
            Break frequency (Hz-equivalent CGS).
        F_nu_brk: float
            Flux density at the break frequency (CGS units).
        p: float
            Power-law index of the electron energy distribution.
        s: float
            Smoothing parameter for the break.

        Returns
        -------
        float or array-like
            The computed SED value at the specified frequency. The output of this SED
            is in CGS units of erg s^-1 cm^-2 Hz^-1.
        """
        return smoothed_BPL(
            nu,
            F_nu_brk,
            nu_brk,
            -(p - 1) / 2,  # optically thin index
            (5 / 2),  # optically thick index
            s,  # smoothing parameter
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_brk: "_UnitBearingArrayLike",
        F_nu_brk: "_UnitBearingArrayLike",
        p: float,
        s: float,
    ):
        r"""
        Compute the synchrotron self-absorbed (SSA) broken power-law SED.

        This is the user-facing interface for the SSA broken power-law spectral
        energy distribution. It validates units, coerces inputs to CGS, and
        dispatches to the optimized low-level backend implementation
        :meth:`_opt_sed`.

        Parameters
        ----------
        nu : float, array-like, or astropy.units.Quantity
            Frequency at which to evaluate the SED. Default units are Hz.
        nu_brk : float or astropy.units.Quantity
            Break frequency of the SED. Default units are Hz.
        F_nu_brk : float or astropy.units.Quantity
            Flux density at the break frequency. Default units are
            erg s^-1 cm^-2 Hz^-1.
        p : float
            Power-law index of the electron energy distribution.
        s : float
            Smoothing parameter controlling the sharpness of the spectral break.

        Returns
        -------
        astropy.units.Quantity
            Flux density evaluated at ``nu`` with units of
            erg s^-1 cm^-2 Hz^-1.
        """
        # --- Unit validation and coercion --- #
        nu = ensure_in_units(nu, u.Hz)
        nu_brk = ensure_in_units(nu_brk, u.Hz)
        F_nu_brk = ensure_in_units(F_nu_brk, u.erg / u.s / u.cm**2 / u.Hz)

        # --- Call optimized backend --- #
        F_nu_cgs = self._opt_sed(
            nu=nu,
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            p=p,
            s=s,
        )

        return F_nu_cgs * u.erg / u.s / u.cm**2 / u.Hz

    # ================================================ #
    # Closure Relations.                               #
    # ================================================ #
    # For this SED, we implement the closure relations to go from
    # the phenomenological parameters (nu_brk, F_nu_brk) to the physical
    # parameters (B, R) and vice versa. This is implemented exactly as
    # done in DeMarchi+22.
    def _opt_from_params_to_physics(
        self,
        nu_brk: Union[float, np.ndarray],
        F_nu_brk: Union[float, np.ndarray],
        distance: Union[float, np.ndarray],
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        r"""
        Compute the magnetic field and radius of the emitting region from a broken power-law synchrotron SED.

        Parameters
        ----------
        nu_brk: float or array-like
            The break frequency :math:`\nu_{\rm brk}` where the synchrotron SED transitions from
            optically thick to optically thin. This should be provided in GHz. May be provided either
            as a scalar float (for a single SED) or as a 1-D array (for multiple SEDs).
        F_nu_brk: float or array-like
            The flux density :math:`F_{\nu_{\rm brk}}` at the break frequency. This should be provided
            in Jansky (Jy). May be provided either as a scalar float (for a single SED) or as a 1-D array (for multiple
            SEDs).
        distance: float or array-like
            The distance to the source in Megaparsecs (Mpc). May be provided either as a scalar float (for a single SED)
            or as a 1-D array (for multiple SEDs).
        p: float or array-like
            The power-law index :math:`p` of the electron energy distribution. By default, this is 3.0. If provided as a
            float, the value is used for all SEDs. If provided as an array, its shape must be compatible with that of
            ``nu_brk``.

            .. warning::

                This function does not handle the case where :math:`p = 2` due to singularities in the underlying
                equations.
                Users must ensure that :math:`p` is not equal to 2 when calling this function.

        f: float or array-like
            The filling factor :math:`f` of the emitting region. Default is ``0.5``. If provided as a float, the value
            is
            used for all SEDs. If provided as an array, its shape must be compatible with that of ``nu_brk``.
        theta: float or array-like
            The pitch angle :math:`\theta` between the magnetic field and the line of sight, in radians.
            Default is ``pi/2`` (i.e., perpendicular). If provided as a float, the value is used for all SEDs.
            If provided as an array, its shape must be compatible with that of ``nu_brk``.
        epsilon_B: float or array-like
            The fraction of post-shock energy in magnetic fields, :math:`\epsilon_B`.
        epsilon_E: float or array-like
            The fraction of post-shock energy in relativistic electrons, :math:`\epsilon_E`.
        gamma_min: float or array-like
            The minimum Lorentz factor :math:`\gamma_{\rm min}` of the electron energy.
        gamma_max: float or array-like
            The maximum Lorentz factor :math:`\gamma_{\rm max}` of the electron energy.

        Returns
        -------
        B: float or array-like
            The computed magnetic field strength :math:`B` in Gauss. The shape matches that of the input parameters.
        R: float or array-like
            The computed radius of the emitting region :math:`R` in cm. The shape matches that of the input parameters.

        Notes
        -----
        See notes in the user-facing wrapper function `compute_BR_from_BPL` for details on the
        underlying physics and assumptions.
        """
        # Validate input parameters. We do NOT check explicitly for shape correctness of the arrays, instead opting
        # to allow an error to rise naturally if the shapes are incompatible during computation. We do want to ensure
        # that all of the constants are cast properly for array operations and masking.
        p, gamma_min, gamma_max = (
            np.asarray(p, dtype="f8"),
            np.asarray(gamma_min, dtype="f8"),
            np.asarray(gamma_max, dtype="f8"),
        )

        # Obtain the synchrotron coefficients relevant for this scenario. We need c_1,c_5, and c_6.
        c_5 = compute_c5_parameter(p)  # Should match shape of p.
        c_6 = compute_c6_parameter(p)  # Should match shape of p.
        c_1 = c_1_cgs  # Scalar constant.

        # Construct the array for delta. See the documentation notes for details on the procedure here / physical
        # motivation. To do this efficiently, we pre-allocate the array and then fill in values based on the conditions.
        # Because we pre-allocate with ones, we only need to fill in values where p < 2. In THIS IMPLEMENTATION, we
        # ignore
        # cases where p == 2 for efficiency; these should be screened for upstream if needed.
        delta = np.ones_like(p)
        _p_lt_2_mask = p < 2
        delta[_p_lt_2_mask] = (gamma_max[_p_lt_2_mask] / gamma_min[_p_lt_2_mask]) ** (
            2 - p[_p_lt_2_mask]
        ) - 1  # Fill in values where p < 2.

        # Construct the "p_norm" term used in the B and R calculations. This allows us to handle the two branches
        # of the solution (p < 2 and p > 2) in a unified way.
        # Note that we take the absolute value here to avoid issues with negative bases and fractional exponents.
        # The p == 2 case is handled upstream.
        p_norm = np.abs(p - 2.0)

        # Compute the electron energy floor using the gamma_min parameter and the standard formula.
        E_l = electron_rest_energy_cgs * gamma_min

        # Compute the magnetic field following equation (16) of DeMarchi+22. We break this into
        # components for clarity. The operation should be heavily CPU bound, so this should not have any impact
        # on optimization.
        #
        # Here nu_brk is in GHz, E_l is in erg, distance is in Mpc, F_nu_brk is in Jy, and B will be in Gauss.
        _B_coeff = 2.50e9 * (nu_brk / 5) * (1 / c_1)
        _B_num = (
            4.69e-23
            * E_l ** (4 - 2 * p)
            * delta**2
            * (epsilon_B / epsilon_E) ** 2
            * c_5
            * np.sin(theta) ** (1 / 2 * (-5 - 2 * p))
        )
        _B_denom = p_norm**2 * distance**2 * (f / 0.5) ** 2 * F_nu_brk * c_6**3

        B = _B_coeff * (_B_num / _B_denom) ** (2 / (13 + 2 * p))

        # Compute the radius following equation (17) of DeMarchi+22. We break this into parts as well on the same
        # basis as above.
        _R_coeff = (2.50e9**-1) * c_1 * (nu_brk / 5) ** -1
        _R_t1 = (12 * epsilon_B) * (c_5 ** (-6 - p)) * (c_6 ** (5 + p))
        _R_t2 = (9.52e25) ** (6 + p) * np.sin(theta) ** 2 * np.pi ** (-5 - p) * distance ** (12 + 2 * p)
        _R_t3 = E_l ** (2 - p) * F_nu_brk ** (6 + p)
        _R_t4 = (epsilon_E * (p_norm) * (f / 0.5)) ** -1

        R = _R_coeff * (_R_t1 * _R_t2 * _R_t3 * _R_t4) ** (1 / (13 + 2 * p))

        return B, R

    def from_params_to_physics(
        self,
        nu_brk: Union[float, np.ndarray, u.Quantity],
        F_nu_brk: Union[float, np.ndarray, u.Quantity],
        distance: Union[float, np.ndarray, u.Quantity],
        *,
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the magnetic field strength and emitting radius from a broken power-law SED.

        This function provides a **user-facing, unit-aware interface** for computing
        the physical properties of a synchrotron-emitting region whose radio spectrum
        exhibits a turnover due to synchrotron self-absorption (SSA). Internally, it
        wraps a low-level optimized routine that implements the analytic inversion
        described in :footcite:t:`demarchiRadioAnalysisSN2004C2022` (DM22).

        The function accepts scalar values, NumPy arrays, or Astropy ``Quantity`` objects
        and performs the necessary unit coercion, validation, and shape checking before
        dispatching to the optimized backend.

        Parameters
        ----------
        nu_brk : float, array-like, or astropy.units.Quantity
            The SSA break (turnover) frequency :math:`\nu_{\rm brk}` separating the
            optically thick and optically thin synchrotron regimes. Default units are GHz, but may
            be overridden by providing ``nu_brk`` as a :class:`astropy.units.Quantity` object. May be
            provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs).

        F_nu_brk : float, array-like, or astropy.units.Quantity
            The flux density :math:`F_{\nu_{\rm brk}}` at the break frequency. Default units
            are Jansky (Jy), but may be overridden by providing ``F_nu_brk`` as a
            :class:`astropy.units.Quantity` object. May be provided as either a scalar
            (for a single SED) or a 1-D array (for multiple SEDs). If provided as an array, shape
            must be compatible with that of ``nu_brk``.

        distance : float, array-like, or astropy.units.Quantity
            The (luminosity) distance to the source. By default, units are Megaparsecs (Mpc),
            but may be overridden by providing ``distance`` as a :class:`astropy.units.Quantity` object.
            May be provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs). If provided
            as an array, shape must be compatible with that of ``nu_brk``.

        p : float or array-like, optional
            The power-law index :math:`p` of the electron Lorentz factor distribution,
            defined by

            .. math::

                N(\Gamma)\, d\Gamma = K_e \Gamma^{-p}\, d\Gamma.

            Default is ``3.0``.

            .. warning::

                This function **does not support** the case :math:`p = 2`, for which
                the electron energy integral is logarithmically divergent and requires
                a separate analytic treatment. If any value of ``p`` is exactly equal
                to 2, a ``ValueError`` is raised.

        f : float or array-like, optional
            Volume filling factor of the synchrotron-emitting region. Default is ``0.5``.

        theta : float or array-like, optional
            Pitch angle :math:`\theta` between the magnetic field and the electron
            velocity, in radians. Default is ``\pi/2`` (isotropic average).

        epsilon_B : float or array-like, optional
            Fraction of post-shock internal energy in magnetic fields,
            :math:`\epsilon_B`. Default is ``0.1``.

        epsilon_E : float or array-like, optional
            Fraction of post-shock internal energy in relativistic electrons,
            :math:`\epsilon_E`. Default is ``0.1``.

        gamma_min : float or array-like, optional
            Minimum electron Lorentz factor :math:`\gamma_{\rm min}`. Default is ``1``.

        gamma_max : float or array-like, optional
            Maximum electron Lorentz factor :math:`\gamma_{\rm max}`. Default is ``1e6``.

            This parameter only affects the calculation when :math:`p < 2`.

        Returns
        -------
        B : astropy.units.Quantity
            The inferred magnetic field strength :math:`B` in **Gauss**.

        R : astropy.units.Quantity
            The inferred radius of the synchrotron-emitting region :math:`R`
            in **cm**.

        Notes
        -----
        This function follows the formalism laid out in :footcite:t:`demarchiRadioAnalysisSN2004C2022` (DM22) to compute
        the magnetic field strength and radius of the synchrotron-emitting region from the observed break frequency and
        flux density of a broken power-law SED. The calculations assume a power-law distribution of electron energies
        characterized by the index :math:`p`.

        Letting :math:`\nu_{\rm brk}` be the break frequency (in GHz) between the SSA-thick and SSA-thin regimes, and
        :math:`F_{\nu_{\rm brk}}` be the flux density (in Jy) at that frequency, the magnetic field strength :math:`B`
        (in Gauss) and radius :math:`R` (in cm) of the emitting region can be computed
        by requiring that the asymptotic behavior of
        the optically thick and thin synchrotron spectra match at :math:`\nu_{\rm brk}`. The equations used are
        equations
        (16) and (17) from DM22 with minor alterations.

        In treating the electron energy distribution, some additional care is taken based on the value of :math:`p`. In
        particular, we assume a power-law distribution of electron Lorentz factors :math:`\Gamma` such that

        .. math::

            N(\Gamma) d\Gamma = K_e \Gamma^{-p} d\Gamma,\;\; \Gamma_{\rm min} \leq \Gamma \leq \Gamma_{\rm max},

        where :math:`K_e` is the normalization constant, :math:`\Gamma_{\rm min}` is the minimum Lorentz factor,
        and :math:`\Gamma_{\rm max}` is the maximum Lorentz factor. For values of :math:`p > 2`, the total energy is
        dominated by electrons near :math:`\Gamma_{\rm min}`, while for :math:`p < 2`, it is dominated by those near
        :math:`\Gamma_{\rm max}`.

        To account for this, when :math:`p > 2`, we enforce :math:`\Gamma_{\rm max} = \infty` in the energy integral,
        while
        for :math:`p < 2`, we enforce the upper limit on the energy integral to be :math:`\Gamma_{\rm max}`. This leads
        to
        a correction factor :math:`\delta` defined as:

        .. math::

            \delta = \begin{cases} 1, & p > 2 \[6pt]
            \left(\frac{\Gamma_{\rm max}}{\Gamma_{\rm min}}\right)^{2 - p} - 1, & p < 2 \end{cases}

        which modifies the expressions for :math:`B` and :math:`R` accordingly.

        References
        ----------

        .. footbibliography::

        """
        # Validate units of all unit bearing quantities and coerce them to the expected
        # units for the optimized backend.
        if hasattr(nu_brk, "units"):
            nu_brk = nu_brk.to_value(u.GHz)
        if hasattr(F_nu_brk, "units"):
            F_nu_brk = F_nu_brk.to_value(u.Jy)
        if hasattr(distance, "units"):
            distance = distance.to_value(u.Mpc)

        # Check the validity of p values. We need to ensure that ``p`` behaves as an array at
        # this point, so we cast it explicitly.
        p = np.asarray(p)
        if np.any(p == 2):
            raise ValueError(
                "compute_BR_from_BPL does not support p = 2. "
                "Use p slightly above or below 2, or implement a dedicated "
                "logarithmic normalization for this case."
            )

        # Dispatch to the optimized backend.
        B, R = self._opt_from_params_to_physics(
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        return B * u.Gauss, R * u.cm

    def _opt_from_physics_to_params(
        self,
        B: Union[float, np.ndarray],
        R: Union[float, np.ndarray],
        distance: Union[float, np.ndarray],
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ):
        r"""
        Compute the synchrotron self-absorption frequency and peak flux.

        This is a **low-level, performance-optimized routine** that assumes all
        inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        The function analytically eliminates the normalization of the electron
        energy distribution using microphysical energy-partition assumptions,
        introducing a correction factor ``delta`` to account for the convergence
        of the electron energy integral when :math:`p < 2`.

        Parameters
        ----------
        B : float or array-like
            Magnetic field strength in Gauss.

        R : float or array-like
            Radius of the emitting region in cm.

        distance : float or array-like
            Distance to the source in cm.

        p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
            See the user-facing function ``compute_BPL_SED_from_BR``.

        Returns
        -------
        nu_brk : float or array-like
            Synchrotron self-absorption break frequency (Hz-equivalent CGS).

        F_nu_brk : float or array-like
            Peak flux density at the break frequency (CGS-equivalent).

        Notes
        -----
        See notes in the user-facing wrapper function `compute_BR_from_BPL` for details on the
        underlying physics and assumptions.
        """
        # Validate input parameters. We do NOT check explicitly for shape correctness of the arrays, instead opting
        # to allow an error to rise naturally if the shapes are incompatible during computation. We do want to ensure
        # that all of the constants are cast properly for array operations and masking.
        p, gamma_min, gamma_max = (
            np.asarray(p, dtype="f8"),
            np.asarray(gamma_min, dtype="f8"),
            np.asarray(gamma_max, dtype="f8"),
        )

        # Obtain the synchrotron coefficients relevant for this scenario. We need c_1,c_5, and c_6.
        c_5 = compute_c5_parameter(p)  # Should match shape of p.
        c_6 = compute_c6_parameter(p)  # Should match shape of p.
        c_1 = c_1_cgs  # Scalar constant.

        # Construct the array for delta. See the documentation notes for details on the procedure here / physical
        # motivation. To do this efficiently, we pre-allocate the array and then fill in values based on the conditions.
        # Because we pre-allocate with ones, we only need to fill in values where p < 2. In THIS IMPLEMENTATION, we
        # ignore
        # cases where p == 2 for efficiency; these should be screened for upstream if needed.
        delta = np.ones_like(p)
        _p_lt_2_mask = p < 2
        delta[_p_lt_2_mask] = (gamma_max[_p_lt_2_mask] / gamma_min[_p_lt_2_mask]) ** (
            2 - p[_p_lt_2_mask]
        ) - 1  # Fill in values where p < 2.

        # Construct the "p_norm" term used in the B and R calculations. This allows us to handle the two branches
        # of the solution (p < 2 and p > 2) in a unified way.
        # Note that we take the absolute value here to avoid issues with negative bases and fractional exponents.
        # The p == 2 case is handled upstream.
        p_norm = np.abs(p - 2.0)

        # Compute the electron energy floor using the gamma_min parameter and the standard formula.
        E_l = electron_rest_energy_cgs * gamma_min

        # Calculate nu_brk. We break this into parts for clarity. See the notes on this function for an explanation of
        # where this formula comes from.
        _nu_brk_coeff = 2 * c_1
        _nu_brk_t1 = R * f * (epsilon_E / (delta * epsilon_B)) * p_norm / (6 * np.pi)
        _nu_brk_t2 = c_6 ** (2 / (p + 4))
        _nu_brk_t3 = np.sin(theta) ** ((p + 2) / (p + 4))
        _nu_brk_E_l_exp = 2 * (p - 2) / (p + 4)
        _nu_brk_B_exp = (p + 6) / (p + 4)

        nu_brk = (
            _nu_brk_coeff
            * (_nu_brk_t1 ** (2 / (p + 4)))
            * (E_l**_nu_brk_E_l_exp)
            * (B**_nu_brk_B_exp)
            * _nu_brk_t2
            * _nu_brk_t3
        )

        # Calculate F_nu_brk by inserting nu_brk into either the optically thick or thin formula.
        # We choose the optically thick formula here (equation 14 of DeMarchi+22) for consistency.
        _F_nu_brk_coeff = (c_5 / c_6) * np.pi * (R / distance) ** 2
        _F_nu_brk_t1 = (B * np.sin(theta)) ** (-1 / 2)
        _F_nu_brk_t2 = (nu_brk / (2 * c_1)) ** (5 / 2)

        F_nu_brk = _F_nu_brk_coeff * _F_nu_brk_t1 * _F_nu_brk_t2

        return nu_brk, F_nu_brk

    def from_physics_to_params(
        self,
        B: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        distance: Union[float, np.ndarray, u.Quantity],
        *,
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ):
        r"""
        Compute the synchrotron self-absorption break frequency and peak flux.

        This is a **low-level, performance-optimized routine** that assumes all
        inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        The function analytically eliminates the normalization of the electron
        energy distribution using microphysical energy-partition assumptions,
        introducing a correction factor ``delta`` to account for the convergence
        of the electron energy integral when :math:`p < 2`.

        Parameters
        ----------
        B : float or array-like
            Magnetic field strength in Gauss.

        R : float or array-like
            Radius of the emitting region in cm.

        distance : float or array-like
            Distance to the source in Mpc.

        p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
            See the user-facing function ``compute_BPL_SED_from_BR``.

        Returns
        -------
        nu_brk : float or array-like
            Synchrotron self-absorption break frequency (Hz-equivalent CGS).

        F_nu_brk : float or array-like
            Peak flux density at the break frequency (CGS-equivalent).

        Notes
        -----
        In the optically thick regime, the synchrotron SED follows a power law

        .. math::

            F_\nu = \frac{c_5}{c_6} \left(B\sin\theta\right)^{-1/2} \left(\frac{\nu}{2c_1}\right)^{5/2}
            \frac{\pi R^2}{d^2}.

        In the optically thin regime, the SED follows a different power law:

        .. math::

            F_\nu = \frac{4\pi f R^3}{3 d^2} c_5 N_0 \left(B\sin \theta\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

        We define the break frequency :math:`\nu_{\rm brk}` as the frequency where these two power laws intersect and
        the
        corresponding peak flux density :math:`F_{\nu_{\rm brk}}`. By equating the two expressions for :math:`F_\nu` at
        :math:`\nu = \nu_{\rm brk}`, we can solve for :math:`\nu_{\rm brk}` and :math:`F_{\nu_{\rm brk}}` in terms of
        the physical parameters :math:`B`, :math:`R`, and :math:`d`.

        Equation these two equations yields

        .. math::

            \nu_{\rm brk} = 2 c_1 \left[\frac{4}{3} c_6 f N_0\right]^{2/(p+4)} R^{2/(p+4)}
            \left(B\sin\theta\right)^{(p+2)/(p+4)}

        Inserting :math:`\nu_{\rm brk}` back into either expression for :math:`F_\nu` gives the peak flux density
        :math:`F_{\nu_{\rm brk}}`.

        The normalization :math:`N_0` of the electron distribution is eliminated
        analytically by equating the total electron energy density to a fraction
        :math:`\epsilon_E` of the post-shock internal energy density, with magnetic
        energy fraction :math:`\epsilon_B`.

        For :math:`p \neq 2`, this introduces a correction factor

        .. math::

            \delta =
            \begin{cases}
            1, & p > 2 \
            (\gamma_{\max}/\gamma_{\min})^{2-p} - 1, & p < 2
            \end{cases}

        which accounts for the convergence properties of the electron energy integral.
        """
        # Validate units of all unit bearing quantities and coerce them to the expected
        # units for the optimized backend.
        if hasattr(B, "units"):
            B = B.to_value(u.Gauss)
        if hasattr(R, "units"):
            R = R.to_value(u.cm)
        if hasattr(distance, "units"):
            distance = distance.to_value(u.cm)

        # Check the validity of p values. We need to ensure that ``p`` behaves as an array at
        # this point, so we cast it explicitly.
        p = np.asarray(p)
        if np.any(p == 2):
            raise ValueError(
                "compute_BR_from_BPL does not support p = 2. "
                "Use p slightly above or below 2, or implement a dedicated "
                "logarithmic normalization for this case."
            )

        # Dispatch to the optimized backend.
        nu, F_nu = self._opt_from_physics_to_params(
            B=B,
            R=R,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        return nu * u.Hz, F_nu * u.erg / (u.s * u.cm**2 * u.Hz)
