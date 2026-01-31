"""
Synchrotron shock models for transient radio sources.

This module contains phenomenological models that describe synchrotron emission from
transient astrophysical sources such as supernovae (SNe) and gamma-ray bursts (GRBs). These models
are designed to be flexible and adaptable, allowing them to be applied in different contexts and scenarios
within the Triceratops ecosystem.
"""

import numpy as np
from astropy import units as u

from triceratops.models.core.base import Model
from triceratops.models.core.parameters import ModelParameter, ModelVariable
from triceratops.radiation.synchrotron.SEDs import SSA_SED_PowerLaw


class SynchrotronShockModel(Model):
    r"""
    Phenomenological synchrotron shock model for radio emission from transient sources.

    This model provides a **time-independent**, physically motivated description of
    synchrotron emission from a shocked region, such as those produced in astrophysical
    transients including supernovae (SNe) and gamma-ray bursts (GRBs). The emitting region
    is characterized by a shock radius :math:`R` and magnetic field strength :math:`B`,
    along with standard microphysical parameters describing particle acceleration and
    magnetic field amplification.

    The model computes the synchrotron self-absorbed (SSA) spectral energy distribution
    (SED) by first determining the effective break frequency and flux normalization
    implied by the physical shock parameters, and then evaluating a smoothly broken
    power-law approximation to the resulting spectrum.

    Formally, the emergent spectrum is written as

    .. math::

        F_{\nu} = F_{\nu,\rm break}
        \left[
            \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_{\rm thin}/s}
            +
            \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_{\rm thick}/s}
        \right]^s,

    where the spectral indices are tied to the electron energy distribution power-law
    index :math:`p` via

    .. math::

        \alpha_{\rm thick} = \frac{5}{2},
        \qquad
        \alpha_{\rm thin} = -\frac{p - 1}{2}.

    The break frequency :math:`\nu_{\rm break}` and corresponding flux
    :math:`F_{\nu,\rm break}` are computed self-consistently from the shock radius,
    magnetic field strength, distance to the source, and microphysical parameters
    following standard synchrotron theory.

    This model does **not** include explicit time evolution of the shock properties.
    Instead, it represents a snapshot of the synchrotron emission at a given epoch.
    Time-dependent behavior can be introduced by embedding this model within a higher-level
    dynamical framework that prescribes the evolution of :math:`R(t)` and :math:`B(t)`.

    This model is therefore most appropriate for:

    - single-epoch radio SED fitting with physical parameter inference,
    - exploratory studies connecting phenomenological SED fits to physical shock properties,
    - use as a building block for time-dependent synchrotron shock models.

    For detailed treatments of synchrotron radiation and self-absorption in shocked
    astrophysical plasmas, see :footcite:t:`rybickilightman`,
    :footcite:t:`1970ranp.book.....P`, and :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``epsilon_e``
             - :math:`\epsilon_e`
             - Fraction of post-shock energy density in relativistic electrons.
           * - ``epsilon_B``
             - :math:`\epsilon_B`
             - Fraction of post-shock energy density in magnetic fields.
           * - ``p``
             - :math:`p`
             - Power-law index of the relativistic electron energy distribution.
           * - ``gamma_min``
             - :math:`\gamma_{\rm min}`
             - Minimum Lorentz factor of the accelerated electron population.
           * - ``gamma_max``
             - :math:`\gamma_{\rm max}`
             - Maximum Lorentz factor of the accelerated electron population.
           * - ``f``
             - :math:`f`
             - Filling factor of the emitting shocked region.
           * - ``theta``
             - :math:`\theta`
             - Characteristic pitch angle of the electron population.
           * - ``D``
             - :math:`D`
             - Luminosity distance to the source.
           * - ``B``
             - :math:`B`
             - Magnetic field strength in the shocked region.
           * - ``R``
             - :math:`R`
             - Radius of the shock front.
           * - ``s``
             - :math:`s`
             - Smoothness parameter controlling the sharpness of the SSA spectral break.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``frequency``
             - :math:`\nu`
             - Observing frequency at which the SED is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux_density``
             - :math:`F_{\nu}`
             - Flux density evaluated at the observing frequency.

    Notes
    -----
    - The spectral break frequency is interpreted as the effective synchrotron
      self-absorption turnover frequency.
    - The optically thick spectral slope of :math:`5/2` corresponds to a homogeneous,
      spherical emitting region.
    - This model assumes isotropic pitch-angle distributions and a homogeneous shock.
    - Without broadband frequency coverage, degeneracies between :math:`R`, :math:`B`,
      and microphysical parameters may remain significant.

    References
    ----------
    .. footbibliography::
    """

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    PARAMETERS = (
        ModelParameter(
            "epsilon_e",
            0.1,
            description="Fraction of shock energy in relativistic electrons.",
            bounds=(0, None),
            latex=r"\epsilon_e",
            base_units="",
        ),
        ModelParameter(
            "epsilon_B",
            0.1,
            description="Fraction of shock energy in magnetic fields.",
            bounds=(0, None),
            latex=r"\epsilon_B",
            base_units="",
        ),
        ModelParameter(
            "p",
            3.0,
            description="Power-law index of the electron energy distribution.",
            bounds=(2, None),
            latex=r"p",
            base_units="",
        ),
        ModelParameter(
            "gamma_max",
            1e7,
            description="The maximum Lorentz factor of the electron population. Only used if p < 2.",
            bounds=(1, None),
            latex=r"\gamma_{\rm max}",
            base_units="",
        ),
        ModelParameter(
            "gamma_min",
            1,
            description="The minimum Lorentz factor of the electron population.",
            bounds=(0, None),
            latex=r"\gamma_{\rm max}",
            base_units="",
        ),
        ModelParameter(
            "f",
            0.5,
            description="Filling factor of the emitting region.",
            bounds=(0, 1),
            latex=r"f",
            base_units="",
        ),
        ModelParameter(
            "theta",
            np.pi / 2,
            description="The pitch angle of the electron population.",
            bounds=(-np.pi, np.pi),
            latex=r"\theta",
            base_units="",
        ),
        ModelParameter("D", 1, description="Distance to the source.", base_units="Mpc", latex=r"D", bounds=(0, None)),
        ModelParameter(
            "s",
            default=-0.5,
            description="The smoothing parameter of the broken power-law.",
            bounds=(None, 0),
            latex=r"s",
            base_units="",
        ),
        ModelParameter(
            "B",
            0.1,
            description="Magnetic field strength in the shocked region.",
            bounds=(0, None),
            latex=r"B",
            base_units="G",
        ),
        ModelParameter(
            "R",
            1e16,
            description="Radius of the shock front.",
            bounds=(0, None),
            latex=r"R",
            base_units="cm",
        ),
    )
    VARIABLES = (
        ModelVariable(
            "frequency", description="Observing frequency at which to evaluate the SED.", base_units="Hz", latex=r"\nu"
        ),
    )
    # =============================================== #
    # Model Metadata Declarations                     #
    # =============================================== #
    # Each model must declare its parameters and variables as class-level attributes.
    OUTPUTS: tuple[str, ...] = ("flux_density",)
    """tuple of str: The names of the model's outputs.

    Each element of :attr:`OUTPUTS` is a string that defines the name of a single output of the model.
    These names correspond to the keys in the dictionary returned by the model's evaluation method.
    """
    UNITS: tuple[u.Unit, ...] = (u.Jy,)
    """tuple of :class:`astropy.units.Unit`: The units of the model's outputs.

    Each element of :attr:`UNITS` is an :class:`astropy.units.Unit` instance that defines the units of a single
    output of the model. The order of the units in :attr:`UNITS` corresponds to the order of the output names
    in :attr:`OUTPUTS`.
    """
    DESCRIPTION: str = ""
    """str: A brief description of the model."""
    REFERENCE: str = ""
    """str: A reference for the model, e.g., a journal article or textbook."""

    # =============================================== #
    # Initialization Method                           #
    # =============================================== #
    def __init__(self, *args, **kwargs):
        # Initialize the base Model class
        super().__init__(*args, **kwargs)

        # Generate our SED object.
        self._SED = SSA_SED_PowerLaw()

    # =============================================== #
    # Model Evaluation Method                         #
    # =============================================== #
    def _forward_model(
        self,
        variables,
        parameters,
    ):
        # First use the parameters to compute the BPL parameters from the
        # core physics parameters.
        nu_brk, F_brk = self._SED._opt_from_physics_to_params(
            parameters["B"],  # Gauss
            parameters["R"],  # cm
            parameters["D"],  # cm
            p=parameters["p"],
            f=parameters["f"],
            theta=parameters["theta"],
            epsilon_B=parameters["epsilon_B"],
            epsilon_E=parameters["epsilon_e"],
            gamma_min=parameters["gamma_min"],
            gamma_max=parameters["gamma_max"],
        )

        # Return the flux density in Jy
        # 1e-23 erg/s/cm^2/Hz = 1 Jy
        return {
            "flux_density": 1e-23
            * self._SED._log_opt_sed(variables["frequency"], nu_brk, F_brk, parameters["p"], parameters["s"])
        }
