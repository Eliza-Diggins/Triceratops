"""
Phenomenological models of synchrotron emission for Triceratops.

This module contains phenomenological models designed to simulate and analyze synchrotron
emission from transient astrophysical sources. These models provide simplified yet effective
descriptions of synchrotron radiation processes, allowing researchers to study radio emissions,
spectral energy distributions (SEDs), and other relevant phenomena without delving into
complex physical details.
"""

from astropy import units as u

from triceratops.models._typing import (
    _ModelOutputRaw,
    _ModelParametersInputRaw,
    _ModelVariablesInputRaw,
)
from triceratops.models.core import Model, ModelParameter, ModelVariable
from triceratops.profiles import smoothed_BPL


class Synchrotron_SSA_SBPL_SED(Model):
    r"""
    Phenomenological synchrotron self-absorbed broken power-law spectral energy distribution (SED).

    This model provides a **time-independent**, smoothly broken power-law approximation to a
    synchrotron self-absorbed (SSA) radio spectrum. It is intended as a lightweight,
    phenomenological description of synchrotron emission from transient astrophysical
    sources such as supernovae (SNe) and gamma-ray bursts (GRBs), without enforcing
    dynamical self-consistency or detailed radiative transfer.

    The SED is parameterized as

    .. math::

        F_{\nu} = F_{\nu,0}
        \left[
            \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_{\rm thick}/s}
            +
            \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_{\rm thin}/s}
        \right]^s,

    where :math:`F_{\nu,0}` is the normalization at the break frequency
    :math:`\nu_{\rm break}`. The spectral indices are tied to the electron energy
    distribution power-law index :math:`p` via

    .. math::

        \alpha_{\rm thick} = \frac{5}{3},
        \qquad
        \alpha_{\rm thin} = -\frac{p - 1}{2}.

    This choice reproduces the canonical optically thick and optically thin synchrotron
    spectral slopes expected for a homogeneous emitting region with a power-law electron
    population.

    This model is **phenomenological** in nature: it does not compute or require physical
    quantities such as shock radius, magnetic field strength, or particle density, and it
    does not include time evolution. It is therefore most appropriate for:

    - single-epoch radio SED fitting,
    - exploratory or diagnostic spectral modeling,
    - comparison against physically motivated, time-dependent synchrotron models.

    For detailed discussions of synchrotron spectral shapes and their physical
    interpretation, see :footcite:t:`rybickilightman` and
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``norm``
             - :math:`F_{\nu,0}`
             - Flux density normalization at the break frequency.
           * - ``nu_break``
             - :math:`\nu_{\rm break}`
             - Synchrotron self-absorption turnover (break) frequency.
           * - ``p``
             - :math:`p`
             - Power-law index of the relativistic electron energy distribution.
           * - ``s``
             - :math:`s`
             - Smoothness parameter controlling the sharpness of the spectral break.

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
    In most scenarios, the break frequency will be ill-constrained without broadband sampling of
    the radio SED. Care should be taken when interpreting fit results, especially when using
    sparse data.
    """

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    # Each model must declare its parameters and variables as class-level attributes. These
    # must each be instances of `ModelParameter` and `ModelVariable`, respectively.
    PARAMETERS: tuple["ModelParameter", ...] = (
        ModelParameter(
            "norm",
            1.0,
            description="Normalization constant of the SED.",
            base_units="Jy",
            bounds=(0.0, None),
            latex=r"F_{\rm nu,0}",
        ),
        ModelParameter(
            "nu_break",
            5.0,
            description="Break frequency of the SED.",
            base_units="GHz",
            bounds=(0.0, None),
            latex=r"\nu_{\rm break}",
        ),
        ModelParameter(
            "p",
            3.0,
            description="Power-law index for the electron energy distribution.",
            base_units="",
            bounds=(0, None),
            latex=r"p",
        ),
        ModelParameter(
            "s",
            -0.5,
            description="The smoothing parameter of the broken power-law.",
            base_units="",
            bounds=(None, 0),
            latex=r"s",
        ),
    )
    """tuple of :class:`ModelParameter`: The model's parameters.

    Each element of :attr:`PARAMETERS` is a :class:`ModelParameter` instance that defines a single
    parameter of the model. These parameters are used to configure the model and control its behavior. Each
    parameter contains information about the base units, default value, and valid range for that parameter.
    """
    VARIABLES: tuple["ModelVariable", ...] = (
        ModelVariable(
            "frequency", description="Observing frequency at which to evaluate the SED.", base_units="GHz", latex=r"\nu"
        ),
    )
    """tuple of :class:`ModelVariable`: The model's variables.

    Each element of :attr:`VARIABLES` is a :class:`ModelVariable` instance that defines a single
    variable of the model. These variables represent the inputs to the model that can vary during
    evaluation. Each variable contains information about the base units, name, etc. of the variable. Notably,
    variables differ from the parameters (:attr:`PARAMETERS`) in that variables are do **NOT** have default
    values, do **NOT** have validity ranges, and are expected to be provided at model evaluation time.
    """

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
    # Model Evaluation Method                         #
    # =============================================== #
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> _ModelOutputRaw:
        # Construct the indices from the value of p.
        p = parameters["p"]
        alpha_2 = 5 / 3
        alpha_1 = -(p - 1) / 2

        result = smoothed_BPL(
            variables["frequency"], parameters["norm"], parameters["nu_break"], alpha_1, alpha_2, parameters["s"]
        )
        return {"flux_density": result}
