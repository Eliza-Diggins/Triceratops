"""
Phenomenological spectral energy distribution (SED) models.

These models represent purely spectral, time-independent SEDs and are
intended for exploratory or diagnostic modeling of broadband radio spectra.
They make no assumptions about the physical emission mechanism or source
dynamics.
"""

from astropy import units as u

from triceratops.models._typing import (
    _ModelOutputRaw,
    _ModelParametersInputRaw,
    _ModelVariablesInputRaw,
)
from triceratops.models.core import Model, ModelParameter, ModelVariable
from triceratops.profiles import broken_power_law, smoothed_BPL


class BrokenPowerLawModel(Model):
    r"""
    Phenomenological broken power-law spectral energy distribution (SED) model.

    This model represents a *purely spectral*, time-independent broken power-law
    SED of the form

    .. math::

        F_\nu =
        \begin{cases}
        F_{\nu,0} \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_1},
        & \nu < \nu_{\rm break} \\
        F_{\nu,0} \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_2},
        & \nu \ge \nu_{\rm break}
        \end{cases}

    where :math:`F_{\nu,0}` is the normalization at the break frequency
    :math:`\nu_{\rm break}`.

    This model is intentionally **phenomenological** and makes no assumptions about
    the physical emission mechanism or source dynamics. It is therefore suitable for:

    - single-epoch SED fitting,
    - exploratory or diagnostic spectral modeling,
    - initial characterization of broadband radio spectra,
    - comparison against physically motivated models.

    It is *not* intended to represent a self-consistent physical model of synchrotron
    emission, shock dynamics, or absorption processes.

    Notes
    -----
    - The spectrum is continuous but not differentiable at
      :math:`\nu = \nu_{\rm break}`.
    - No time dependence is included; if temporal evolution is required, this model
      should be used only at fixed epochs or embedded within a higher-level inference
      framework.
    - This model is mathematically equivalent to a sharp broken power-law commonly used
      to approximate synchrotron spectra in the optically thick and thin regimes.

    See Also
    --------
    triceratops.models.phenomenological :
        Additional time-independent and weakly physical SED models.
    triceratops.models.emission :
        Physically motivated synchrotron emission models.
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
            "alpha_1",
            2.0,
            description="Spectral index before the break frequency.",
            base_units="",
            bounds=(None, None),
            latex=r"\alpha_1",
        ),
        ModelParameter(
            "alpha_2",
            -0.5,
            description="Spectral index after the break frequency.",
            base_units="",
            bounds=(None, None),
            latex=r"\alpha_2",
        ),
    )
    """tuple of :class:`ModelParameter`: The model's parameters.

    Each element of :attr:`PARAMETERS` is a :class:`ModelParameter` instance that defines a single
    parameter of the model. These parameters are used to configure the model and control its behavior. Each
    parameter contains information about the base units, default value, and valid range for that parameter.
    """
    VARIABLES: tuple["ModelVariable", ...] = (
        ModelVariable(
            "frequency", description="Frequency at which to evaluate the SED.", base_units="GHz", latex=r"\nu"
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
        variables: "_ModelVariablesInputRaw",
        parameters: "_ModelParametersInputRaw",
    ) -> "_ModelOutputRaw":
        """
        Compute the model's outputs based on the provided variables and parameters.

        Parameters
        ----------
        variables: dict of str, array-like
            The model's input variables. Each variable must be provided as a key-value pair in the
            dictionary, where the key is the variable name and the value is the variable's value. Each element
            must either be a float or an array-like object. Standard numpy broadcasting rules are applied
            throughout.
        parameters: dict of str, array-like
            The model's parameters. Each parameter must be provided as a key-value pair in the
            dictionary, where the key is the parameter name and the value is the parameter's value. Each element
            must either be a float or an array-like object. Standard numpy broadcasting rules are applied
            throughout. All parameters must be provided; there are no default values at this level.

        Returns
        -------
        outputs: dict of str, array-like
            The model's outputs. Each output is provided as a key-value pair in the dictionary, where the key
            is the output name and the value is the computed output value. Each element will either be a float
            or an array-like object, depending on the inputs. Standard numpy broadcasting rules are applied throughout.
        """
        return {
            "flux_density": broken_power_law(
                variables["frequency"],
                parameters["norm"],
                parameters["nu_break"],
                parameters["alpha_1"],
                parameters["alpha_2"],
            )
        }


class SmoothedBPLModel(Model):
    r"""
    Phenomenological smoothed broken power-law spectral energy distribution (SED) model.

    This model represents a *purely spectral*, time-independent broken power-law
    SED of the form

    .. math::

        F_\nu = F_{\rm nu,0} \left[\left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_1/s} +
        \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_2/s} \right]^{s}

    where :math:`F_{\nu,0}` is the normalization at the break frequency
    :math:`\nu_{\rm break}`.

    This model is intentionally **phenomenological** and makes no assumptions about
    the physical emission mechanism or source dynamics. It is therefore suitable for:

    - single-epoch SED fitting,
    - exploratory or diagnostic spectral modeling,
    - initial characterization of broadband radio spectra,
    - comparison against physically motivated models.

    It is *not* intended to represent a self-consistent physical model of synchrotron
    emission, shock dynamics, or absorption processes.

    Notes
    -----
    - The spectrum is continuous but not differentiable at
      :math:`\nu = \nu_{\rm break}`.
    - No time dependence is included; if temporal evolution is required, this model
      should be used only at fixed epochs or embedded within a higher-level inference
      framework.
    - This model is mathematically equivalent to a sharp broken power-law commonly used
      to approximate synchrotron spectra in the optically thick and thin regimes.

    See Also
    --------
    triceratops.models.phenomenological :
        Additional time-independent and weakly physical SED models.
    triceratops.models.emission :
        Physically motivated synchrotron emission models.
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
            "alpha_1",
            2.0,
            description="Spectral index before the break frequency.",
            base_units="",
            bounds=(None, None),
            latex=r"\alpha_1",
        ),
        ModelParameter(
            "alpha_2",
            -0.5,
            description="Spectral index after the break frequency.",
            base_units="",
            bounds=(None, None),
            latex=r"\alpha_2",
        ),
        ModelParameter(
            "s",
            0.5,
            description="Smoothing parameter controlling the sharpness of the break.",
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
            "frequency", description="Frequency at which to evaluate the SED.", base_units="GHz", latex=r"\nu"
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
        variables: "_ModelVariablesInputRaw",
        parameters: "_ModelParametersInputRaw",
    ) -> "_ModelOutputRaw":
        """
        Compute the model's outputs based on the provided variables and parameters.

        Parameters
        ----------
        variables: dict of str, array-like
            The model's input variables. Each variable must be provided as a key-value pair in the
            dictionary, where the key is the variable name and the value is the variable's value. Each element
            must either be a float or an array-like object. Standard numpy broadcasting rules are applied
            throughout.
        parameters: dict of str, array-like
            The model's parameters. Each parameter must be provided as a key-value pair in the
            dictionary, where the key is the parameter name and the value is the parameter's value. Each element
            must either be a float or an array-like object. Standard numpy broadcasting rules are applied
            throughout. All parameters must be provided; there are no default values at this level.

        Returns
        -------
        outputs: dict of str, array-like
            The model's outputs. Each output is provided as a key-value pair in the dictionary, where the key
            is the output name and the value is the computed output value. Each element will either be a float
            or an array-like object, depending on the inputs. Standard numpy broadcasting rules are applied throughout.
        """
        return {
            "flux_density": smoothed_BPL(
                variables["frequency"],
                parameters["norm"],
                parameters["nu_break"],
                parameters["alpha_1"],
                parameters["alpha_2"],
                parameters["s"],
            )
        }
