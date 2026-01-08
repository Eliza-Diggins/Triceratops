"""
Abstract base classes for Triceratops models.

This module defines the abstract base class structure for all Triceratops models. It is the
foundation upon which specific physical models are built, ensuring consistency and
standardization across different implementations.

At their core, all of the models in Triceratops inherit from the :class:`Model` abstract base class, which
effectively declares a set of ``parameters``, a set of ``variables``, and a method to compute the model's outputs
based on those parameters and variables.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from astropy import units as u

from .._typing import (
    _ModelOutput,
    _ModelOutputRaw,
    _ModelParametersInput,
    _ModelParametersInputRaw,
    _ModelVariablesInput,
    _ModelVariablesInputRaw,
)
from .parameters import ModelParameter, ModelVariable

# --- TYPE ALIASES --- #
# These are type aliases to allow for cleaner type hints throughout the codebase. They are
# used only for type hinting and do not affect runtime behavior.


class Model(ABC):
    r"""
    Abstract base class for all Triceratops models.

    At their core, all models in Triceratops are wrappers around a set of physical
    parameters and variables, along with a method to compute the model's outputs based on
    those parameters and variables. Formally, a model maps as set of input variables :math:`{\bf x}` and
    a set of model parameters :math:`\boldsymbol{\Theta}` to a set of outputs :math:`{\bf y}` via a function :math:`f`

    .. math::

        {\bf y} = f({\bf x}, \boldsymbol{\Theta}).

    This abstract base class defines the structure that all Triceratops models must adhere to. Specifically, each
    model must declare its parameters and variables as class-level attributes, and implement the
    :meth:`_forward_model` method to compute the model's outputs.
    """

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    # Each model must declare its parameters and variables as class-level attributes. These
    # must each be instances of `ModelParameter` and `ModelVariable`, respectively.
    PARAMETERS: tuple[ModelParameter, ...] = ()
    """tuple of :class:`ModelParameter`: The model's parameters.

    Each element of :attr:`PARAMETERS` is a :class:`ModelParameter` instance that defines a single
    parameter of the model. These parameters are used to configure the model and control its behavior. Each
    parameter contains information about the base units, default value, and valid range for that parameter.
    """
    VARIABLES: tuple[ModelVariable, ...] = ()
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
    OUTPUTS: tuple[str, ...] = ()
    """tuple of str: The names of the model's outputs.

    Each element of :attr:`OUTPUTS` is a string that defines the name of a single output of the model.
    These names correspond to the keys in the dictionary returned by the model's evaluation method.
    """
    UNITS: tuple[u.Unit, ...] = ()
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
    # Initialization and Instantiation                #
    # =============================================== #
    def __init_subclass__(cls, **kwargs):
        """Validate that the subclass has properly defined its parameters and variables."""
        # Hand off to the super-class initializer to ensure that any work
        # done there is not skipped.
        super().__init_subclass__(**kwargs)

        # Ensure that the parameters and variables are properly defined
        # and compatible.
        if len(cls.VARIABLES) == 0:
            raise ValueError("At least one VARIABLE must be defined.")

        # Ensure that the model parameters and variables are of the correct type.
        if not all(isinstance(param, ModelParameter) for param in cls.PARAMETERS):
            raise TypeError("All PARAMETERS must be instances of ModelParameter.")
        if not all(isinstance(var, ModelVariable) for var in cls.VARIABLES):
            raise TypeError("All VARIABLES must be instances of ModelVariable.")

        # Validate that OUTPUTS and UNITS have the same length
        if len(cls.OUTPUTS) != len(cls.UNITS):
            raise ValueError("OUTPUTS and UNITS must have the same length.")

        # Ensure the class's output units are actually converted to
        # astropy units.
        for i, unit in enumerate(cls.UNITS):
            if not isinstance(unit, u.UnitBase):
                raise TypeError(f"UNITS[{i}] is not an astropy Unit.")

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the model with given parameters and variables."""
        # At the base class level, there is no specific initialization logic.
        pass

    # =============================================== #
    # Model Evaluation                                #
    # =============================================== #
    # Each model must implement the evaluate method to compute its outputs. This is where the
    # core physics of the model is written and implemented. Each subclass must provide its own implementation
    # of this method.
    #
    # DEV NOTE: In general, it is best to place generic physics routines in the ``physics`` module and
    # then import them at the level of the model. This keeps the model code clean and focused on
    # the high-level structure of the model rather than the low-level physics details.
    @abstractmethod
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> _ModelOutputRaw:
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
        pass

    def forward_model(
        self,
        variables: _ModelVariablesInput,
        parameters: _ModelParametersInput,
    ) -> _ModelOutput:
        """
        Compute the model's outputs based on the provided variables and parameters.

        Parameters
        ----------
        variables: dict of str, array-like or Quantity
            The model's input variables. Each variable must be provided as a key-value pair in the
            dictionary, where the key is the variable name and the value is the variable's value. Each element
            must either be a float, an array-like object, or an :class:`astropy.units.Quantity`. Standard numpy
            broadcasting rules are applied throughout.
        parameters: dict of str, array-like or Quantity
            The model's parameters. Each parameter must be provided as a key-value pair in the
            dictionary, where the key is the parameter name and the value is the parameter's value. Each element
            must either be a float, an array-like object, or an :class:`astropy.units.Quantity`. Standard numpy
            broadcasting rules are applied throughout. If a parameter is missing, it will be filled by the
            default value defined in the :attr:`PARAMETERS` class attribute.

        Returns
        -------
        outputs: dict of str, array-like or Quantity
            The model's outputs. Each output is provided as a key-value pair in the dictionary, where the key
            is the output name and the value is the computed output value. If an output has units, then it
            will be returned as an astropy quantity, otherwise, it will be a numpy array or float. Standard numpy
            broadcasting rules are applied throughout.
        """
        # --- Coerce the variables and parameters --- #
        # The first task in the forward model is to coerce the input variables and parameters
        # into raw numerical values (i.e., strip off any units) and ensure that all of the
        # required parameters and variables are provided. This is passed off to the helper functions
        # ``coerce_model_variables`` and ``coerce_model_parameters``. We re-declare so that we do not
        # have mutability issues.
        _input_variables, _input_parameters = (
            self.coerce_model_variables(variables),
            self.coerce_model_parameters(parameters),
        )

        # --- Check bounds --- #
        # There are two chances for bounds to be violated: either because one of the parameters is
        # invalid (``check_parameter_bounds``), or because of a model-level bounds check (``check_model_bounds``).
        # If either of these trip for any parameter or variable, we raise a ValueError.
        self._check_parameter_bounds_base(_input_parameters)
        self._check_model_bounds_base(_input_variables, _input_parameters)

        # --- Compute the raw outputs --- #
        # With the inputs coerced and validated, we can now compute the raw outputs by passing
        # the inputs to the model's core physics routine, ``_forward_model``.
        _raw_outputs = self._forward_model(_input_variables, _input_parameters)

        # --- Attach units to outputs --- #
        # Finally, we need to attach units to the outputs where appropriate. We cycle through
        # each of the outputs defined in ``OUTPUTS`` and ``UNITS`` and attach units where
        # defined.
        _final_outputs: _ModelOutput = {}
        for output_name, output_unit in zip(self.OUTPUTS, self.UNITS, strict=False):
            raw_value = _raw_outputs[output_name]
            if output_unit is not None:
                _final_outputs[output_name] = raw_value * output_unit
            else:
                _final_outputs[output_name] = raw_value

        return _final_outputs

    def __call__(self, variables, parameters):
        """
        Evaluate the model with the given variables and parameters.

        Parameters
        ----------
        variables: dict of str, array-like or Quantity
            The model's input variables. Each variable must be provided as a key-value pair in the
            dictionary, where the key is the variable name and the value is the variable's value. Each element
            must either be a float, an array-like object, or an :class:`astropy.units.Quantity`. Standard numpy
            broadcasting rules are applied throughout.
        parameters: dict of str, array-like or Quantity
            The model's parameters. Each parameter must be provided as a key-value pair in the
            dictionary, where the key is the parameter name and the value is the parameter's value. Each element
            must either be a float, an array-like object, or an :class:`astropy.units.Quantity`. Standard numpy
            broadcasting rules are applied throughout. If a parameter is missing, it will be filled by the
            default value defined in the :attr:`PARAMETERS` class attribute.

        Returns
        -------
        outputs: dict of str, array-like or Quantity
            The model's outputs. Each output is provided as a key-value pair in the dictionary, where the key
            is the output name and the value is the computed output value. If an output has units, then it
            will be returned as an astropy quantity, otherwise, it will be a numpy array or float. Standard numpy
            broadcasting rules are applied throughout.
        """
        return self.forward_model(variables, parameters)

    # =============================================== #
    # Variables, Bounds, Parameters, etc.             #
    # =============================================== #
    def coerce_model_variables(self, variables: _ModelVariablesInput) -> _ModelVariablesInputRaw:
        """
        Coerce model variables into raw numerical values in base units.

        This method validates that all required model variables are provided and
        converts each variable into its raw numerical value in base units. Any
        astropy Quantity inputs are stripped of units during this process.

        Parameters
        ----------
        variables : dict of str -> float, numpy.ndarray, or astropy.units.Quantity
            Dictionary mapping variable names to their values. All variables
            declared in :attr:`VARIABLES` must be present.

        Returns
        -------
        dict of str -> float or numpy.ndarray
            Dictionary mapping variable names to raw numerical values in base units.

        Raises
        ------
        ValueError
            If a required variable is missing or cannot be coerced into base units.
        """
        # Declare the coerced variables dictionary to ensure that
        # there are no mutability issues.
        coerced_variables: _ModelVariablesInputRaw = {}

        # Cycle through each of the variables in the model and attempt to coerce. If we cannot find
        # the variable in the input dictionary, raise an error.
        for var in self.VARIABLES:
            # Check for the variable in the input variables dictionary.
            if var.name not in variables:
                raise ValueError(f"Missing required model variable '{var.name}'.")

            # Attempt to coerce the variable to its base value.
            try:
                coerced_variables[var.name] = var.to_base_value(variables[var.name])
            except Exception as exc:
                raise ValueError(f"Error coercing model variable '{var.name}': {exc}") from exc

        return coerced_variables

    def coerce_model_parameters(self, parameters: _ModelParametersInput) -> _ModelParametersInputRaw:
        """
        Coerce model parameters into raw numerical values in base units.

        This method fills in missing parameters using their default values,
        converts all parameters into base units, and checks parameter bounds
        where defined.

        Parameters
        ----------
        parameters : dict of str -> float, numpy.ndarray, or astropy.units.Quantity
            Dictionary mapping parameter names to their values. Missing parameters
            will be filled using defaults defined in :class:`ModelParameter`.

        Returns
        -------
        dict of str -> float or numpy.ndarray
            Dictionary mapping parameter names to raw numerical values in base units.

        Raises
        ------
        ValueError
            If a parameter value cannot be coerced or violates its bounds.
        """
        # Declare the coerced variables dictionary to ensure that
        # there are no mutability issues.
        coerced_parameters: _ModelParametersInputRaw = {}

        # Cycle through each of the parameters and attempt to coerce. If a parameter is missing,
        # we place the default value.
        for param in self.PARAMETERS:
            # Set the value either to the provided value or to the default.
            coerced_parameters[param.name] = parameters.get(param.name, param.default)

            # Attempt to coerce the value to the base value.
            try:
                coerced_parameters[param.name] = param.to_base_value(coerced_parameters[param.name])
            except Exception as exc:
                raise ValueError(f"Error coercing model parameter '{param.name}': {exc}") from exc

        return coerced_parameters

    # =============================================== #
    # Bounds Checking                                 #
    # =============================================== #

    def _check_parameter_bounds_base(
        self,
        parameters: _ModelParametersInputRaw,
    ) -> None:
        """
        Check parameter-level bounds in base units.

        This method enforces all parameter bounds defined by each
        :class:`ModelParameter`. Bounds are treated as *hard constraints*:
        if any parameter violates its bounds, a ``ValueError`` is raised.

        Parameters
        ----------
        parameters : dict of str -> float or numpy.ndarray
            Parameter values in base units.

        Raises
        ------
        ValueError
            If any parameter value violates its bounds.
        """
        for param in self.PARAMETERS:
            value = parameters[param.name]

            # Scalar case
            if np.isscalar(value):
                valid = param.check_bounds(value)

            # Array case: all values must be valid
            else:
                valid = np.all(param.check_bounds(value))

            if not valid:
                raise ValueError(
                    f"Model parameter '{param.name}' with value {value} violates its bounds {param.bounds}."
                )

    def _check_model_bounds_base(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> None:
        """
        Check model-level bounds in base units.

        This method is intended to be optionally overridden by subclasses
        to enforce *model-scale* constraints involving multiple parameters
        and/or variables.

        By default, no model-level bounds are enforced.

        Parameters
        ----------
        variables : dict of str -> float or numpy.ndarray
            Model variables in base units.
        parameters : dict of str -> float or numpy.ndarray
            Model parameters in base units.

        Notes
        -----
        - This method should raise ``ValueError`` for *hard* violations.
        - For physics-driven invalid regions, it is generally preferable
          to allow ``_forward_model`` to return NaNs or infinities instead.
        """
        return None

    def check_parameter_bounds(
        self,
        parameters: _ModelParametersInput,
    ) -> None:
        """
        Check parameter-level bounds with unit handling.

        This is a user-facing wrapper around
        :meth:`_check_parameter_bounds_base` that coerces parameter
        values into base units before checking.

        Parameters
        ----------
        parameters : dict of str -> float, numpy.ndarray, or Quantity
            Parameter values to validate.

        Raises
        ------
        ValueError
            If any parameter violates its bounds.
        """
        coerced = self.coerce_model_parameters(parameters)
        self._check_parameter_bounds_base(coerced)

    def check_model_bounds(
        self,
        variables: _ModelVariablesInput,
        parameters: _ModelParametersInput,
    ) -> None:
        """
        Check model-level bounds with unit handling.

        This method coerces both variables and parameters into base units
        and then applies any model-level constraints implemented by the
        subclass.

        Parameters
        ----------
        variables : dict of str -> float, numpy.ndarray, or Quantity
            Model variables.
        parameters : dict of str -> float, numpy.ndarray, or Quantity
            Model parameters.

        Raises
        ------
        ValueError
            If a model-level constraint is violated.
        """
        coerced_vars = self.coerce_model_variables(variables)
        coerced_params = self.coerce_model_parameters(parameters)
        self._check_model_bounds_base(coerced_vars, coerced_params)

    # =============================================== #
    # Dunder Methods                                  #
    # =============================================== #
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} Model: {self.DESCRIPTION}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """Compute the total number of parameters and variables."""
        return len(self.PARAMETERS) + len(self.VARIABLES)

    def __getitem__(self, item: str) -> Union[ModelParameter, ModelVariable]:
        """Get a parameter or variable by name."""
        for param in self.PARAMETERS:
            if param.name == item:
                return param
        for var in self.VARIABLES:
            if var.name == item:
                return var
        raise KeyError(f"'{item}' is not a valid parameter or variable name.")

    def __iter__(self):
        """Iterate over the model's parameters and variables."""
        yield from self.PARAMETERS
        yield from self.VARIABLES

    # =============================================== #
    # Help Methods                                    #
    # =============================================== #
    @property
    def parameter_names(self) -> tuple[str, ...]:
        """str: The names of the model's parameters."""
        return tuple(p.name for p in self.PARAMETERS)

    @property
    def variable_names(self) -> tuple[str, ...]:
        """str: The names of the model's variables."""
        return tuple(v.name for v in self.VARIABLES)

    @property
    def output_names(self) -> tuple[str, ...]:
        """str: The names of the model's outputs."""
        return self.OUTPUTS
