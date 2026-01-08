"""Abstract base class for likelihood models."""

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from triceratops.models.core.base import Model

if TYPE_CHECKING:
    from triceratops.models._typing import (
        _ModelParametersInput,
        _ModelParametersInputRaw,
    )

# Defining the __all__ variable.
__all__ = ["Likelihood"]


# ============================================================ #
# Likelihood Base Class                                        #
# ============================================================ #
class Likelihood(ABC):
    """
    Abstract base class for likelihood functions.

    The :class:`Likelihood` class defines the interface and core semantics for
    evaluating the probability of observational data given a physical model.
    It provides a structured separation between:

    - **Model physics**, implemented in subclasses of :class:`~models.core.base.Model`
    - **Observational data**, encapsulated in container classes
    - **Statistical assumptions**, encoded in concrete likelihood subclasses

    A likelihood object binds together a *specific model* and a *specific data
    container*, validates their compatibility, and performs any required
    preprocessing of the data into a numerical form suitable for efficient
    likelihood evaluation.

    Design Philosophy
    -----------------
    This class is intentionally **not** a fully generic "black box" likelihood.
    Instead, it enforces explicit compatibility between models and data
    containers via class-level declarations. This prevents subtle misuse while
    still allowing a high degree of extensibility.

    The likelihood workflow is divided into three conceptual stages:

    1. **Compatibility validation**
       Ensures that the supplied model and data container are appropriate for
       the likelihood implementation.

    2. **Data reduction and coercion**
       The input data container is processed exactly once during initialization
       via :meth:`_process_input_data`. This step extracts relevant quantities,
       converts them to consistent base units, and stores them in a
       :class:`types.SimpleNamespace` for fast reuse.

    3. **Likelihood evaluation**
       The core statistical computation is implemented in :meth:`_log_likelihood`,
       which assumes that model parameters are already in their raw, unit-consistent
       format.

    Public vs. Internal API
    -----------------------
    The public method :meth:`log_likelihood` is intended for user-facing calls
    and accepts parameters in any format supported by the model. These parameters
    are coerced into the model's raw format before being passed to
    :meth:`_log_likelihood`.

    For inference backends (e.g., MCMC, nested sampling), the private
    :meth:`_log_likelihood` method should be used directly, as it avoids repeated
    parameter coercion and assumes a consistent internal representation.

    Extensibility
    -------------
    Concrete likelihood implementations should subclass :class:`Likelihood` and:

    - Declare compatible models via :attr:`COMPATIBLE_MODELS`
    - Declare compatible data containers via :attr:`COMPATIBLE_DATA_CONTAINERS`
    - Implement :meth:`_process_input_data`
    - Implement :meth:`_log_likelihood`

    This structure allows for clean extension to multiple likelihood forms
    (e.g., Gaussian errors, censored data, hierarchical likelihoods) while
    maintaining a uniform interface for inference engines.

    Notes
    -----
    This class makes no assumptions about priors, posteriors, or sampling
    strategies. Those concerns are handled at the inference layer, which
    composes likelihoods with priors and samplers.
    """

    # ============================================================ #
    # Class Semantics                                              #
    # ============================================================ #
    # Here we establish the permissible models and data structures which
    # can be initialized in this likelihood model. This permits some degree
    # of reduced boilerplate when constructing specific likelihood implementations,
    # but still ensures we cannot cross inconsistent models and data.
    #
    # When writing custom likelihoods, it is often acceptable to have MANY
    # compatible models but only ONE compatible data container (e.g. a
    # photometry likelihood that works with any SED model, but only
    # photometry data). In other cases, one may have a SINGLE compatible
    # model that works with multiple data containers.
    #
    # For this reason, both compatibility declarations are tuples.
    COMPATIBLE_MODELS: tuple[type[Model], ...] = ()
    """tuple of :class:`~triceratops.models.core.base.Model`

    Model classes compatible with this likelihood. The provided model instance
    must be an instance of one of these classes.
    """

    COMPATIBLE_DATA_CONTAINERS: tuple[type, ...] = ()
    """tuple of types

    Data container classes compatible with this likelihood. The provided data
    object must be an instance of one of these types.
    """

    # ============================================================ #
    # Initialization                                               #
    # ============================================================ #
    def __init_subclass__(cls, **kwargs):
        """Validate likelihood subclass definitions at class creation time."""
        super().__init_subclass__(**kwargs)

        if not isinstance(cls.COMPATIBLE_MODELS, tuple):
            raise NotImplementedError(f"{cls.__name__} must define COMPATIBLE_MODELS as a tuple.")
        if not isinstance(cls.COMPATIBLE_DATA_CONTAINERS, tuple):
            raise NotImplementedError(f"{cls.__name__} must define COMPATIBLE_DATA_CONTAINERS as a tuple.")

        if len(cls.COMPATIBLE_MODELS) == 0:
            raise NotImplementedError(f"{cls.__name__} must declare at least one compatible model.")
        if len(cls.COMPATIBLE_DATA_CONTAINERS) == 0:
            raise NotImplementedError(f"{cls.__name__} must declare at least one compatible data container.")

        for model_cls in cls.COMPATIBLE_MODELS:
            if not issubclass(model_cls, Model):
                raise TypeError(f"{model_cls} listed in COMPATIBLE_MODELS is not a Model subclass.")

    def __init__(self, model: Model, data: Any, **kwargs):
        """
        Initialize the likelihood object.

        This performs:

        1. Compatibility validation between model and data.
        2. Storage of the raw model and data container.
        3. One-time preprocessing of the data into numerical form suitable
           for fast likelihood evaluation.

        Parameters
        ----------
        model : ~models.core.base.Model
            The astrophysical model to be evaluated. This must be a subclass of :class:`~models.core.base.Model` and
            be compatible with the likelihood class (see :attr:`COMPATIBLE_MODELS`). This defines the forward model
            being fit against.
        data:
            The data container object holding the observational data. This must be of a type compatible
            with the likelihood class (see :attr:`COMPATIBLE_DATA_CONTAINERS`). This defines the observations being
            compared against.
        **kwargs
            Additional keyword arguments passed to the data processing method.

        Raises
        ------
        TypeError:
            If the provided model or data are not compatible with this likelihood.

        Notes
        -----
        Upon initialization, the likelihood object processes the input data
        via the :meth:`_process_input_data` method. The results are cached in
        ``self._data`` for efficient reuse during likelihood evaluations. All data is coerced
        into consistent base units during this step.
        """
        self._validate_input_model_and_data(model, data)

        self._model: Model = model
        self._data_container: Any = data

        # Process and cache numerical data needed for likelihood evaluation.
        self._data: SimpleNamespace = self._process_input_data(**kwargs)

    @classmethod
    def _validate_input_model_and_data(cls, model: Model, data: Any) -> None:
        """Validate model–data compatibility."""
        if not any(isinstance(model, m) for m in cls.COMPATIBLE_MODELS):
            raise TypeError(
                f"Model of type {type(model)} is not compatible with "
                f"{cls.__name__}. Compatible models: {cls.COMPATIBLE_MODELS}"
            )

        if not any(isinstance(data, d) for d in cls.COMPATIBLE_DATA_CONTAINERS):
            raise TypeError(
                f"Data of type {type(data)} is not compatible with "
                f"{cls.__name__}. Compatible data containers: "
                f"{cls.COMPATIBLE_DATA_CONTAINERS}"
            )

    # ============================================================ #
    # Data Processing                                              #
    # ============================================================ #
    @abstractmethod
    def _process_input_data(self, **kwargs) -> SimpleNamespace:
        """
        Convert the input data container into a numerical representation suitable for likelihood evaluation.

        This method is called exactly once during initialization and should:

        - Extract relevant quantities from the data container
        - Convert all values into consistent base units
        - Store the results in a :class:`types.SimpleNamespace`

        The returned namespace is cached as ``self._data`` and should be treated
        as immutable.

        Returns
        -------
        types.SimpleNamespace
            Namespace containing all numerical arrays required for computing
            the likelihood.
        """
        raise NotImplementedError

    # ============================================================ #
    # Likelihood Evaluation                                        #
    # ============================================================ #
    @abstractmethod
    def _log_likelihood(
        self,
        parameters: dict[str, "_ModelParametersInputRaw"],
    ) -> float:
        """
        Compute the log-likelihood for the given model parameters.

        Parameters
        ----------
        parameters : dict
            Dictionary mapping model parameter names to values in base units,
            as defined by the model. These parameters should be in the **raw format**
            accepted by the model: array-like objects in the correct base units.

        Returns
        -------
        float
            Log-likelihood value.
        """
        raise NotImplementedError

    def log_likelihood(self, parameters: dict[str, "_ModelParametersInput"]) -> float:
        """
        Compute the log-likelihood for the given model parameters.

        Parameters
        ----------
        parameters: dict
            Dictionary mapping model parameter names to values. These parameters
            may be in any acceptable format for the model; they will be coerced
            into the raw format by this method.

        Returns
        -------
        float
            Log-likelihood value.


        Notes
        -----
        For inference, the private method :meth:`_log_likelihood` should be used,
        as it assumes that the parameters are already in the correct raw format. This is a user-facing
        wrapper that performs the necessary coercion.
        """
        return self._log_likelihood(self._model.coerce_model_parameters(parameters))
