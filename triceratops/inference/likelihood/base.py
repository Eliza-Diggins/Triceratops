"""Abstract base class for likelihood models."""

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from scipy.stats import norm

from triceratops.data.core import DataContainer
from triceratops.models.core.base import Model
from triceratops.utils.log import triceratops_logger
from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops.models._typing import (
        _ModelParametersInput,
        _ModelParametersInputRaw,
    )

# Defining the __all__ variable.
__all__ = ["Likelihood", "GaussianCensoredLikelihoodStencil", "GaussianLikelihoodXY"]


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

    def __init__(self, model: Model, data: DataContainer, **kwargs):
        """
        Instantiate the likelihood object.

        Parameters
        ----------
        model: ~models.core.base.Model
            The astrophysical model to be evaluated. This must be a subclass of :class:`~models.core.base.Model` and
            be compatible with the likelihood class (see :attr:`COMPATIBLE_MODELS`). This defines the forward model
            used in the likelihood evaluation.
        data: ~data.core.DataContainer
            The data container object holding the observational data. This must be of a type compatible
            with the likelihood class (see :attr:`COMPATIBLE_DATA_CONTAINERS`). This defines the observations being
            compared against.
        kwargs:
            Additional keyword arguments that may be used during configuration. These are passed
            directly to the :meth:`_configure` method immediately after assigning the model and data attributes.
            These can therefore be used to customize likelihood behavior during initialization.
        """
        # Declare the model and the data.
        if not isinstance(model, Model):
            raise TypeError("The 'model' argument must be an instance of Model.")
        if not isinstance(data, DataContainer):
            raise TypeError("The 'data' argument must be an instance of DataContainer.")

        self._model: Model = model
        self._data_container: DataContainer = data

        # Configure the likelihood by passing off to the ``_configure`` method. This
        # allows for custom namespace configuration before we perform any validation
        # or other steps.
        self._configure(**kwargs)

        # With the model and data assigned and the namespace configured, we now
        # perform the model and data validation step.
        self._validate_model_and_data()

        # Finalize the initialization step by generating the data.
        self._data = self._process_input_data()

    @abstractmethod
    def _configure(self, **kwargs):
        """
        Configure the likelihood namespace.

        This method is called immediately after the model and data attributes
        have been assigned. It can be used to set up any additional attributes
        or perform any configuration steps needed before validation and data
        processing.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments passed during initialization.
        """
        raise NotImplementedError

    @abstractmethod
    def _validate_model_and_data(self):
        """
        Validate model–data compatibility.

        This method is called after the namespace has been configured.
        It should raise an error if the model and data are not compatible
        with this likelihood implementation.
        """
        if not any(isinstance(self._model, m) for m in self.__class__.COMPATIBLE_MODELS):
            raise TypeError(
                f"Model of type {type(self._model)} is not compatible with "
                f"{self.__class__.__name__}. Compatible models: {self.__class__.COMPATIBLE_MODELS}"
            )

        if not any(isinstance(self._data_container, d) for d in self.__class__.COMPATIBLE_DATA_CONTAINERS):
            raise TypeError(
                f"Data of type {type(self._data_container)} is not compatible with "
                f"{self.__class__.__name__}. Compatible data containers: "
                f"{self.__class__.COMPATIBLE_DATA_CONTAINERS}"
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


class GaussianCensoredLikelihoodStencil(Likelihood, ABC):
    r"""
    Gaussian likelihood class.

    The :class:`_GaussianLikelihoodStencil` class implements a standard Gaussian
    likelihood function, assuming that the observational data have normally
    distributed errors. This class can be combined with specific model and
    data container compatibility via multiple inheritance to create concrete
    likelihood implementations.

    Formally, we assume a forward model :math:`M({\bf x}; \boldsymbol{\Theta}) \to y`,
    where :math:`{\bf x}` are the independent variable(s), :math:`\boldsymbol{\Theta}` are the model parameters,
    and :math:`y \in \mathbb{R}` is the predicted dependent variable. Given observational data
    :math:`\{(x_i, y_i, \sigma_i)\}_{i=1}^N`, where :math:`\sigma_i` are the measurement uncertainties,
    the Gaussian likelihood is given by:

    .. math::

        \ln \mathcal{L}(\boldsymbol{\Theta}) = -\frac{1}{2} \sum_{i=1}^N \left[
            \frac{(y_i - M(x_i; \boldsymbol{\Theta}))^
            {2}}{\sigma_i^2} + \ln(2 \pi \sigma_i^2)
        \right].

    Additionally, data for which only upper and / or lower limits are provided
    are treated as censored observations. For a data point with an upper limit
    :math:`y_i^{\rm upper}` and uncertainty :math:`\sigma_i`, the contribution to the log-likelihood is:

    .. math::

        \ln \mathcal{L}_i^{\rm upper}(\boldsymbol{\Theta}) = \ln \left[
            \frac{1}{2} \left( 1 + \operatorname{erf} \left(
                \frac{y_i^{\rm upper} - M(x_i; \boldsymbol{\Theta})}
                {\sigma_i \sqrt{2}}
            \right) \right) \right].

    Similarly, for a data point with a lower limit :math:`y_i^{\rm lower}`, the contribution is:

    .. math::

        \ln \mathcal{L}_i^{\rm lower}(\boldsymbol{\Theta}) = \ln \left[
            \frac{1}{2} \left( 1 - \operatorname{erf} \left(
                \frac{y_i^{\rm lower} - M(x_i; \boldsymbol{\Theta})}
                {\sigma_i \sqrt{2}}
            \right) \right) \right].
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
    COMPATIBLE_MODELS: tuple[type[Model], ...] = (Model,)
    """tuple of :class:`~triceratops.models.core.base.Model`

    Model classes compatible with this likelihood. The provided model instance
    must be an instance of one of these classes. Because this likelihood is
    an agnostic base-class, we accept any :class:`~triceratops.models.core.base.Model`.
    """

    COMPATIBLE_DATA_CONTAINERS: tuple[type, ...] = (DataContainer,)
    """tuple of types

    Data container classes compatible with this likelihood. For the standard
    Gaussian likelihood, we require :class:`~triceratops.data.core.XYDataContainer`,
    so that we have both x and y data with associated uncertainties.
    """

    # ============================================================ #
    # Data Processing                                              #
    # ============================================================ #
    @abstractmethod
    def _process_input_data(self, **kwargs) -> SimpleNamespace:
        """
        Process the input data.

        For the gaussian likelihood, we require that the following in the namespace:

        - ``x``: array-like dict of str, array (shapes (N, ), ...)
            Independent variable data points. These should be in base units.
        - ``y_err``: array-like (N,)
            Uncertainties on the dependent variable data points. These should be in base units. These
            must be complete. We do not accept censored data without uncertainties attached.
        - ``y_upper``: array-like (N,)
            Upper limits on the dependent variable data points. These should be in base units.
            Entries without upper limits should be set to ``np.nan``.
        - ``y_lower``: array-like (N,)
            Lower limits on the dependent variable data points. These should be in base units.
            Entries without lower limits should be set to ``np.nan``.
        - ``y_lower_mask``: array-like (N,)
            Boolean mask indicating which entries in ``y_lower`` are valid lower limits.
        - ``y_upper_mask``: array-like (N,)
            Boolean mask indicating which entries in ``y_upper`` are valid upper limits.

        With these quantities defined, the Gaussian likelihood can be computed
        in the :meth:`_log_likelihood` method. As such, new subclasses of this likelihood
        need only ensure that they correctly implement this and the machinery will be complete.
        """
        raise NotImplementedError

    # ============================================================ #
    # Likelihood Evaluation                                        #
    # ============================================================ #
    def _log_likelihood(
        self,
        parameters: dict[str, "_ModelParametersInputRaw"],
    ) -> float:
        """
        Compute the Gaussian log-likelihood for the given model parameters.

        Parameters
        ----------
        parameters : dict
            Dictionary mapping model parameter names to values in base units,
            as defined by the model.

        Returns
        -------
        float
            Log-likelihood value.
        """
        # ------------------------------------------------------------
        # Forward model evaluation
        # ------------------------------------------------------------
        # We assume the model returns the dependent variable as the
        # first element of the tuple.
        model_y = self._model._forward_model_tupled(self._data.x, parameters)[0]

        # ------------------------------------------------------------
        # Split data by detection / censoring type
        # ------------------------------------------------------------
        det_mask = ~self._data.y_upper_mask & ~self._data.y_lower_mask
        ul_mask = self._data.y_upper_mask
        ll_mask = self._data.y_lower_mask

        # Detections
        y_det = self._data.y[det_mask]
        y_err_det = self._data.y_err[det_mask]
        model_y_det = model_y[det_mask]

        # Upper limits (y < y_ul)
        y_ul = self._data.y_upper[ul_mask]
        y_err_ul = self._data.y_err[ul_mask]
        model_y_ul = model_y[ul_mask]

        # Lower limits (y > y_ll)
        y_ll = self._data.y_lower[ll_mask]
        y_err_ll = self._data.y_err[ll_mask]
        model_y_ll = model_y[ll_mask]

        log_likelihood = 0.0

        # ------------------------------------------------------------
        # Gaussian likelihood for detections
        # ------------------------------------------------------------
        if y_det.size > 0:
            resid = (y_det - model_y_det) / y_err_det
            log_likelihood += -0.5 * (np.sum(resid**2) + np.sum(np.log(2.0 * np.pi * y_err_det**2)))

        # ------------------------------------------------------------
        # Upper limits: P(y < y_ul)
        # ------------------------------------------------------------
        if y_ul.size > 0:
            logcdf = norm.logcdf(
                y_ul,
                loc=model_y_ul,
                scale=y_err_ul,
            )
            # Guard against numerical underflow
            log_likelihood += np.sum(np.where(np.isfinite(logcdf), logcdf, -np.inf))

        # ------------------------------------------------------------
        # Lower limits: P(y > y_ll)
        # ------------------------------------------------------------
        if y_ll.size > 0:
            logsf = norm.logsf(
                y_ll,
                loc=model_y_ll,
                scale=y_err_ll,
            )
            # Guard against numerical underflow
            log_likelihood += np.sum(np.where(np.isfinite(logsf), logsf, -np.inf))

        return log_likelihood

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


# ============================================================ #
# Generic Likelihood Classes                                   #
# ============================================================ #
# Here we define a few drop-in likelihood classes that can be used to
# generate other likelihoods via multiple inheritance. These classes
# implement common likelihood functions (e.g., Gaussian errors) but do
# not define any specific model or data container compatibility.
# Specific likelihood implementations can then combine these generic
# likelihoods with specific model/data compatibility via multiple inheritance.
class GaussianLikelihoodXY(
    GaussianCensoredLikelihoodStencil,
):
    """
    Gaussian likelihood for XY data.

    This class combines the standard Gaussian likelihood with compatibility
    for :class:`~triceratops.data.core.XYDataContainer`, allowing for
    straightforward construction of Gaussian likelihoods for XY data. This avoids
    needing to write boilerplate code for common use cases.
    """

    COMPATIBLE_DATA_CONTAINERS: tuple[type, ...] = (DataContainer,)
    """tuple of types

    Data container classes compatible with this likelihood. For the standard
    Gaussian likelihood, we require :class:`~triceratops.data.core.XYDataContainer`,
    so that we have both x and y data with associated uncertainties.
    """

    # ============================================================ #
    # Initialization                                               #
    # ============================================================ #
    def _validate_model_and_data(self):
        """
        Validate model–data compatibility.

        In addition to the base class validation, which will ensure we have
        an XY data container, we also need to ensure that we do not have
        any errors on the independent variable, as the Gaussian likelihood
        does not support that. We raise a warning in this scenario because it
        might be an intended use-case, but its worth flagging.
        """
        super()._validate_model_and_data()

        # Check for independent variable errors.
        if any(
            _col is not None
            for _col in [
                self._data_container.X_ERROR_COLUMN,
                self._data_container.X_LOWER_LIMIT_COLUMN,
                self._data_container.X_UPPER_LIMIT_COLUMN,
            ]
        ):
            triceratops_logger.warning(
                "Likelihood class `GaussianLikelihoodXY` doesn't support errors on the"
                " independent variable(s)\nThey will be ignored if they are present."
            )

        # Ensure that we DO have an X, a Y, and a Y error column.
        if self._data_container.Y_ERROR_COLUMN is None:
            raise ValueError("Likelihood class `GaussianLikelihoodXY` requires a Y error column in the data container.")
        if self._data_container.Y_COLUMN is None:
            raise ValueError("Likelihood class `GaussianLikelihoodXY` requires a Y column in the data container.")
        if self._data_container.X_COLUMN is None:
            raise ValueError("Likelihood class `GaussianLikelihoodXY` requires an X column in the data container.")

        # Check that the model does not have more than 1 independent variable.
        if len(self._model.VARIABLES) > 1:
            raise NotImplementedError(
                "Likelihood class `GaussianLikelihoodXY` only supports models with a single independent variable."
            )

    def _configure(self, **kwargs):
        pass

    # ============================================================ #
    # Data Processing                                              #
    # ============================================================ #
    def _process_input_data(self, **kwargs) -> SimpleNamespace:
        """
        Process the input data.

        For the gaussian likelihood, we require the following in the namespace:

        - ``x``: array-like dict of str, array (shapes (N, ), ...)
            Independent variable data points. These should be in base units.
        - ``y``: array-like (N,)
            Dependent variable data points. These should be in base units.
        - ``y_err``: array-like (N,)
            Uncertainties on the dependent variable data points. These should be in base units. These
            must be complete. We do not accept censored data without uncertainties attached.
        - ``y_upper``: array-like (N,)
            Upper limits on the dependent variable data points. These should be in base units.
            Entries without upper limits should be set to ``np.nan``.
        - ``y_lower``: array-like (N,)
            Lower limits on the dependent variable data points. These should be in base units.
            Entries without lower limits should be set to ``np.nan``.
        - ``y_lower_mask``: array-like (N,)
            Boolean mask indicating which entries in ``y_lower`` are valid lower limits.
        - ``y_upper_mask``: array-like (N,)
            Boolean mask indicating which entries in ``y_upper`` are valid upper limits.

        With these quantities defined, the Gaussian likelihood can be computed
        in the :meth:`_log_likelihood` method. As such, new subclasses of this likelihood
        need only ensure that they correctly implement this and the machinery will be complete.
        """
        # Extract the model units for x and y so that we can ensure
        # unit coercion is done correctly.
        model_x_unit = self._model.VARIABLES[0].base_units
        model_y_unit = self._model.UNITS[0]

        x_units = model_x_unit if model_x_unit != u.dimensionless_unscaled else self._data_container.x.unit
        y_units = model_y_unit if model_y_unit != u.dimensionless_unscaled else self._data_container.y.unit

        # Pull X and Y from the data container. We ensure that they get
        # converted into the correct base units.
        x = ensure_in_units(self._data_container.x, x_units)
        y = ensure_in_units(self._data_container.y, y_units)
        y_err = ensure_in_units(self._data_container.y_error, y_units)

        if self._data_container.Y_UPPER_LIMIT_COLUMN is not None:
            y_upper = ensure_in_units(
                self._data_container.y_upper_limit,
                y_units,
            )
            y_upper_mask = self._data_container.y_upper_lim_mask
        else:
            y_upper = np.full_like(y, np.nan)
            y_upper_mask = np.zeros_like(y, dtype=bool)

        if self._data_container.Y_LOWER_LIMIT_COLUMN is not None:
            y_lower = ensure_in_units(
                self._data_container.y_lower_limit,
                y_units,
            )
            y_lower_mask = self._data_container.y_lower_lim_mask
        else:
            y_lower = np.full_like(y, np.nan)
            y_lower_mask = np.zeros_like(y, dtype=bool)

        # Now combine the processed data into a namespace and return it.
        return SimpleNamespace(
            x={self._model.VARIABLES[0].name: x},
            y=y,
            y_err=y_err,
            y_upper=y_upper,
            y_lower=y_lower,
            y_upper_mask=y_upper_mask,
            y_lower_mask=y_lower_mask,
        )
