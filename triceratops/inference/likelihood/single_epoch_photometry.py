"""
Likelihood models for single-epoch photometric data in Triceratops.

This module defines likelihood classes designed to compare theoretical
spectral energy distribution (SED) models against single-epoch photometric
observations. Both detections and non-detections (upper limits) are handled
in a statistically consistent manner.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from triceratops.data.photometry import RadioPhotometryContainer
from triceratops.models.core.base import Model

from .base import Likelihood

# Type checking imports
if TYPE_CHECKING:
    from triceratops.models._typing import _ModelParametersInputRaw

# Defining the __all__ variable.
__all__ = [
    "GaussianPhotometryLikelihood",
]


# ============================================================ #
# Likelihood Classes                                           #
# ============================================================ #
class GaussianPhotometryLikelihood(Likelihood):
    r"""
    A Gaussian likelihood model for fitting single-epoch photometric data.

    This likelihood model is designed to compare theoretical spectral energy
    distribution (SED) models against observed photometric data at a single epoch.
    It accounts for both detections and non-detections (upper limits) in a
    statistically consistent manner.

    Notes
    -----
    Formally, the log-likelihood is broken into two components: one for the true detections
    and one for the upper limits. For detected data points, we assume Gaussian errors,
    leading to the standard Gaussian log-likelihood formulation. For non-detections, we
    treat the upper limits using the cumulative distribution function (CDF) of the
    Gaussian distribution.

    The detection component of the log-likelihood is given by:

    .. math::

        \log \mathcal{L}_{\rm det} = -\frac{1}{2} \sum_{i}
        \left[ \frac{(F_{\rm obs, i} - F_{\rm model, i})^2}{\sigma_i^2} +
        \log(2 \pi \sigma_i^2) \right]

    where :math:`F_{\rm obs, i}` is the observed flux density, :math:`F_{\rm model, i}`
    is the model-predicted flux density, and :math:`\sigma_i` is the measurement error. These correspond to the
    ``freq``, ``flux_density``, and ``flux_density_error`` fields in the
    :class:`~data.photometry.RadioPhotometryContainer`.

    The upper limit component of the log-likelihood is given by:

    .. math::

        \log \mathcal{L}_{\rm upper} = \sum_{j}
        \log \left[ \Phi\left( \frac{F_{\rm limit, j} - F_{\rm model, j}}{\sigma_j} \right) \right],

    where :math:`F_{\rm limit, j}` is the upper limit flux density, :math:`F_{\rm model, j}`
    is the model-predicted flux density, :math:`\sigma_j` is the noise level, and :math:`\Phi` is the CDF of
    the standard normal distribution:

    .. math::

        \Phi(x) = \frac{1}{2} \left[ 1 + {\rm erf}\left( \frac{x}{\sqrt{2}} \right) \right].

    **Missing Errors on Non-Detections**:

    It is common for non-detections to lack explicit error estimates. In such cases, we
    infer the noise level from the provided upper limit using a specified number of
    standard deviations, :math:`n_\sigma`. Specifically, if the upper limit is given
    as :math:`F_{\rm limit}`, we estimate the noise level as:

    .. math::

        \sigma = \frac{F_{\rm limit}}{n_\sigma}.

    By default, :math:`n_\sigma = 3`, but this can be adjusted via the
    ``n_sigma_upper_limit`` parameter during initialization. If the ``flux_density_error``
    field is provided for non-detections, any ``np.nan`` fields will be filled with the predicted uncertainty from
    this calculation.
    """

    # ============================================================ #
    # Class Semantics                                              #
    # ============================================================ #

    # TODO: Currently, we are permitting all models to be used with this likelihood.
    #       In the future, we may want to restrict this to only SED-type models, but we
    #       don't currently have any other models.
    COMPATIBLE_MODELS: tuple[type[Model], ...] = (Model,)
    """tuple of :class:`~triceratops.models.core.base.Model`

    Model classes compatible with this likelihood. The provided model instance
    must be an instance of one of these classes.
    """

    COMPATIBLE_DATA_CONTAINERS: tuple[type, ...] = (RadioPhotometryContainer,)
    """tuple of types

    Data container classes compatible with this likelihood. The provided data
    object must be an instance of one of these types.
    """

    # ============================================================ #
    # Initialization                                               #
    # ============================================================ #
    # We overwrite the __init__ to handle specific options for handling of
    # non-detections.
    def __init__(self, model: Model, data, n_sigma_upper_limit: float = 3.0, **kwargs):
        """
        Initialize the :class:`GaussianPhotometryLikelihood` object.

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
        n_sigma_upper_limit : float, optional
            The number of standard deviations above the noise level to consider
            as the upper limit for non-detections. Default is 3.0. See the class notes
            (:class:`GaussianPhotometryLikelihood`) for more details on how upper limits
            are treated in the likelihood.
        **kwargs
            Additional keyword arguments passed to the data processing method. These are unused.

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
        # Pass off to the base class initializer.
        super().__init__(model, data, **kwargs)

        # Store the upper limit sigma level.
        self.n_sigma_upper_limit = n_sigma_upper_limit
        """float: Number of standard deviations above the noise level to consider as the upper limit for non-detections.

        This value is used in the likelihood evaluation to handle non-detections
        appropriately. See the class notes (:class:`GaussianPhotometryLikelihood`) for more details.
        """

    # ============================================================ #
    # Data Processing                                              #
    # ============================================================ #
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
        # Get the model units for the flux and the frequency.
        flux_unit = self._model.UNITS[0]  # Assume first output is flux density (its the only output of the model).
        frequency_unit = self._model["frequency"].base_units

        # Now extract the flux, frequency, error, and upper units from the data container,
        # converting to the model units as we go.
        freq, flux, err, upper_lim = (
            self._data_container.freq.to_value(frequency_unit),
            self._data_container.flux_density.to_value(flux_unit),
            self._data_container.flux_density_error.to_value(flux_unit),
            self._data_container.flux_upper_limit.to_value(flux_unit),
        )

        # Extract the detection mask.
        detection_mask = self._data_container.detection_mask

        # Correct any missing non-detection errors. We do this by building a mask of nan errors and
        # taking the intersection with the non-detection mask.
        missing_errors_mask = np.isnan(err)
        if np.any(missing_errors_mask & detection_mask):
            raise ValueError("Detected data points cannot have missing errors.")
        if np.any(missing_errors_mask & ~detection_mask):
            # Compute the inferred errors.
            inferred_errors = upper_lim[missing_errors_mask & ~detection_mask] / self.n_sigma_upper_limit
            # Fill them in.
            err[missing_errors_mask & ~detection_mask] = inferred_errors

        # Now fill everything into the namespace and return it.
        return SimpleNamespace(
            frequency=freq,
            flux_density=flux,
            flux_density_err=err,
            flux_density_upper_limits=upper_lim,
            detection_mask=detection_mask,
        )

    # ============================================================ #
    # Likelihood Evaluation                                        #
    # ============================================================ #
    def _log_likelihood(
        self,
        parameters: dict[str, "_ModelParametersInputRaw"],
    ) -> float:
        """
        Compute the total log-likelihood for the given model parameters.

        The likelihood is composed of two parts:

        1. **Detections**
           Gaussian likelihood using measured flux densities and errors.

        2. **Non-detections (upper limits)**
           Censored Gaussian likelihood computed via the normal CDF.

        Parameters
        ----------
        parameters : dict
            Dictionary mapping model parameter names to values in base units.

        Returns
        -------
        float
            Log-likelihood value.
        """
        # Allocate the log likelihood variable.
        log_likelihood = 0.0

        # --- Perform the forward modeling step --- #
        # We send parameters down to the model to compute the forward
        # modeling step. This provides the model flux densities at the
        # observed frequencies.
        model_flux_density = self._model._forward_model({"frequency": self._data.frequency}, parameters)["flux_density"]

        # --- Compute the detection log-likelihood component --- #
        # We start with the log-likelihood for the detected bands. This is
        # a standard Gaussian log-likelihood.
        detected_flux = self._data.flux_density[self._data.detection_mask]
        detected_err = self._data.flux_density_err[self._data.detection_mask]
        model_detected_flux = model_flux_density[self._data.detection_mask]

        # Compute the exponential term first.
        _ll = -0.5 * np.sum(((detected_flux - model_detected_flux) / detected_err) ** 2)
        # Now add the normalization term.
        _ll += -0.5 * np.sum(np.log(2 * np.pi * detected_err**2))

        # Accumulate before moving on to the upper limits.
        log_likelihood += _ll

        # --- Handle the upper limit log-likelihood component --- #
        # We use the error function CDF to compute the log-likelihood for the upper limits.
        non_detected_upper_limits = self._data.flux_density_upper_limits[~self._data.detection_mask]
        non_detected_err = self._data.flux_density_err[~self._data.detection_mask]
        model_non_detected_flux = model_flux_density[~self._data.detection_mask]

        _ll = np.sum(norm.logcdf(non_detected_upper_limits, loc=model_non_detected_flux, scale=non_detected_err))
        log_likelihood += _ll
        return log_likelihood
