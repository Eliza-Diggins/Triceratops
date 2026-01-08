"""
Inference priors for use in Bayesian inference.

This module provides a collection of commonly used prior probability
distributions for Bayesian inference in astrophysical modeling and
related scientific applications.

All priors are implemented as subclasses of :class:`Prior` and expose
a **callable interface returning the log-prior**, ``log p(x)``, for a
given parameter value. Outside of their support, priors must return
``-np.inf``.

In addition to the log-prior, priors may optionally expose their
probability density function (PDF) and cumulative distribution function
(CDF). These are provided as conveniences for diagnostic, visualization,
and advanced inference workflows (e.g., prior predictive checks or
analytic marginalizations), but they are **not required** for standard
MCMC-based inference.

Usage within Triceratops
------------------------
When defining an :class:`~inference.problem.InferenceProblem`,
users may specify priors in one of two ways:

1. By providing an instance of a :class:`Prior` subclass defined in this
   module.
2. By providing a custom callable that evaluates ``log p(x)`` directly.

The inference machinery treats both options equivalently, enabling
flexibility without sacrificing clarity or performance.

Design principles
-----------------
The design of this module follows a few core principles:

- **Lightweight**: Priors are simple, stateless callable objects.
- **Explicit support**: Every prior explicitly defines its support and
  returns ``-np.inf`` outside of it.
- **Physical parameter space**: Priors operate on physical parameter
  values; parameter transformations are handled elsewhere.
- **Sampler-agnostic**: Priors are compatible with MCMC, nested sampling,
  variational inference, and grid-based approaches.
- **Extensible**: New priors can be implemented by defining a single
  method that returns a log-prior callable.

Improper priors are discouraged and should only be introduced deliberately
and explicitly by the user.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import numpy as np
from scipy.special import erf, gamma

__all__ = [
    "Prior",
    "UniformPrior",
    "LogUniformPrior",
    "NormalPrior",
    "TruncatedNormalPrior",
    "HalfNormalPrior",
    "LogNormalPrior",
    "GammaPrior",
    "BetaPrior",
]


# ============================================================ #
# BASE PRIOR CLASS                                             #
# ============================================================ #
class Prior(ABC):
    """
    Abstract base class for prior probability distributions.

    This class defines the minimal interface required for Bayesian inference
    while allowing optional extensions (PDFs and CDFs) for more advanced
    inference workflows.

    Design philosophy
    -----------------
    - Priors are lightweight, callable objects.
    - The *log-prior* is the primary quantity used during inference.
    - All priors operate in **physical parameter space**.
    - Parameter transformations are handled externally.
    - Probability density (PDF) and cumulative distribution (CDF) are optional.
    - Outside the support, priors must return ``-np.inf``.

    Subclasses are expected to:

    - Accept any required parameters at construction time
    - Implement ``_generate_log_prior``
    - Optionally implement ``_generate_prior`` and ``_generate_cum_prior``

    Notes
    -----
    This class is sampler-agnostic and can be used with MCMC, nested sampling,
    variational inference, or grid-based approaches.
    """

    # ------------------------------------------------------------ #
    # Initialization                                               #
    # ------------------------------------------------------------ #
    def __init__(self, **parameters):
        """
        Initialize the prior distribution.

        Parameters
        ----------
        parameters
            Keyword arguments defining the parameters of the prior distribution.
            These are stored and passed to the generator methods so that subclasses
            may flexibly define their behavior.
        """
        # Store the defining parameters for introspection and debugging.
        self._parameters = parameters

        # Generate the callable interfaces.
        self._log_prior = self._generate_log_prior(**parameters)
        self._prior = self._generate_prior(**parameters)
        self._cum_prior = self._generate_cum_prior(**parameters)

    # ------------------------------------------------------------ #
    # Generator methods                                            #
    # ------------------------------------------------------------ #
    @abstractmethod
    def _generate_log_prior(self, **parameters) -> Callable[[float], float]:
        """
        Generate the log-prior callable.

        This method *must* be implemented by all subclasses.

        Returns
        -------
        callable
            A function ``logp(x)`` that evaluates ``log p(x)`` and returns
            ``-np.inf`` outside the support.
        """
        raise NotImplementedError

    def _generate_prior(self, **parameters) -> Optional[Callable[[float], float]]:
        """
        Generate the probability density function (PDF) callable.

        This method is optional. If not implemented, calling ``pdf`` will
        raise ``NotImplementedError``.

        Returns
        -------
        callable or None
            A function ``p(x)`` evaluating the PDF, or ``None`` if unavailable.
        """
        return None

    def _generate_cum_prior(self, **parameters) -> Optional[Callable[[float], float]]:
        """
        Generate the cumulative distribution function (CDF) callable.

        This method is optional. If not implemented, calling ``cdf`` will
        raise ``NotImplementedError``.

        Returns
        -------
        callable or None
            A function ``P(X ≤ x)``, or ``None`` if unavailable.
        """
        return None

    # ------------------------------------------------------------ #
    # Public evaluation interface                                  #
    # ------------------------------------------------------------ #
    def logp(self, x: float) -> float:
        """
        Evaluate the log-prior at a given value.

        Parameters
        ----------
        x : float
            Parameter value in physical units.

        Returns
        -------
        float
            Log-prior value.
        """
        return self._log_prior(x)

    def pdf(self, x: float) -> float:
        """
        Evaluate the probability density function at a given value.

        Raises
        ------
        NotImplementedError
            If the PDF is not defined for this prior.
        """
        if self._prior is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not define a PDF.")
        return self._prior(x)

    def cdf(self, x: float) -> float:
        """
        Evaluate the cumulative distribution function at a given value.

        Raises
        ------
        NotImplementedError
            If the CDF is not defined for this prior.
        """
        if self._cum_prior is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not define a CDF.")
        return self._cum_prior(x)

    # ------------------------------------------------------------ #
    # Dunder methods                                               #
    # ------------------------------------------------------------ #
    def __call__(self, x: float) -> float:
        """
        Alias for :meth:`logp`.

        This allows prior instances to be used as callable log-priors.
        """
        return self.logp(x)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self._parameters.items())
        return f"{self.__class__.__name__}({params})"

    def __str__(self) -> str:
        if self._parameters:
            params = ", ".join(f"{k}={v}" for k, v in self._parameters.items())
            return f"{self.__class__.__name__} with parameters ({params})"
        return f"{self.__class__.__name__}"

    # ------------------------------------------------------------ #
    # Serialization support                                        #
    # ------------------------------------------------------------ #
    @staticmethod
    def _all_subclasses():
        """Recursively yield all subclasses of Prior."""
        work = list(Prior.__subclasses__())
        seen = set(work)

        while work:
            cls = work.pop()
            yield cls
            for sub in cls.__subclasses__():
                if sub not in seen:
                    seen.add(sub)
                    work.append(sub)

    def to_dict(self) -> dict:
        """
        Serialize the prior to a dictionary.

        Returns
        -------
        dict
            A JSON-serializable representation of the prior.
        """
        return {
            "prior_class": self.__class__.__name__,
            "parameters": self._parameters,
            "reconstructible": True,
        }

    @classmethod
    def from_dict(cls, prior_dict: dict) -> "Prior":
        """
        Load a prior from a serialized dictionary.

        Parameters
        ----------
        prior_dict : dict
            A dictionary representation of the prior, as produced by
            :meth:`to_dict`.
        """
        try:
            class_name = prior_dict["prior_class"]
            parameters = prior_dict.get("parameters", {})
        except KeyError as exc:
            raise ValueError("Invalid prior serialization dictionary.") from exc

        if not prior_dict.get("reconstructible", True):
            raise ValueError(f"Prior '{class_name}' is not reconstructible and must be recreated manually.")

        # Find subclass recursively
        for subclass in cls._all_subclasses():
            if subclass.__name__ == class_name:
                return subclass(**parameters)

        raise ValueError(f"Unknown prior class '{class_name}'.")

    def to_string(self) -> str:
        """Serialize the prior to a JSON string."""
        import json

        return json.dumps(self.to_dict())

    @classmethod
    def from_string(cls, prior_string: str) -> "Prior":
        """Deserialize a prior from a JSON string."""
        import json

        prior_dict = json.loads(prior_string)
        return cls.from_dict(prior_dict)


class CallablePrior(Prior):
    """
    Wrapper for user-supplied log-prior callables.

    This allows custom priors without requiring users to subclass `Prior`,
    while preserving introspection and serialization metadata.
    """

    def __init__(self, log_prior_callable: Callable[[float], float], parameters: Optional[dict] = None):
        # Ensure that the callable is valid.
        if not callable(log_prior_callable):
            raise ValueError("log_prior_callable must be callable.")

        # Store the callable and metadata.
        self._user_log_prior = log_prior_callable
        super().__init__(**(parameters or {}))

    def _generate_log_prior(self, **parameters) -> Callable[[float], float]:
        # Return the user-supplied callable directly.
        return self._user_log_prior

    # ------------------------------------------------------------ #
    # Serialization support                                        #
    # ------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """
        Serialize the callable prior to a dictionary.

        Note: The actual callable cannot be serialized. This method
        only captures metadata.

        Returns
        -------
        dict
            A JSON-serializable representation of the prior.
        """
        return {
            "prior_class": "CallablePrior",
            "parameters": self._parameters,
            "reconstructible": False,
            "note": "This prior wraps a user-supplied callable and cannot be fully reconstructed.",
        }

    @classmethod
    def from_dict(cls, prior_dict):
        raise NotImplementedError("CallablePrior cannot be reconstructed from serialized form.")

    def __repr__(self):
        return f"CallablePrior(parameters={self._parameters}, reconstructible=False)"


# ============================================================ #
# UNIFORM PRIORS                                               #
# ============================================================ #
class UniformPrior(Prior):
    r"""
    Uniform (top-hat) prior distribution.

    This prior assigns equal probability density to all values within a
    finite interval and zero probability outside it.

    Mathematically, the probability density is

    .. math::

        P(x) =
        \begin{cases}
        \dfrac{1}{b - a} & a \le x \le b \\
        0 & \text{otherwise}
        \end{cases}

    where :math:`a` is the lower bound and :math:`b` is the upper bound.

    Support
    -------
    :math:`x \in [a,b]`

    Interpretation
    --------------
    The uniform prior represents *complete prior ignorance* **within a
    specified range**, but it still encodes strong assumptions:

    - Values outside the bounds are *impossible*
    - All values inside the bounds are equally likely

    Typical uses
    ------------
    - Poorly constrained parameters with known physical limits
    - Bounding parameters to prevent unphysical solutions
    - Simple exploratory modeling

    Caveats
    -------
    - Uniform priors are **not invariant under reparameterization**
      (e.g., uniform in ``x`` is not uniform in ``log x``).
    - For scale parameters spanning orders of magnitude, a log-uniform
      prior is usually more appropriate.
    """

    def __init__(self, lower: float, upper: float):
        if upper <= lower:
            raise ValueError("UniformPrior requires upper > lower.")
        super().__init__(lower=float(lower), upper=float(upper))

    def _generate_log_prior(self, *, lower: float, upper: float):
        log_norm = -np.log(upper - lower)

        def logp(x: float) -> float:
            if lower <= x <= upper:
                return log_norm
            return -np.inf

        return logp


class LogUniformPrior(Prior):
    r"""
    Log-uniform (Jeffreys-type) prior distribution.

    This prior assigns equal probability per decade in the parameter,
    making it invariant under changes of scale.

    Mathematically, the density is

    .. math::

        P(x) =
        \begin{cases}
        \dfrac{1}{x \ln(b/a)} & a \le x \le b \\
        0 & \text{otherwise}
        \end{cases}

    where :math:`a > 0` and :math:`b > a`.

    Support
    -------
    :math:`x \in [a,b]` , with ``x > 0``

    Interpretation
    --------------
    This prior expresses ignorance over *orders of magnitude* rather than
    linear differences. Equal probability mass is assigned to intervals
    like ``[1, 10]`` and ``[10, 100]``.

    Typical uses
    ------------
    - Normalization parameters
    - Scale parameters (luminosity, mass, density)
    - Any strictly positive quantity spanning many decades

    Caveats
    -------
    - Requires explicit lower and upper bounds
    - Diverges as :math:`x\to 0` without a lower cutoff
    - Should not be used for parameters that can take zero or negative values
    """

    def __init__(self, lower: float, upper: float):
        if lower <= 0 or upper <= 0:
            raise ValueError("LogUniformPrior bounds must be > 0.")
        if upper <= lower:
            raise ValueError("LogUniformPrior requires upper > lower.")
        super().__init__(lower=float(lower), upper=float(upper))

    def _generate_log_prior(self, *, lower: float, upper: float):
        log_norm = -np.log(np.log(upper / lower))

        def logp(x: float) -> float:
            if lower <= x <= upper:
                return log_norm - np.log(x)
            return -np.inf

        return logp


# ============================================================ #
# GAUSSIAN PRIORS                                              #
# ============================================================ #
class NormalPrior(Prior):
    r"""
    Normal (Gaussian) prior distribution.

    This prior assumes that the parameter follows a normal distribution
    with specified mean and standard deviation.
    Mathematically,

    .. math::

        P(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left[
        -\frac{(x - \mu)^2}{2\sigma^2}
        \right]

    """

    def __init__(self, mean: float, sigma: float):
        if sigma <= 0:
            raise ValueError("NormalPrior requires sigma > 0.")
        super().__init__(mean=float(mean), sigma=float(sigma))

    def _generate_log_prior(self, *, mean: float, sigma: float):
        log_norm = -0.5 * np.log(2.0 * np.pi * sigma**2)

        def logp(x: float) -> float:
            return log_norm - 0.5 * ((x - mean) / sigma) ** 2

        return logp


class TruncatedNormalPrior(Prior):
    r"""
    Truncated normal (Gaussian) prior distribution.

    This prior is a normal distribution restricted to a finite interval
    and renormalized accordingly.

    Mathematically,

    .. math::

        P(x) =
        \begin{cases}
        \dfrac{1}{Z}
        \exp\left[
        -\dfrac{(x - \mu)^2}{2\sigma^2}
        \right]
        & a \le x \le b \\
         0 & \text{otherwise}
        \end{cases}

    where :math:`Z` is the normalization constant.

    Support
    -------
    :math:`x \in [a,b]`

    Interpretation
    --------------
    This prior combines informative Gaussian beliefs with hard physical
    bounds.

    Typical uses
    ------------
    - Parameters with known physical limits
    - Regularizing weakly constrained fits
    - Quantities that must remain finite or positive

    Caveats
    -------
    - Behavior near the boundaries can influence inference
    - Poorly chosen bounds can introduce bias
    """

    def __init__(self, mean: float, sigma: float, lower: float, upper: float):
        if sigma <= 0:
            raise ValueError("TruncatedNormalPrior requires sigma > 0.")
        if upper <= lower:
            raise ValueError("TruncatedNormalPrior requires upper > lower.")
        super().__init__(
            mean=float(mean),
            sigma=float(sigma),
            lower=float(lower),
            upper=float(upper),
        )

    def _generate_log_prior(self, *, mean, sigma, lower, upper):
        alpha = (lower - mean) / (np.sqrt(2) * sigma)
        beta = (upper - mean) / (np.sqrt(2) * sigma)
        Z = 0.5 * (erf(beta) - erf(alpha))

        if Z <= 0:
            raise ValueError("Invalid truncation range for TruncatedNormalPrior.")

        log_norm = -np.log(Z * np.sqrt(2.0 * np.pi) * sigma)

        def logp(x: float) -> float:
            if lower <= x <= upper:
                return log_norm - 0.5 * ((x - mean) / sigma) ** 2
            return -np.inf

        return logp


class HalfNormalPrior(Prior):
    r"""
    Half-normal prior distribution.

    This is a normal distribution restricted to non-negative values.

    Mathematically,

    .. math::

        P(x) =
        \begin{cases}
        \dfrac{\sqrt{2}}{\sqrt{\pi}\sigma}
        \exp\left(
        -\dfrac{x^2}{2\sigma^2}
        \right)
        & x \ge 0 \\
            0 & \text{otherwise}
        \end{cases}

    Support
    -------
    :math:`x \ge 0`

    Interpretation
    --------------
    The half-normal prior favors small positive values while allowing
    occasional large excursions.

    Typical uses
    ------------
    - Noise amplitudes
    - Scatter or variance parameters
    - Regularization terms

    Caveats
    -------
    - Strongly informative near zero
    - Scale parameter must be chosen carefully
    """

    def __init__(self, sigma: float):
        if sigma <= 0:
            raise ValueError("HalfNormalPrior requires sigma > 0.")
        super().__init__(sigma=float(sigma))

    def _generate_log_prior(self, *, sigma: float):
        log_norm = np.log(2.0) - 0.5 * np.log(2.0 * np.pi * sigma**2)

        def logp(x: float) -> float:
            if x >= 0:
                return log_norm - 0.5 * (x / sigma) ** 2
            return -np.inf

        return logp


# ============================================================ #
# LOG-NORMAL PRIORS                                            #
# ============================================================ #
class LogNormalPrior(Prior):
    r"""
    Log-normal prior distribution.

    This prior assumes that the logarithm of the parameter is normally
    distributed.

    Mathematically,

    .. math::

        \ln x \sim \mathcal{N}(\mu,\sigma^2)

    or equivalently,

    .. math::

        P(x) =
        \frac{1}{x\sqrt{2\pi}\sigma}
        \exp\left[
        -\frac{(\ln x - \mu)^2}{2\sigma^2}
        \right]

    Support
    -------
    :math:`x > 0`

    Interpretation
    --------------
    The log-normal prior models multiplicative uncertainty and naturally
    describes quantities spanning orders of magnitude.

    Typical uses
    ------------
    - Scale parameters with asymmetric uncertainty
    - Physical quantities constrained to be positive
    - Growth or decay processes

    Caveats
    -------
    - Mean and variance are not intuitive in linear space
    - Strongly asymmetric for large :math:`\sigma`
    """

    def __init__(self, mean: float, sigma: float):
        if sigma <= 0:
            raise ValueError("LogNormalPrior requires sigma > 0.")
        super().__init__(mean=float(mean), sigma=float(sigma))

    def _generate_log_prior(self, *, mean: float, sigma: float):
        log_norm = -0.5 * np.log(2.0 * np.pi * sigma**2)

        def logp(x: float) -> float:
            if x > 0:
                y = np.log(x)
                return log_norm - np.log(x) - 0.5 * ((y - mean) / sigma) ** 2
            return -np.inf

        return logp


# ============================================================ #
# GAMMA PRIORS                                                 #
# ============================================================ #
class GammaPrior(Prior):
    r"""
    Gamma prior distribution.

    The Gamma distribution is defined as

    .. math::

        P(x) =
        \begin{cases}
        \dfrac{1}{\Gamma(k)\theta^k}
        x^{k-1} \exp(-x/\theta)
        & x > 0 \\
        0 & \text{otherwise}
        \end{cases}

    where :math:`k > 0` is the shape parameter and :math:`\theta > 0`
    is the scale parameter.

    Support
    -------
    :math:`x > 0`

    Interpretation
    --------------
    The Gamma prior is flexible and can interpolate between exponential-like
    and approximately Gaussian behavior.

    Typical uses
    ------------
    - Rates and intensities
    - Variance or precision parameters
    - Hierarchical scale parameters

    Caveats
    -------
    - Shape and scale parameters may be unintuitive
    - Can become highly informative if poorly chosen
    """

    def __init__(self, shape: float, scale: float):
        if shape <= 0 or scale <= 0:
            raise ValueError("GammaPrior requires shape > 0 and scale > 0.")
        super().__init__(shape=float(shape), scale=float(scale))

    def _generate_log_prior(self, *, shape: float, scale: float):
        log_norm = -shape * np.log(scale) - np.log(gamma(shape))

        def logp(x: float) -> float:
            if x > 0:
                return log_norm + (shape - 1.0) * np.log(x) - x / scale
            return -np.inf

        return logp


# ============================================================ #
# BETA PRIORS                                                  #
# ============================================================ #
class BetaPrior(Prior):
    r"""
    Beta prior distribution.

    This prior is defined on the unit interval.

    Mathematically,

    .. math::

        P(x) =
        \begin{cases}
        \dfrac{1}{B(\alpha,\beta)}
        x^{\alpha - 1} (1 - x)^{\beta - 1}
        & 0 \le x \le 1 \\
        0 & \text{otherwise}
        \end{cases}

    where :math:`\alpha > 0` and :math:`\beta > 0`.

    Support
    -------
    :math:`x \in [0,1]`

    Interpretation
    --------------
    The Beta prior is highly flexible and can represent uniform, peaked,
    or boundary-favoring beliefs.

    Typical uses
    ------------
    - Efficiencies
    - Fractions
    - Probabilities

    Caveats
    -------
    - Singular behavior at boundaries when :math:`\alpha < 1` or
      :math:`\beta < 1`
    - Must not be used outside the unit interval
    """

    def __init__(self, alpha: float, beta: float):
        if alpha <= 0 or beta <= 0:
            raise ValueError("BetaPrior requires alpha > 0 and beta > 0.")
        super().__init__(alpha=float(alpha), beta=float(beta))

    def _generate_log_prior(self, *, alpha: float, beta: float):
        log_norm = np.log(gamma(alpha + beta)) - np.log(gamma(alpha)) - np.log(gamma(beta))

        def logp(x: float) -> float:
            if 0.0 <= x <= 1.0:
                return log_norm + (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log(1.0 - x)
            return -np.inf

        return logp
