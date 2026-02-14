"""
Broken power-law models and associated utilities.

This module provides generic phenomenological curve models commonly used in
astrophysical spectral and time-domain analysis. Implementations include:

- Sharp two-segment broken power laws
- Smoothly broken two-segment power laws
- Sharp three-segment broken power laws
- Smooth three-segment broken power laws

All models:

- Map a single independent variable ``x`` to a single dependent variable ``y``.
- Are compatible with generic XY likelihood frameworks.
- Support unit-aware modeling via the ``Model`` base class.

These forms are widely used for modeling:

- Synchrotron spectra
- GRB afterglows
- TDE fallback curves
- Non-thermal emission processes
- Multi-phase lightcurves
"""

from collections import namedtuple

import numpy as np
from astropy import units as u

from triceratops.models.core import Model, ModelParameter, ModelVariable

__all__ = [
    "BrokenPowerLaw",
    "SmoothedBrokenPowerLaw",
    "TripleBrokenPowerLaw",
    "SmoothedTripleBrokenPowerLaw",
]

# ============================================================
# Shared Output Definitions
# ============================================================
_BPLOutputs = namedtuple("BPLOutputs", ["y"])
_BPLUnits = namedtuple("BPLUnits", ["y"])

# ============================================================
# Sharp Broken Power Law
# ============================================================


class BrokenPowerLaw(Model):
    r"""
    Sharp two-segment broken power-law model.

    This model describes a piecewise power law with a discontinuous derivative
    at a single break location :math:`x_b`.

    The functional form is

    .. math::

        y(x) =
        \begin{cases}
            A \left( \frac{x}{x_b} \right)^{\alpha_1}, & x < x_b \\
            A \left( \frac{x}{x_b} \right)^{\alpha_2}, & x \ge x_b
        \end{cases}

    where:

    - :math:`A` is the normalization defined at :math:`x = x_b`
    - :math:`\alpha_1` is the low-:math:`x` slope
    - :math:`\alpha_2` is the high-:math:`x` slope

    Properties
    ----------

    - Continuous at :math:`x_b`
    - First derivative is discontinuous
    - Computationally inexpensive

    Asymptotic Limits
    -----------------

    For :math:`x \ll x_b`:

    - :math:`y \propto x^{\alpha_1}`

    For :math:`x \gg x_b`:

    - :math:`y \propto x^{\alpha_2}`

    Notes
    -----
    - Canonical phenomenological broken power-law form
    - Frequently used in spectral and lightcurve modeling
    """

    VARIABLES = (
        ModelVariable(
            name="x",
            base_units=u.dimensionless_unscaled,
            description="Independent variable.",
            latex=r"$x$",
        ),
    )

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=u.dimensionless_unscaled, bounds=(0.0, None)),
        ModelParameter("x_b", 1.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("alpha_1", -1.0, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_2", -2.0, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
    )

    OUTPUTS = _BPLOutputs(y="y")
    UNITS = _BPLUnits(y=u.dimensionless_unscaled)

    DESCRIPTION = "Sharp two-segment broken power law."
    REFERENCE = "Standard phenomenological broken power law."

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        A = parameters["A"]
        xb = parameters["x_b"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]

        y = np.empty_like(x)

        mask = x < xb
        y[mask] = A * (x[mask] / xb) ** a1
        y[~mask] = A * (x[~mask] / xb) ** a2

        return {"y": y}


# ============================================================
# Smoothed Broken Power Law
# ============================================================


class SmoothedBrokenPowerLaw(Model):
    r"""
    Smoothly broken two-segment power-law model.

    This model replaces the sharp transition at :math:`x_b`
    with a differentiable break controlled by a smoothing
    parameter :math:`s`.

    The functional form is

    .. math::

        y(x) =
        A \left[
            \left( \frac{x}{x_b} \right)^{\alpha_1 / s}
            +
            \left( \frac{x}{x_b} \right)^{\alpha_2 / s}
        \right]^s

    where:

    - :math:`A` sets the normalization at :math:`x = x_b`
    - :math:`x_b` is the break location
    - :math:`\alpha_1` and :math:`\alpha_2` are the asymptotic slopes
    - :math:`s > 0` controls the sharpness of the transition

    Behavior of the Smoothing Parameter
    ------------------------------------

    - Small :math:`s` → sharper transition
    - Large :math:`s` → broader transition
    - As :math:`s \to 0`, the model approaches the sharp broken power law

    Properties
    ----------

    - Continuous everywhere
    - Differentiable everywhere
    - Suitable for gradient-based inference

    Asymptotic Limits
    -----------------

    For :math:`x \ll x_b`:

    - :math:`y \propto x^{\alpha_1}`

    For :math:`x \gg x_b`:

    - :math:`y \propto x^{\alpha_2}`
    """

    VARIABLES = (ModelVariable("x", base_units=u.dimensionless_unscaled),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=u.dimensionless_unscaled, bounds=(0.0, None)),
        ModelParameter("x_b", 1.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("alpha_1", -1.0, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_2", -2.0, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("s", 0.1, base_units=u.dimensionless_unscaled, bounds=(1e-6, None)),
    )

    OUTPUTS = _BPLOutputs(y="y")
    UNITS = _BPLUnits(y=u.dimensionless_unscaled)

    DESCRIPTION = "Smooth two-segment broken power law."
    REFERENCE = "Standard smooth break prescription."

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        A = parameters["A"]
        xb = parameters["x_b"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        s = parameters["s"]

        term = ((x / xb) ** (a1 / s) + (x / xb) ** (a2 / s)) ** s
        return {"y": A * term}


# ============================================================
# Triple Broken Power Law
# ============================================================


class TripleBrokenPowerLaw(Model):
    r"""
    Three-segment sharp broken power-law model.

    This model consists of three power-law regimes joined
    continuously at two break locations :math:`x_{b,1}` and
    :math:`x_{b,2}`.

    The piecewise form is

    .. math::

        y(x) =
        \begin{cases}
            A \left( \frac{x}{x_{b,1}} \right)^{\alpha_1},
            & x < x_{b,1} \\
            A \left( \frac{x}{x_{b,1}} \right)^{\alpha_2},
            & x_{b,1} \le x < x_{b,2} \\
            A \left( \frac{x_{b,2}}{x_{b,1}} \right)^{\alpha_2}
            \left( \frac{x}{x_{b,2}} \right)^{\alpha_3},
            & x \ge x_{b,2}
        \end{cases}

    where:

    - :math:`A` is defined at :math:`x_{b,1}`
    - :math:`\alpha_1`, :math:`\alpha_2`, :math:`\alpha_3`
      are the slopes in the three regimes

    Properties
    ----------

    - Continuous at both breaks
    - Derivative is discontinuous at both breaks
    - Allows modeling of multi-phase behavior

    Typical Applications
    --------------------

    - Multi-phase GRB afterglows
    - Broken synchrotron spectra
    - TDE fallback phases
    - Composite non-thermal emission processes
    """

    VARIABLES = (ModelVariable("x", base_units=u.dimensionless_unscaled),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=u.dimensionless_unscaled, bounds=(0.0, None)),
        ModelParameter("x_b1", 1.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("x_b2", 10.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("alpha_1", -0.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_2", -1.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_3", -2.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
    )

    OUTPUTS = _BPLOutputs(y="y")
    UNITS = _BPLUnits(y=u.dimensionless_unscaled)

    DESCRIPTION = "Sharp three-segment broken power law."
    REFERENCE = "Standard multi-break phenomenological model."

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        A = parameters["A"]
        b1 = parameters["x_b1"]
        b2 = parameters["x_b2"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        a3 = parameters["alpha_3"]

        y = np.empty_like(x)

        m1 = x < b1
        m2 = (x >= b1) & (x < b2)
        m3 = x >= b2

        y[m1] = A * (x[m1] / b1) ** a1
        y[m2] = A * (x[m2] / b1) ** a2
        y[m3] = A * (b2 / b1) ** a2 * (x[m3] / b2) ** a3

        return {"y": y}


# ============================================================
# Smoothed Triple Broken Power Law
# ============================================================


class SmoothedTripleBrokenPowerLaw(Model):
    r"""
    Smooth three-segment broken power-law model.

    This model introduces differentiable transitions at two
    break locations using a shared smoothing parameter :math:`s`.

    The construction is equivalent to nesting two smoothed
    broken power laws, ensuring continuity and differentiability
    across all regimes.

    Parameters
    ----------
    - :math:`A` – overall normalization
    - :math:`x_{b,1}`, :math:`x_{b,2}` – break locations
    - :math:`\alpha_1`, :math:`\alpha_2`, :math:`\alpha_3` – asymptotic slopes
    - :math:`s` – smoothing parameter controlling break sharpness

    Behavior of the Smoothing Parameter
    ------------------------------------

    - Small :math:`s` → sharper breaks
    - Large :math:`s` → broader transitions
    - As :math:`s \to 0`, the model approaches the sharp triple broken power law

    Properties
    ----------

    - Continuous everywhere
    - Differentiable everywhere
    - Recommended for inference workflows requiring gradients

    Asymptotic Limits
    -----------------

    For :math:`x \ll x_{b,1}`:

    - :math:`y \propto x^{\alpha_1}`

    For :math:`x_{b,1} \ll x \ll x_{b,2}`:

    - :math:`y \propto x^{\alpha_2}`

    For :math:`x \gg x_{b,2}`:

    - :math:`y \propto x^{\alpha_3}`
    """

    VARIABLES = (ModelVariable("x", base_units=u.dimensionless_unscaled),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=u.dimensionless_unscaled, bounds=(0.0, None)),
        ModelParameter("x_b1", 1.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("x_b2", 10.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("alpha_1", -0.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_2", -1.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_3", -2.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("s", 0.1, base_units=u.dimensionless_unscaled, bounds=(1e-6, None)),
    )

    OUTPUTS = _BPLOutputs(y="y")
    UNITS = _BPLUnits(y=u.dimensionless_unscaled)

    DESCRIPTION = "Smooth three-segment broken power law."
    REFERENCE = "Nested smooth break prescription."

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        A = parameters["A"]
        b1 = parameters["x_b1"]
        b2 = parameters["x_b2"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        a3 = parameters["alpha_3"]
        s = parameters["s"]

        t12 = ((x / b1) ** (a1 / s) + (x / b1) ** (a2 / s)) ** s
        t23 = ((x / b2) ** (a2 / s) + (x / b2) ** (a3 / s)) ** s

        y = A * t12 * (t23 / (x / b2) ** a2)

        return {"y": y}
