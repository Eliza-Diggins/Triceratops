"""
Broken power-law models and associated utilities.

These models provide generic phenomenological curve forms commonly used in
astrophysics and time-domain modeling, including sharp and smoothed broken
power laws with two or three segments.

All models map a single independent variable ``x`` to a single dependent
variable ``y`` and are compatible with generic XY likelihoods.
"""

import numpy as np
from astropy import units as u

from triceratops.models.core import Model, ModelParameter, ModelVariable


class BrokenPowerLaw(Model):
    r"""
    Sharp broken power-law model.

    This model describes a two-segment power law with a sharp transition at
    a break location :math:`x_b`:

    .. math::

        y(x) =
        \begin{cases}
            A \left( \frac{x}{x_b} \right)^{\alpha_1}, & x < x_b \\
            A \left( \frac{x}{x_b} \right)^{\alpha_2}, & x \ge x_b
        \end{cases}

    The normalization :math:`A` is defined such that the model is continuous
    at the break.

    Parameters
    ----------
    A : float
        Overall normalization.
    x_b : float
        Break location.
    alpha_1 : float
        Power-law slope below the break.
    alpha_2 : float
        Power-law slope above the break.

    Notes
    -----
    - The transition is *not* differentiable at :math:`x_b`.
    - This form is appropriate when the data resolve a sharp change in slope.

    See Also
    --------
    SmoothedBrokenPowerLaw
        Differentiable alternative with a smooth transition.
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
        ModelParameter(
            name="A",
            default=1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="Normalization evaluated at the break (i.e. y(x_b) = A).",
            latex=r"$A$",
        ),
        ModelParameter(
            name="x_b",
            default=1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="Break location.",
            latex=r"$x_b$",
        ),
        ModelParameter(
            name="alpha_1",
            default=-1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="Slope below the break.",
            latex=r"$\alpha_1$",
        ),
        ModelParameter(
            name="alpha_2",
            default=-2.0,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="Slope above the break.",
            latex=r"$\alpha_2$",
        ),
    )

    OUTPUTS = ("y",)
    UNITS = (u.dimensionless_unscaled,)

    def __init__(self):
        super().__init__()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        A = parameters["A"]
        xb = parameters["x_b"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]

        y = np.empty_like(x)

        mask_lo = x < xb
        mask_hi = ~mask_lo

        y[mask_lo] = A * (x[mask_lo] / xb) ** a1
        y[mask_hi] = A * (x[mask_hi] / xb) ** a2

        return {"y": y}


class SmoothedBrokenPowerLaw(Model):
    r"""
    Smoothed broken power-law model.

    This model replaces the sharp transition with a differentiable smoothing
    parameter :math:`s`:

    .. math::

        y(x) = A \left[
            \left( \frac{x}{x_b} \right)^{\alpha_1 / s}
            +
            \left( \frac{x}{x_b} \right)^{\alpha_2 / s}
        \right]^s

    In the limit :math:`s \to 0`, the model approaches a sharp broken power law.

    Parameters
    ----------
    A : float
        Overall normalization.
    x_b : float
        Break location.
    alpha_1 : float
        Low-:math:`x` slope.
    alpha_2 : float
        High-:math:`x` slope.
    s : float
        Smoothing parameter controlling transition sharpness.

    Notes
    -----
    - The model is continuous and differentiable everywhere.
    - Smaller values of ``s`` produce sharper transitions.
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
        ModelParameter(
            name="A",
            default=1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="Normalization (sets the overall amplitude scale).",
            latex=r"$A$",
        ),
        ModelParameter(
            name="x_b",
            default=1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="Break location.",
            latex=r"$x_b$",
        ),
        ModelParameter(
            name="alpha_1",
            default=-1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="Low-x asymptotic slope.",
            latex=r"$\alpha_1$",
        ),
        ModelParameter(
            name="alpha_2",
            default=-2.0,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="High-x asymptotic slope.",
            latex=r"$\alpha_2$",
        ),
        ModelParameter(
            name="s",
            default=0.1,
            base_units=u.dimensionless_unscaled,
            bounds=(1e-3, 10.0),
            description="Smoothing parameter (smaller → sharper transition).",
            latex=r"$s$",
        ),
    )

    OUTPUTS = ("y",)
    UNITS = (u.dimensionless_unscaled,)

    def __init__(self):
        super().__init__()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        A = parameters["A"]
        xb = parameters["x_b"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        s = parameters["s"]

        t1 = (x / xb) ** (a1 / s)
        t2 = (x / xb) ** (a2 / s)

        return {"y": A * (t1 + t2) ** s}


class TripleBrokenPowerLaw(Model):
    r"""
    Three-segment broken power-law model with two sharp breaks.

    The model consists of three power-law segments joined continuously at
    :math:`x_{b,1}` and :math:`x_{b,2}`.

    Parameters
    ----------
    A : float
        Normalization at the first break.
    x_b1, x_b2 : float
        First and second break locations.
    alpha_1, alpha_2, alpha_3 : float
        Slopes in the three regions.

    Notes
    -----
    - The model is continuous but not differentiable at the breaks.
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
        ModelParameter(
            name="A",
            default=1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="Normalization evaluated at the first break (y(x_b1) = A).",
            latex=r"$A$",
        ),
        ModelParameter(
            name="x_b1",
            default=1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="First break location.",
            latex=r"$x_{b,1}$",
        ),
        ModelParameter(
            name="x_b2",
            default=10.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="Second break location (should be > x_b1 for typical usage).",
            latex=r"$x_{b,2}$",
        ),
        ModelParameter(
            name="alpha_1",
            default=-0.5,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="Slope for x < x_b1.",
            latex=r"$\alpha_1$",
        ),
        ModelParameter(
            name="alpha_2",
            default=-1.5,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="Slope for x_b1 <= x < x_b2.",
            latex=r"$\alpha_2$",
        ),
        ModelParameter(
            name="alpha_3",
            default=-2.5,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="Slope for x >= x_b2.",
            latex=r"$\alpha_3$",
        ),
    )

    OUTPUTS = ("y",)
    UNITS = (u.dimensionless_unscaled,)

    def __init__(self):
        super().__init__()

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


class SmoothedTripleBrokenPowerLaw(Model):
    r"""
    Smoothed three-segment broken power-law.

    This model is constructed as a nested smoothed broken power law,
    ensuring continuity and differentiability across both breaks.

    Parameters
    ----------
    A : float
        Normalization.
    x_b1, x_b2 : float
        Break locations.
    alpha_1, alpha_2, alpha_3 : float
        Power-law slopes.
    s : float
        Smoothing parameter.

    Notes
    -----
    - Reduces to the sharp triple broken power law as ``s → 0``.
    - Numerically stable and differentiable everywhere.
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
        ModelParameter(
            name="A",
            default=1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="Overall normalization.",
            latex=r"$A$",
        ),
        ModelParameter(
            name="x_b1",
            default=1.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="First break location.",
            latex=r"$x_{b,1}$",
        ),
        ModelParameter(
            name="x_b2",
            default=10.0,
            base_units=u.dimensionless_unscaled,
            bounds=(0.0, None),
            description="Second break location.",
            latex=r"$x_{b,2}$",
        ),
        ModelParameter(
            name="alpha_1",
            default=-0.5,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="Low-x asymptotic slope.",
            latex=r"$\alpha_1$",
        ),
        ModelParameter(
            name="alpha_2",
            default=-1.5,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="Intermediate asymptotic slope.",
            latex=r"$\alpha_2$",
        ),
        ModelParameter(
            name="alpha_3",
            default=-2.5,
            base_units=u.dimensionless_unscaled,
            bounds=(-10.0, 10.0),
            description="High-x asymptotic slope.",
            latex=r"$\alpha_3$",
        ),
        ModelParameter(
            name="s",
            default=0.1,
            base_units=u.dimensionless_unscaled,
            bounds=(1e-3, 10.0),
            description="Smoothing parameter controlling break sharpness.",
            latex=r"$s$",
        ),
    )

    OUTPUTS = ("y",)
    UNITS = (u.dimensionless_unscaled,)

    def __init__(self):
        super().__init__()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        A = parameters["A"]
        b1 = parameters["x_b1"]
        b2 = parameters["x_b2"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        a3 = parameters["alpha_3"]
        s = parameters["s"]

        # First smooth transition
        t12 = ((x / b1) ** (a1 / s) + (x / b1) ** (a2 / s)) ** s

        # Second smooth transition
        t23 = ((x / b2) ** (a2 / s) + (x / b2) ** (a3 / s)) ** s

        # Combine
        y = A * t12 * (t23 / (x / b2) ** a2)

        return {"y": y}
