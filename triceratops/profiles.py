"""
Physical profiles for use in Triceratops models.

This module defines a collection of analytic profile functions commonly used
in astrophysical modeling, including power laws, broken power laws, smoothly
broken power laws, Band spectra, and exponential cutoffs.

All profiles are fully vectorized, NumPy-broadcastable, and safe for use with
scalar inputs, arrays, or higher-dimensional grids.
"""

from typing import Union

import numpy as np

_ArrayLike = Union[float, np.ndarray]


# ============================================================ #
# Power-Law Profiles                                           #
# ============================================================ #
def power_law(
    x: _ArrayLike,
    y0: float = 1.0,
    x_0: float = 1.0,
    a: float = -1.0,
) -> _ArrayLike:
    r"""
    Power-law profile.

    .. math::

        f(x) = y_0 \left( \frac{x}{x_0} \right)^a

    Parameters
    ----------
    x : array_like
        Independent variable.
    y0 : float, optional
        Normalization constant. Default is 1.0.
    x_0 : float, optional
        Reference scale. Default is 1.0.
    a : float, optional
        Power-law index. Default is -1.0.

    Returns
    -------
    array_like
        Power-law evaluated at ``x``.

    Notes
    -----
    - Fully vectorized and NumPy-broadcastable.
    - No bounds checking is performed on ``x``.
    """
    return y0 * (np.asarray(x, dtype=float) / x_0) ** a


# ============================================================ #
# Broken Power-Law Profiles                                    #
# ============================================================ #
def broken_power_law(
    x: _ArrayLike,
    y0: float = 1.0,
    x_0: float = 1.0,
    a_1: float = -1.0,
    a_2: float = -2.5,
) -> _ArrayLike:
    r"""
    Sharp broken power-law profile.

    .. math::

        f(x) =
        \begin{cases}
        y_0 (x/x_0)^{a_1}, & x < x_0 \\
        y_0 (x/x_0)^{a_2}, & x \ge x_0
        \end{cases}

    Parameters
    ----------
    x : array_like
        Independent variable.
    y0 : float, optional
        Normalization constant. Default is 1.0.
    x_0 : float, optional
        Break location. Default is 1.0.
    a_1 : float, optional
        Power-law index below the break.
    a_2 : float, optional
        Power-law index above the break.

    Returns
    -------
    array_like
        Broken power-law evaluated at ``x``.

    Notes
    -----
    - Continuous but not differentiable at ``x_0``.
    - Vectorized using ``np.where``.
    """
    x = np.asarray(x, dtype=float)
    r = x / x_0
    return y0 * np.where(x < x_0, r**a_1, r**a_2)


def smoothed_BPL(
    x: _ArrayLike,
    y0: float = 1.0,
    x_0: float = 1.0,
    a_1: float = -1.0,
    a_2: float = -2.5,
    s: float = 5.0,
) -> _ArrayLike:
    r"""
    Smoothed broken power-law profile.

    .. math::

        f(x) = y_0
        \left[
        \left( \frac{x}{x_0} \right)^{a_1 /s}
        + \left( \frac{x}{x_0} \right)^{a_2 /s}
        \right]^{s}

    Parameters
    ----------
    x : array_like
        Independent variable.
    y0 : float, optional
        Normalization constant.
    x_0 : float, optional
        Break location.
    a_1 : float, optional
        Power-law index below the break.
    a_2 : float, optional
        Power-law index above the break.
    s : float, optional
        Smoothness parameter. Larger values give sharper transitions.

    Returns
    -------
    array_like
        Smoothed broken power-law evaluated at ``x``.

    Notes
    -----
    - Smooth and differentiable everywhere.
    - Reduces to a sharp break as ``s → ∞``.
    """
    x = np.asarray(x, dtype=float)
    r = x / x_0
    return y0 * (r ** (a_1 / s) + r ** (a_2 / s)) ** (s)


# ============================================================ #
# Double Broken Power-Law Profiles                              #
# ============================================================ #
def double_BPL(
    x: _ArrayLike,
    y0: float = 1.0,
    x_1: float = 1.0,
    x_2: float = 10.0,
    a_1: float = 1.0,
    a_2: float = -1.0,
    a_3: float = -2.5,
) -> _ArrayLike:
    """
    Double broken power-law profile with sharp breaks.

    Parameters
    ----------
    x : array_like
        Independent variable.
    y0 : float, optional
        Normalization constant.
    x_1 : float, optional
        First break location.
    x_2 : float, optional
        Second break location.
    a_1 : float
        Power-law index for ``x < x_1``.
    a_2 : float
        Power-law index for ``x_1 <= x < x_2``.
    a_3 : float
        Power-law index for ``x >= x_2``.

    Returns
    -------
    array_like
        Double broken power-law evaluated at ``x``.

    Notes
    -----
    - Continuous but not differentiable at the break points.
    """
    x = np.asarray(x, dtype=float)

    r1 = x / x_1
    r2 = x / x_2

    return y0 * np.where(
        x < x_1,
        r1**a_1,
        np.where(
            x < x_2,
            r1**a_2,
            (x_2 / x_1) ** a_2 * r2**a_3,
        ),
    )


def double_smoothed_BPL(
    x: _ArrayLike,
    y0: float = 1.0,
    x_1: float = 1.0,
    x_2: float = 10.0,
    a_1: float = 1.0,
    a_2: float = -1.0,
    a_3: float = -2.5,
    s1: float = 5.0,
    s2: float = 5.0,
) -> _ArrayLike:
    """
    Double smoothed broken power-law profile.

    Parameters
    ----------
    x : array_like
        Independent variable.
    y0 : float
        Normalization constant.
    x_1, x_2 : float
        Break locations.
    a_1, a_2, a_3 : float
        Power-law indices.
    s1, s2 : float
        Smoothness parameters for each break.

    Returns
    -------
    array_like
        Double smoothed broken power-law evaluated at ``x``.

    Notes
    -----
    - Smooth and differentiable everywhere.
    - Reduces to ``double_BPL`` as ``s1, s2 → ∞``.
    """
    x = np.asarray(x, dtype=float)

    r1 = x / x_1
    r2 = x / x_2

    f1 = (r1 ** (a_1 * s1) + r1 ** (a_2 * s1)) ** (-1.0 / s1)
    f2 = (1.0 + r2 ** ((a_2 - a_3) * s2)) ** (-1.0 / s2)

    return y0 * f1 * f2


# ============================================================ #
# Miscellaneous Profiles                                       #
# ============================================================ #
def band_profile(
    x: _ArrayLike,
    y0: float = 1.0,
    x_0: float = 1.0,
    alpha: float = -1.0,
    beta: float = -2.5,
) -> _ArrayLike:
    """
    Band function profile (GRB spectral shape).

    Parameters
    ----------
    x : array_like
        Independent variable (e.g., photon energy).
    y0 : float
        Normalization constant.
    x_0 : float
        Characteristic energy scale.
    alpha : float
        Low-energy spectral index.
    beta : float
        High-energy spectral index.

    Returns
    -------
    array_like
        Band function evaluated at ``x``.

    Notes
    -----
    - Continuous and differentiable at the break energy.
    - Standard form used in GRB prompt emission modeling.
    """
    x = np.asarray(x, dtype=float)
    x_break = (alpha - beta) * x_0

    prefactor = (alpha - beta) ** (alpha - beta) * np.exp(beta - alpha)

    return y0 * np.where(
        x < x_break,
        (x / x_0) ** alpha * np.exp(-x / x_0),
        prefactor * (x / x_0) ** beta,
    )


def power_law_exp_cutoff(
    x: _ArrayLike,
    y0: float = 1.0,
    x_0: float = 1.0,
    a: float = -1.0,
    x_cut: float = 10.0,
) -> _ArrayLike:
    r"""
    Power law with exponential cutoff.

    .. math::

        f(x) = y_0 \left( \frac{x}{x_0} \right)^a
        \exp\left(-\frac{x}{x_{\mathrm{cut}}}\right)

    Parameters
    ----------
    x : array_like
        Independent variable.
    y0 : float
        Normalization constant.
    x_0 : float
        Reference scale.
    a : float
        Power-law index.
    x_cut : float
        Exponential cutoff scale.

    Returns
    -------
    array_like
        Power law with exponential cutoff evaluated at ``x``.

    Notes
    -----
    Commonly used for particle distributions and synchrotron spectra.
    """
    x = np.asarray(x, dtype=float)
    return y0 * (x / x_0) ** a * np.exp(-x / x_cut)
