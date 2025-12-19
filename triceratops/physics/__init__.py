"""
Physics tools and methods for use in radio modeling.

This module contains most of the pure-physics related tools for Triceratops. This includes
the radiative processes, shock dynamics, and other physical calculations needed for
modeling synchrotron emission from supernova shocks and similar phenomena.

.. note::

    For separation of concerns reasons, this module does **NOT** include any modeling. The methods here
    are simply physics building-blocks which can be combined to generate models elsewhere in the library.
"""

import numpy as np
from astropy import constants
from scipy.special import gamma


def compute_c5_parameter(p: float = 3.0) -> float:
    r"""
    Compute the :math:`c_5(p)` coefficient for synchrotron emission from a power-law population.

    Parameters
    ----------
    p: float, optional
        The power-law index of the electron energy distribution. Default is 3.0.

    Returns
    -------
    float
        The :math:`c_5(p)` coefficient for the given power-law index. This is returned in CGS units.

    Notes
    -----
    For a power-law distribution of electrons such that

    .. math::

        N(\\Gamma) d\\Gamma = K_e \\Gamma^{-p} d\\Gamma,

    the synchrotron emissivity is the integrated power of each electron over the distribution:

    .. math::

        j_\nu = \\int P(\nu, \\Gamma) N(\\Gamma) d\\Gamma =

    Using the synchrotron power per electron and integrating over the power-law distribution
    yields the expression for :math:`j_\nu` in terms of :math:`c_5(p)`:

    .. math::

        j_\nu = c_5(p) K_e B^{(p + 1)/2} \nu^{-(p - 1)/2}.

    """
    # Compute the unit-bearing coefficient.
    c5_0 = np.sqrt(3) / (16 * np.pi) * (constants.e.esu**3 / (constants.m_e * constants.c**2)).cgs.value

    # Compute the dimensionless part.
    dimless_part = (p + 7 / 3) / (p + 1) * gamma((3 * p - 1) / 12) * gamma((3 * p + 7) / 12)

    return c5_0 * dimless_part
