"""Constants for various physical calculations across the physics module."""

import numpy as np
from astropy import constants as const
from astropy import units as u

# ========================================= #
# Synchrotron Radiation Constants           #
# ========================================= #
c_1: u.Quantity = (3 / (4 * np.pi)) * (const.e.esu / const.m_e * const.c)
r"""astropy.units.Quantity: Synchrotron radiation constant :math:`c_1`.

The :math:`c_1` constant is the coefficient appearing in the synchrotron frequency

.. math::

    \nu_c = \frac{3e}{4\pi m_e c} B\sin \alpha \Gamma^2 = c_1 B \sin \alpha \Gamma^2.
"""
c_1_cgs: float = c_1.cgs.value
"""float: Synchrotron radiation constant :math:`c_1` in CGS units."""

electron_rest_energy: u.Quantity = const.m_e * const.c**2
r"""astropy.units.Quantity: Electron rest energy :math:`E_0`."""
electron_rest_energy_cgs: float = electron_rest_energy.cgs.value
"""float: Electron rest energy :math:`E_0` in CGS units."""
