"""
Numerical synchrotron SED engines.

This subpackage provides numerical engines for computing synchrotron SEDs from
arbitrary electron distributions via direct kernel integration. Supports
synchrotron self-absorption, relativistic bulk motion, and multi-zone geometries.

See Also
--------
:mod:`trilobite.radiation.synchrotron.SEDs.numerical.core`
    Base engine class providing kernel tabulation and low-level radiative-transfer machinery.
:mod:`trilobite.radiation.synchrotron.SEDs.numerical.aspherical`
    On-axis axisymmetric (aspherical) multi-zone engine.
:mod:`trilobite.radiation.synchrotron.SEDs.numerical.one_zone`
    Spherical one-zone geometry wrappers and high-level SED classes.
:mod:`trilobite.radiation.synchrotron.SEDs.one_zone`
    Analytic one-zone SED models for power-law electron populations.
"""

from .aspherical import OnAxisAsymmetricSynchrotronEngine
from .core import NumericalSynchrotronEngine
from .one_zone import (
    NonRelativisticSphericalSynchrotronEngine,
    Numerical_PL_SSA_SED,
    Numerical_Thermal_PL_SSA_SED,
    Numerical_Thermal_SSA_SED,
    UltraRelativisticSphericalSynchrotronEngine,
)

__all__ = [
    "NumericalSynchrotronEngine",
    "Numerical_PL_SSA_SED",
    "Numerical_Thermal_SSA_SED",
    "Numerical_Thermal_PL_SSA_SED",
    "NonRelativisticSphericalSynchrotronEngine",
    "UltraRelativisticSphericalSynchrotronEngine",
    "OnAxisAsymmetricSynchrotronEngine",
]
