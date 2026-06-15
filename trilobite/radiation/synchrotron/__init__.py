"""
Synchrotron Radiation Module.

This module provides tools and functions to model and analyze synchrotron radiation
processes in astrophysical contexts. It includes implementations for calculating
synchrotron emissivity, absorption, and related phenomena based on physical parameters
such as magnetic fields, electron energy distributions, and observational frequencies.
These tools can be integrated with dynamical models of astrophysical transients to
produce comprehensive simulations of radiation from events like supernovae and gamma-ray bursts.
"""

__all__ = [
    "compute_gyrofrequency",
    "compute_nu_critical",
    "compute_synchrotron_frequency",
    "compute_synchrotron_gamma",
    "PowerLaw_SSA_SynchrotronSED",
    "compute_averaged_first_synchrotron_kernel",
    "compute_first_synchrotron_kernel",
    "PowerLaw_Cooling_SSA_SynchrotronSED",
    "PowerLaw_Cooling_SynchrotronSED",
    "InverseComptonCoolingEngine",
    "SynchrotronCoolingEngine",
    "equipartition_magnetic_field",
    "BrokenPowerLaw",
    "PowerLaw",
    "MaxwellJuettner",
    "ElectronDistribution",
]

# Import the various core module items.
from trilobite.radiation.synchrotron.SEDs import (
    PowerLaw_Cooling_SSA_SynchrotronSED,
    PowerLaw_Cooling_SynchrotronSED,
    PowerLaw_SSA_SynchrotronSED,
)

from .cooling import InverseComptonCoolingEngine, SynchrotronCoolingEngine
from .core import (
    compute_averaged_first_synchrotron_kernel,
    compute_first_synchrotron_kernel,
    compute_gyrofrequency,
    compute_nu_critical,
    compute_synchrotron_frequency,
    compute_synchrotron_gamma,
)
from .electron_distributions import (
    BrokenPowerLaw,
    ElectronDistribution,
    MaxwellJuettner,
    PowerLaw,
    equipartition_magnetic_field,
)
