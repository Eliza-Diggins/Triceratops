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
    "microphysics",
    "frequencies",
    "SEDs",
    "utils",
    "PowerLaw_Cooling_SSA_SynchrotronSED",
    "PowerLaw_Cooling_SynchrotronSED",
    "PowerLaw_SSA_SynchrotronSED",
]

from . import SEDs, frequencies, microphysics, utils
from .SEDs import (
    PowerLaw_Cooling_SSA_SynchrotronSED,
    PowerLaw_Cooling_SynchrotronSED,
    PowerLaw_SSA_SynchrotronSED,
)
