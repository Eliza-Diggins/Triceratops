"""
Synchrotron Radiation Module.

This module provides tools and functions to model and analyze synchrotron radiation
processes in astrophysical contexts. It includes implementations for calculating
synchrotron emissivity, absorption, and related phenomena based on physical parameters
such as magnetic fields, electron energy distributions, and observational frequencies.
These tools can be integrated with dynamical models of astrophysical transients to
produce comprehensive simulations of radiation from events like supernovae and gamma-ray bursts.
"""

__all__ = ["distributions", "frequencies", "SEDs", "utils"]

from . import SEDs, distributions, frequencies, utils
