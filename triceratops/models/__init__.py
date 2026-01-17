"""
Models of radio emission from astrophysical transients.

This module includes various models for simulating and analyzing the spectral energy distributions (SEDs
of astrophysical transient events such as supernovae, gamma-ray bursts, and tidal disruption events. Models
are distributed across several submodules for better organization and maintainability.
"""

__all__ = ["core", "transients", "emission", "curves"]
from . import core, curves, emission, transients
