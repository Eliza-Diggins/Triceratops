"""
Generic models of mathematical curves.

These models can be used to perform generic curve fitting tasks.
"""

__all__ = ["BrokenPowerLaw", "SmoothedBrokenPowerLaw", "TripleBrokenPowerLaw", "SmoothedTripleBrokenPowerLaw"]

from .broken_power_law import (
    BrokenPowerLaw,
    SmoothedBrokenPowerLaw,
    SmoothedTripleBrokenPowerLaw,
    TripleBrokenPowerLaw,
)
