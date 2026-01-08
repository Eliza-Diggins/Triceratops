"""
General transient models for the Triceratops package.

These models are applicable to a wide range of transient astrophysical systems,
including supernovae, gamma-ray bursts, and tidal disruption events. They provide
a phenomenological framework for modeling transient phenomena without being tied
to a specific transient type.
"""

__all__ = [
    "SynchrotronShockModel",
]

from .synchotron_shock import SynchrotronShockModel
