"""
Transient system specific models for Triceratops.

These models are specific to transient astrophysical systems such as supernovae
and gamma-ray bursts. Models are further subdivided by the type of transient they are
designed to represent. In some cases, models may be applicable to multiple transient types. In
this case, they are either placed into the :mod:`general` submodule or placed into the module
with which they have the greatest relevance.
"""

__all__ = [
    "GRBs",
    "TDEs",
    "general",
    "supernovae",
    "SynchrotronShockModel",
]

from triceratops.models.transients.general import SynchrotronShockModel

from . import GRBs, TDEs, general, supernovae
