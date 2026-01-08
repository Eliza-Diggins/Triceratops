"""
Dynamical modeling functions for use in Triceratops models.

These modules provide tools for computing the dynamical evolution of
astrophysical transients, including supernovae and gamma-ray bursts. These can be coupled with
radiation processes in the :mod:`radiation` module to produce self-consistent physical models in
the :mod:`models` module.
"""

__all__ = [
    "rankine_hugoniot",
    "shock_engine",
    "supernovae",
]
from triceratops.dynamics import rankine_hugoniot, shock_engine, supernovae
