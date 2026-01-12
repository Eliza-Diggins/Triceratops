"""
Likelihood structures for use in Triceratops model inference.

This module provides classes and functions to define and compute likelihoods
for comparing astrophysical models to observational data. It includes support
for various types of likelihoods, including Gaussian, Poisson, and custom
likelihoods tailored to specific data types.
"""

__all__ = [
    "Likelihood",
    "GaussianPhotometryLikelihood",
]
from .base import Likelihood
from .single_epoch_photometry import GaussianPhotometryLikelihood
