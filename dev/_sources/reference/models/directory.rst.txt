.. _model_directory:
=========================================
Triceratops Model Directory
=========================================
Triceratops provides an ever expanding set of models for forward modeling and inversion. The models are organized
in a directory structure that allows users to easily navigate and select models for their specific needs. In this
document, we'll provide a complete listing of all of the different models that are available in Triceratops, along with
links to their respective documentation pages for more detailed information.

.. hint::

    Even if we don't have the exact model you're looking for, many models can be easily adapted or extended to suit
    your requirements. It's always worth looking for a model that can be subclassed or modified to fit your needs.

Curve-Fitting Models
--------------------

The first set of models we'll highlight are the curve-fitting models. These models are designed to fit observational data
using various mathematical functions and techniques. They are particularly useful for analyzing light curves, spectra,
and other time-series data during the early stages of analysis to get a quick understanding of the underlying trends.

.. currentmodule:: triceratops.models.curves

.. autosummary::
   :nosignatures:

   BrokenPowerLawModel
   SmoothedBrokenPowerLawModel

Transient-Specific Models
--------------------------

Supernova Models
^^^^^^^^^^^^^^^^
