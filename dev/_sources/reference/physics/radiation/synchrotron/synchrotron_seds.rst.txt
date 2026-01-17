.. _synchrotron_seds:
=========================================
Synchrotron Spectral Energy Distributions
=========================================

In Triceratops, synchrotron radiation plays a key role in the emission from various transient sources. The spectral
energy distribution (SED) of this emission depends on a number of factors as described in the documentation on
`synchrotron theory <synchrotron_theory>`__. In this document, we'll discuss the various SEDs that are implemented
in Triceratops and the tools available to work with them when building models.

.. important::

    This guide assumes familiarity with the theory of synchrotron radiation as described in the
    :ref:`synchrotron_theory` section of the documentation.

.. contents::
    :local:
    :depth: 2

Overview
---------

The observed SED from synchrotron radiation is, in general, a complicated function of the underlying dynamics, the
constituent microphysics, and the environment surrounding the source. It is therefore **NOT** the intention of this
module to provide an exhaustive implementation of **ALL** possible synchrotron SEDs. Nonetheless, there are a number
of SEDs which appear frequently in the literature and are widely application to use in supernova modeling, GRBs, and
TDEs. We therefore provide implementations of these commonly used SEDs.

For a given SED, :mod:`~radiation.synchrotron.SEDs` provides (a) functions for simply calculating the expected
flux as a function of frequency and (b), when possible, closure relations which allow for the calculation of the
spectral parameters from the underlying dynamics and microphysics or vice-versa.

The Synchrotron SED Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

Because of the variety of SEDs which arise in practice and the fact that they are easily confused with one
another, Triceratops opts for a class-based SED implementation. Each SED is a subclass of the
:class:`~radiation.synchrotron.SEDs.SynchrotronSED` base class which provides a common interface for
calculating fluxes and spectral parameters. Each subclass implements the specific details of the SED it represents.

Each SED class has a number of important methods which can be used when constructing models and
complies with Triceratops' standard public / private API design. The central functionality of the SED class is to

1. Provide the generic SED function :math:`F_\nu(\nu) = f(\nu, \boldsymbol{\Theta})`, where
   :math:`\boldsymbol{\Theta}` is the set of spectral parameters for the SED. These are generally things like
   :math:`\nu_c`, :math:`\nu_m`, :math:`\nu_a`, and :math:`F_{\nu, \mathrm{max}}`.
2. Provide bi-directional closure methods to relate the *spectral* parameters (critical fluxes, normalization, etc.)
   to the *physical* parameters (energy, density, microphysical parameters, etc.) of the system.

A particular SED class may implement many closures or none at all, depending on the intent of the SED and
its implementation.

.. important::

    **Developer Note**: For documentation regarding implementing custom SEDs, please see the
    class documentation ::class:`~radiation.synchrotron.SEDs.SynchrotronSED`.

.. rubric:: Synchrotron SED API

As with all of Triceratops, each SED class provides both a high-level and low-level API for interacting with it.

.. currentmodule:: radiation.synchrotron.SEDs

.. tab-set::

    .. tab-item:: High Level API

        At the high level, each SED class provides :meth:`SynchrotronSED.sed` method which calculates the flux
        density at a given frequency for a set of spectral parameters. This method is designed to be used in model construction and
        provides a simple interface for calculating fluxes.

        Example usage:

        .. code-block:: python

            from radiation.synchrotron.SEDs import ExampleSED

            # Define spectral parameters
            spectral_params = {
                'nu_c': 1e14,  # Cooling frequency in Hz
                'nu_m': 1e12,  # Minimum frequency in Hz
                'nu_a': 1e10,  # Self-absorption frequency in Hz
                'F_nu_max': 1.0  # Maximum flux density in mJy
            }

            # Create an instance of the FastCoolingSED class
            sed = ExampleSED()

            # Calculate flux density at a specific frequency
            frequency = 1e11  # Frequency in Hz
            flux_density = sed.sed(frequency, **spectral_params)
            print(f"Flux Density at {frequency} Hz: {flux_density} mJy")

        If available, these classes will also implement :meth:`SynchrotronSED.from_params_to_physics` and
        :meth:`SynchrotronSED.from_physics_to_params` methods for converting between spectral and physical parameters.
        These are not required to be implemented in every SED class, however.

    .. tab-item:: Low Level API

        At the low level, each SED class provides a method for computing the flux density directly without
        unit coercion or validation. This method is named ``_opt_sed`` and is intended for internal use or for developers
        who need more control over the calculation. Likewise, the forward and backward parameter closures are
        implemented as ``_opt_from_params_to_physics`` and ``_opt_from_physics_to_params`` methods.

Available SEDs
---------------

While it may be necessary for some users to implement custom SEDs for their specific applications, Triceratops
provides a number of commonly used SEDs out-of-the-box. These include:

- :class:`~radiation.synchrotron.SEDs.SSA_SED_PowerLaw`: The de-facto standard synchrotron SED with self-absorption
  included for power-law distributed electrons (see e.g. :footcite:t:`demarchiRadioAnalysisSN2004C2022`).


In depth descriptions of each of the SED classes, including references and derivation of their forms, can be found
in their respective class documentation.

References
----------
.. footbibliography::
