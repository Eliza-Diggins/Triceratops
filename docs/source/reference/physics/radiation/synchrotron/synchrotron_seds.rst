.. _synchrotron_seds:
=========================================
Synchrotron Spectral Energy Distributions
=========================================

In Triceratops, synchrotron radiation plays a key role in the emission from various transient sources. The spectral
energy distribution (SED) of this emission depends on a number of factors as described in our document on
`synchrotron theory <synchrotron_theory>`__, and in all its gruesome detail in
`synchrotron SED theory <synch_sed_theory>`__. In this document, we'll discuss how Triceratops implements all of the
various synchrotron SEDs commonly used in the literature and how to use them in your own modeling.

.. important::

    This guide assumes familiarity with the theory of synchrotron radiation as described in the
    :ref:`synchrotron_theory` section of the documentation. Likewise, it is recommended that users read the
    :ref:`synch_sed_theory` document for a more in-depth understanding of the various SEDs implemented here, their
    derivations, and their applications.

.. contents::
    :local:
    :depth: 2

Overview
---------

The core functionality for synchrotron SEDs in Triceratops is encapsulated in the
:mod:`~radiation.synchrotron.SEDs` module.
Triceratops represents synchrotron SEDs as compositions of scale-free, log-space shape functions,
which are assembled into physically meaningful spectra based on frequency ordering and theoretical asymptotes.
These spectra are exposed through lightweight SED classes that optionally implement analytic closure relations,
cleanly separating numerical stability, spectral theory, and physical interpretation. Broadly speaking, there
are 2 levels of abstraction available in the module:

- **Low-Level**:
- **High-Level** (Object Oriented):

The High Level Interface
-------------------------

- High level view of the trickiness of the operation: compute frequencies, order them, select SED shape, normalize,
  etc. (Many functions)! These can be encapsulated in a class.
- Discuss the idea of the high level interface.
- Wrap the various SEDs in relevant classes which isolate the logic of determining
  SED shape from the ordering of break frequencies.
- We choose a class based on the physics we want included (cooling, SSA, etc.)
- Each class implements the details of its normalization, SED shape, etc. as well as certain closures for
  physical parameters where possible.

The Synchrotron SED Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Why we encapsulate the SEDs in classes.
- The core functionality of the class and its intention.
- Encapsulation and simplification.

- Example usage of the class

Available SED Classes
^^^^^^^^^^^^^^^^^^^^^

Power Law SED
~~~~~~~~~~~~~~~~~~

Power Law + Cooling SED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Power Law + SSA SED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Power Law + Cooling + SSA SED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Low-Level Interface
------------------------

The low-level interface provides a more direct way to access the individual components that make up the synchrotron
SEDs. For the most part, this interface is of greatest utility to those defining custom SEDs or wishing to understand
the inner workings of the Triceratops synchrotron SED implementation. The low-level interface is organized into
several key categories of functions:

- **Shape Functions**: These functions define the scale-free shapes of the components of the SEDs. This includes
  the smoothed broken power law (SBPL) shapes used to model transitions between different spectral regimes and
  the scale-free SBPL functions that define allow for smooth transitions between asymptotic power-law segments.
- **SED Functions**: These functions implement the various SED shapes (broken power law, smoothed broken power law,
  etc.) used in the modeling of synchrotron emission. For each physical scenario (e.g., with or without cooling,
  with or without synchrotron self-absorption), there may be a **number of relevant SEDs** that can be constructed
  based on the ordering of characteristic frequencies. Each of these is implemented (at the low level) as a separate
  function.
- **Normalization Functions**: These functions handle the normalization of the SEDs based on physical parameters such as
  the peak flux density, characteristic frequencies, and electron distribution properties.
- **Regime Determination Functions**: These functions determine the ordering of characteristic frequencies and
  select the appropriate SED shape function to use based on the physical scenario being modeled.

These are generally combined to produce the high-level interface for the SED calculations. In the documentation below,
we'll discuss the various functions available in each of these categories.

Shape Functions
^^^^^^^^^^^^^^^

.. currentmodule:: radiation.synchrotron.SEDs

.. rubric:: Shape Function API
.. autosummary::
    :toctree: ../../../../_as_gen
    :nosignatures:

    log_smoothed_BPL
    log_smoothed_SFBPL
    log_exp_cutoff_sed

SED Functions (BPL and SBPL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As described in :ref:`synch_sed_theory`, each physical scenario (e.g., with or without cooling, with or without
synchrotron self-absorption) can give rise to multiple possible SED shapes depending on the ordering of characteristic
frequencies. In Triceratops, these various SED shapes are implemented as individual functions within the
:mod:`~radiation.synchrotron.SEDs` module. Each function corresponds to a specific SED shape and is named
according to the physical scenario it represents and the spectrum number as defined in the theory documentation.

Because there are so many possible SED shapes, we organize adopt a standardized naming convention for each:

.. code-block::

    _log_<electron_pop_type>_<SED_func_type>_sed_<physics tags ...>_<spectrum number>_<truncation tag>,

where:

- `<electron_pop_type>` indicates what type of electron population is being modeled. In all of the current SEDs, this is
  simply ``powerlaw`` to indicate that we are assuming a standard DSA informed power-law distribution of electrons.
- `<SED_func_type>` indicates the type of SED function being used. This is either ``bpl`` for broken power law
  functions or ``sbpl`` for smoothed broken power law functions.
- `<physics tags ...>` is a series of tags indicating the physical processes included in the SED. Currently, the
  following tags are used:

  - ``cool``: Indicates that cooling effects are included in the SED.
  - ``ssa``: Indicates that synchrotron self-absorption effects are included in the SED.

- `<spectrum number>` is an integer indicating which specific spectrum (based on frequency ordering) is being
  implemented. This is directly taken from :ref:`synch_sed_theory`.
- `<truncation tag>` indicates whether or not the SED includes an exponential cutoff at high frequencies. If the
  ``trunc`` tag is present, then the SED function takes an additional :math:`\nu_{\rm max}` parameter
  and applies an exponential cutoff beyond that frequency in accordance with the high frequency behavior described
  in the theory documentation.



SBPL SED Functions
~~~~~~~~~~~~~~~~~~

BPL SED Functions
~~~~~~~~~~~~~~~~~~

Normalization Functions
^^^^^^^^^^^^^^^^^^^^^^^^

Regime Determination Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^







References
----------
.. footbibliography::
