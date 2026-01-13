.. _synchrotron_theory:
===================================
Synchrotron Radiation Theory
===================================

Synchrotron radiation is one of the most fundamental radiative processes in the world of transient astrophysics. It is
responsible for a wide variety of observed phenomena, from the radio afterglows of gamma-ray bursts (GRBs) to the emission from
active galactic nuclei (AGN) jets. Because Triceratops is so heavily reliant on synchrotron radiation for modeling
transient sources, it is important to have a solid understanding of the theory; particularly the conventions and
terminology used throughout the Triceratops documentation.

In this documentation, we seek to provide a concise overview of the topics one needs to understand to begin using
Triceratops for synchrotron modeling. We cannot hope to provide a comprehensive review of synchrotron radiation theory;
however, we provide references throughout to more detailed treatments of the subject and the relevant literature
for various results.

.. contents::
    :local:
    :depth: 2

Synchrotron From a Single Electron
----------------------------------

- some overview notes here, the basic idea. The importance of relativistic beaming.

The Cyclotron Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^

- Introduce the cyclotron frequency and its relevance to synchrotron radiation.
- Introduce the relativistic gyrofrequency.
- Discuss the pitch angle.

Synchrotron Power
^^^^^^^^^^^^^^^^^^^^^

- Use Larmor formula to derive power emitted by a single electron.
- The total power of a population from Larmor.

The Frequency of Synchrotron Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Derive the critical frequency using the pulse-time formalism.
- Introduce the c_1 constant.

The Single Electron Synchrotron Spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Follow the detailed derivation of Rybicki & Lightman to get the single-electron synchrotron spectrum.
- Introduce the synchrotron kernel.
- asymptotic regimes.

Synchrotron From A Population of Electrons
------------------------------------------

- General principles of integration over a distribution of electrons.
- Introduce the distribution function of electron gamma.

Microphysical Closures and Equipartition
----------------------------------------

- Discuss the considerations that lead us to want a closure and the issues with actually
  predicting this ab initio.
- Introduce the concept of equipartition and the microphysical parameters.
- Define epsilon_e and epsilon_B.
- Discuss typical values and references.

Power-Law Electron Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Introduce power-law distributions and the relevant parameters (p, gamma_min, gamma_max).
- Introduce the normalization of the distribution and derive the equations in
  terms of the B field and U_thermal.
- Discuss typical values of p and references.

The Spectrum of a Power-Law Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Derive the spectrum from a power-law distribution of electrons.
- Introduce the c_5 and c_6 constants.

Cooling of Electrons
^^^^^^^^^^^^^^^^^^^^^^^

- How electron cooling can change the distribution.
- Introduce the cooling Lorentz factor.
- Discuss typical regimes (fast vs slow cooling).

Synchrotron Cooling
~~~~~~~~~~~~~~~~~~~

IC Cooling
~~~~~~~~~~

Synchrotron Self-Compton Cooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Absorption Processes in Synchrotron Radiation
---------------------------------------------

Synchrotron Emissivity and Absorption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Free-Free Absorption
~~~~~~~~~~~~~~~~~~~~~~


Spectral Regimes of Synchrotron SEDs
------------------------------------

Introduce and derive each of these spectral regimes in detail, with references.
