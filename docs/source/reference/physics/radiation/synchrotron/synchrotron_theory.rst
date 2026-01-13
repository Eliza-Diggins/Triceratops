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

The Spectrum of a Power-Law Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We discuss in detail why this is the most common choice for astrophysical sources below.

- Derive the spectrum from a power-law distribution of electrons.
- Introduce the c_5 and c_6 constants.


Microphysical Closures and Equipartition
----------------------------------------

A critical element of most synchrotron modeling pipelines is a treatment of the coupling between the dynamics
of the radiating material and the resulting synchrotron emission. A detailed treatment of this coupling is generally
beyond the scope of most modeling efforts, and so we instead rely on a set of microphysical parameters that
parameterize our ignorance of the detailed physics at play.

Equipartition
^^^^^^^^^^^^^^

The most common approach to microphysical closure is to assume some form of equipartition between the energy
in the magnetic fields and relativistic electrons and the thermal energy of the shocked plasma. This approach
is motivated by the idea that shocks are efficient at converting kinetic energy into thermal energy, and that
some fraction of this thermal energy is then partitioned into magnetic fields and relativistic particles.

A common choice throughout the modern literature (e.g. :footcite:t:`Margutti2019COW`,
:footcite:t:`demarchiRadioAnalysisSN2004C2022`, :footcite:t:`wuDelayedRadioEmission2025`, etc.) is to introduce
the parameters :math:`\epsilon_e` and :math:`\epsilon_B`, which represent the fraction of the thermal energy density
that goes into relativistic electrons and magnetic fields, respectively.

Given a thermal energy density :math:`U_{\rm thermal}`, we can then write the energy densities in relativistic electrons
and magnetic fields as:

.. math::

    U_e = \epsilon_e U_{\rm thermal},

and

.. math::

    U_B = \epsilon_B U_{\rm thermal} = \frac{B^2}{8 \pi},

where :math:`\epsilon_e` and :math:`\epsilon_B` are dimensionless parameters typically in the range
:math:`0 < \epsilon_e, \epsilon_B < 1`, and :math:`U_B` is the magnetic energy density given by

.. math::

    U_B = \frac{B^2}{8 \pi},

and :math:`B` is the magnetic field strength.

.. note::

    The resulting implications for the emission of synchrotron is dictated by the way that :math:`U_e` is distributed
    into the population of relativistic electrons. In the most common scenario, where we have a power-law distribution of
    relativistic electrons, equipartition can uniquely determine the normalization of the distribution given
    :math:`\epsilon_e`, :math:`\epsilon_B`, and :math:`U_{\rm thermal}`.


Power-Law Electron Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A central assumption in most synchrotron emission models is that the population of relativistic
electrons accelerated by a shock follows a power-law distribution in Lorentz factor (and energy). This assumption
is motivated both by theoretical considerations of diffusive shock
acceleration:footcite:p:`caprioliParticleAccelerationShocks2023` and by its empirical success in explaining non-thermal
emission across a wide range of astrophysical environments.

We write the electron distribution function as

.. math::

    \frac{dN}{d\gamma} = N_0 \gamma^{-p},
    \qquad
    \gamma_{\min} \le \gamma \le \gamma_{\max},

where :math:`p` is the power-law index, :math:`N` is the number (density), :math:`\gamma_{\min}`
and :math:`\gamma_{\max}` define the support of the distribution, and :math:`N_0` is a normalization constant.
Outside this range, the distribution is assumed to vanish.

.. note::

    In some instances, authors refer instead to the energy distribution of electrons:

    .. math::

        \frac{dN}{dE} = N_{E,0} E^{-p}.

    Triceratops adopts the **Lorentz factor** formulation as the **canonical standard**; however, both are
    implemented in the relevant API (see :mod:`radiation.synchrotron.microphysics`).

.. note::

    :math:`N` may be either the total number of electrons or the number density of electrons, depending on context.
    Triceratops generally works with number densities to facilitate coupling with hydrodynamical quantities. The
    conversion between the two is just a division by the relevant volume.

Equipartition for Power-Law Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming a power-law distribution of electrons, we can derive normalization of the distribution
from our closure relationship. Here, we describe the procedure when assuming equipartition with some
choice of :math:`\epsilon_e` and :math:`\epsilon_B`.

Given a thermal energy density :math:`U_{\rm thermal}`, the energy density in relativistic electrons is (by
equipartition)

.. math::

    U_e = \epsilon_e U_{\rm thermal}.

The total energy density of the electron population is

.. tab-set::

    .. tab-item:: In Terms of Lorentz Factor

        .. math::

            U_e = m_e c^2 \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma \frac{dN}{d\gamma} d\gamma
            = N_0 m_e c^2 \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{1 - p} d\gamma.

    .. tab-item:: In Terms of Energy

        .. math::

            U_e = \int_{E_{\min}}^{E_{\max}} E \frac{dN}{dE} dE = N_{E,0} \int_{E_{\min}}^{E_{\max}} E^{1 - p} dE.

Letting

.. math::

    M^{(\ell)}_{\gamma} = N_0 \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{\ell} \gamma^{-p} d\gamma,

be the :math:`\ell`-th moment of the Lorentz factor distribution, we can write the energy density as

.. math::

    \boxed{
    U_e = m_e c^2 N_0 M^{(1)}_{\gamma}.
    }

Solving for :math:`N_0`, we find

.. math::

    \boxed{
    N_0 = \frac{\epsilon_e U_{\rm thermal}}{m_e c^2 M^{(1)}_{\gamma}}.
    }

In terms of the magnetic field, we also have

.. math::

    B = \sqrt{8 \pi \epsilon_B U_{\rm thermal}} \implies N_0 = \frac{\epsilon_e B^2}{8 \pi \epsilon_B m_e c^2 M^{(1)}_{\gamma}}.

.. important::

    This is the canonical way to convert dynamics into synchrotron emission in Triceratops when assuming
    equipartition and a power-law distribution of electrons. See :mod:`radiation.synchrotron.microphysics`
    for the relevant API.

Another useful computation which is made possible with equipartition is the **total emitted power** from synchrotron.
From the Larmor formula, we previously derived that the total power of a **single electron** is

.. math::

    P_{\rm synch} = \frac{4}{3} \sigma_T c \gamma^2 \beta^2 U_B.

Thus, the total power from a **population** of electrons in the relativistic limit (:math:`\beta \approx 1`) is

.. math::

    P_{\rm total} = \frac{4}{3} \sigma_T c U_B \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^2 \frac{dN}{d\gamma} d\gamma
    = \frac{4}{3} \sigma_T c U_B N_0 M^{(2)}_{\gamma}.

From equipartition, we know :math:`N_0` in terms of :math:`U_{\rm thermal}` and :math:`\epsilon_e`, :math:`\epsilon_B`, so we can write

.. math::

    \boxed{
    P_{\rm total} = \frac{4}{3} \sigma_T c \frac{\epsilon_e \epsilon_B u_{\rm therm}^2}{m_e c^2} \left(
        \frac{M^{(2)}_{\gamma}}{M^{(1)}_{\gamma}}
    \right).
    }

.. note::

    One frequently uses the fact that :math:`N_{\rm eff} = N_0 M^{(2)}_{\gamma}` is the effective number of
    radiating electrons in the population (see :ref:`synchrotron_seds` for details). This allows us to write

    .. math::

        P_{\rm total} = \frac{4}{3} \sigma_T c U_B N_{\rm eff}.

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

References
-----------
.. footbibliography::
