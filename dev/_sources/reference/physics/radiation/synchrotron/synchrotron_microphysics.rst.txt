.. _synchrotron_microphysics:
=========================================
Synchrotron Microphysics
=========================================

As discussed in :ref:`synchrotron_theory`, it is, in most cases, not possible to uniquely determine the microphysical
processes which govern the synchrotron emission from first principles. Instead, we rely on a set of phenomenological
parameters / closures which encapsulate our ignorance of the detailed physics at play. These microphysical parameters are
critical to the modeling of synchrotron emission, as they directly influence the resulting spectra and light curves.

In this document, we'll describe the available microphysical closures implemented in Triceratops, how to use them,
and provide references for further reading on the subject.

Electron Distributions
----------------------

Synchrotron emission in Triceratops is computed by combining macrophysical dynamical quantities
(e.g. shock velocities, densities, pressures) with a phenomenological model for the population of
relativistic electrons responsible for the radiation.

At present, Triceratops implements a **single, canonical electron distribution**: a power-law in
Lorentz factor,

.. math::

    \frac{dN}{d\gamma} = N_0 \gamma^{-p},
    \qquad
    \gamma_{\min} \le \gamma \le \gamma_{\max},

where:

- :math:`p` is the power-law index,
- :math:`\gamma_{\min}` and :math:`\gamma_{\max}` define the support of the distribution,
- :math:`N_0` is the normalization constant.

This choice is motivated by both theoretical arguments for diffusive shock acceleration and its
empirical success in modeling non-thermal emission from a wide range of astrophysical sources.
A detailed discussion of the physical origin of this distribution can be found in
:ref:`synchrotron_theory`.

.. hint::

    From the perspective of **model building**, the important point is that essentially all synchrotron
    observables depend on *moments* of this distribution rather than on its detailed shape. Thus, there are
    a number of useful functions in the :mod:`radiation.synchrotron` module for computing these moments
    directly without needing to manipulate the distribution itself.

.. note::

    While only a power-law distribution is currently implemented, the modular nature of Triceratops
    makes it straightforward to implement additional electron distributions in the future.
    Contributions from the community are welcome!

.. rubric:: API Reference

*current module*: :mod:`radiation.synchrotron.microphysics`

.. tab-set::

    .. tab-item:: High-Level API

        The following high-level helper functions are provided for working with power-law electron
        distributions:

        .. currentmodule:: triceratops.radiation.synchrotron.microphysics
        .. autosummary::
           :toctree: generated
           :nosignatures:

           compute_electron_gamma_PL_moment
           compute_electron_energy_PL_moment
           compute_mean_gamma_PL
           compute_mean_energy_PL
           compute_PL_total_number_density
           compute_PL_effective_number_density

    .. tab-item:: Low-Level API


        Mirroring the high-level functions, the following low-level functions are provided for
        working directly with the power-law electron distribution:

        - ``_opt_compute_PL_moment``, which computes the generic
          moment of the power-law distribution:

          .. math::

                I = \int_{x_0}^{x_1} x^{(n-p)} dx,

          where :math:`x` is the variable of integration (e.g. :math:`\gamma` or :math:`E`), and
          :math:`n` is the order of the moment.
        - ``_opt_compute_PL_n_total``, which computes the total number of electrons in the power-law
          distribution:

          .. math::

              N_{\rm total} = \int_{\gamma_{\min}}^{\gamma_{\max}} \frac{dN}{d\gamma} d\gamma = N_0 M^{(1)}_\gamma.

        - ``_opt_compute_PL_n_eff``, which computes the effective number of radiating electrons in the
          distribution (see :ref:`synchrotron_theory` for details):

          .. math::

                N_{\rm eff} = N_0 M^{(2)}_\gamma. = N_{\rm total} \frac{M^{(2)}_\gamma}{M^{(1)}_\gamma}.

----

Equipartition Closure
----------------------

The most common closure mechanism used in the literature (and the only primarily used in Triceratops) is the
**equipartition closure**. In this framework, we assume that a fixed fraction of the thermal energy density
behind the shock is partitioned into relativistic electrons and magnetic fields. Specifically, we define two
dimensionless parameters:

- :math:`\epsilon_e`, the fraction of thermal energy in relativistic electrons,
- :math:`\epsilon_B`, the fraction of thermal energy in magnetic fields.

As a result, we can (given knowledge of the shock dynamics) compute the normalization of the electron
distribution and the magnetic field strength directly from the macrophysical quantities. This approach
has been widely used in the literature to model synchrotron emission from a variety of astrophysical
sources (e.g. :footcite:t:`Margutti2019COW`,
:footcite:t:`demarchiRadioAnalysisSN2004C2022`, :footcite:t:`wuDelayedRadioEmission2025`, etc.)

In Triceratops, the helper functions for performing equipartition closure are provided in the
:mod:`radiation.synchrotron.microphysics` module. Before highlighting some of these functions, we'll provide
the API in the tab-set below:

.. tab-set::

    .. tab-item:: High-Level API

        The following high-level helper functions are provided for performing equipartition closure:

        .. currentmodule:: triceratops.radiation.synchrotron.microphysics
        .. autosummary::
           :toctree: generated
           :nosignatures:

           compute_PL_norm_from_magnetic_field
           compute_PL_norm_from_thermal_energy_density
           compute_equipartition_magnetic_field
           compute_bol_emissivity
           compute_bol_emissivity_from_thermal_energy_density

    .. tab-item:: Low-Level API

        Mirroring the high-level functions, the following low-level functions are provided for
        performing equipartition closure:

        - ``_opt_normalize_PL_from_magnetic_field``, which computes the normalization of a power-law
          electron distribution given a magnetic field strength and the equipartition parameter
          :math:`\epsilon_e`.
        - ``_opt_normalize_PL_from_thermal_energy_density``, which computes the normalization of a
          power-law electron distribution given a thermal energy density and the equipartition parameter
          :math:`\epsilon_e`.
        - ``_opt_normalize_energy_PL_from_magnetic_field``, which computes the normalization of a power-law
          electron energy distribution given a magnetic field strength and the equipartition parameter
          :math:`\epsilon_e`.
        - ``_opt_normalize_energy_PL_from_thermal_energy_density``, which computes the normalization of a
          power-law electron energy distribution given a thermal energy density and the equipartition parameter
          :math:`\epsilon_e`.
        - ``_opt_compute_equipartition_magnetic_field``, which computes the magnetic field strength given a
          thermal energy density and the equipartition parameter :math:`\epsilon_B`.
        - ``_opt_compute_bol_emiss_from_magnetic_field``, which computes the bolometric synchrotron emissivity
          given a magnetic field strength and a power-law electron distribution normalization.
        - ``_opt_compute_bol_emiss_from_thermal_energy_density``, which computes the bolometric synchrotron
          emissivity given a thermal energy density and the equipartition parameters :math:`\epsilon_e` and
          :math:`\epsilon_B`.
        - ``_opt_compute_bol_emiss_from_thermal_energy_density_full``, which computes the bolometric synchrotron
          emissivity given a thermal energy density and the equipartition parameters :math:`\epsilon_e` and
          :math:`\epsilon_B`, **without requiring knowledge of the PL normalization**.

The most important of these functions is :func:`compute_PL_norm_from_thermal_energy_density`, which computes
the normalization of a power-law
electron distribution given a thermal energy density and the equipartition parameter :math:`\epsilon_e`. This
is generally the correct way to convert macrophysical quantities into microphysical ones when building
synchrotron models.

For use in models, users should instead use the low-level CGS version of the function
(``_opt_normalize_PL_from_thermal_energy_density``), which is optimized for performance.
