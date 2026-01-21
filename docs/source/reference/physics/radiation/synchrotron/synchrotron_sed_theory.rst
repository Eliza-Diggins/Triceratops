.. _synch_sed_theory:
===========================================
Theory of Synchrotron SEDs
===========================================

Synchrotron radiation is one of the most powerful and widely used diagnostics of
non-thermal processes in astrophysics. From radio emission in supernova remnants
and relativistic jets to broadband afterglows of gamma-ray bursts and fast radio
transients, observed spectra are often interpreted through the lens of
synchrotron theory. However, while the underlying microphysics of synchrotron
emission from individual electrons is well understood, the construction of
realistic **synchrotron spectral energy distributions (SEDs)** is a substantially
more involved problem.

The observed SED of a synchrotron source is not determined by a single physical
process, but rather by the **interplay between multiple ingredients**: the energy
distribution of relativistic electrons, the magnetic field strength and geometry,
radiative and adiabatic cooling, and absorption processes within the emitting
region. Each of these elements introduces characteristic spectral features—such
as breaks, cutoffs, and turnovers—that together define the shape of the emergent
spectrum.

A further complication arises from the fact that astrophysical synchrotron
sources are rarely static. Electron populations evolve in time due to continuous
injection and energy losses, magnetic fields may decay or be amplified, and the
emitting plasma may expand or decelerate. As a result, synchrotron SEDs are often
**time-dependent**, with break frequencies and normalizations that evolve over
dynamical timescales. In many contexts, especially for transients, the observed
spectrum represents only a snapshot of this ongoing evolution.

Over the past several decades, a small number of “canonical” synchrotron SED
regimes have emerged as effective phenomenological descriptions of these systems.
These regimes—commonly labeled by the ordering of characteristic frequencies such
as the synchrotron peak, cooling break, and self-absorption turnover—provide a
compact framework for interpreting observations and connecting them to physical
parameters. Despite their apparent simplicity, these spectral forms encode a
rich set of assumptions about electron injection, cooling, and radiative transfer.

In this section, we synthesize the theoretical results developed in previous
chapters into a coherent framework for understanding and constructing synchrotron
SEDs. We begin by reviewing the critical physical ingredients that shape the
spectrum, including cooling and absorption processes. We then introduce the
characteristic break frequencies that organize the spectrum and discuss how their
ordering defines the classic synchrotron regimes. Finally, we derive the standard
spectral forms for single electrons and power-law electron populations, leading
to the five canonical synchrotron SEDs commonly used in astrophysical modeling.

Throughout this discussion, we emphasize physical intuition, scaling relations,
and the assumptions underlying each result, while providing references to the
foundational literature (e.g. :footcite:t:`RybickiLightman`,
:footcite:t:`1970ranp.book.....P`, :footcite:t:`GranotSari2002SpectralBreaks`,
:footcite:t:`Chevalier1998SynchrotronSelfAbsorption`) where more
detailed derivations may be found.

.. contents::

Review of Critical Results
--------------------------

.. hint::

    For a more detailed discussion of the physics of synchrotron radiation, including
    derivations of key results, see :ref:`synchrotron_theory`.

Before we proceed with a detailed treatment of the synchrotron SEDs, we briefly
review several critical results from previous sections that will be used
throughout this discussion. The most important of these results is the power
of a single electron radiating in a magnetic field, given by:

.. math::

    \boxed{
    P(\nu) = \frac{\sqrt{3}q^3 B \sin\alpha}{m c^2} F\left(\frac{\nu}{\nu_{\rm char}}\right),
    }

where :math:`F(x)` is the **synchrotron kernel function** and :math:`\nu_{\rm char}` is the
**characteristic synchrotron frequency** of the electron:

.. math::

    \boxed{
    \nu_{\rm char} = \frac{3 q B \gamma^2 \sin\alpha}{4 \pi m c}.
    }

For a population of electrons, the corresponding emissivity is given by integrating
over the electron energy distribution:

.. math::

    \boxed{
    j_\nu = \frac{1}{4\pi} \int d\gamma \, n(\gamma) P(\nu, \gamma).
    }

For a power-law distribution of electrons, :math:`N(\gamma) = N_0 \gamma^{-p}` (in :math:`\gamma` or :math:`E`), the
resulting emissivity is also a power-law in frequency:

.. tab-set::

    .. tab-item:: Lorentz Factor Distribution

        .. math::

            j_\nu(\alpha)
            =
            c_5(p)\,N_0\,(m_e c^2)^{p-1}
            \left(B\sin\alpha\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

    .. tab-item:: Energy Distribution

        .. math::

            j_\nu(\alpha)
            =
            c_5(p)\,N_{0,E}
            \left(B\sin\alpha\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

Complementary to the emissivity is the absorption coefficient, which for the same power-law
distribution of electrons is given by
(see :footcite:t:`RybickiLightman`, Chapter 6):

.. math::

    \boxed{
    \alpha_\nu
    =
    -\frac{1}{8\pi m_e \nu^2}
    \int_{\gamma_{\min}}^{\gamma_{\max}}
    P(\nu,\gamma)\,
    \gamma^2
    \frac{\partial}{\partial\gamma}
    \left[
        \frac{1}{\gamma^2}
        \frac{dN}{d\gamma}
    \right]
    d\gamma
    }.

The corresponding absorption coefficient for a power-law distribution is also a power-law
in frequency:

.. tab-set::

    .. tab-item:: Lorentz Factor Distribution

        .. math::

            \alpha_\nu
            =
            c_6(p)\,N_0\,(m_e c^2)^{p-1}
            (B\sin\alpha)^{(p+2)/2}
            \left(\frac{\nu}{2 c_1}\right)^{-(p+4)/2},

    .. tab-item:: Energy Distribution

        .. math::

            \alpha_\nu
            =
            c_6(p)\,N_{0,E}
            (B\sin\alpha)^{(p+2)/2}
            \left(\frac{\nu}{2 c_1}\right)^{-(p+4)/2}.

From these results, we will be able to fully derive the behavior of synchrotron SEDs
under a variety of physical assumptions.

The Single Electron SED
-----------------------

.. hint::

    The single electron SED can be found in the :mod:`radiation.synchrotron.core` module.

Let us now describe the simplest synchrotron SED: that of a single electron. As described above, the
emission from a single electron is characterized by the synchrotron kernel function
:math:`F(x)`, where :math:`x = \nu/\nu_{\rm char}`. The resulting SED takes the form:

.. math::

    \boxed{
    P(\nu) = \frac{\sqrt{3}q^3 B \sin\alpha}{m c^2} F\left(\frac{\nu}{\nu_{\rm char}}\right),
    }

Importantly, there are two asymptotic regimes of the single-electron SED:

The Low Frequency Regime
^^^^^^^^^^^^^^^^^^^^^^^^

In the low-frequency regime, the synchrotron kernel takes the form

.. math::

    F(x) \approx \frac{4\pi}{\sqrt{3}\,\Gamma\left(\frac{1}{3}\right)} \left(\frac{x}{2}\right)^{1/3} \propto x^{1/3}.

As such, the corresponding SED takes the form

.. math::

    P(\nu) \approx \frac{4\pi q^3}{m_e c^2} \left(B\sin\alpha\right)
    \Gamma\left(\frac{1}{3}\right)^{-1} \left(\frac{\nu}{2\nu_{\rm char}}\right)^{1/3} \propto \nu^{1/3}.

The High Frequency Regime
^^^^^^^^^^^^^^^^^^^^^^^^^

In the high-frequency regime, the synchrotron kernel takes the form

.. math::

    F(x) \approx \sqrt{\frac{\pi x}{2}} e^{-x} \propto x^{1/2} e^{-x}.

As such, the corresponding SED takes the form

.. math::

    P(\nu) \approx \frac{\sqrt{3}\pi^{1/2} q^3}{\sqrt{2} m_e c^2} \left(B\sin\alpha\right)
    \left(\frac{\nu}{\nu_{\rm char}}\right)^{1/2} e^{-\nu/\nu_{\rm char}} \propto \nu^{1/2} e^{-\nu/\nu_{\rm char}}.

At high frequencies, the SED exhibits an exponential cutoff beyond the characteristic frequency
:math:`\nu_{\rm char}`.

The Power-Law Electron Distribution SED
---------------------------------------

We may now extend our discussion to the more complex case of a population of electrons
with a power-law energy distribution. As described above, the emissivity and absorption
coefficients for such a population are also power-laws in frequency. The resulting SED can be
constructed by considering the optically thin and optically thick regimes separately. In this case, we are
required to consider the effects of both emission and absorption processes. We therefore need a detailed understanding
of the radiative transfer effects at play and the cooling processes impacting the underlying electron distribution.

Power-Law SEDs with Absorption
---------------------------------------

Power-Law SEDs with Cooling
---------------------------


Power-Law SEDs with Cooling and Absorption
------------------------------------------

Having derived the synchrotron spectra for single electrons and for power-law
electron populations—and having extended these results to include the effects
of radiative cooling and synchrotron self-absorption—we are now equipped to
discuss the **full synchrotron spectral energy distributions (SEDs)** that arise
from the combined action of these processes.

At this stage, the problem becomes substantially more complex. Realistic
synchrotron spectra are shaped by the **interplay of multiple physical
mechanisms**, each of which introduces its own characteristic scale. In
particular, both the **electron energy distribution** and the **radiative
transfer** imprint distinct spectral breaks, leading to SEDs composed of
multiple power-law segments separated by transition regions.

Despite this apparent complexity, the structure of synchrotron SEDs is governed
by a small number of organizing principles. Chief among these are a set of
**characteristic break frequencies** that mark transitions between different
spectral regimes. These breaks arise from:

- the injection of relativistic electrons (e.g. :math:`\nu_m`),
- radiative cooling of the electron population (e.g. :math:`\nu_c`), and
- absorption processes at low frequencies (e.g. :math:`\nu_a`).

Once these frequencies are identified and ordered, the synchrotron SED may be
constructed as a sequence of asymptotic power-law segments. The full spectrum
is then obtained by **smoothly connecting** these segments using the broken
power-law prescriptions described earlier.

Methodology
^^^^^^^^^^^

The construction of a synchrotron SED proceeds through the following steps:

1. **Identify the relevant break frequencies**, including those associated with
   electron injection, cooling, and absorption.
2. **Determine the ordering** of these frequencies, which defines the sequence
   of spectral regimes.
3. **Derive the spectral slopes** in each regime based on the underlying electron
   distribution and radiative physics.
4. **Normalize the spectrum** using a single characteristic flux scale.
5. **Stitch together the regimes** using smoothly broken power-law functions to
   obtain a continuous SED.

While the algebra required to carry out these steps can be involved, the
underlying logic is straightforward: each spectral segment reflects a distinct
physical regime, and the observed SED is simply the superposition of these
regimes across frequency space.


SED Surgery
~~~~~~~~~~~

To be precise, and to avoid confusion in our derivations below, we adopt a few standard notations. First,
the flux density between **any two adjacent regions** (i.e., power-law segments) will be connected
using a **smoothly broken power-law** (SBPL) of the form:

.. math::

    F_{\nu}^{(1,2)} = F_{\nu,0}^{(1,2)} \left[
        \left(\frac{\nu}{\nu_{brk}}\right)^{\alpha_1/s_{(1,2)}} +
        \left(\frac{\nu}{\nu_{brk}}\right)^{\alpha_2/s_{(1,2)}}
    \right]^{s_{(1,2)}},

where :math:`\alpha_1` and :math:`\alpha_2` are the spectral indices in the two regions, :math:`\nu_{brk}` is the break
frequency between them, :math:`F_{\nu,0}^{(1,2)}` is the normalization constant for the broken
power-law, and :math:`s_{(1,2)}` is the smoothness parameter that controls the sharpness of the transition.

We likewise define the **scale-free** SBPL between two adjacent regions as:

.. math::

    \tilde{F}_{\nu}^{(1,2)} = \left[
        1 +
        \left(\frac{\nu}{\nu_{brk}}\right)^{(\alpha_2-\alpha_1)/s_{(1,2)}}
    \right]^{s_{(1,2)}}.

Thus, we may be precise about our notion of **SED surgery** by recognizing that a spectrum composed of
multiple power-law segments may be constructed by multiplying together the scale-free SBPLs
between each adjacent pair of regions and then normalizing the entire SED with a single flux scale.

Break Frequencies
^^^^^^^^^^^^^^^^^

The most important features of synchrotron SEDs are the characteristic break frequencies that define
the transitions between different spectral regimes. These break frequencies arise from various physical processes,
including the injection of relativistic electrons, radiative cooling, and self-absorption. Critically, there are
two important things that can cause breaks:

1. Changes in the **electron distribution** (e.g., injection breaks, cooling breaks), which tend to be impactful
   at the higher frequencies for which synchrotron emission is optically thin,
2. Changes in the **radiative transfer** (e.g., self-absorption), which tend to be impactful at the lower frequencies
   for which synchrotron emission is optically thick.

.. tab-set::

    .. tab-item:: Critical Frequency(ies)


        Recall that, from our discussion of the single electron SED, each electron emits most of its synchrotron power
        at its characteristic frequency:

        .. math::

            \nu_{\rm char} = \frac{3 q B \gamma^2 \sin\alpha}{4 \pi m c}.

        We therefore expect that, for the most generic case of a power-law distribution of electrons, there will be a
        characteristic frequency associated with the **minimum Lorentz factor** :math:`\gamma_{\min}` of the distribution.
        This frequency is commonly referred to as the **synchrotron peak frequency**, **injection frequency**,
        **minimum frequency**, or the **critical frequency**:

        .. math::

            \boxed{
            \nu_m
            =
            \frac{q B}{2 \pi m c} \gamma_{\min}^2.
            }

        .. important::

            We have here adopted the *de-facto* standard convention used in :footcite:t:`demarchiRadioAnalysisSN2004C2022`
            and many other works of defining :math:`\nu_m` without the factor of :math:`\sin\alpha` present in the
            characteristic frequency. :footcite:t:`ChevalierFranssonHandbook` use a slightly different convention wherein
            one assumes an isotropic distribution of pitch angles and replaces :math:`\sin\alpha` with its average value
            of :math:`\pi/4`. This results in a slightly different numerical prefactor (:math:`3/16`) instead of (:math:`1/2`).
            Likewise, one could perfectly well justify just using :math:`(3/4)`, which would correspond to assuming
            that all electrons have a pitch angle of :math:`\pi/2`.

        It is also the case that, (if there is a maximum Lorentz factor :math:`\gamma_{\max}` in the electron distribution),
        there will be a corresponding **maximum frequency**:

        .. math::

            \boxed{
            \nu_{\max}
            =
            \frac{q B}{2 \pi m c} \gamma_{\max}^2.
            }

        In most cases, however, this frequency lies well beyond the observational window of interest, and so we will not
        consider it further in this discussion.

        The important implication of **both these frequencies** is that they encode breaks in the SED. Below the
        minimum frequency :math:`\nu_m`, one picks up emission only from the low-energy tail of the single-electron SED,
        leading to a characteristic spectral slope of :math:`F_\nu \propto \nu^{1/3}`.
        Above :math:`\nu_m`, one samples the full power-law distribution of electrons, leading to a spectral slope of
        :math:`F_\nu \propto \nu^{-(p-1)/2}`. If one ventures beyond :math:`\nu_{\max}`, the SED will again steepen due to the
        exponential cutoff in the single-electron SED.

        Even in the absence of cooling or absorption processes, therefore, we expect the synchrotron SED
        to exhibit at least one break frequency, :math:`\nu_m`, associated with the minimum Lorentz factor of
        the electron distribution.

    .. tab-item:: Cooling Frequency

        .. hint::

            A detailed discussion of synchrotron cooling can be found in :ref:`synchrotron_cooling_theory`.

        In addition to the injection break at :math:`\nu_m`, synchrotron SEDs often exhibit a second break frequency
        associated with radiative cooling of the electron population. Fundamentally, this is a SED break reflective of the
        change in the underlying electron distribution due to energy losses.

        Given a cooling process with a cooling rate :math:`\Lambda(\gamma)`, one can define a
        **cooling Lorentz factor** :math:`\gamma_c` as the Lorentz factor for which the cooling timescale
        equals the dynamical timescale :math:`t_{\rm dyn}` of the system:

        .. math::

            \boxed{
            t_{\rm cool}(\gamma_c)
            =
            \frac{m_e c^2 \gamma_c}{\Lambda(\gamma_c)}
            =
            t_{\rm dyn}.
            }

        Thus,

        .. math::

            \boxed{
            \gamma_c
            =
            \frac{m_e c^2}{\Lambda(\gamma_c) t_{\rm dyn}}.
            }

        The corresponding **cooling frequency** is then given by

        .. math::

            \boxed{
            \nu_c
            =
            \frac{q B}{2 \pi m c} \gamma_c^2.
            }

        The most important cooling processes for relativistic electrons in astrophysical synchrotron sources are
        synchrotron cooling and inverse Compton cooling. In both cases, we may write the cooling rate in the form

        .. math::

            \Lambda(\gamma) = \Lambda_0 \gamma^2,

        where :math:`\Lambda_0` is a constant that depends on the specific cooling mechanism. See :ref:`synchrotron_cooling_theory`
        for details on the derivation of :math:`\Lambda_0` for these processes.


    .. tab-item:: SSA

        .. hint::

            A detailed discussion of synchrotron self-absorption can be found in :ref:`synchrotron_self_absorption_theory`.

    .. tab-item:: Stratified SSA

        .. hint::

            A detailed discussion of synchrotron self-absorption in stratified sources can be found in :ref:`stratified_absorption`.

Spectral Breaks
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: S1

        - *Break Frequency*: :math:`\nu_a`
        - *Slopes*: :math:`\alpha_1 = 2`, :math:`\alpha_2 = 1/3`

        This spectral break occurs when :math:`\nu_a < \nu_m < \nu_c` and corresponds to the transition from
        the **optically-thin low-frequency tail** of the single-electron SED to the **optically-thick
        self-absorbed** regime.

        On the high-frequency side of the break, :math:`F_\nu` scales as :math:`\nu^{1/3}`, reflecting the low-frequency
        asymptote of the single-electron SED. On the low-frequency side of the break, self-absorption dominates, meaning
        that :math:`F_\nu` scales as :math:`S_\nu`. From our previous discussion of synchrotron self-absorption, recall
        that below :math:`\nu_m`, :math:`S_\nu \propto \nu^2`, leading to the low-frequency slope of :math:`\alpha_1 = 2`.

    .. tab-item:: S2

        - *Break Frequency*: :math:`\nu_m`
        - *Slopes*: :math:`\alpha_1 = 1/3`, :math:`\alpha_2 = (1-p)/2`

        This spectral break occurs when :math:`\nu_a < \nu_m < \nu_c` and corresponds to the transition from
        the **optically-thin low-frequency tail** of the single-electron SED to the regime dominated by the
        **power-law distribution of electrons**. As described for the single-electron SED, below :math:`\nu_m`,
        :math:`F_\nu` scales as :math:`\nu^{1/3}`. Above :math:`\nu_m`, the full power-law distribution of electrons
        contributes to the emission, leading to a spectral slope of :math:`\alpha_2 = (1-p)/2`.

    .. tab-item:: S3

        - *Break Frequency*: :math:`\nu_c`
        - *Slopes*: :math:`\alpha_1 = (1-p)/2`, :math:`\alpha_2 = -p/2`

        This slope corresponds to the break between the **uncooled** and **cooled** regimes of the electron
        distribution. Below :math:`\nu_c`, the electrons responsible for the emission have not cooled significantly,
        leading to a spectral slope of :math:`\alpha_1 = (1-p)/2`. Above :math:`\nu_c`, the electrons have cooled,
        resulting in a steeper spectral slope of :math:`\alpha_2 = -p/2`. This is only the case if the domain
        is optically thin. If self-absorption occurs above this break, it will disappear since the source function
        is independent of the electron distribution index :math:`p`.

    .. tab-item:: S4

        - *Break Frequency*: :math:`\nu_m`
        - *Slopes*: :math:`\alpha_1 = 2`, :math:`\alpha_2 = 5/2`

        This spectral break occurs when :math:`\nu_m < \nu_a < \nu_c` and corresponds to the transition from
        the **optically-thick low-frequency tail** to the **optically-thick power-law regime**. Below :math:`\nu_m`,
        self-absorption dominates, leading to a spectral slope of :math:`\alpha_1 = 2`. Above :math:`\nu_m`, the
        full power-law distribution of electrons contributes to the emission, resulting in a spectral slope of
        :math:`\alpha_2 = 5/2`.

    .. tab-item:: S5

        - *Break Frequency*: :math:`\nu_{a}`
        - *Slopes*: :math:`\alpha_1 = 5/2`, :math:`\alpha_2 = (1-p)/2`

        This spectral break occurs when :math:`\nu_m < \nu_a < \nu_c` and corresponds to the transition from
        the **optically-thick power-law regime** to the **optically-thin power-law regime**. Below :math:`\nu_a`,
        self-absorption dominates, leading to a spectral slope of :math:`\alpha_1 = 5/2`. Above :math:`\nu_a`, the
        emission becomes optically thin, resulting in a spectral slope of :math:`\alpha_2 = (1-p)/2`.

    .. tab-item:: S6

        - *Break Frequency*: :math:`\nu_a`
        - *Slopes*: :math:`\alpha_1 = 5/2`, :math:`\alpha_2 = -p/2`

        The break ``S6`` is very similar to ``S5``, except that it occurs when :math:`\nu_m < \nu_c < \nu_a`.
        Below :math:`\nu_a`, self-absorption dominates, leading to a spectral slope of :math:`\alpha_1 = 5/2`.
        Above :math:`\nu_a`, the emission becomes optically thin, and since we are now in the cooled regime,
        the spectral slope is :math:`\alpha_2 = -p/2`.

        It is worth remembering that, for **any power-law** electron distribution, the optically-thick slope within
        the injection bounds of the power-law will always be :math:`\alpha = 5/2`. This is a direct consequence of the
        fact that, in this regime, the source function scales as :math:`S_\nu \propto \nu^{5/2}` regardless of the
        power-law index.

    .. tab-item:: S7

        - *Break Frequency*: :math:`\nu_{ac}`
        - *Slopes*: :math:`\alpha_1 = 2`, :math:`\alpha_2 = 11/8`

        In cases where cooling is **strong**, electron may be able to cool below even the injection frequency
        :math:`\nu_m` and will then pile up at lower energy. The corresponding break frequency is then
        :math:`\nu_{ac}` (introduced above). This break corresponds to the transition from the **optically-thick
        low-frequency tail** to the **optically-thick cooled regime**. Below :math:`\nu_{ac}`, self-absorption dominates,
        leading to a spectral slope of :math:`\alpha_1 = 2`. Above :math:`\nu_{ac}`, the cooled electrons contribute to the
        emission. Because cooled electrons develop a steady state distribution of :math:`N(\gamma) \propto \gamma^{-2}`,

    .. tab-item:: S8

        - *Break Frequency*: :math:`\nu_a`
        - *Slopes*: :math:`\alpha_1 = 2`, :math:`\alpha_2 = 1/3`

Spectral Regimes
^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Regime I

        In Progress

    .. tab-item:: Regime II

        In Progress

    .. tab-item:: Regime III

        In Progress

    .. tab-item:: Regime IV

        In Progress

    .. tab-item:: Regime V

        In Progress



References
----------
.. footbibliography::
