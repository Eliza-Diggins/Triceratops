.. _synch_sed_theory:
===========================================
Theory of Synchrotron SEDs
===========================================

Having established the foundations of synchrotron radiation theory in :ref:`synchrotron_theory` this document is
intended to develop the theory behind Triceratops' implementation of **synchrotron SEDs**. This is a critical task
on the basis that the literature, spanning some 40 years at this point, is highly fractured in its methodology concerning
these SEDs and their construction (see e.g. :footcite:t:`Chevalier1998SynchrotronSelfAbsorption`, :footcite:t:`ChevalierFranssonHandbook`,
:footcite:t:`GranotSari2002SpectralBreaks`, :footcite:t:`GaoSynchrotronReview2013`, :footcite:t:`2025ApJ...992L..18S`, and
references therein). Our goal in this presentation is to (a) provide a background to users who are not familiar with
the details of this theory and, more importantly, (b) to establish our methodology in as robust a manner as possible ensuring
that Triceratops remains extensible, reproducible, and accurate.

.. contents::

Overview
--------

.. note::

    For readers unfamiliar with elementary theory of synchrotron radiation, it is worthwhile to
    first read :ref:`synchrotron_theory` before proceeding.

In general, the SEDs produced by synchrotron emission from transients are well described by **broken power-law** profiles.
More precisely, **smoothed broken power-laws** have been determined to be a well suited option for interpolating between
the standard power-law regimes of each SED.

For any given scenario, the SED is characterized by a set of **break frequencies** :math:`(\nu_1,\nu_2,\ldots)` between
which the SED follows standard asymptotic behaviors characterized by a set of spectral slopes :math:`(\alpha_{1,2},
\alpha_{2,3},\ldots)`. As described in :ref:`synchrotron_sed_methods`, we consider a total of 4 break frequencies in
Triceratops:

- The **minimum injection frequency** :math:`\nu_m` determined by the characteristic synchrotron frequency of the lowest energy
  electrons injected by the shock.
- The **maximum injection frequency** :math:`\nu_{\rm max}` determined by the synchrotron frequency of the most energetic
  electrons injected by the shock.
- The **cooling frequency** :math:`\nu_c` corresponding to the frequency at which cooling is efficient enough to have
  lead to significant cooling within the dynamical time.
- The **self absorption frequency** :math:`\nu_a` determined by the frequency at which the optical depth to self-absorption
  is unity.

.. hint::

    It's not always the case that one wants to use SEDs which consider **all** of these breaks. We therefore implement
    various combinations in Triceratops as well (i.e. SSA but no cooling or cooling but no SSA). Likewise, because the
    maximum injection frequency is often irrelevant, SEDs without it are available.

In order to determine the SED for a particular scenario, one needs to, self-consistently,

1. Determine the **ordering of the relevant frequencies**,
2. **Identify the correct SED** based on the ordering,
3. **Normalize the SED** based on the emission geometry,
4. Calculate the SED.

As mentioned above, there are a great many implementations of this general scheme with significant variety in the
exact methodology for elements like the calculation of the break frequencies, normalization, etc. In most cases,
these methods are consistent with one another up to an order of unity.


.. _synchrotron_sed_methods:
Methodology
-----------

We now begin our discussion of the methodology used in our construction of the synchrotron SEDs. In the subsections
below, we will describe in detail each element of SED construction; however, for those familiar with the literature, it
should be noted that we follow the formulation of :footcite:t:`GranotSari2002SpectralBreaks` to construct our SEDs and
to describe the various power-law components. Because we include an additional break frequency (:math:`\nu_{\rm max}`) a
few additional scenarios are considered here which are not therein mentioned.

Because :footcite:t:`GranotSari2002SpectralBreaks` use a methodology for normalization which is not fully generalizable without
numerical quadrature, we choose instead to follow the approximations described in :footcite:t:`sari1999jets` and used throughout
the literature (see e.g. :footcite:t:`2025ApJ...992L..18S`).

The Shape of SEDs
^^^^^^^^^^^^^^^^^

To begin, we introduce the formal notation used to describe each of the SEDs which is to be constructed in
this document. We present (and implement in the code) two versions of any given SED:

- The **Smoothed** SED, which uses smoothed broken power laws,
- The **Discrete** SED, which uses piecewise defined power laws.

This is done to allow for easy comparison with the literature since both are used. In either case, the normalizations
of the SEDs and the positions of the breaks are the same.

To be precise, and to avoid confusion in our derivations below, we adopt a few standard notations. First,
the flux density between **any two adjacent regions** (i.e., power-law segments [PLSs]) will be connected
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

Once each segment has been identified, we can then construct the resulting smoothed SED by multiplying a single
scaled SED segment with a number of other *scaled* SED segments in a process we call **SED surgery**.
We may be precise about our notion of **SED surgery** by recognizing that a spectrum composed of
multiple power-law segments may be constructed by multiplying together the scale-free SBPLs
between each adjacent pair of regions and then normalizing the entire SED with a single flux scale. Thus, for
a break :math:`\nu_0` with known flux normalization :math:`F_{\nu,0}`, and additional breaks
at :math:`\nu_1, \nu_2, \ldots, \nu_n`, the full SED may be written as:

.. math::
    :label: full_sed_surgery

    F_\nu = F_{\nu,0} \prod_{i=0}^{n-1} \tilde{F}_{\nu}^{(i,i+1)}.

SED Normalization
^^^^^^^^^^^^^^^^^

While seemingly a simple undertaking, the normalization of the SEDs is a complex element of the theory. This is, in
part, because various approaches have been used in the literature ranging from exact calculations using numerical quadrature
to a variety of approximate schemes. In our case, we select a scheme based off of that described in
:footcite:t:`1998ApJ...497L..17S`, which is an approximate scheme based off the notion that, at an (optically thin) break frequency
:math:`\nu_{\rm brk}`, the synchrotron emission is dominated a single set of electrons with lorentz factors :math:`\gamma_{\brk}`
and number density :math:`n_e(\gamma_{\rm brk}`. One may, therefore, use this approximation to estimate the normalization
to within a factor of order unity.

To be more precise, consider a population of electrons with a distribution of Lorentz factors :math:`dN/d\gamma(\gamma)`. The
emissivity of synchrotron emission for such a population (see :ref:`synchrotron_theory`) is

.. math::

    j_\nu = \frac{\sqrt{3} q^3 B \sin \alpha}{4\pi mc^2} \int_{\gamma_{\rm min}}^{\gamma_{\rm max}} \frac{dN}{d\gamma} F\left(
        \frac{\nu}{\nu_c(\gamma)} \right) d\gamma,

where :math:`q` is the electron charge, :math:`B` is the magnetic field strength, :math:`\gamma_{\rm min}` and :math:`\gamma_{\rm max}`
are the bounding Lorentz factors for the population, and :math:`F` is the synchrotron spectrum. :math:`\nu_c(\gamma)` is
the critical frequency function

.. math::

    \nu_c(\gamma) = \frac{3q}{4\pi m c} B \sin \alpha \gamma^2 = C_\nu (B\sin\alpha) \gamma^2.

Letting :math:`x(\gamma)\equiv \nu/\nu_c(\gamma)`, we have :math:`dx/d\gamma = -2x/\gamma` and thus
:math:`d\gamma = -(\gamma/2x)\,dx`. Since :math:`F(x)` is sharply peaked near :math:`x\sim\mathcal{O}(1)`,
the integral is dominated by :math:`\gamma=\gamma_\nu` satisfying :math:`\nu\sim\nu_c(\gamma_\nu)`, and the remaining
kernel integral contributes only an order-unity constant. Therefore,

.. math::

    j_\nu \approx \frac{\sqrt{3} q^3 B \sin \alpha}{4\pi m_e c^2}\,\left.\frac{dN}{d\gamma}\right|_{\gamma=\gamma_\nu}\,\gamma_\nu.

.. note::

    We here, as in :footcite:t:`1998ApJ...497L..17S`, suppress an order-unity factor stemming from integration
    of the kernel.

Using :math:`\sigma_T = \frac{8\pi}{3}\frac{q^4}{m_e^2c^4}` we may rewrite the pre-factor as

.. math::

    j_\nu \approx \left(\frac{3 \sqrt{3}}{32\pi^2}\right)\frac{m_e c^2}{q}\,\sigma_T\,(B\sin\alpha)\,\left.\frac{dN}{d\gamma}\right|_{\gamma=\gamma_\nu}\,\gamma_\nu.

Assuming an **effective emitting volume** V at a luminosity distance :math:`D_L`,

.. math::

    F_\nu \approx \frac{V}{D_L^2} j_\nu.

.. hint::

    A common approach in the literature which can be used, but is not *required* is to assume a spherical emitting
    region of radius :math:`R` and a *filling factor* :math:`f`, such that :math:`V = f (4\pi R^3/3)`.

This is therefore the basis of our normalization scheme with a number of caveats:

1. The approximation made above is only really sensible when used to describe the emissivity of the **maximal population
   of electrons**. We therefore always anchor our normalization to the **peak emission frequency**; however,
2. The presence of absorption may **obscure the optically thin peak**. To correct for this, we use the known shape
   of the corresponding SED and the relevant break frequencies to "extrapolate" the optically thin peak down to the
   absorption break and then back along the correct PLS to correct the normalization for absorption effects.

To be more precise about which population of electrons we use to normalize the SED, we note that the determination of the
correct population of electrons is determined by the cooling regime, with fast-cooling and slow-cooling scenarios corresponding
to populations peaking at :math:`\nu_c` and :math:`\nu_m`, respectively. In the tabs below, we describe the normalization
for each case:

.. tab-set::

    .. tab-item:: Fast Cooling

        In the fast cooling scenario, the electron population has enough time to relax to a steady state distribution function
        of the form

        .. math::

            \frac{dN}{d\gamma} = \begin{cases}
                0, & \gamma < \gamma_c \\
                K_c  \left(\frac{\gamma}{\gamma_c}\right)^{-2}, & \gamma_c \le \gamma < \gamma_{\min} \\
                K_c  \left(\frac{\gamma_{\rm min}}{\gamma_c}\right)^{-2}
                     \left(\frac{\gamma}{\gamma_{\rm min}}\right)^{-(p+1)}, & \gamma \ge \gamma_{\min}
            \end{cases},

        where :math:`K_c` is the normalization constant of the cooled electron distribution. The correct optically thin flux
        is therefore

        .. math::

            F_{c,0} \approx \left[\left(\frac{3 \sqrt{3}}{32\pi^2}\right)
                                \frac{m_e c^2}{q}\,\sigma_T\,
                                (B\sin\alpha)\,
                                K_c
                                \gamma_c\right]\frac{V}{D_L^2},

        or, letting :math:`\gamma = (2\pi m_e c \nu / 3 q B \sin\alpha)^{1/2}`,

        .. math::
            :label: cooling_norm

            F_{c,0} \approx \left[\left(\frac{3 \sqrt{3}}{32\pi^2}\right)
                                \left(\frac{4\pi}{3}\right)^{1/2}
                                \frac{m_e^{3/2} c^{5/2}}{q^{3/2}}\,\sigma_T\,
                                (B\sin\alpha)^{1/2}\,
                                K_c
                                \nu_c^{1/2}
                                \right]\frac{V}{D_L^2}.

    .. tab-item:: Slow Cooling

        When **slow-cooling** applies, i.e. :math:`\gamma_c > \gamma_{\min}`, the peak emission is at
        :math:`\nu_m = \nu_c(\gamma_{\min})` and the electron population follows
        :math:`dN/d\gamma = N_0 \gamma^{-p}`. Evaluating the emissivity at
        :math:`\gamma=\gamma_{\min}` therefore yields

        .. math::

            F_{m,0} \approx
            \left[\left(\frac{3 \sqrt{3}}{32\pi^2}\right)
            \frac{m_e c^2}{q}\,\sigma_T\,
            (B\sin\alpha)\,
            N_0\,\gamma_{\min}^{1-p}
            \right]\frac{V}{D_L^2},

        or, expressing :math:`\gamma_{\min}` in terms of the peak frequency :math:`\nu_m`,

        .. math::
            :label: slow_cooling_norm

            F_{m,0} \approx
            \left[\left(\frac{3 \sqrt{3}}{32\pi^2}\right)
            \left(\frac{4\pi}{3}\right)^{(1-p)/2}
            \frac{m_e^{(3-p)/2} c^{(5-p)/2}}{q^{(3-p)/2}}
            \sigma_T\,
            (B\sin\alpha)^{(p+1)/2}\,
            N_0\,\nu_m^{(1-p)/2}
            \right]\frac{V}{D_L^2}.

Break Frequencies
^^^^^^^^^^^^^^^^^

Before proceeding to discuss the construction of various SEDs, it is necessary to describe the precise methodology
with while we compute the various break frequencies used in the SEDs. In each of the following sections, we describe
this methodology in detail.

.. hint::

    In :ref:`synchrotron_seds`, we describe the actual code implementation of all of the SEDs. It is worth noting that,
    in general, we provide implementations for SEDs in terms of the relevant frequencies so that, should one wish to
    do so, any prescription for computing the break frequencies may be used.

The Injection Frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~

The injection frequencies are determined by the characteristic synchrotron frequency of electrons at the minimum
and maximum Lorentz factors of the electron distribution. These are given by

.. math::
    :label: injection_freq_min

    \boxed{
    \nu_m
    =
    \frac{q B \sin \alpha}{2 \pi m c} \gamma_{\min}^2,
    }

and

.. math::
    :label: injection_freq_max

    \boxed{
    \nu_{\max}
    =
    \frac{q B \sin \alpha}{2 \pi m c} \gamma_{\max}^2.
    }

In practice, the minimum injection frequency is typically treated as a free parameter in a model and fit for, while
the maximum injection frequency is often neglected entirely due to its location at very high frequencies. One may
also choose :math:`\gamma_{\rm max}` as a hyper-parameter in a model if desired, or treat it as a free parameter just
like :math:`\gamma_{\min}`.

The Cooling Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a population of electrons subject to a cooling process with a cooling rate :math:`\Lambda(\gamma)`. Electrons
with energy :math:`E = m_e c^2 \gamma` will then cool on a timescale

.. math::

    t_{\rm cool}(\gamma) = \frac{E}{\Lambda(\gamma)} = \frac{m_e c^2 \gamma}{\Lambda(\gamma)}.

If, in order to cool significantly, the dynamical time must exceed the cooling time for a particular energy, we can
define the **cooling Lorentz factor** :math:`\gamma_c` as the Lorentz factor for which the cooling timescale
equals the dynamical timescale :math:`t_{\rm dyn}` of the system:

.. math::

    \boxed{
    t_{\rm cool}(\gamma_c)
    =
    \frac{m_e c^2 \gamma_c}{\Lambda(\gamma_c)}
    =
    t_{\rm dyn}.
    }

This then implies

.. math::
    :label: cooling_lorentz_factor

    \boxed{
    \gamma_c
    =
    \frac{m_e c^2}{\Lambda(\gamma_c) t_{\rm dyn}}.
    }

The corresponding **cooling frequency** is then given by

.. math::
    :label: cooling_frequency

    \boxed{
    \nu_c
    =
    \frac{q B \sin \alpha}{2 \pi m c} \gamma_c^2 = \frac{q m_e c^3}{2\pi} \left(\frac{B\sin \alpha}{\Lambda(\gamma_c)^2 t_{\rm dyn}^2}\right).
    }

The precise value of the cooling frequency should be determined from the dominant cooling process affecting the electron
population. In most astrophysical synchrotron sources, the most important cooling processes are synchrotron cooling
and inverse Compton cooling.

The Absorption Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~
The self-absorption frequency is a less trivial quantity to compute, as it depends on the radiative transfer
properties of the source and is therefore dependent on the SED one is using. This creates a circular dependency
which must be resolved by considering every possible SED configuration given a known :math:`\nu_m` and :math:`\nu_c`,
computing the value of :math:`\nu_a` in each case and then checking for self-consistency with the assumed SED and
its assumptions.

In the most rigorous sense, the absorption frequency :math:`\nu_a` is determined by the condition that the
optical depth to self-absorption equals unity:

.. math::

    \tau_{\nu_{a}} = \alpha_{\nu_a} L = 1,

The form of :math:`\alpha_\nu` depends explicitly on the structure of the absorbing electron population (see
:ref:`synchrotron_theory` for details). One could, in principle, perform these computations in full detail; however,
an alternative approach has been developed in the literature :footcite:p:`duran2013radius` which allows for
approximate expressions for :math:`\nu_a`.

We assume, as was done in the development of the normalization approach, that the absorption at a particular frequency
is dominated by a mono-energetic population of electrons. In such a case, the optically thick emission from the source
should be well approximated by a blackbody with effective temperature :math:`kT_{\rm eff} = \gamma_\nu m_e c^2`, where
:math:`\gamma_\nu` is the Lorentz factor of electrons emitting at frequency :math:`\nu`. This corresponds to a source
function :math:`S_\nu = 2\nu^2 m_e \gamma_\nu`. The corresponding flux :math:`F_\nu` should then be

.. math::

    F_\nu = 2\nu^2 m_e \gamma_\nu \frac{A}{D_L^2},

where :math:`A` is the effective radiating area of the source. Equating this to the **optically thin** flux
from the normalized SED at :math:`\nu_a` then allows one to solve for :math:`\nu_a`.


The Single Electron SED
-----------------------

.. hint::

    The single electron SED can be found in the :mod:`radiation.synchrotron.core` module.

.. note::

    In this case, it is effectively non-sensical to describe a volume emitting flux or intensity. We therefore
    simply describe the **spectral power density** (i.e., power per unit frequency) :math:`P(\nu)` of a single electron.

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


Power Law Synchrotron SEDs
---------------------------
Having now established all of the relevant theory and mathematical tools, we are
finally in a position to derive the classic synchrotron SEDs used in astrophysical
modeling. In the sections that follow, we will derive each of the standard synchrotron SED regimes
and describe the normalization in each case.

Asymptotic Regimes
^^^^^^^^^^^^^^^^^^

Because various broadband SEDs have segments with the same SPL slopes, we begin by listing the different
possible SPL segments that can arise in synchrotron SEDs from power-law electron distributions. This follows
the naming convention of :footcite:t:`GranotSari2002SpectralBreaks`.

.. tab-set::

    .. tab-item:: SPL A (:math:`F_\nu \propto \nu^{5/2}`)

        SPL A occurs in the optically thick regime below the self-absorption frequency :math:`\nu_a`, but above
        the minimum electron frequency :math:`\nu_m`. In this regime, the SED is dominated by self-absorbed synchrotron
        emission from the full power-law distribution of electrons. The resulting SED is derived in most references
        in synchrotron emission, (e.g. :footcite:t:`RybickiLightman`, Chapter 6; :footcite:t:`1970ranp.book.....P`).

    .. tab-item:: SPL B (:math:`F_\nu \propto \nu^{2}`)

        SPL B occurs in the optically thick regime below both the self-absorption frequency :math:`\nu_a` and
        the minimum electron frequency :math:`\nu_m`. In this regime, the SED is dominated by self-absorbed synchrotron
        emission from the low-energy tail of the single-electron SED. This results in a characteristic spectral slope
        of :math:`F_\nu \propto \nu^{2}`.

        More precisely, the absorption coefficient to synchrotron self-abortion is (:eq:`absorption_general`):

        .. math::

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
            d\gamma.

        In the low-frequency limit, the integral is dominated by the lowest energy electrons in the population and
        the frequency is in the low-frequency tail of those single-electron SEDs. In that case, the entire integral term
        scales as :math:`\nu^{1/3}` and the absorption coefficient scales as :math:`\alpha_\nu \propto \nu^{-5/3}`. The
        emissivity in this case scales as :math:`j_\nu \propto \nu^{1/3}` as well. The source function is then
        :math:`S_\nu = j_\nu/\alpha_\nu \propto \nu^{2}`, leading to the characteristic SPL B slope.

    .. tab-item:: SPL C (:math:`F_\nu \propto \nu^{11/8}`)

        SPL C is a special case first identified in :footcite:t:`2000ApJ...534L.163G` in the context of GRB afterglows.
        It arises in scenarios with fast cooling such that the post-shock electrons are able to cool very rapidly
        compared to the dynamical timescale. This leads to a stratified structure in the source, with a thin layer
        of uncooled electrons near the shock front and a larger volume of cooled electrons behind them. In this case,
        the self-absorption is dominated by the cooler, downstream electrons, leading to a modified self-absorption
        regime with a characteristic slope of
        :math:`F_\nu \propto \nu^{11/8}`.

        To see a derivation of this result, see the theory note: :ref:`stratified_absorption`.

    .. tab-item:: SPL D (:math:`F_\nu \propto \nu^{1/3}`)

        SPL D occurs in the optically thin regime below the minimum electron frequency :math:`\nu_m`. In this
        regime, the SED is dominated by synchrotron emission from the low-energy tail of the single-electron SED.
        This results in a characteristic spectral slope of :math:`F_\nu \propto \nu^{1/3}`.

    .. tab-item:: SPL E (:math:`F_\nu \propto \nu^{1/3}`)

        SPL E has the same slope as SPL D, but occurs in the optically thin regime between the cooling frequency
        :math:`\nu_c` and the SSA frequency :math:`\nu_a`. In this regime, the SED is dominated by synchrotron emission
        from the low-energy tail of the single-electron SED, similar to SPL D.

    .. tab-item:: SPL F (:math:`F_\nu \propto \nu^{-1/2}`)

        SPL E occurs in the optically thin regime above the cooling frequency :math:`\nu_c` but
        below the minimum electron frequency :math:`\nu_m`. In this regime, the SED is dominated by
        synchrotron emission from the cooled portion of the electron distribution. Because this population has
        a characteristic electron index of :math:`p=2`, the emissivity scales as :math:`j_\nu \propto \nu^{-1/2}`.

    .. tab-item:: SPL G (:math:`F_\nu \propto \nu^{-(p-1)/2}`)

        SPL E occurs in the optically thin regime above the minimum electron frequency :math:`\nu_m` but
        below the cooling frequency :math:`\nu_c` (if present). In this regime, the SED is dominated by
        synchrotron emission from the full power-law distribution of electrons that have not yet cooled.
        This results in a characteristic spectral slope of :math:`F_\nu \propto \nu^{-(p-1)/2}`.

    .. tab-item:: SPL H (:math:`F_\nu \propto \nu^{-p/2}`)

        SPL H occurs in the optically thin regime above the cooling frequency :math:`\nu_c`. In this
        regime, the SED is dominated by synchrotron emission from the cooled portion of the electron
        distribution. This results in a characteristic spectral slope of :math:`F_\nu \propto \nu^{-p/2}`.

    .. tab-item:: SPL I (:math:`F_\nu \propto \nu^{1/2} \exp(-\nu/\nu_{\rm char})`)

        SPL I occurs in the extreme high-frequency regime above the maximum electron frequency :math:`\nu_{\max}`.
        In this regime, the SED is dominated by the exponential cutoff in the single-electron SEDs of the highest
        energy electrons. This results in a characteristic spectral slope of
        :math:`F_\nu \propto \nu^{1/2} \exp(-\nu/\nu_{\rm char})`.

Spectral Breaks
^^^^^^^^^^^^^^^

As with the SPL segments, we can list the different possible spectral breaks combining two such segments and
a given break frequency. Again, we follow the naming convention of :footcite:t:`GranotSari2002SpectralBreaks`.

.. tab-set::

    .. tab-item:: 1

        *Slopes*: SPL B to SPL D (:math:`F_\nu \propto \nu^{2}` to :math:`F_\nu \propto \nu^{1/3}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_1_SBPL

            F_{\nu}^{(B,D)} = F^{(B,D)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{2/s_{(B,D)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{(1/3)/s_{(B,D)}}
            \right]^{s_{(B,D)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_1_scale_free_SBPL

            \tilde{F}_{\nu}^{(B,D)} = \left[
                1+
                \left(\frac{\nu}{\nu_a}\right)^{(-5/3)/s_{(B,D)}}
            \right]^{s_{(B,D)}}.


    .. tab-item:: 2

        *Slopes*: SPL D to SPL G (:math:`F_\nu \propto \nu^{1/3}` to :math:`F_\nu \propto \nu^{-(p-1)/2}`)

        *Break Frequency*: :math:`\nu_m`

        This break occurs at the minimum electron frequency :math:`\nu_m`, transitioning from the optically thin
        SPL D regime to the optically thin SPL F regime. The corresponding SBPL is

        .. math::
            :label: break_2_SBPL

            F_{\nu}^{(D,G)} = F^{(D,G)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_m}\right)^{(1/3)/s_{(D,G)}}
                +
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2s_{(D,G)}}
            \right]^{s_{(D,G)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_2_scale_free_SBPL

            \tilde{F}_{\nu}^{(D,G)} = \left[
                1 +
                \left(\frac{\nu}{\nu_m}\right)^{(1-3p)/6s_{(D,G)}}
            \right]^{s_{(D,G)}}.

    .. tab-item:: 3

        *Slopes*: SPL G to SPL H (:math:`F_\nu \propto \nu^{-(p-1)/2}` to :math:`F_\nu \propto \nu^{-p/2}`)

        *Break Frequency*: :math:`\nu_c`

        This break occurs at the cooling frequency :math:`\nu_c`, transitioning from the optically thin
        SPL G regime to the optically thin SPL H regime. The corresponding SBPL is

        .. math::
            :label: break_3_SBPL

            \tilde{F}_{\nu}^{(G,H)} = F^{(G,H)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_c}\right)^{-(p-1)/2s_{(G,H)}}
                +
                \left(\frac{\nu}{\nu_c}\right)^{-p/2s_{(G,H)}}
            \right]^{s_{(G,H)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_3_scale_free_SBPL

            \tilde{F}_{\nu}^{(G,H)} = \left[
                1 +
                \left(\frac{\nu}{\nu_c}\right)^{-1/2s_{(G,H)}}
            \right]^{s_{(G,H)}}.

    .. tab-item:: 4

        *Slopes*: SPL B to SPL A (:math:`F_\nu \propto \nu^{2}` to :math:`F_\nu \propto \nu^{5/2}`)

        *Break Frequency*: :math:`\nu_m`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_4_SBPL

            \tilde{F}_{\nu}^{(B,A)} = F^{(B,A)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{2/s_{(B,A)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{5/2s_{(B,A)}}
            \right]^{s_{(B,A)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_4_scale_free_SBPL

            \tilde{F}_{\nu}^{(B,A)} = \left[
                1 +
                \left(\frac{\nu}{\nu_a}\right)^{1/2s_{(B,A)}}
            \right]^{s_{(B,A)}}.

    .. tab-item:: 5

        *Slopes*: SPL A to SPL G (:math:`F_\nu \propto \nu^{5/2}` to :math:`F_\nu \propto \nu^{-(p-1)/2}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_5_SBPL

            \tilde{F}_{\nu}^{(A,G)} = F^{(A,G)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{5/2s_{(A,G)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{-(p-1)/2s_{(A,G)}}
            \right]^{s_{(A,G)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_5_scale_free_SBPL

            \tilde{F}_{\nu}^{(A,G)} = \left[
                1 +
                \left(\frac{\nu}{\nu_a}\right)^{-(2+4)/2s_{(A,G)}}
            \right]^{s_{(A,G)}}.

    .. tab-item:: 6

        *Slopes*: SPL A to SPL H (:math:`F_\nu \propto \nu^{5/2}` to :math:`F_\nu \propto \nu^{-p/2}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_6_SBPL

            \tilde{F}_{\nu}^{(A,H)} = F^{(A,H)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{5/2s_{(A,H)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{-p/2s_{(A,H)}}
            \right]^{s_{(A,H)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_6_scale_free_SBPL

            \tilde{F}_{\nu}^{(A,H)} = \left[
                1 +
                \left(\frac{\nu}{\nu_a}\right)^{-(p+5)/2s_{(A,H)}}
            \right]^{s_{(A,H)}}.

    .. tab-item:: 7

        *Slopes*: SPL B to SPL C (:math:`F_\nu \propto \nu^{2}` to :math:`F_\nu \propto \nu^{11/8}`)

        *Break Frequency*: :math:`\nu_{ac}`

        This break occurs at the stratified self-absorption frequency :math:`\nu_{ac}`, transitioning from the optically thick
        SPL B regime to the stratified self-absorption SPL C regime. The corresponding SBPL is

        .. math::
            :label: break_7_SBPL

            \tilde{F}_{\nu}^{(B,C)} = F^{(B,C)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_{ac}}\right)^{2/s_{(B,C)}}
                +
                \left(\frac{\nu}{\nu_{ac}}\right)^{(11/8)/s_{(B,C)}}
            \right]^{s_{(B,C)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_7_scale_free_SBPL

            \tilde{F}_{\nu}^{(B,C)} = \left[
                1+
                \left(\frac{\nu}{\nu_{ac}}\right)^{(-5/8)/s_{(B,C)}}
            \right]^{s_{(B,C)}}.

    .. tab-item:: 8

        *Slopes*: SPL C to SPL F (:math:`F_\nu \propto \nu^{11/8}` to :math:`F_\nu \propto \nu^{-1/2}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_8_SBPL

            \tilde{F}_{\nu}^{(C,F)} = F^{(C,F)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{(11/8)/s_{(C,F)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{-1/2s_{(C,F)}}
            \right]^{s_{(C,F)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_8_scale_free_SBPL

            \tilde{F}_{\nu}^{(C,F)} = \left[
                1 +
                \left(\frac{\nu}{\nu_a}\right)^{(-15/8)/s_{(C,F)}}
            \right]^{s_{(C,F)}}.

    .. tab-item:: 9

        *Slopes*: SPL F to SPL H (:math:`F_\nu \propto \nu^{-1/2}` to :math:`F_\nu \propto \nu^{-p/2}`)

        *Break Frequency*: :math:`\nu_m`

        This break occurs at the minimum electron frequency :math:`\nu_m`, transitioning from the optically thin
        SPL D regime to the optically thin SPL F regime. The corresponding SBPL is

        .. math::
            :label: break_9_SBPL

            \tilde{F}_{\nu}^{(F,H)} = F^{(F,H)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_m}\right)^{-1/2s_{(F,H)}}
                +
                \left(\frac{\nu}{\nu_m}\right)^{-p/2s_{(F,H)}}
            \right]^{s_{(F,H)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_9_scale_free_SBPL

            \tilde{F}_{\nu}^{(F,H)} = \left[
                1 +
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2s_{(F,H)}}
            \right]^{s_{(F,H)}}.

    .. tab-item:: 10

        *Slopes*: SPL C to SPL E (:math:`F_\nu \propto \nu^{11/8}` to :math:`F_\nu \propto \nu^{1/3}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_10_SBPL

            \tilde{F}_{\nu}^{(C,E)} = F^{(C,E)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{(11/8)/s_{(C,E)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{(1/3)/s_{(C,E)}}
            \right]^{s_{(C,E)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_10_scale_free_SBPL

            \tilde{F}_{\nu}^{(C,E)} = \left[
                \left(\frac{\nu}{\nu_a}\right)^{(-25/24)/s_{(C,E)}}
                +
                1
            \right]^{s_{(C,E)}}.

    .. tab-item:: 11

        *Slopes*: SPL E to SPL F (:math:`F_\nu \propto \nu^{1/3}` to :math:`F_\nu \propto \nu^{-1/2}`)

        *Break Frequency*: :math:`\nu_c`

        This break occurs at the cooling frequency :math:`\nu_c`, transitioning from the optically thin
        SPL G regime to the optically thin SPL H regime. The corresponding SBPL is

        .. math::
            :label: break_11_SBPL

            \tilde{F}_{\nu}^{(E,F)} = F^{(E,F)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_c}\right)^{(1/3)/s_{(E,F)}}
                +
                \left(\frac{\nu}{\nu_c}\right)^{-1/2s_{(E,F)}}
            \right]^{s_{(E,F)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_11_scale_free_SBPL

            \tilde{F}_{\nu}^{(E,F)} = \left[
                \left(\frac{\nu}{\nu_c}\right)^{-5/6s_{(E,F)}}
                +
                1
            \right]^{s_{(E,F)}}.

Our final set of spectral breaks occur as one ventures into the asymptotic high-frequency regime beyond the
maximum electron frequency :math:`\nu_{\max}`. In this regime, the SED transitions from any of the optically thin
segments (SPL F, SPL G, or SPL H) to the exponential cutoff segment (SPL I). Because the
exponential cutoff is not a power law, we do not provide SBPL representations for these breaks, but instead
provide **exponential cutoff functions**. For discrete (non-smooth) representations of SEDs, we use the function
:math:`\Phi(\nu,\nu_{\rm max})` to denote the cutoff:

.. math::

    \Phi(\nu,\nu_{\rm max}) = \left(\frac{\nu}{\nu_{\max}}\right)^{1/2}
    \exp\left(1 -\frac{\nu}{\nu_{\max}}\right).

In the smoothed case described above, we instead need a **scale-free exponential cutoff function** which does
not interfere with the normalization of the SED at lower frequencies. We therefore define:

.. math::

    \tilde{\Phi}(\nu,\nu_{\max}) = \begin{cases}
        1, & \nu < \nu_{\max} \\
        \left(\frac{\nu}{\nu_{\max}}\right)^{1/2}
        \exp\left(1 -\frac{\nu}{\nu_{\max}}\right), & \nu \geq \nu_{\max}.
    \end{cases}


Broadband SEDs
^^^^^^^^^^^^^^

For each of the broadband SEDs described below, we provide two different formulations of the SED: one utilizing
the smoothed broken power-law (SBPL) construction described in :ref:`sed_surgery`, and one providing the full piecewise
definition of the SED. In each case, the normalization of the SED is provided according to the LKIN scheme described
in :ref:`sed_normalization`. We report two different normalizations for the SED: one for a **single pitch angle** and
one for an **isotropic pitch angle distribution**. The difference between these two normalizations is an averaging factor.

The Power Law SED
~~~~~~~~~~~~~~~~~

We start with the simplest power-law SED: that of a power-law distribution of electrons with no cooling and
no absorption. In this case, the only break frequencies are the minimum and maximum electron frequencies, leading
to segments of SPL H, SPL F, and SPL D. The smoothed SED may be constructed as:

.. math::

    F_\nu = F^{(D,G)}_\nu \tilde{\Phi}(\nu,\nu_{\max}),

The discrete SED segments are:

.. math::

    F_\nu = F_{\nu,0} \begin{cases}
        \left(\frac{\nu}{\nu_m}\right)^{1/3}, & \nu < \nu_m \quad \text{(SPL D)}\\
        \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2}, & \nu_m \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
        \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
        \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}
    \end{cases}

Normalization in this case is simple as the peak frequency is also in the optically thin regime. Therefore,

.. math::

    F_{\nu,0} = F_{\nu_m,0},

where :math:`F_{\nu_m,0}` is given in equation :eq:`slow_cooling_norm`.

The SSA Power Law SED
~~~~~~~~~~~~~~~~~~~~~~

We now progress to the case with SSA but no cooling. In this case, there are 2(3) orderings of the break frequencies:

1. :math:`\nu_a < \nu_m < \nu_{\max}`: In this case, the SED segments are SPL B, SPL D, SPL F, and SPL H.
2. :math:`\nu_m < \nu_a < \nu_{\max}`: In this case, the SED segments are SPL B, SPL A, SPL F, and SPL H.
3. :math:`\nu_m < \nu_{\max} < \nu_a`: This scenario is non-physical as self-absorption requires electrons with energies
   at or near the characteristic energy of the absorbed frequency. Since there are no electrons above the maximum
   cutoff, there is no way to self-absorb at those frequencies.

.. tab-set::

    .. tab-item:: Spectrum 1

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_a`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_a \leq \nu < \nu_m`
              - SPL D
              - :math:`1/3`
            * - 3
              - :math:`\nu_m \leq \nu < \nu_{\max}`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = \tilde{F}^{(B,D)}_\nu F^{(D,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_a}\right)^{2}\left(\frac{\nu_a}{\nu_m}\right)^{1/3}, & \nu < \nu_a \quad \text{(SPL B)}\\
                \left(\frac{\nu}{\nu_m}\right)^{1/3},& \nu_a < \nu < \nu_m \quad \text{(SPL D)}\\
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2}, & \nu_m \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
                \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
                \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}\\
            \end{cases}

        The normalization is set at :math:`\nu_m` using the LKIN scheme:

        .. math::

            F_{\nu,0} = F_{\nu_m,0},

        where :math:`F_{\nu_m,0}` is given in equation :eq:`slow_cooling_norm`.

    .. tab-item:: Spectrum 2

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_m`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_m \leq \nu < \nu_a`
              - SPL A
              - :math:`5/2`
            * - 3
              - :math:`\nu_a \leq \nu < \nu_{\max}`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(B,A)}_\nu \tilde{F}^{(A,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (B,A) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_m}\right)^{2}, & \nu < \nu_m \quad \text{(SPL B)}\\
                \left(\frac{\nu}{\nu_m}\right)^{5/2}, & \nu_m < \nu < \nu_a \quad \text{(SPL A)}\\
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu}{\nu_a}\right)^{-(p-1)/2}, & \nu_a \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
                \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}\\
            \end{cases}

        In this case, we are required to rely on the power-law propagation method described above as our chosen
        normalization frequency is obscured by the effects of self-absorption. This then produces

        .. math::

            F_{\nu,0} = F_{\nu_m,0}
            \left(\frac{\nu_a}{\nu_m}\right)^{-(p-1)/2}
            \left(\frac{\nu_m}{\nu_a}\right)^{5/2},

        where :math:`F_{\nu_m,0}` is given in equation :eq:`slow_cooling_norm`.

Cooling Power Law SEDs
~~~~~~~~~~~~~~~~~~~~~~

The other simple scenario worth considering is the SED from a synchrotron source with non-negligible cooling
and no SSA. In this case, the three relevant break frequencies are :math:`\nu_m`, :math:`\nu_c`, and
:math:`\nu_{\rm max}`. There are 3 possible configurations

1. :math:`\nu_c < \nu_m < \nu_{\rm max}`: The **fast-cooling** regime. The SED here is composed of segments SPL E,
   SPL F, SPL H, and SPL I with slopes :math:`1/3, 1/2, -p/2, {\rm exp}`. The maximum in this case occurs at
   :math:`\nu_c`, and so we use that point to normalize.
2. :math:`\nu_m < \nu_c < \nu_{\rm max}`: The **slow-cooling** regime. The SED here is composed of segments SPL D,
   SPL G, SPL H, and SPL I with slopes :math:`1/3, -(p-1)/2, -p/2, {\rm exp}`.
   The maximum in this case occurs at :math:`\nu_m`, and so we use that point to normalize.
3. :math:`\nu_m < \nu_{\rm max} < \nu_c`: The **uncooled regime**. The SED here is identical to the standard
   power-law SED and is therefore not described in any further detail.

.. tab-set::

    .. tab-item:: Spectrum 1

        In this spectrum, there are 4 SPL segments connected by 3 breaks. Because the population is rapidly cooled,
        the bulk of electrons are effectively reduced to :math:`\gamma_c` and the corresponding peak in the spectrum
        occurs at :math:`\nu_c`.

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_c`
              - SPL E
              - :math:`1/3`
            * - 2
              - :math:`\nu_c < \nu < \nu_m`
              - SPL F
              - :math:`-1/2`
            * - 3
              - :math:`\nu_m \leq \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(E,F)}_\nu \tilde{F}^{(F,H)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we will normalize at :math:`\nu_c` using the cooled population and corresponding electron
        distribution function. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_c}\right)^{1/3}, & \nu < \nu_c \quad \text{(SPL E)}\\
                \left(\frac{\nu}{\nu_c}\right)^{-1/2}, & \nu_c < \nu < \nu_m \quad \text{(SPL F)}\\
                \left(\frac{\nu_m}{\nu_c}\right)^{-1/2}
                \left(\frac{\nu}{\nu_m}\right)^{-p/2}, & \nu_m < \nu < \nu_{\rm max} \quad \text{(SPL H)}\\
                \left(\frac{\nu_m}{\nu_c}\right)^{-1/2}
                \left(\frac{\nu_{\rm max}}{\nu_m}\right)^{-p/2}
                \left(\frac{\nu}{\nu_{\rm max}}\right)^{-1/2}
                \exp\left(1-\frac{\nu}{\nu_{\rm max}}\right), & \nu > \nu_{\rm max} \quad \text{(SPL I)}.
            \end{cases}

        The dominant electron population at the spectrum peak is the population of **cooled electrons**. As such,
        the normalization takes the form of :eq:`cooling_norm`:

        .. math::

            F_{\nu,0} = F_{c,0}

    .. tab-item:: Spectrum 2

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_c`
              - SPL D
              - :math:`1/3`
            * - 2
              - :math:`\nu_c < \nu < \nu_m`
              - SPL G
              - :math:`-(p-1)/2`
            * - 3
              - :math:`\nu_m \leq \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(D,G)}_\nu \tilde{F}^{(G,H)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_m}\right)^{1/3}, & \nu < \nu_m \quad \text{(SPL D)}\\
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2}, & \nu_m < \nu < \nu_c \quad \text{(SPL G)}\\
                \left(\frac{\nu}{\nu_c}\right)^{-p/2}
                \left(\frac{\nu_c}{\nu_m}\right)^{-(p-1)/2}, & \nu_c < \nu < \nu_{\rm max} \quad \text{(SPL H)}\\
                \left(\frac{\nu_c}{\nu_m}\right)^{-(p-1)/2}
                \left(\frac{\nu_{\rm max}}{\nu_c}\right)^{-p/2}
                \left(\frac{\nu}{\nu_{\rm max}}\right)^{1/2}
                \exp\left(1-\frac{\nu}{\nu_{\rm max}}\right), & \nu > \nu_{\rm max} \quad \text{(SPL I)}.
            \end{cases}

        The dominant electron population at the spectrum peak is the population of **uncooled electrons**. As such,
        the normalization takes the form of :eq:`slow_cooling_norm`:

        .. math::

            F_{\nu,0} = F_{\nu_m,0}

Cooling+SSA Power Law SEDs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are now prepared to introduce the complete set of synchrotron SEDs relevant to the most generic scenarios in which
both SSA and cooling are relevant. We therefore have the break frequencies :math:`\nu_m`, :math:`\nu_a`, :math:`\nu_c`,
and :math:`\nu_{\rm max}`. In addition, for absorption dominated regimes with fast cooling, we have the additional
break frequency :math:`\nu_{\rm ac}` from stratified SSA (see the theory note: :ref:`stratified_absorption`). This leads
to 8 regimes characterized by the cooling state and the radiation transfer state at maximum:

- A spectrum is either **fast cooling** (:math:`\nu_c < \nu_m`), **slow cooling** (:math:`\nu_m <\nu_c < \nu_{\rm max}`)
  or **no cooling** (:math:`\nu_c > \nu_{\rm max}`).
- A spectrum is optically **thick** at maximum if :math:`\nu_a > \rm{min}(\nu_a,\nu_c)` and is optically **thin** at
  peak if :math:`\nu_a < \rm{min}(\nu_a,\nu_c)`.

The resulting spectra are

1. :math:`(\nu_a < \nu_m < \nu_{\rm max} < \nu_c)`: This is the **thin, no cooling** spectrum.
2. :math:`(\nu_m < \nu_a < \nu_{\rm max} < \nu_c)`: This it the **thick, no cooling** spectrum.
3. :math:`(\nu_a < \nu_m < \nu_c < \nu_{\rm max})`: This is the **thin, slow cooling** spectrum.
4. :math:`(\nu_m < \nu_a < \nu_c < \nu_{\rm max})`: This is the **thick, slow cooling** spectrum.
5. :math:`(\nu_a < \nu_c < \nu_m < \nu_{\rm max})`: This is the **thin, fast cooling** spectrum.
6. :math:`(\nu_c < \nu_a < \nu_m < \nu_{\rm max})`: This is the **thick, fast cooling** spectrum.
7. :math:`(\nu_c, \nu_m < \nu_a < \nu_{\rm max})`: This is the **extremely thick, fast cooling** spectrum.

In the tab set below, we'll go through each of these and discuss the normalization and the corresponding SEDs for the
various cases.

.. tab-set::

    .. tab-item:: Spectrum 1 :math:`(\nu_a < \nu_m < \nu_{\rm max} < \nu_c)`

        This is the **SSA-only** spectrum in which cooling is irrelevant over the
        emitting band because :math:`\nu_c` lies above the high-energy cutoff
        :math:`\nu_{\max}`. It is therefore equivalent to spectrum 1 from our discussion above
        regarding non-cooling synchrotron SEDs.

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_a`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_a \leq \nu < \nu_m`
              - SPL D
              - :math:`1/3`
            * - 3
              - :math:`\nu_m \leq \nu < \nu_{\max}`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = \tilde{F}^{(B,D)}_\nu F^{(D,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_a}\right)^{2}\left(\frac{\nu_a}{\nu_m}\right)^{1/3}, & \nu < \nu_a \quad \text{(SPL B)}\\
                \left(\frac{\nu}{\nu_m}\right)^{1/3},& \nu_a < \nu < \nu_m \quad \text{(SPL D)}\\
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2}, & \nu_m \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
                \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
                \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}\\
            \end{cases}

        The dominant electron population at the spectrum peak is the population of **uncooled electrons**. As such,
        the normalization takes the form of :eq:`slow_cooling_norm`:

        .. math::

            F_{\nu,0} = F_{\nu_m,0}


    .. tab-item:: Spectrum 2 :math:`(\nu_m < \nu_a < \nu_{\rm max} < \nu_c)`

        This is the **SSA-only** spectrum in which cooling is irrelevant over the
        emitting band because :math:`\nu_c` lies above the high-energy cutoff
        :math:`\nu_{\max}`. It is therefore equivalent to spectrum 2 from our discussion above
        regarding non-cooling synchrotron SEDs.

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_m`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_m \leq \nu < \nu_a`
              - SPL A
              - :math:`5/2`
            * - 3
              - :math:`\nu_a \leq \nu < \nu_{\max}`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(B,A)}_\nu \tilde{F}^{(A,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (B,A) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_m}\right)^{2}, & \nu < \nu_m \quad \text{(SPL B)}\\
                \left(\frac{\nu}{\nu_m}\right)^{5/2}, & \nu_m < \nu < \nu_a \quad \text{(SPL A)}\\
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu}{\nu_a}\right)^{-(p-1)/2}, & \nu_a \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
                \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}\\
            \end{cases}

        In this case, we are required to rely on the power-law propagation method described above as our chosen
        normalization frequency is obscured by the effects of self-absorption. Without self-absorption, the normalization
        would be

        .. math::

            F_{\nu,0} = F_{\nu_m,0}

        This corresponds to a flux at :math:`\nu_a` of

        .. math::

            F_{\nu_a} = F_{\nu_m,0}
                        \left(\frac{\nu_a}{\nu_m}\right)^{-(p-1)/2}.

        If we then propagate this flux back to :math:`\nu_m` using the self-absorbed SPL A slope (5/2), we find

        .. math::

            F_{\nu,0} = F_{\nu_a}
                        \left(\frac{\nu_m}{\nu_a}\right)^{5/2}
                        =
                        F_{\nu_m,0}
                        \left(\frac{\nu_a}{\nu_m}\right)^{-(p-1)/2}
                        \left(\frac{\nu_m}{\nu_a}\right)^{5/2}.

    .. tab-item:: Spectrum 3 :math:`(\nu_a < \nu_m < \nu_c < \nu_{\rm max})`

        This is the standard **slow-cooling + SSA** spectrum with all three breaks
        present in-band.

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_a`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_a < \nu < \nu_m`
              - SPL D
              - :math:`1/3`
            * - 3
              - :math:`\nu_m \le \nu < \nu_c`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu_c \le \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 5
              - :math:`\nu \ge \nu_{\max}`
              - SPL I
              - cutoff

        The SBPL SED may be constructed as:

        .. math::

            F_\nu
            =
            F_{\nu_m,0}
            \,
            \tilde{F}_\nu^{(B,D)}(\nu;\nu_a)
            \,
            F_\nu^{(D,G)}(\nu;\nu_m)
            \,
            \tilde{F}_\nu^{(G,H)}(\nu;\nu_c)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        The corresponding discrete SED takes the form

        .. math::

            F_\nu = F_{\nu,0}\begin{cases}
            \left(\dfrac{\nu_a}{\nu_m}\right)^{1/3}\left(\dfrac{\nu}{\nu_a}\right)^2,
            & \nu < \nu_a \quad \text{(SPL B)},\\[6pt]
            \left(\dfrac{\nu}{\nu_m}\right)^{1/3},
            & \nu_a \le \nu < \nu_m \quad \text{(SPL D)},\\[6pt]
            \left(\dfrac{\nu}{\nu_m}\right)^{-(p-1)/2},
            & \nu_m \le \nu < \nu_c \quad \text{(SPL G)},\\[6pt]
            \left(\dfrac{\nu_c}{\nu_m}\right)^{-(p-1)/2}
            \left(\dfrac{\nu}{\nu_c}\right)^{-p/2},
            & \nu_c \le \nu < \nu_{\max} \quad \text{(SPL H)},\\[6pt]
            \left(\dfrac{\nu_c}{\nu_m}\right)^{-(p-1)/2}
            \left(\dfrac{\nu_{\max}}{\nu_c}\right)^{-p/2}
            \Phi(\nu,\nu_{\rm max}),& \nu > \nu_{\rm max}  \quad \text{(SPL I)}
            \end{cases}

        The dominant electron population at the spectrum peak is the population of **uncooled electrons**. As such,
        the normalization takes the form of :eq:`slow_cooling_norm`:

        .. math::

            F_{\nu,0} = F_{\nu_m,0}


    .. tab-item:: Spectrum 4 :math:`(\nu_m < \nu_a < \nu_c < \nu_{\rm max})`

        This is the **slow-cooling + SSA** spectrum with :math:`\nu_a` above
        :math:`\nu_m`, producing an optically thick :math:`\nu^{5/2}` segment
        between :math:`\nu_m` and :math:`\nu_a`.

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_m`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_m < \nu < \nu_a`
              - SPL A
              - :math:`5/2`
            * - 3
              - :math:`\nu_a \le \nu < \nu_c`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu_c \le \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 5
              - :math:`\nu \ge \nu_{\max}`
              - SPL I
              - cutoff

        We anchor the SED at :math:`\nu_m` so that the SED takes the form

        .. math::

            F_\nu
            =
            F_{\nu_m,0}
            \,
            F_\nu^{(B,A)}(\nu;\nu_m)
            \,
            \tilde{F}_\nu^{(A,G)}(\nu;\nu_a)
            \,
            \tilde{F}_\nu^{(G,H)}(\nu;\nu_c)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        The corresponding discrete SED takes the form

        .. math::

            F_\nu = F_{\nu,0}\begin{cases}
            \left(\frac{\nu}{\nu_m}\right)^2,&\nu<\nu_m \quad \text{(SPL B)}\\
            \left(\frac{\nu}{\nu_m}\right)^{5/2},&\nu_m<\nu<\nu_a \quad \text{(SPL A)}\\
            \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
            \left(\frac{\nu}{\nu_a}\right)^{-(p-1)/2},& \nu_a < \nu < \nu_c \quad \text{(SPL G)}\\
            \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
            \left(\frac{\nu_c}{\nu_a}\right)^{-(p-1)/2}
            \left(\frac{\nu}{\nu_c}\right)^{-p/2},& \nu_c < \nu < \nu_{\rm max} \quad \text{(SPL H)}\\
            \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
            \left(\frac{\nu_c}{\nu_a}\right)^{-(p-1)/2}
            \left(\frac{\nu}{\nu_c}\right)^{-p/2}
            \Phi(\nu,\nu_{\rm max}),& \nu > \nu_{\rm max} \quad \text{(SPL I)}\\
            \end{cases}

        Because the anchor point for the SED is specified in the optically thick regime, we cannot use
        the uncorrected normalization. We therefore use our power-law propagation technique to transport the
        expected (without absorption) flux at :math:`\nu_m` down to :math:`\nu_a` and then back along the correct
        power-law back to :math:`\nu_m`. The resulting normalization is

        .. math::

            F_{\nu,0} = F_{\nu_m,0}
            \left(\frac{\nu_a}{\nu_m}\right)^{-(p-1)/2}
            \left(\frac{\nu_m}{\nu_a}\right)^{5/2}\frac{V}{D^2}


    .. tab-item:: Spectrum 5 :math:`(\nu_a < \nu_c < \nu_m < \nu_{\rm max})`

        Spectrum 5 is the first of the two spectra in this formalism which is subject to the
        effects of **stratified SSA**, which introduces an additional SSA break at a frequency
        :math:`\nu_{\rm ac}`. The segments of the spectrum are

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_{ac}`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_{ac} < \nu < \nu_a`
              - SPL C
              - :math:`11/8`
            * - 3
              - :math:`\nu_a < \nu < \nu_c`
              - SPL E
              - :math:`1/3`
            * - 4
              - :math:`\nu_c \le \nu < \nu_m`
              - SPL F
              - :math:`-1/2`
            * - 5
              - :math:`\nu_m \le \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 6
              - :math:`\nu \ge \nu_{\max}`
              - SPL I
              - cutoff

        The SBPL SED may be constructed as:

        .. math::

            F_\nu
            =
            \,
            \tilde{F}_\nu^{(B,C)}(\nu;\nu_{\rm ac})
            \,
            \tilde{F}_\nu^{(C,E)}(\nu;\nu_a)
            \,
            F_\nu^{(E,F)}(\nu;\nu_c)
            \,
            \tilde{F}_\nu^{(F,H)}(\nu;\nu_m)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        Piecewise spectrum (normalized at :math:`\nu_c`):

        .. math::

            F_\nu = F_{\nu,0}\begin{cases}
            \left(\frac{\nu_a}{\nu_c}\right)^{1/3}
            \left(\frac{\nu_{\rm ac}}{\nu_a}\right)^{11/8}
            \left(\frac{\nu}{\nu_{\rm ac}}\right)^2,& \nu < \nu_{\rm ac},\quad \text{(SPL B)}\\[6pt]
            \left(\frac{\nu_a}{\nu_c}\right)^{1/3}
            \left(\frac{\nu}{\nu_a}\right)^{11/8},& \nu_{\rm ac} < \nu < \nu_a,\quad \text{(SPL C)}\\[6pt]
            \left(\dfrac{\nu}{\nu_c}\right)^{1/3},& \nu_a \le \nu < \nu_c,\quad \text{(SPL E)}\\[6pt]
            \left(\dfrac{\nu}{\nu_c}\right)^{-1/2},& \nu_c \le \nu < \nu_m,\quad \text{(SPL F)}\\[6pt]
            \left(\dfrac{\nu_m}{\nu_c}\right)^{-1/2}
            \left(\dfrac{\nu}{\nu_m}\right)^{-p/2},& \nu_m \le \nu < \nu_{\max},\quad \text{(SPL H)}\\[6pt]
            \left(\dfrac{\nu_m}{\nu_c}\right)^{-1/2}
            \left(\dfrac{\nu_{\max}}{\nu_m}\right)^{-p/2}
            \Phi_{\rm cut}(\nu;\nu_{\max}),& \nu \ge \nu_{\max} \quad \text{(SPL I)}.
            \end{cases}


        The dominant electron population at the spectrum peak is the population of **cooled electrons**. As such,
        the normalization takes the form of :eq:`cooling_norm`:

        .. math::

            F_{\nu,0} = F_{c,0}


    .. tab-item:: Spectrum 6 :math:`(\nu_c < \nu_a < \nu_m < \nu_{\rm max})`

        Spectrum 6 is the second case in which the SSA break due to stratified SSA appears at
        :math:`\nu_{\rm ac}`. Additionally, because :math:`\nu_c` is obscured by SSA, we also have to
        perform power-law propagation to correct the normalization, making this one of the trickier of the
        SED cases.

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - PLS Type
              - Slope
            * - 1
              - :math:`\nu < \nu_{\rm ac}`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_{\rm ac} < \nu < \nu_a`
              - SPL C
              - :math:`11/8`
            * - 3
              - :math:`\nu_a \le \nu < \nu_m`
              - SPL F
              - :math:`-1/2`
            * - 4
              - :math:`\nu_m \le \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 5
              - :math:`\nu \ge \nu_{\max}`
              - SPL I
              - cutoff

        The SBPL SED may be constructed as:

        .. math::

            F_\nu
            =
            \,
            \tilde{F}_\nu^{(B,C)}(\nu;\nu_{\rm ac})
            \,
            F_\nu^{(C,F)}(\nu;\nu_a)
            \,
            \tilde{F}_\nu^{(F,H)}(\nu;\nu_m)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        The corresponding discrete SED is

        .. math::

            F_\nu = F_{\nu,0}\begin{cases}
            \left(\frac{\nu_{\rm ac}}{\nu_{\rm a}}\right)^{11/8}
            \left(\frac{\nu}{\nu_{\rm ac}}\right)^2,& \nu < \nu_{\rm ac},\quad \text{(SPL B)}\\[6pt]
            \left(\frac{\nu}{\nu_{\rm a}}\right)^{11/8},& \nu_{\rm ac} < \nu < \nu_a,\quad \text{(SPL C)}\\[6pt]
            \left(\dfrac{\nu}{\nu_a}\right)^{-1/2},& \nu_a \le \nu < \nu_m,\quad \text{(SPL F)}\\[6pt]
            \left(\dfrac{\nu_m}{\nu_a}\right)^{-1/2}
            \left(\dfrac{\nu}{\nu_m}\right)^{-p/2},& \nu_m \le \nu < \nu_{\max},\quad \text{(SPL H)}\\[6pt]
            \left(\dfrac{\nu_m}{\nu_a}\right)^{-1/2}
            \left(\dfrac{\nu_{\max}}{\nu_m}\right)^{-p/2}
            \Phi_{\rm cut}(\nu;\nu_{\max}),& \nu \ge \nu_{\max} \quad \text{(SPL I)}.
            \end{cases}

        The dominant electron population at the spectrum peak is the population of **cooled electrons**. As such,
        the normalization takes the form of :eq:`cooling_norm`:

        .. math::

            F_{\nu,0} = F_{c,0};

        however, the spectrum is not optically thin at :math:`\nu_c`, meaning that we must propagate from :math:`\nu_c`
        to :math:`\nu_a` using the cooled slope. We then specify the normalization of the SED relative to the
        absorption peak. Thus,

        .. math::

            F_{\nu,0} = F_{c,0} \left(\frac{\nu_c}{\nu_a}\right)^{-1/2}.


    .. tab-item:: Spectrum 7 :math:`(\nu_c, \nu_m < \nu_a < \nu_{\rm max})`

        This spectrum corresponds to scenarios where SSA is dominant over both cooling and
        the minimum injection break. In this regime, the relative ordering of :math:`\nu_m` and
        :math:`\nu_c` is irrelevant because the post-shock material becomes optically thick to
        SSA immediately and so cooled material does not have the ability to contribute to the
        spectrum. We therefore see the traditional low-energy tail :math:`\nu^2` up to the
        minimum injection energy :math:`\nu_m`, beyond which we obtain the standard
        :math:`\nu^{5/2}` scaling. Finally, beyond the absorption break, we have optically
        thin emission from the steady state cooled population of electrons deeper in the
        post-shock material producing the typical :math:`\nu^{-p/2}`.

        In this spectrum, the regimes are as follows

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_m`
              - SPL B
              - :math:`2`
            * - 3
              - :math:`\nu_m \le \nu < \nu_a`
              - SPL A
              - :math:`5/2`
            * - 4
              - :math:`\nu_a \le \nu < \nu_{\rm max}`
              - SPL H
              - :math:`-p/2`
            * - 5
              - :math:`\nu \ge \nu_{\rm max}`
              - SPL I
              - cutoff

        The SBPL SED may be constructed as:

        .. math::

            F_\nu
            =
            F_{\nu_m,0}
            \,
            F_\nu^{(B,A)}(\nu;\nu_m)
            \,
            \tilde{F}_\nu^{(A,H)}(\nu;\nu_a)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        The discrete SED is therefore

        .. math::

            F_\nu
            =
            F_{\nu,0}
            \begin{cases}
                \left(\frac{\nu}{\nu_m}\right)^{2},
                & \nu < \nu_m
                \quad \text{(SPL B)}
                \\[6pt]
                \left(\frac{\nu}{\nu_m}\right)^{5/2},
                & \nu_m < \nu < \nu_a
                \quad \text{(SPL A)}
                \\[6pt]
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu}{\nu_a}\right)^{-p/2},
                & \nu_a < \nu < \nu_{\rm max}
                \quad \text{(SPL H)}
                \\[6pt]
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu_{\rm max}}{\nu_a}\right)^{-p/2}
                \Phi(\nu,\nu_{\rm max})
                & \nu > \nu_{\rm max}
                \quad \text{(SPL I)}.
            \end{cases}

        Normalization of this SED should be done thoughtfully as, at first glance, it may
        seem that the normalization should depend on the ordering of :math:`\nu_m` and
        :math:`\nu_c`. Critically, for this SED we conclude that the SSA photosphere must
        correspond to the initial shock surface at which point, the electrons have not yet had
        an opportunity to cool. As such, we propagate from :math:`\nu_m` to :math:`\nu_a` using
        the uncooled slope

        .. math::

            F_{\nu,0}
            =
            \left[
            c_5(p)\,N_0\,(m_e c^2)^{p-1}
            \left(B\sin\alpha\right)^{(p+1)/2}
            \left(\frac{\nu_m}{2c_1}\right)^{-(p-1)/2}
            \right]\left(\frac{\nu_a}{\nu_m}\right)^{-(p-1)/2}
            \left(\frac{\nu_m}{\nu_a}\right)^{5/2}\frac{V}{D^2}



References
----------
.. footbibliography::
