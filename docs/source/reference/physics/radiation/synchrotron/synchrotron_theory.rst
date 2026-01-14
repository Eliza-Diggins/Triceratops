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

To describe, from first principles, the relevant synchrotron theory used in Triceratops, we first need to develop
the emission of a single relativistic electron spiraling in a magnetic field. With that in hand, it will be a
relatively straightforward process to extend to a population of electrons with some distribution function.

Synchrotron emission is the relativistic generalization of cyclotron emission, which is the radiation emitted
by a charged particle spiraling in a magnetic field. The key difference is that in the relativistic case, the
emission is beamed in the direction of motion of the particle, leading to a characteristic spectrum and angular distribution.

From the Fourier uncertainty principle, the shorter a pulse of radiation lasts in time, the broader its frequency
composition must be. Thus, for the relativistic electrons in this case, the emission is concentrated in short pulses
as the electron's velocity vector sweeps past the observer. This leads to a characteristic synchrotron spectrum which
can extend to very high frequencies.

The Cyclotron Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^

In the classical case, we consider a magnetic field :math:`{\bf B} = B_0 \hat{\bf z}` and a particle of
charge :math:`q` bound to the :math:`x-y`plane. The **Lorentz Force** dictates that

.. math::

    {\bf F} = m {\bf a} = \frac{q}{c}{\bf v} \times {\bf B}.

This will cause gyroscopic motion about the field lines. We imagine constructing our coordinate system
so that :math:`{\bf v}_0 = v_0 \hat{\bf x}.` Then we achieve a force always directed inward of magnitude

.. math::

    F = \frac{qB_0v_0}{c} = m\omega^2 r \implies \omega = \frac{qB_0}{mc}.

This is the **cyclotron frequency** of the particle:

.. math::

    \boxed{
    \omega_{\rm cyclotron} = \frac{qB_0}{mc},\;\text{(gaussian units)}.
    }

In the relativistic case, we must account for time dilation. The relativistic gyrofrequency is therefore

.. math::

    \boxed{
    \omega_B = \frac{qB_0}{\gamma mc},\;\text{(gaussian units)}.
    }

Synchrotron Power
^^^^^^^^^^^^^^^^^^^^^

Already, because we know the acceleration of the electron, we can compute the total power emitted from synchrotron
radiation using the Larmor formula. The Larmor formula states that the power emitted by an accelerating charge is

.. math::

    P = \frac{2}{3} \frac{q^2 a^2}{c^3}.

Given that the acceleration from the Lorentz force must maintain circular motion, we have

.. math::

    a = \omega_B^2 r = \omega_B^2 v_{\perp} = \frac{qB_0 v_\perp}{\gamma m c}

so,

.. math::

    P = \frac{2}{3} \frac{q^4 \gamma^2 B^2}{c^5 m^2} v_{\perp}^2 = \frac{4}{3} \sigma_T c \beta^2 \gamma^2 U_B.

In the relativistic limit (:math:`\beta \approx 1`), this becomes

.. math::

    \boxed{
    P_{\rm synch} = \frac{4}{3} \sigma_T c \gamma^2 U_B,
    }

where :math:`U_B = B^2/8\pi` is the magnetic energy density.

The Characteristic Synchrotron Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The critical physical insight that distinguishes **synchrotron radiation** from its
non-relativistic cyclotron counterpart is the emergence of **strong relativistic beaming**.
Although an accelerated charged particle emits radiation continuously in its instantaneous
rest frame, relativistic aberration causes that emission to be concentrated into a narrow
forward cone of angular width :math:`\Delta\theta \sim 1/\gamma` in the lab frame.

As a relativistic particle spirals along a magnetic field line, this forward emission cone
does not remain pointed toward a distant observer. Instead, the observer receives radiation
only during the brief interval when the cone sweeps across the line of sight. The observed
signal therefore takes the form of a **temporally localized pulse**, rather than a continuous
sinusoidal wave.

From Fourier theory, a signal confined to a short interval in time necessarily corresponds to
a **broad frequency spectrum**. Thus, even though the underlying motion of the particle is
periodic, relativistic beaming transforms what would otherwise be narrow-band cyclotron
emission into the broadband synchrotron spectrum characteristic of relativistic sources.

The particle emits
radiation into a cone of angular width :math:`\Delta\theta \sim 2/\gamma`. As the particle
moves along its curved trajectory of radius :math:`a`, the arc length over which the observer
remains inside the beam is approximately

.. math::

    \Delta s \simeq \frac{2a}{\gamma}.

The radius of the particle’s helical trajectory is

.. math::

    r = \frac{v}{\omega_B \sin\varphi},

where :math:`\varphi` is the pitch angle between the particle velocity and the magnetic field,
and :math:`\omega_B = qB/(\gamma m_e c)` is the relativistic gyrofrequency.

The pulse begins when the edge of the emission cone first intersects the observer’s line of
sight and ends when it exits the cone. The corresponding emission time interval is therefore

.. math::

    \Delta t_{\rm emit}
    \simeq
    \frac{2}{\gamma \omega_B \sin\varphi}.

Because photons emitted at the beginning and end of the pulse originate from different
locations along the trajectory, there is an additional light-travel-time compression factor.
Accounting for this effect yields an observed pulse duration

.. math::

    \Delta t_{\rm obs}
    \simeq
    \frac{2}{\gamma \omega_B \sin\varphi}(1-\beta).

In the ultra-relativistic limit, :math:`1-\beta \simeq 1/(2\gamma^2)`, so the pulse duration
scales as

.. math::

    \Delta t_{\rm obs}
    \sim
    \frac{1}{\gamma^3 \omega_B \sin\varphi}.

This timescale defines a characteristic upper frequency in the synchrotron spectrum,
corresponding to the inverse pulse duration. We therefore define the **critical synchrotron
frequency** as

.. math::

    \omega_c
    =
    \frac{3}{2}\,\gamma^3 \omega_B \sin\varphi.

Expressed as a linear frequency, this becomes

.. math::

    \boxed{
    \nu_c
    =
    \frac{3}{4\pi}\,\gamma^3 \omega_B \sin\varphi
    =
    \frac{3}{4\pi}\,\gamma^2\,\frac{qB}{m_e c}\,\sin\varphi.
    }

Although :math:`\nu_c` is often colloquially referred to as the “synchrotron frequency” or
“fundamental frequency,” it does **not** correspond to the peak of the emitted spectrum.
Instead, the single-electron synchrotron spectrum peaks at approximately
:math:`\nu \simeq 0.29\,\nu_c`. Nevertheless, :math:`\nu_c` provides a convenient and physically
meaningful scale that sets the overall frequency range of synchrotron emission.

It is common to define the constant

.. math::

    c_1 \equiv \frac{3e}{4\pi m_e c},

in terms of which the characteristic frequency may be written compactly as

.. math::

    \nu_c = c_1\,B\,\sin\varphi\,\gamma^2.

.. important::

    For populations of electrons, we often define the characteristic frequency as :math:`\nu_c(\gamma)` for a
    particular choice of :math:`\gamma`. In particular, when considering power-law distributions of electrons,
    we often define the characteristic frequency as that corresponding to the minimum Lorentz factor.

    The convention for this frequency varies by a constant of order unity in the literature. See the section below
    for the Triceratops convention.


The Single Electron Synchrotron Spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we will derive the synchrotron spectrum of a single electron spiraling in a magnetic field. From
elementary electrodynamics (see :footcite:t:`RybickiLightman` for a detailed treatment), the power radiated per unit
frequency per unit solid angle is given by

.. math::

    \frac{dE}{dt\,d\omega d\Omega} = \frac{q^2 \omega^2}{4\pi^2 c}\left|
    \int_{-\infty}^{\infty} \frac{\hat{n} \times (\hat{n} \times \beta)}{1-\hat{n} \cdot \beta}
    e^{i\omega(t' - \hat{n} \cdot r(t')/c)} dt' \right|^2.

Consider an electron such that, at a retarded time :math:`t' = 0`, it is a distance :math:`a` from the origin and has
velocity along the x-axis. Let
the observer be oriented along :math:`\hat{n}` and let the vector from the electron's position to the origin be
:math:`\boldsymbol{\epsilon}_\perp`. Let :math:`\boldsymbol{\epsilon}_\parallel = \hat{n} \times
\boldsymbol{\epsilon}_\perp` be the vector perpendicular to both :math:`\hat{n}` and :math:`\boldsymbol{\epsilon}_\perp`.
Finally, let :math:`\theta` be the angle between :math:`\hat{\bf n}` and the x-axis.

In this geometry, the phase angle of the electron in its orbit is just

.. math::

    \phi = \frac{vt'}{a},

and the velocity vector is

.. math::

    \boldsymbol{\beta} = \beta \left(\cos \phi \hat{\bf x} + \sin \phi \boldsymbol{\epsilon}_\perp \right).

Thus, the cross product in the integrand becomes

.. math::

    \hat{n} \times (\hat{n} \times \beta) = \beta\left(\cos \phi \sin \theta \boldsymbol{\epsilon}_\parallel -
    \sin \phi \boldsymbol{\epsilon}_\perp \right).

The phase term in the exponential is

.. math::

    t - \frac{\hat{n} \cdot r(t')}{c} = t' - \frac{a}{c} \cos \theta \sin \phi.

After some simplifications (see the drop down), we arrive at the final result for the power radiated per unit frequency
per unit solid angle:

.. dropdown:: Small-Angle Expansion and Evaluation of the Master Integral

    We now evaluate the master expression for the radiated energy in the relativistic limit.
    For ultra-relativistic particles we take :math:`\beta \simeq 1`, and because synchrotron
    emission is concentrated into short pulses and strongly beamed, we may expand all angular
    quantities to lowest nontrivial order.

    In particular, for small angles :math:`\theta` and :math:`\phi` we use

    .. math::

        \cos\theta \simeq 1 - \frac{\theta^2}{2},
        \qquad
        \sin\phi \simeq \phi - \frac{\phi^3}{6},

    which implies

    .. math::

        \cos\theta\,\sin\phi
        \simeq
        \phi - \frac{\phi\theta^2}{2} - \frac{\phi^3}{6}.

    Using the orbital phase relation :math:`\phi = vt'/a`, we obtain

    .. math::

        \cos\theta\,\sin\phi
        \simeq
        \frac{vt'}{a}\left(1 - \frac{\theta^2}{2}\right)
        - \frac{v^3 t'^3}{6a^3}.

    Substituting this into the phase factor of the exponential yields

    .. math::

        t - \frac{\hat{n}\cdot r(t')}{c}
        =
        (1 - \beta)t'
        + \frac{\beta\theta^2}{2}t'
        + \frac{\beta^3 c^2 t'^3}{6a^2}.

    Factoring out :math:`\Gamma^{-2} \simeq 2(1-\beta)`, this may be written as

    .. math::

        t
        =
        \frac{1}{2\Gamma^2}
        \left[
            \left(1 + \Gamma^2\theta^2\right)t'
            + \frac{c^2\Gamma^2 t'^3}{3a^2}
        \right].

    The master formula may now be decomposed into its perpendicular and parallel polarization
    components. After simplification, we find

    .. math::

        \begin{aligned}
        \frac{dE_\perp}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2}{4\pi^2 c}
        \left|
        \int_{-\infty}^{\infty}
        \frac{ct'}{a}
        \exp\!\left[
            \frac{i\omega}{2\Gamma^2}
            \left(
                \Theta_\gamma^2 t'
                + \frac{c^2\Gamma^2 t'^3}{3a^2}
            \right)
        \right]
        dt'
        \right|^2, \\[6pt]
        \frac{dE_\parallel}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2\theta^2}{4\pi^2 c}
        \left|
        \int_{-\infty}^{\infty}
        \exp\!\left[
            \frac{i\omega}{2\Gamma^2}
            \left(
                \Theta_\gamma^2 t'
                + \frac{c^2\Gamma^2 t'^3}{3a^2}
            \right)
        \right]
        dt'
        \right|^2,
        \end{aligned}

    where we have defined

    .. math::

        \Theta_\gamma^2 \equiv 1 + \Gamma^2\theta^2.

    Although these integrals appear formidable, they may be reduced to standard forms by
    introducing the dimensionless variables

    .. math::

        y = \Gamma \frac{ct'}{a\Theta_\gamma},
        \qquad
        \eta = \frac{\omega a \Theta_\gamma^3}{3c\Gamma^3}.

    In terms of these variables, the expressions simplify to

    .. math::

        \boxed{
        \begin{aligned}
        \frac{dE_\perp}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2}{4\pi^2 c}
        \left(\frac{a\Theta_\gamma^2}{\Gamma^2 c}\right)^2
        \left|
        \int_{-\infty}^{\infty}
        y
        \exp\!\left[
            \frac{3i\eta}{2}
            \left(
                y + \frac{y^3}{3}
            \right)
        \right]
        dy
        \right|^2, \\[6pt]
        \frac{dE_\parallel}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2\theta^2}{4\pi^2 c}
        \left(\frac{a\Theta_\gamma}{\Gamma c}\right)^2
        \left|
        \int_{-\infty}^{\infty}
        y
        \exp\!\left[
            \frac{3i\eta}{2}
            \left(
                y + \frac{y^3}{3}
            \right)
        \right]
        dy
        \right|^2.
        \end{aligned}
        }

    These integrals are well known and may be expressed in terms of modified Bessel functions.
    Evaluating them yields the standard synchrotron expressions

    .. math::

        \boxed{
        \begin{aligned}
        \frac{dE_\perp}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2}{3\pi c}
        \left(\frac{a\Theta_\gamma^2}{\Gamma^2 c}\right)^2
        K_{2/3}^2(\eta), \\[6pt]
        \frac{dE_\parallel}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2\theta^2}{3\pi^2 c}
        \left(\frac{a\Theta_\gamma}{\Gamma c}\right)^2
        K_{1/3}^2(\eta),
        \end{aligned}
        }

    where :math:`K_\nu` denotes the modified Bessel function of the second kind. Because the
    emission is strongly beamed, the dominant contribution arises near :math:`\theta \simeq 0`,
    in which case

    .. math::

        \eta \simeq \frac{\omega}{2\omega_c},

    with :math:`\omega_c` the characteristic synchrotron frequency.

.. math::

    \boxed{
    \begin{aligned}
    \frac{dE_\perp}{dt\,d\omega\,d\Omega}
    &=
    \frac{q^2\omega^2}{3\pi c}
    \left(\frac{a\Theta_\gamma^2}{\Gamma^2 c}\right)^2
    K_{2/3}^2(\eta), \\[6pt]
    \frac{dE_\parallel}{dt\,d\omega\,d\Omega}
    &=
    \frac{q^2\omega^2\theta^2}{3\pi^2 c}
    \left(\frac{a\Theta_\gamma}{\Gamma c}\right)^2
    K_{1/3}^2(\eta),
    \end{aligned}
    }

where :math:`K_\nu` denotes the modified Bessel function of the second kind. Because the
emission is strongly beamed, the dominant contribution arises near :math:`\theta \simeq 0`,
in which case

.. math::

    \eta \simeq \frac{\omega}{2\omega_c},

with :math:`\omega_c` the characteristic synchrotron frequency.

Because the radiation is strongly beamed, it will only emit into a thin band of solid angles with width approximately
:math:`1/\gamma` centered around the cone :math:`1/\alpha` (see :footcite:t:`RybickiLightman` for details). Integrating
over solid angle, we find the total power radiated per unit frequency is

.. dropdown:: Details

.. math::

    \begin{aligned}
        \frac{dE_\perp}{dt\;d\omega} &= \frac{\sqrt{3} q^2 \Gamma \sin \alpha}{2c}\left(F(x)+G(x)\right)\\
        \frac{dE_\parallel}{dt\;d\omega} &= \frac{\sqrt{3}q^2 \Gamma \sin \alpha}{2c} \left(F(x)-G(x)\right)
    \end{aligned}

where

.. math::

    F(x) = x \int_x^\infty K_{5/3}(\xi)\;d\xi,\;\;G(x) = xK_{2/3}(x).

Summing these two contributions, the total power radiated per unit frequency is

.. math::

    \boxed{
    P(\omega) = \frac{\sqrt{3}q^3 B \sin\alpha}{2\pi m c^2} F\left(\frac{\omega}{\omega_c}\right),
    }

In terms of frequency,

.. math::

    \boxed{
    P(\nu) = \frac{\sqrt{3}q^3 B \sin\alpha}{m c^2} F\left(\frac{\nu}{\nu_c}\right),
    }

where :math:`\nu_c` is the characteristic synchrotron frequency defined previously.

.. important::

    In Triceratops, :math:`F(x)` is referred to as the **first synchrotron kernel** and :math:`G(x)` as
    the **second synchrotron kernel**.



Synchrotron From A Population of Electrons
------------------------------------------

With the all-important single-electron synchrotron spectrum in hand, we can now extend our arguments to the
emission from a population of electrons. Let the distribution function be :math:`\frac{dN}{d\gamma}`. Then,

.. math::

    P(\nu) = \int_{\gamma_{\rm min}}^{\gamma_{\rm max}} \frac{dN}{d\gamma} P(\nu, \gamma) d\gamma =
    \frac{\sqrt{3} q^3 B \sin \alpha}{mc^2} \int_{\gamma_{\rm min}}^{\gamma_{\rm max}} \frac{dN}{d\gamma} F\left(
        \frac{\nu}{\nu_c(\gamma)} \right) d\gamma,

where :math:`P(\nu, \gamma)` is the single-electron synchrotron power derived previously.

Now, in general, this integration is not analytically tractable for arbitrary distribution functions; however, there
are some scenarios which are of particular relevance to astrophysical sources where analytic results can be obtained.

.. note::

    Support for performing these integrations numerically for arbitrary distribution functions is a planned
    feature of Triceratops, but not currently implemented.

The Spectrum of a Power-Law Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most important case in which the above formalism has a known closure is that of a power-law distribution
of electrons. This is the most common assumption in astrophysical synchrotron modeling, as it is both
theoretically motivated by diffusive shock acceleration:footcite:p:`caprioliParticleAccelerationShocks2023` and
empirically successful in explaining non-thermal emission across a wide range of astrophysical environments.

Assuming a power-law distribution of electrons

.. math::

    \frac{dN}{d\gamma} = N_0 \gamma^{-p},\;\;\gamma_{\min} \le \gamma \le \gamma_{\max},

we can compute the integral above to find the famous result (see e.g. :footcite:t:`RybickiLightman`,
:footcite:t:`demarchiRadioAnalysisSN2004C2022` etc):

.. math::

    \boxed{
    P(\nu) = c_5(p) N_0 B^{(p+1)/2} \nu^{-(p-1)/2},
    }

where :math:`c_5(p)` is one of a number of constants which appear in the literature to disguise the complex
expressions involving gamma functions and powers of fundamental constants:

.. math::

    c_5(p) = \frac{\sqrt{3}}{16 \pi} \left(\frac{e^3}{m_ec^2}\right) \frac{p+7/3}{p+1} \Gamma\left(\frac{3p-1}{12}\right)
    \Gamma\left(\frac{3p + 7}{12}\right).

.. important::

    Conventions differ on the definition of this :math:`c_5(p)` function. By default, Triceratops adopts the convention
    of :footcite:t:`1970ranp.book.....P`, which has become the general standard in the astrophysical literature
    (see e.g. :footcite:t:`demarchiRadioAnalysisSN2004C2022`, :footcite:t:`Margutti2019COW`, etc.). However,
    other versions do arise in the literature (e.g. :footcite:t:`RybickiLightman`), so care should be taken
    when comparing results from different sources.

It should be noted that, because :math:`N(\gamma)` is (traditionally) thought of as the number *density* of electrons,
the above expression for :math:`P(\nu)` is the power emitted per unit volume per unit frequency. It is therefore more
precisely a power spectral density.

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

.. hint::

    The relevant API in Triceratops is in the :mod:`radiation.synchrotron.microphysics` module. See
    :ref:`synchrotron_microphysics` for details on use.

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

.. _electron_cooling:
Cooling of Electrons
^^^^^^^^^^^^^^^^^^^^

As a population, relativistic electrons will cool and thereby evolve away from their injected energy distribution.
This evolution can imprint additional spectral breaks and modify the normalization of the synchrotron emission.

Let :math:`P(\gamma,t)` denote the differential electron distribution (number or number density per unit Lorentz factor),
so that :math:`P(\gamma,t)\,d\gamma` is the number of electrons in :math:`(\gamma,\gamma+d\gamma)`. Suppose electrons are
injected at a rate :math:`Q(\gamma)` (per unit time per unit :math:`\gamma`) and cool according to

.. math::

    \dot{\gamma} \equiv \frac{d\gamma}{dt} = H(\gamma),

with :math:`H(\gamma) < 0` for radiative cooling. Conservation of particle number in energy space implies that, for any
interval :math:`\gamma\in[\gamma_1,\gamma_2]`,

.. math::

    \frac{d}{dt}\int_{\gamma_1}^{\gamma_2} P(\gamma,t)\,d\gamma
    =
    \int_{\gamma_1}^{\gamma_2} Q(\gamma)\,d\gamma
    - \left[ H(\gamma)\,P(\gamma,t)\right]_{\gamma_1}^{\gamma_2}.

Taking :math:`\gamma_2\to\gamma_1` yields the continuity equation

.. math::

    \frac{\partial P(\gamma,t)}{\partial t}
    +
    \frac{\partial}{\partial\gamma}\!\left[H(\gamma)\,P(\gamma,t)\right]
    =
    Q(\gamma).

.. note::

    This first-order PDE is solvable by the method of characteristics for general :math:`H(\gamma)` and :math:`Q(\gamma)`.
    In this documentation we focus on steady-state solutions commonly used in synchrotron modeling.

.. note::

    We have here ignored the effects of adiabatic cooling, which can be important in expanding systems.

Steady-state power-law injection with power-law cooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In steady state (:math:`\partial P/\partial t=0`), the continuity equation reduces to

.. math::

    \frac{d}{d\gamma}\!\left[H(\gamma)\,P(\gamma)\right] = Q(\gamma).

Assuming :math:`P(\gamma)\to 0` sufficiently fast as :math:`\gamma\to\infty`, integration gives

.. math::

    H(\gamma)\,P(\gamma) = -\int_{\gamma}^{\infty} Q(\gamma')\,d\gamma'.

Now take a power-law injection spectrum

.. math::

    Q(\gamma) = Q_0\,\gamma^{-p}, \qquad p>1,

and a power-law cooling law

.. math::

    H(\gamma) = -\Lambda\,\gamma^k, \qquad \Lambda>0.

Then

.. math::

    \int_{\gamma}^{\infty} Q(\gamma')\,d\gamma'
    =
    \frac{Q_0}{p-1}\,\gamma^{-(p-1)},

and the steady-state distribution becomes

.. math::

    \boxed{
    P_{\rm SS}(\gamma)
    =
    \frac{Q_0}{\Lambda\,(p-1)}\,\gamma^{-(p+k-1)}.
    }

Thus, radiative cooling steepens the injected spectrum by :math:`k-1`. For synchrotron/Thomson IC cooling,
:math:`\dot{\gamma}\propto -\gamma^2` (i.e. :math:`k=2`), so the steady-state slope steepens by one:
:math:`P\propto \gamma^{-(p+1)}` above the cooling break.

Cooling break at :math:`\gamma_c`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many applications, only electrons above a critical Lorentz factor :math:`\gamma_c` cool efficiently within the
relevant timescale (often the dynamical time). A common phenomenological approximation is:

- for :math:`\gamma>\gamma_c`, electrons are in the cooled steady-state regime, :math:`P\propto \gamma^{-(p+k-1)}`;
- for :math:`\gamma<\gamma_c`, cooling is negligible and the distribution retains the injection slope, :math:`P\propto\gamma^{-p}`.

Imposing continuity at :math:`\gamma=\gamma_c` yields the broken power-law form

.. math::

    \boxed{
    P_{\rm SS}(\gamma)
    =
    \begin{cases}
        \dfrac{Q_0}{\Lambda\,(p-1)}\,\gamma^{-(p+k-1)}, & \gamma > \gamma_c, \\[10pt]
        \dfrac{Q_0}{\Lambda\,(p-1)}\,\gamma_c^{-(k-1)}\,\gamma^{-p}, & \gamma < \gamma_c.
    \end{cases}
    }

Synchrotron Cooling
~~~~~~~~~~~~~~~~~~~

The most obvious source of cooling from relativistic electrons in a synchrotron-emitting plasma is the synchrotron
radiation itself. The power radiated by a single electron due to synchrotron emission is

.. math::

    P_{\rm synch} = \frac{4}{3} \sigma_T c \gamma^2 \beta^2 U_B,

where :math:`\sigma_T` is the Thomson cross section, :math:`c` is the speed of light, :math:`\gamma` is the
Lorentz factor of the electron, :math:`\beta` is the dimensionless velocity (:math:`\beta = v/c`),
and :math:`U_B` is the magnetic energy density. In the relativistic limit (:math:`\beta \approx 1`), this simplifies to

.. math::

    P_{\rm synch} \approx \frac{4}{3} \sigma_T c \gamma^2 U_B.

.. hint::

    Since :math:`P \propto \gamma^2`, the resulting steady state distribution steepens by one power in the
    synchrotron-cooled regime (see previous section).

In this case, the cooling timescale is

.. math::

    \tau_{\rm synch, cool} = \frac{\gamma m_e c^2}{P_{\rm synch}} = \frac{3 m_e c}{4 \sigma_T U_B \gamma}.

If we then define a dynamical timescale :math:`t_{\rm dyn}`, we can solve for the cooling Lorentz factor:

.. math::

    \boxed{
    \gamma_c = \frac{3 m_e c}{4 \sigma_T U_B t_{\rm dyn}} = \frac{6 \pi m_e c}{\sigma_T B^2 t_{\rm dyn}}.
    }

The corresponding synchrotron frequency is

.. math::

    \nu_c = \frac{18 \pi m_e c e}{\sigma_T^2 B^3 t_{\rm dyn}^2}.

.. hint::

    This is implemented in the :mod:`radiation.synchrotron.frequencies` module!


IC Cooling
~~~~~~~~~~

In addition to synchrotron radiation, relativistic electrons may also lose energy through
**inverse Compton (IC) scattering**, in which electrons transfer energy to ambient photon fields.
This process is particularly important in environments with strong radiation backgrounds,
such as dense star-forming regions, AGN environments, or compact transients.

In the **Thomson regime**, where the photon energy in the electron rest frame satisfies
:math:`\gamma h\nu \ll m_e c^2`, the power radiated by a single electron due to inverse Compton
scattering is

.. math::

    P_{\rm IC}
    =
    \frac{4}{3}\,\sigma_T c\,\gamma^2 \beta^2 U_{\rm rad},

where :math:`U_{\rm rad}` is the energy density of the target photon field. In the relativistic
limit (:math:`\beta \approx 1`), this simplifies to

.. math::

    P_{\rm IC}
    \approx
    \frac{4}{3}\,\sigma_T c\,\gamma^2 U_{\rm rad}.

This expression is formally identical to the synchrotron power, with the magnetic energy
density :math:`U_B` replaced by the radiation energy density :math:`U_{\rm rad}`. As a result,
inverse Compton cooling produces the same qualitative effect on the electron distribution as
synchrotron cooling.

The corresponding inverse Compton cooling timescale is

.. math::

    \tau_{\rm IC, cool}
    =
    \frac{\gamma m_e c^2}{P_{\rm IC}}
    =
    \frac{3 m_e c}{4 \sigma_T U_{\rm rad}\,\gamma}.

Defining a dynamical timescale :math:`t_{\rm dyn}`, we may introduce the **inverse Compton cooling
Lorentz factor**

.. math::

    \boxed{
    \gamma_{c,{\rm IC}}
    =
    \frac{3 m_e c}{4 \sigma_T U_{\rm rad}\,t_{\rm dyn}}.
    }

Electrons with :math:`\gamma > \gamma_{c,{\rm IC}}` cool efficiently via inverse Compton scattering
within a dynamical time, while lower-energy electrons remain effectively uncooled.

.. note::

    In many astrophysical environments, electrons cool through **both** synchrotron and inverse
    Compton losses. In this case, the total cooling rate is additive:

    .. math::

        P_{\rm tot}
        =
        P_{\rm synch} + P_{\rm IC}
        =
        \frac{4}{3}\,\sigma_T c\,\gamma^2\left(U_B + U_{\rm rad}\right).

    The effective cooling Lorentz factor is therefore

    .. math::

        \gamma_c
        =
        \frac{3 m_e c}{4 \sigma_T \left(U_B + U_{\rm rad}\right) t_{\rm dyn}}.

    This regime is often parameterized by the **Compton \(Y\)-parameter**,
    :math:`Y \equiv U_{\rm rad}/U_B`, which quantifies the relative importance of IC cooling
    compared to synchrotron cooling.

.. warning::

    At sufficiently high electron energies, inverse Compton scattering enters the
    **Klein–Nishina regime**, where the scattering cross section is reduced and the
    :math:`\gamma^2` scaling of the cooling rate breaks down. Triceratops currently assumes
    Thomson-regime IC cooling; Klein–Nishina corrections are not yet implemented.

.. hint::

    This is implemented in the :mod:`radiation.synchrotron.frequencies` module! See
    :func:`~radiation.synchrotron.frequencies.compute_IC_cooling_gamma`,
    :func:`~radiation.synchrotron.frequencies.compute_IC_cooling_frequency`, and the
    associated low-level API.

Absorption Processes in Synchrotron Radiation
---------------------------------------------

It is not always the case that radiation observed by an observer is simply the emitted synchrotron radiation; it may
be modified by absorption processes in the emitting plasma or along the line of sight. Two important absorption
processes that can affect synchrotron radiation are synchrotron self-absorption (SSA) and free-free absorption.

In this section, we'll discuss both processes and derive the critical results.

Synchrotron Emissivity and Absorption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have already shown that, for a population of electrons with some :math:`N(\gamma)`, the synchrotron emissivity is

.. math::

    j_\nu = \frac{1}{4\pi} \int_{\gamma_{\min}}^{\gamma_{\max}} N(\gamma) P(\nu, \gamma) d\gamma,







Free-Free Absorption
~~~~~~~~~~~~~~~~~~~~~~


Spectral Regimes of Synchrotron SEDs
------------------------------------

The primary task of the synchrotron backend in Triceratops is to take a set of physical parameters including
(potentially) both the dynamics of the radiating material and the microphysical parameters governing the
synchrotron emission, and produce an accurate computation of the radiation. One element of this is correctly computing
the spectrum of the resultant radiation.

Unfortunately, the synchrotron spectrum is a complicated beast. Even in the ideal case (see Chapter 6 of
:footcite:t:`RybickiLightman`) is a non-trivial computation. More detailed treatments reveal a number of additional
complications including absorption processes (both synchrotron self-absorption and free-free absorption), cooling of
the electron population, and the presence of spectral breaks due to characteristic frequencies in the system.

In this section, we'll describe the various spectral regimes that can arise in synchrotron SEDs, and how they
are treated in Triceratops.

.. note::

    For detailed descriptions in the literature, we encourage the reader to consult
    :footcite:t:`demarchiRadioAnalysisSN2004C2022` for the "classical" synchrotron SED treatment. For the multi-domain
    treatment, we suggest :footcite:t:`GranotSari2002SpectralBreaks`, :footcite:t:`PiranGammaRayBursts2004` for
    introductory reading. :footcite:t:`GaoSynchrotronReview2013` provides an exhaustive reference of the
    synchrotron theory relevant here.

.. admonition:: Big Idea

    In this section, we are going to consider the shape of the synchrotron spectrum depending on the ordering of
    various characteristic frequencies in the system. These frequencies include:

    - The peak frequency :math:`\nu_m`, corresponding to the minimum Lorentz factor of the electron distribution.
    - The cooling frequency :math:`\nu_c`, corresponding to the Lorentz factor at which electrons cool on the dynamical
      timescale.
    - The self-absorption frequency :math:`\nu_a`, below which the spectrum is self-absorbed.

    Depending on the order of the frequencies, we can have a variety of different spectral shapes. Here we will discuss
    the details of each of these.

Review of Important Frequencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before proceeding with the relevant theory, we will here briefly review the important frequencies
that will determine the shape of the synchrotron spectrum.

The Peak Frequency
~~~~~~~~~~~~~~~~~~

For a single-electron, the synchrotron spectrum is dominated by emission around the "peak frequency"

.. math::

    \nu_{\rm crit} = \frac{3}{4\pi} \gamma^3 \omega_B \sin \alpha \sim \frac{e}{2\pi m_e c} \gamma^2 B,

where :math:`\omega_B` is the relativistic gyrofrequency (:math:`\omega_B = eB/\gamma m_e c`), :math:`\gamma` is the
Lorentz factor of the electron, and :math:`\alpha` is the pitch angle between the
magnetic field and the electron velocity.

.. note::

    The pedagogical reference (:footcite:t:`RybickiLightman`) uses the first expression, while the
    second expression is more common in the astrophysical literature (e.g. :footcite:t:`demarchiRadioAnalysisSN2004C2022`).
    Either way, these are an order unity correction of one another.

For a population of electrons with a power-law distribution in Lorentz factor, the emission is dominated by the
lowest energy electrons, we therefore define the **characteristic peak frequency** as

.. math::

    \boxed{
    \nu_m = \nu_{\rm crit}(\gamma_{\min}) = \frac{1}{2\pi} \gamma_{\min}^2 \frac{eB}{m_e c}.
    }

This is the first important frequency in determining the shape of the synchrotron spectrum. In general, one expects
that the spectrum will peak around :math:`\nu_m`, with different power-law segments above and below this frequency.

The Cooling Frequency
~~~~~~~~~~~~~~~~~~~~~
Another important frequency in determining the shape of the synchrotron spectrum is the **cooling frequency**,
:math:`\nu_c`. This frequency corresponds to the Lorentz factor at which electrons cool on the dynamical timescale
of the system.

Given a cooling process with rate :math:`\Lambda` per electron, we can define the cooling time as a function of the
Lorentz factor to be

.. math::

    \tau(\gamma) = \frac{\gamma m_e c^2}{\Lambda(\gamma)}.

Relative to a dynamical timescale :math:`t_{\rm dyn}`, we can then define the cooling Lorentz factor

.. math::

    \gamma_c : \tau(\gamma_c) = t_{\rm dyn} \implies \gamma_c = \frac{m_e c^2}{\Lambda(\gamma_c) t_{\rm dyn}}.

This corresponds to a characteristic frequency (the characteristic frequency of electrons with Lorentz factor
:math:`\gamma_c`) of

.. math::

    \boxed{
    \nu_c = \frac{m_e c^3 e}{2\pi} B \Lambda^{-2} t_{\rm dyn}^{-2}
    }

.. note::

    Depending on the mechanism of cooling, :math:`\Lambda` will take different forms. For pure synchrotron
    cooling, we have

    .. math::

        \nu_c = \frac{18 \pi m_e c e}{\sigma_T^2 t_{\rm dyn}^2 B^3},

    while for other mechanisms (e.g. inverse Compton), the expression will differ. See :ref:`electron_cooling`
    above for details.

The SSA Frequency
~~~~~~~~~~~~~~~~~~~~

The final important frequency in determining the shape of the synchrotron spectrum is the **self-absorption frequency**,
:math:`\nu_a`. A more detailed discussion of synchrotron self-absorption is provided in :ref:`ssa`; however, it suffices
in this discussion to note that we define the self-absorption frequency as the frequency at which the optical depth
to synchrotron self-absorption is unity:

.. math::

    \boxed{
    \tau_{\nu_a} = 1.
    }

Given that :math:`\tau_\nu = \alpha_\nu L`, where :math:`\alpha_\nu` is the synchrotron absorption coefficient
and :math:`L` is the path length through the emitting region, we can compute :math:`\nu_a` (in principle) by
solving

.. math::

    \alpha_{\nu_a} L = - \frac{1}{8\pi m_e \nu_a^2} \int_{\gamma_{\min}}^{\gamma_{\max}} P(\nu_a, \gamma)
    \gamma^2 \frac{\partial}{\partial \gamma} \left[ \frac{1}{\gamma^2}
    \frac{dN}{d\gamma} \right] d\gamma \cdot L = 1.

References
-----------
.. footbibliography::
