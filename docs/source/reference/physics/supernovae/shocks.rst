.. _supernova_shocks_theory:
==========================
Supernova Shocks
==========================

One of the primary modeling goals of Triceratops is to model the emission of synchrotron radiation from supernova shocks
as they interact with the circumstellar medium (CSM). This interaction accelerates particles to relativistic speeds and
amplifies magnetic fields, leading to the production of radio emission that can be observed with radio telescopes.

The shock dynamics are typically modeled using self-similar solutions, such as those described by
:footcite:t:`ChevalierXRayRadioEmission1982` or :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`. These
are; however, not the only solutions available, and a number of approaches are taken to the self-consistent modeling
of shock dynamics in supernovae.

The resulting synchrotron emission is calculated based on the shock properties, CSM density profile, and microphysical
parameters that describe the efficiency of particle acceleration and magnetic field amplification. The synchrotron
spectrum is typically modeled using a power-law distribution of electron energies, with a characteristic spectral index
that depends on the shock properties and the CSM environment. The details of our synchrotron modeling can be found in the
:ref:`synchrotron_radiation` section.

Overview
---------

There are 3 canonical stages to the evolution of a supernova shock remnant:

1. Free Expansion Phase: In this initial phase, the supernova ejecta expands freely into the surrounding CSM. The shock
   velocity is high, and the shock radius increases linearly with time. The radio emission during this phase is typically
   dominated by synchrotron radiation from relativistic electrons accelerated at the shock front.
2. Sedov-Taylor Phase: As the shock sweeps up more CSM material, it enters the Sedov-Taylor phase, where the dynamics
   are governed by the conservation of energy. The shock radius increases as a power-law function of time, and the
   shock velocity decreases. The radio emission during this phase is still dominated by synchrotron radiation,
   but the spectral properties may change due to the evolving shock conditions.
3. Radiative Phase: In the final phase, the shock cools radiatively, leading to the formation of a dense shell of
   material behind the shock front. The shock velocity decreases further, and the radio emission may be dominated by
   thermal bremsstrahlung radiation from the shocked CSM.

The majority of radio supernovae are observed during the free expansion phase, where the synchrotron emission is most
prominent; however, Triceratops is intended to (eventually) model all phases of supernova shock evolution.

.. hint::

    Importantly, modeling of a specific remnant will require identifying the appropriate phase of evolution and
    selecting the relevant physical parameters and models accordingly.

Homologous Expansion (Free Expansion Phase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *earliest stage* of a supernova remnant’s evolution begins immediately after the stellar envelope has been
ejected and the forward shock has broken out of the progenitor. At this point, the ejecta moves
**almost unimpeded** into the surrounding circumstellar medium (CSM).
Critically, the energy is **kinetic** and the expansion is **ejecta-momentum** dominated not pressure dominated
like the Sedov scenario.

.. note::

    This phase is sometimes referred to as the "ejecta-dominated phase" or "free expansion phase" in the literature. It
    is worth thinking about this phase as the period in which the remnant is still thermalizing all of the ejecta
    energy, at which point the Sedov-Taylor phase will begin.

Given a total ejected mass :math:`M_{\rm ej}` with velocity :math:`v_{\rm ej}`, this phase *conserves momentum*,
so

.. math::

    R_{\rm shock} \sim v_{\rm ej} t = 1.02 \left(\frac{v_{\rm ej}}{\rm 10^4\; km\;s^{-1}}\right)\left(\frac{t}
    {\rm 100\;yr}\right)\; {\rm pc}.

This, of course, corresponds to a corresponding ejecta energy :math:`E_{\rm ej}`,

.. math::

    E_{0} = \frac{1}{2} M_{\rm ej} v_{\rm ej}^2 \implies v_{\rm ej} =
    10^4\;\left(\frac{M_{\rm ej}}{M_\odot}\right)^{-1/2} \left(\frac{E_0}{10^{51}\;{\rm erg}}\right)^{1/2}\;
    {\rm km\;s^{-1}}

Once the explosion has broken out of the star, the ejecta can move ballistically, so, for any given fluid element :math:`i`,

.. math::

    r_i(t) = v_i t,\;\text{and}\; dv_i/dt = 0

Thus, each parcel will have

.. math::

    \boxed{
    v(r,t) = \frac{r}{t},
    }

corresponding to so-called **homologous expansion**.

The obvious question to ask is *at what point does deceleration matter?*
To answer this, we require that the swept up mass of the ICM/ISM be similar
to the ejecta mass, meaning that we have been forced to accelerate a non-trivial amount of mass.
In this scenario,

.. math::

    M_{\rm ej} \sim M_{\rm ism}(t) = \frac{4}{3}\pi \rho_0 R_{\rm shock}^3 = \frac{4}{3}\pi \rho_0 v_{\rm ej}^3 t^3.


We can therefore identify a time at the *end of free expansion* for which

.. math::

    \boxed{
    t \sim \left(\frac{3M_{\rm ej}}{4\pi \rho_0}\right)^{1/3} \frac{1}{v_{\rm ej}}, \;R\sim \left(\frac{3 M_{\rm ej}}{4\pi \rho}\right)^{1/3}.
    }

For standard scalings, this becomes

.. math::

    \boxed{
    \begin{aligned}
                R_{\rm FE} &\sim 2.5 \left(\frac{M_{\rm ej}}{M_\odot}\right)^{1/3} \left(\frac{\rho_0}{10^{-24}\;{\rm g/cm^3}}\right)^{-1/3}\; {\rm pc}. \;\text{(FE)}\\
            t_{\rm FE} &\sim 244\;\left(\frac{v_{\rm ej}}{10^4\;{\rm km/s}}\right)^{-1}\left(\frac{M_{\rm ej}}{M_\odot}\right)^{1/3} \left(\frac{\rho_0}{10^{-24}\;{\rm g/cm^3}}\right)^{-1/3}\; {\rm yr}. \;\text{(FE)}
    \end{aligned}
    }


Beyond this point, the swept-up mass begins to dominate the dynamics, and the remnant enters the adiabatic,
energy-conserving stage described by the **Sedov–Taylor solution**.

.. note::

    This is actually an aggressive choice
    for the truncation time. We might prefer something on the order of 500 - 1000 years from a more
    sophisticated analysis.

Sedov-Taylor Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the free expansion phase, the supernova remnant enters the Sedov-Taylor phase, where the dynamics are governed
by the conservation of energy. In this phase, the shock radius increases as a power-law function of time, and the
shock velocity decreases.

The solution for the shock radius as a function of time was first derived independently by
:footcite:t:`sedov1946propagation` and :footcite:t:`taylor1950formation`, and is given by

.. math::

    R_{\rm shock}(t) = \xi_0\left(\frac{E_0t^2}{\rho_0}\right)^{1/5},

where :math:`\xi_0` is a dimensionless constant that depends on the adiabatic index of the gas
(for a monatomic ideal gas, :math:`\gamma = 5/3`, :math:`\xi_0 \approx 1.15`).

If we calculate these for the typical scalings relevant to \textbf{supernova remnants}, we will find that

.. math::

    R_{\rm shock} \approx 2.3 \left(\frac{E_0}{10^{51}\;\rm erg}\right)^{1/5}
    \left(\frac{\rho_0}{10^{-24}\;\rm g\;cm^{-3}}\right)^{-1/5}\left(\frac{t}{1000\;\rm yr}\right)^{2/5}\;{\rm pc}.

and

.. math::

    u_{\rm shock} \approx 9 \times 10^3 \left(\frac{E_0}{10^{51}\;\rm erg}\right)^{1/5}
    \left(\frac{\rho_0}{10^{-24}\;\rm g\;cm^{-3}}\right)^{-1/5}\left(\frac{t}{1000\;\rm yr}\right)^{-3/5}\;{\rm km/s.}

Eventually, the Sedov--Taylor (ST) description breaks down because \textbf{the post--shock gas cools efficiently.  }
The key comparison is between the \textbf{cooling time} of the shocked gas and the \textbf{dynamical (expansion) time}
of the remnant.

As the remnant continues to expand beyond the Sedov–Taylor stage, the post-shock gas cools to progressively lower temperatures.
During the Sedov phase the shock remains hot ($T_s \gtrsim 10^6\,$K), and the cooling time greatly exceeds
the expansion time, \textbf{ensuring that radiative losses are dynamically negligible.}
However, because the shock velocity decreases as $v_s \propto t^{-3/5}$, the post-shock temperature eventually
falls into the regime where radiative cooling---particularly via metal line emission---becomes extremely efficient.
As we have seen, when the losses due to cooling are no longer negligible, we have \textbf{left the Sedov phase} and
entered the \textbf{radiative phase} (snowplow phase).

The immediate consequence of rapid cooling is that the shocked interstellar material \textbf{collapses into a thin,
dense shell just behind the blast wave.}
The shock itself continues to sweep up and thermalize ambient material, but the post-shock gas now loses
its thermal energy almost instantaneously and is deposited into the shell at temperatures of order $10^2$--$10^4\,$K.
Behind this cold shell lies the still-hot interior plasma---the relic of the Sedov phase---which has not yet
cooled appreciably.
The interface between the high-pressure interior and the \textbf{thin radiative shell is highly unstable}, and
Rayleigh--Taylor fingers develop as the lighter, overpressured interior gas pushes outward against the heavy shell.
These instabilities contribute to the complex filamentary morphology seen in many middle-aged supernova remnants.

Because the cooled shell is extremely dense and thin, its thermal pressure is negligible compared to its bulk kinetic
energy. The remnant can therefore be approximated as a \textbf{momentum-conserving “snowplow”} in
which the dynamics are governed by the conservation of radial momentum rather
than total energy.

Shock Modeling in Homologous Expansion
--------------------------------------

Our first implementation of supernova shock modeling in Triceratops focuses on the homologous expansion phase.
This is primarily because the majority of radio supernovae are observed during this phase, where the synchrotron
emission is most prominent. There are a number of ways to treat the relevant physics, and we have implemented
a few different models to capture the shock dynamics and resulting synchrotron emission. We will discuss them
in order of complexity.

Remark: The Morphology of the Shock
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the supernova shock has escaped the stellar progenitor, it generically separates into four dynamical regions:

- The \textbf{unshocked CSM (region 4)} — the ambient interstellar or circumstellar gas, initially at
  rest with density $\rho_0$ and pressure $P_0$.
- The \textbf{shocked CSM (region III)} — material swept up by the forward shock and heated to high
  temperatures ($T \sim 10^{6}$–$10^{8}$ K).
- The \textbf{shocked ejecta (region II)} — ejecta that have passed through the reverse shock, now
  moving subsonically in the local frame.
- The \textbf{unshocked ejecta (region I)} — freely expanding stellar material,
  often still cold and dense relative to the shocked regions.

Between regions (II) and (III) lies a \textbf{contact discontinuity} that separates the shocked CSM
from the shocked ejecta. The two shocks (forward and reverse) and the contact surface evolve self-consistently
as the explosion expands into the surrounding medium.

The jump conditions across a strong adiabatic shock (Mach number $M \gg 1$) follow from the conservation of mass,
momentum, and energy (\textbf{Rankine–Hugoniot Conditions}):

.. math::

    \begin{aligned}
    \rho_1 v_1 &= \rho_2 v_2, \\
    P_2 + \rho_2 v_2^2 &= P_1 + \rho_1 v_1^2, \\
    \frac{1}{2}v_1^2 + \frac{\gamma}{\gamma-1}\frac{P_1}{\rho_1} &= \frac{1}{2}v_2^2 + \frac{\gamma}{\gamma-1}\frac{P_2}{\rho_2}.
    \end{aligned}

Here subscripts 1 and 2 refer to pre- and post-shock quantities respectively. Solving these for a given adiabatic
index $\gamma$ yields the compression, pressure, and temperature jumps across the shock.

For a strong shock in a monatomic gas ($\gamma = 5/3$), one finds:

.. math::

    \frac{\rho_2}{\rho_1} = \frac{\gamma+1}{\gamma-1} = 4,

.. math::

    \frac{P_2}{P_1} = \frac{2\gamma M_1^2 - (\gamma - 1)}{\gamma + 1} \approx \frac{2\gamma}{\gamma + 1}
    M_1^2 \quad (M_1 \gg 1),

and thus for $\gamma=5/3$,

.. math::

    \frac{P_2}{P_1} \approx \frac{5}{4} M_1^2.

Finally, the post-shock temperature is given by

.. math::

    T_2 \approx \frac{3}{16}\frac{\mu m_p}{k_B} v_s^2.

For $v_s = 10^4\,{\rm km\,s^{-1}}$, this yields $T_2 \sim 10^9\,{\rm K}$, sufficient to produce
strong thermal bremsstrahlung and line emission in the X-ray band.

The Self-Similar Model
^^^^^^^^^^^^^^^^^^^^^^^^

The canonical approach to modeling the shock dynamics during the homologous expansion phase is to
use self-similar solutions as described by :footcite:t:`ChevalierXRayRadioEmission1982` and
:footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`. These models make the following assumptions:

- The ejecta density profile follows a power-law distribution in velocity space

  .. math::

        \rho(r,t) = A_1 t^{-3} \left(\frac{r}{t}\right)^{-n},

- The CSM density profile also follows a power-law distribution in radius

  .. math::

        \rho_{\rm csm}(r) = A_4 \left(r\right)^{-s},

- The shock dynamics are self-similar, meaning that the shock radius and velocity can be expressed as power-law
  functions of time.

- The shocks are strong and adiabatic, allowing the use of the Rankine-Hugoniot jump conditions to relate
  pre- and post-shock quantities.

- The forward and backward shocks are sufficiently close together that the entire shock may be treated as a
  "thin shell" for the purposes of calculating synchrotron emission. In fact,
  :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982` did not rely on this assumption and instead solved the
  self-similar ODE system directly to obtain the full structure of the shocked region. It has become standard practice;
  however, to use the thin-shell approximation for simplicity following :footcite:t:`ChevalierXRayRadioEmission1982`.

The Shock Radius and Velocity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a self-similar framework, there can be no intrinsic timescale or length scale, and so ratios of physical quantities
must remain constant. Requiring that the ratio of the CSM to the ejecta density be constant with time at the
shock radius leads to the following scaling for the shock radius:

.. math::

    R_{\rm shock}(t) = \left[\frac{A_4}{A_1} \zeta\right]^{1/(s-n)} t^{(3-n)/(s-n)} = A_{\rm shock} t^{\lambda},

where :math:`\zeta` is a dimensionless constant determined by solving the self-similar ODEs, and
:math:`\lambda = (n-3)/(n-s)`. The shock velocity is then given by

.. math::

    v_{\rm shock}(t) = \frac{dR_{\rm shock}}{dt} = \lambda A_{\rm shock} t^{\lambda - 1} = \lambda R_{\rm shock}/t.

The normalization constant :math:`A_{\rm shock}` is (unfortunately), quite difficult to obtain in closed form,
as it depends on the detailed structure of the ejecta and CSM density profiles as well as the solution to the
self-similar ODEs. This observation motivates the use of the **thin-shell** approximation, where we treat the shocked
region as a thin shell, which makes it possible to derive approximate expressions for :math:`A_{\rm shock}` as shown
in the dropdown below.

.. dropdown:: Shock Normalization

    In the thin-shell approximation, the pressure difference across the shock must balance the momentum
    flux into the shell. Thus,

    .. math::

        \frac{dp}{dt} = [M_2+M_3]\frac{d}{dt}\left( \dot{R}_{\rm shock}\right)= 4\pi R_{\rm shock}^2 (P_3 -P_2).

    It can be shown that the CSM mass swept up by the forward shock is

    .. math::

        M_3(R_{\rm shock}) = \frac{4\pi A_4}{3-s} R_{\rm shock}^{3-s}.

    The mass swept up by the reverse shock corresponds to all of the ejecta mass with velocities larger than

    .. math::

        \tilde{v} = \frac{R_{\rm shock}}{t}.

    Therefore, at any time :math:`t`, the material beyond :math:`\tilde{v}` has been shocked, and the mass
    in this region is

    .. math::

        M_2(t) = \int_{R_{\rm shock}}^{\infty} 4\pi \xi^2\;\rho_1(\xi,t)\;d\xi.

    Integrating,

    .. math::

        M_2(t) = 4\pi A_1 t^{-3}\int_{R_{\rm shock}}^{\infty} \xi^{2-n} \;d\xi =
        \frac{4\pi A_1 t^{n-3}}{3-n} \left[ - R_{\rm shock}^{3-n}\right].


    Since we generally are interested in :math:`n>3`, the \textbf{dominant term} is

    .. math::

        M_2(t) \simeq \frac{4\pi A_1 t^{n-3}}{n-3} R_{\rm shock}^{3-n}.

    Clearly, the total mass is

    .. math::

        M_2 + M_3 = 4\pi\left[\frac{A_4}{3-s}R_{\rm shock}^{3-s} + \frac{A_1 t^{n-3}}{n-3}R_{\rm shock}^{3-n}\right],

    If we substitute our ansatz for :math:`R_{\rm shock}`,

    .. math::

        M_{\rm sh} = 4\pi \left(\frac{A_4}{3-s} A_{\rm shock}^{3-s}t^{\lambda(3- s)} +
        \frac{A_1}{n-3} A_{\rm shock}^{3-n} t^{(\lambda-1)(3-n) }\right).

    With $\lambda = (n-3)/(n-s)$, this becomes

    .. math::

        \boxed{
            M_{\rm sh} = K_{\rm sh} t^{\gamma}= 4\pi t^{\gamma} \left[\frac{A_4}{3-s}A_{\rm shock}^{3-s}
            + \frac{A_1}{n-3} A_{\rm shock}^{3-n}\right],\;\gamma = \frac{(n-3)(3-s)}{n-s}.
            }

    In order to use the thin-shell approximation to close the problem, we need expressions for the pressures
    behind the forward and reverse shocks. Using the strong shock jump conditions, these are

    .. math::

        P_2 = \frac{3}{4}\rho_1(R_{\rm sh},t) \left(\tilde{v} - \dot{R}_{\rm sh}\right)^2,

    .. math::

        P_3 = \frac{3}{4}\rho_4(R_{\rm sh}) \left(\dot{R}_{\rm sh}\right)^2.

    .. note::

        In some texts, specifically :footcite:t:`ChevalierXRayRadioEmission1982` , these pressures are
        given with a factor of 1 instead of 3/4. This is because the impact of the factor is of order unity and
        does little to influence the overall dynamics. We have chosen to use the more accurate 3/4 factor here.

    Returning to the momentum equation, the RHS becomes

    .. math::

        4\pi R_{\rm sh}^2 \left(P_2-P_3\right) = 3\pi R_{\rm sh}^2\left(\rho_1 \left(\tilde{v}
        -\dot{R}_{\rm sh}\right)^2 - \rho_4 \dot{R}_{\rm sh}^2 \right).

    Using the self-similar ansatz, the time dependency can be separated out, yielding

    .. math::

        4\pi (\lambda -1)\lambda A_{\rm shock}\left[\frac{A_4}{3-s}A_{\rm shock}^{3-s} +
        \frac{A_1}{n-3} A_{\rm shock}^{3-n}\right] = 3\pi A_{\rm shock}^{4-s}
        \left[A_1(1-\lambda)^2 A_{\rm shock}^{s-n} - A_4 \lambda^2\right]

    Collecting terms, we have

    .. math::

        \begin{aligned}
        \underbrace{4\pi A_4 \frac{\lambda(\lambda  -1) }{3-s}}_{A}A_{\rm shock}^{4-s} + \underbrace{4\pi A_1
        \frac{\lambda(\lambda  -1)}{n-3}}_{B}A_{\rm shock}^{4-n} &=\\
        -\underbrace{3\pi \lambda^2 A_4}_C A_{\rm shock}^{4-s} + \underbrace{3\pi A_1 (1-\lambda)^2}_D A_{\rm shock}^{4-n}
        \end{aligned}

    The equation reduces to the form

    .. math::

        \zeta = \frac{C+A}{D-B} =\frac{3\lambda^2 +4\frac{ (\lambda -1)\lambda}{3-s} }
                    { 3(1-\lambda)^2-4\frac{ (\lambda  -1)\lambda}{n-3}}

    .. note::

        Without the 3/4 factor, this reduces to the expression

        .. math::

            \zeta = \frac{C+A}{D-B} =\frac{\lambda^2 +\frac{ (\lambda -1)\lambda}{3-s} }
                        { (1-\lambda)^2-\frac{ (\lambda  -1)\lambda}{n-3}}.

        Furthermore, for a wind-like CSM (:math:`s=2`), this expression simplifies to

        .. math::

            \zeta = \frac{(n-4)(n-3)}{2},

        Which is the expression given in :footcite:t:`ChevalierXRayRadioEmission1982`.

The resulting value for :math:`\zeta` is

.. math::

    \boxed{
    \zeta = \frac{3\lambda^2 +4\frac{ (\lambda -1)\lambda}{3-s} }
                { 3(1-\lambda)^2-4\frac{ (\lambda  -1)\lambda}{n-3}}.
    }

With this, the entire dynamics of the shock are specified. This approach is implemented in the
:class:`~dynamics.supernovae.shock_dynamics.ChevalierSelfSimilarShockEngine` and
:class:`~dynamics.supernovae.shock_dynamics.ChevalierSelfSimilarWindShockEngine` classes. You can read more about
how to use these models in the :ref:`user_guide` sections on shock engines.

Connection To Explosion Energetics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Providing the normalization constants :math:`A_1` and :math:`A_4` requires connecting the density profiles
to physical parameters of the explosion and CSM. For the ejecta, we can relate :math:`A_1` to the total ejecta mass
:math:`M_{\rm ej}` and energy :math:`E_{\rm ej}`.

As described in :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`, the ejecta velocity profile
is well described by a broken power-law. During homologous expansion, :math:`r = vt` implies that the density
of the ejecta is likewise a broken power-law in velocity space:

.. math::

    \rho(r,t) = Kt^{-3} \begin{cases}
        v^{-\delta}, & v < v_t \\
        v_t^{n-\delta} v^{-n}, & v \geq v_t,
    \end{cases}

where :math:`v_t` is the transition velocity between the inner and outer ejecta profiles. The total mass of
the ejecta is given by :math:`M_{\rm ej}` and must be conserved:

.. math::

    M_{\rm ej} = \int_0^{\infty} 4\pi r^2 \rho(r,t) dr = 4\pi K v_t^{3-\delta} \frac{n-\delta}{(3-\delta)(n-3)}.

Similarly, the total kinetic energy of the ejecta is given by :math:`E_{\rm ej}` and must also be conserved:

.. math::

    E_{\rm ej} = \int_0^{\infty} \frac{1}{2} 4\pi r^2 \rho(r,t) v^2 dr =
    2\pi K v_t^{5-\delta} \frac{n-\delta}{(5-\delta)(n-5)}.

In terms of the energy per unit mass, :math:`E_{\rm ej}/M_{\rm ej}`, these two equations can be combined to
solve for the transition velocity:

.. math::

    v_t^2 = \frac{2(5-\delta)(n-5)}{(3-\delta)(n-3)} \frac{E_{\rm ej}}{M_{\rm ej}}.

Finally, substituting this back into the mass equation allows us to solve for the normalization constant
:math:`K_1`:

.. math::

    K_1 = \frac{1}{4\pi} \left(\frac{(3-\delta)(n-3)}{(n-\delta)}\right) \frac{M_{\rm ej}}{v_t^{3-\delta}}.

This is implemented in
:meth:`~dynamics.supernovae.shock_dynamics.ChevalierSelfSimilarShockEngine.compute_v_t_and_K_from_energetics`.


The Thin-Shell Numerical Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One downside of the self-similar approach is that it relies on a number of assumptions about the ejecta and CSM density
profiles, which may not always be valid. To address this, we implement a purely numerical thin-shell model which
permits much more general density profiles for both the ejecta and CSM.

In this approach, we imagine the shock as a thin shell with an equation of motion dictated by the conservation of
momentum in the same form as described in the dropdown above. However, instead of relying on self-similar scalings, we
numerically integrate the equation of motion using arbitrary density profiles for the ejecta and CSM. This allows
us to capture more complex shock dynamics that may arise from non-power-law density profiles or other physical effects.

The momentum equation takes the form

.. math::

    \frac{dp}{dt} = [M_2+M_3]\frac{d}{dt}\left( \dot{R}_{\rm shock}\right)=
    4\pi R_{\rm shock}^2 (P_3 -P_2)
    + \dot{M}_{\rm 2} v_{\rm ej}
    - \dot{M}_{\rm 3} v_{\rm shock}.

We previously showed that the Rankine-Hugoniot jump conditions give us expressions for the pressures behind the forward
and reverse shocks:

.. math::

    4\pi R_{\rm sh}^2 \left(P_2-P_3\right) = 3\pi R_{\rm sh}^2\left(\rho_1 \left(\tilde{v}
    -\dot{R}_{\rm sh}\right)^2 - \rho_4 \dot{R}_{\rm sh}^2 \right).

We also showed that the swept-up masses could be computed in quadrature; however, for numerical implementation, this
produces a slow algorithm as each timestep requires an integral evaluation.

A more mathematically rich approach relies again on the **homologous expansion** in the early phase of the supernova
expansion. Because this is true regardless of the ejecta profile, it remains the case that if the ejecta have some
distribution of velocities such that

.. math::

    \frac{dM_{\rm ej}}{dv} = f(v),


then the density must be of the form

.. math::

    \rho_{\rm ej}(r,t) = t^{-3} G(v = r/t),

where

.. math::

    G(v) = \frac{f(v)}{4\pi v^2}.

.. hint::

    In the self-similar model, we allow only scenarios where :math:`f(v) \propto v^{-n}`; however, here we can
    consider arbitrary distributions.

Now, at a given time :math:`t`, the mass swept up in \textbf{region 2} (post-shock ejecta) is the same as all of
the mass with

.. math::

    r>R_{\rm sh} \iff v> R_{\rm sh}/t = \tilde{v},

where :math:`\tilde{v}` is homologous equivalent velocity at the shock radius. Thus,

.. math::

    M_2(t) = \int_{v_{\rm ej}}^{\infty} f(v) dv,

which means that (using Leibniz rule), noting that

.. math::

    \frac{dv_{\rm ej}{dt} = \frac{d}{dt}\left(\frac{R_{\rm sh}}{t}\right)
     = \frac{\dot{R}_{\rm sh}}{t} - \frac{R_{\rm sh}}{t^2}

.. math::

    \frac{dM_2}{dt} = \frac{dM_2}{dv_{\rm ej}} \frac{dv_{\rm ej}}{dt} = -4\pi \frac{R_{\rm sh}^2}{t^3}
    G\left[\frac{R_{\rm sh}}{t}\right] \left(v_{\rm sh}- \frac{R_{\rm sh}}{t}\right).

Crucially, this allows us to avoid time-step integration. Using

.. math::

    \frac{dM_3}{dt} = 4\pi \rho_{\rm csm}[R_{\rm sh}] R_{\rm sh}^2 v_{\rm sh},

we have

.. math::

    \frac{dM_{\rm sh}}{dt} = 4\pi \left\{\rho_{\rm csm}(R_{\rm sh}) R_{\rm sh}^2 v_{\rm sh}
    - \frac{R_{\rm sh}^2}{t^3} G\left[\tilde{v}\right] \left(v_{\rm sh}- \tilde{v}\right)\right\}.

We therefore solve the set of differential equation

.. math::

    \begin{aligned}
    \frac{dR}{dt} &= v\\
    \frac{dv}{dt} &=  -\frac{7\pi R^2}{M} \left(\rho_{\rm csm} v^2-t^{-3} G(v_{\rm ej}) \Delta^2\right)\\
        \frac{dM}{dt} &= 4\pi R^2 \left\{\rho_{\rm csm} v + \frac{1}{t^3} G(v_{\rm ej})\Delta\right\},
    \end{aligned}

where :math:`\Delta = \tilde{v} - v_{\rm sh}`.

Some additional effort can be taken to make this numerically stable at early times as described in the dropdown below;
however, the overall approach is straightforward to implement using any standard ODE integrator. This approach is implemented
in the :class:`~dynamics.supernovae.shock_dynamics.NumericalThinShellShockEngine` class. You can read more about how to use
this model in the :ref:`user_guide` sections on shock engines.

.. dropdown:: Numerical Details

    With :math:`\xi = R_{\rm sh}/t`, :math:`\tau = \log t`, and :math:`\Delta = \xi - v_{\rm sh}`, we have

    .. math::

        \begin{aligned}
            \frac{d\xi}{d\tau} &= -\Delta\\
            \frac{d\Delta}{d\tau} &= -\Delta + \frac{7\xi^2\pi}{M}\left\{t^3\rho_{\rm csm}(\xi t)
            (\xi - \Delta)^2 - G(\xi) \Delta (4\xi - \Delta)\right\}\\
            \frac{dM}{d\tau} &= 4\pi \xi^2 \left\{t^3\rho_{\rm csm}(\xi t)(\xi -\Delta) + G(\xi) \Delta\right\}
        \end{aligned}


References
----------

.. footbibliography::
