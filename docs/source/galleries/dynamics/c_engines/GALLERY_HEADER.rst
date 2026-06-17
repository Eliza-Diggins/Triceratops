Numerical Shock Engines
========================

These examples demonstrate Trilobite's thin-shell shock engines — the
numerical integrators that evolve blast-wave radius, velocity, and swept-up
mass through an arbitrary circumstellar medium (CSM) density profile.

Three engines are covered:

- :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
  — the standard non-relativistic pressure-driven thin shell
- :class:`~trilobite.dynamics.shocks.numerical.RelPressureDrivenThinShellShockEngine`
  — the fully relativistic variant; shows where :math:`\Gamma\beta \gtrsim 1`
  corrections become essential
- CSM environment comparison — same ejecta launched into ISM, wind, truncated
  wind, and shell density profiles to build intuition for how density structure
  imprints itself on the shock trajectory

Use these examples when you need to evolve a shock through a realistic,
non-power-law density profile.

**API:** :ref:`dynamics_numerical_engines`, :ref:`dynamics_csm_utils`
