import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from trilobite.dynamics.shocks import MechanicalShockEngine, make_homologous_stationary_sources
from trilobite.dynamics.profiles import BrokenPowerLawEjectaProfile, WindCSMProfile
from trilobite.utils.plot_utils import set_plot_style

K, v_t  = BrokenPowerLawEjectaProfile.normalize(1e51 * u.erg, 5.0 * u.Msun, n=10.0, delta=1.0)
G_ej    = BrokenPowerLawEjectaProfile.as_optimized_callable(n=10.0, delta=1.0, K=K, v_t=v_t)
rho_csm = WindCSMProfile.as_optimized_callable(mass_loss_rate=1e-5 * u.Msun / u.yr, wind_velocity=100.0 * u.km / u.s)
rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(G_ej, rho_csm)

engine = MechanicalShockEngine()
t_0    = 1.0 * u.day
ic = engine.infer_initial_conditions(
    R_cd_0=1e14 * u.cm, v_cd_0=1e9 * u.cm / u.s,
    t_0=t_0, rho_1=rho_1, rho_4=rho_4, u_1=u_1, u_4=u_4,
)

time  = np.geomspace(1, 1000, 300) * u.day
state = engine.compute_shock_properties(
    time=time,
    rho_1=rho_1, rho_4=rho_4, u_1=u_1, u_4=u_4,
    initial_conditions=ic, t_0=t_0,
)

set_plot_style()
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
t_days = time.to_value(u.day)

axes[0].loglog(t_days, state.post_shock_temperature_fs.to_value(u.K), label="Forward shock")
axes[0].loglog(t_days, state.post_shock_temperature_rs.to_value(u.K), label="Reverse shock")
axes[0].set_ylabel("Post-shock temperature (K)")
axes[0].legend()

axes[1].loglog(t_days, state.post_shock_pressure_fs.to_value(u.dyn / u.cm**2), label="Forward shock")
axes[1].loglog(t_days, state.post_shock_pressure_rs.to_value(u.dyn / u.cm**2), label="Reverse shock")
axes[1].set_ylabel(r"Post-shock pressure (dyn cm$^{-2}$)")
axes[1].set_xlabel("Time (days)")
axes[1].legend()

plt.tight_layout()
plt.show()