import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import WindBubbleCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e19, 800) * u.cm
rho = WindBubbleCSMProfile.eval(
    r,
    A=5e11 * u.g / u.cm,
    r_wind_termination=3e16 * u.cm,
    rho_bubble=1e-25 * u.g / u.cm**3,
    r_shell_inner=3e17 * u.cm,
    r_shell_outer=4e17 * u.cm,
    rho_shell=1e-21 * u.g / u.cm**3,
    rho_ism=1e-24 * u.g / u.cm**3,
)
fig, ax = plt.subplots()
ax.loglog(r.value, rho.value)
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Wind Bubble CSM Profile')
for x, lbl in [(3e16, r'$R_\mathrm{ts}$'), (3e17, r'$R_\mathrm{sh,in}$'),
               (4e17, r'$R_\mathrm{sh,out}$')]:
    ax.axvline(x, ls='--', color='gray', alpha=0.6)
    ax.text(x * 1.05, 1e-19, lbl, rotation=90, fontsize=8, va='top')
plt.tight_layout()