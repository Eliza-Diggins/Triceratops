import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import ShellCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 600) * u.cm
rho = ShellCSMProfile.eval(
    r,
    r_inner=3e15 * u.cm,
    r_outer=1e16 * u.cm,
    shell_density=1e-18 * u.g / u.cm**3,
    density_floor=1e-23 * u.g / u.cm**3,
)
fig, ax = plt.subplots()
ax.loglog(r.value, rho.value)
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Top-Hat Shell CSM Profile')
plt.tight_layout()