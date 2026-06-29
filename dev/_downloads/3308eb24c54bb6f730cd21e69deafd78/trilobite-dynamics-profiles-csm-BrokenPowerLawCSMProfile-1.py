import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import BrokenPowerLawCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 400) * u.cm
rho = BrokenPowerLawCSMProfile.eval(
    r,
    density_break=1e-18 * u.g / u.cm**3,
    radius_break=1e16 * u.cm,
    slope_inner=2.0,
    slope_outer=0.0,
)
fig, ax = plt.subplots()
ax.loglog(r.value, rho.value)
ax.axvline(1e16, ls='--', color='gray', label=r'$r_b$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Broken Power-Law CSM Profile')
ax.legend()
plt.tight_layout()