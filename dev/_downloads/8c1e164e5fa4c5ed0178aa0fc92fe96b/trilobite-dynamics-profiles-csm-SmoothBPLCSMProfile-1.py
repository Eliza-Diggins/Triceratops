import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import (
    SmoothBPLCSMProfile, BrokenPowerLawCSMProfile,
)
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 400) * u.cm
kw = dict(
    density_break=1e-18 * u.g / u.cm**3,
    radius_break=1e16 * u.cm,
    slope_inner=2.0,
    slope_outer=0.0,
)
rho_sharp = BrokenPowerLawCSMProfile.eval(r, **kw)
fig, ax = plt.subplots()
ax.loglog(r.value, rho_sharp.value, 'k--', label='Sharp BPL')
for delta, ls in [(0.1, ':'), (0.5, '-.'), (2.0, '-')]:
    rho = SmoothBPLCSMProfile.eval(r, smoothness=delta, **kw)
    ax.loglog(r.value, rho.value, ls=ls, label=fr'$\Delta={delta}$')
ax.axvline(1e16, ls=':', color='gray', alpha=0.5, label=r'$r_b$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Smooth Broken Power-Law CSM Profile')
ax.legend()
plt.tight_layout()