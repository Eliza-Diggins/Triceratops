import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import PowerLawCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 300) * u.cm
r_ref = 1e16 * u.cm
rho_ref = 1e-18 * u.g / u.cm**3
fig, ax = plt.subplots()
for s, ls in [(0.0, ':'), (1.0, '--'), (2.0, '-'), (3.0, '-.')]:
    rho = PowerLawCSMProfile.eval(r, rho_ref=rho_ref, r_ref=r_ref, slope=s)
    ax.loglog(r.value, rho.value, ls=ls, label=fr'$s={s}$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Power-Law CSM Profile')
ax.legend()
plt.tight_layout()