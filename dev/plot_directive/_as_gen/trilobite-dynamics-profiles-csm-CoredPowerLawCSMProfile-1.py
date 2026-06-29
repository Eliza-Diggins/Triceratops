import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import CoredPowerLawCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e13, 1e18, 400) * u.cm
r_core = 1e15 * u.cm
rho_0 = 1e-18 * u.g / u.cm**3
fig, ax = plt.subplots()
for s, ls in [(1.0, '--'), (2.0, '-'), (3.0, ':')]:
    rho = CoredPowerLawCSMProfile.eval(r, rho_0=rho_0, r_core=r_core, slope=s)
    ax.loglog(r.value, rho.value, ls=ls, label=fr'$s={s}$')
ax.axvline(r_core.value, ls=':', color='gray', alpha=0.5, label=r'$r_c$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Cored Power-Law CSM Profile')
ax.legend()
plt.tight_layout()