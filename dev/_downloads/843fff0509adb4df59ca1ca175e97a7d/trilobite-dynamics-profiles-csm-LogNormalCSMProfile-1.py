import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import LogNormalCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 500) * u.cm
r_peak = 1e16 * u.cm
rho_0 = 1e-18 * u.g / u.cm**3
fig, ax = plt.subplots()
for sigma, ls in [(0.3, '--'), (0.6, '-'), (1.2, ':')]:
    rho = LogNormalCSMProfile.eval(r, rho_0=rho_0, r_peak=r_peak, sigma=sigma)
    ax.loglog(r.value, rho.value, ls=ls, label=fr'$\sigma={sigma}$')
ax.axvline(r_peak.value, ls=':', color='gray', alpha=0.5, label=r'$r_0$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Log-Normal CSM Profile')
ax.legend()
plt.tight_layout()