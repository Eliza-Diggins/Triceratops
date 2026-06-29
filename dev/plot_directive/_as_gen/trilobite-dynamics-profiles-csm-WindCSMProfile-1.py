import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import WindCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 300) * u.cm
fig, ax = plt.subplots()
for A_star, ls in [(0.1, '--'), (1.0, '-'), (10.0, ':')]:
    rho = WindCSMProfile.eval(r, A_star=A_star)
    ax.loglog(r.to(u.cm).value, rho.to(u.g / u.cm**3).value,
              ls=ls, label=fr'$A_* = {A_star}$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Wind CSM Profile')
ax.legend()
plt.tight_layout()