import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import UniformCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 300) * u.cm
fig, ax = plt.subplots()
for rho0, ls in [(1e-24, '--'), (1e-23, '-'), (1e-22, ':')]:
    rho = UniformCSMProfile.eval(r, rho_0=rho0)
    ax.loglog(r.value, rho.value, ls=ls,
              label=fr'$\rho_0 = 10^{{{int(np.log10(rho0))}}}\ \mathrm{{g\,cm^{{-3}}}}$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Uniform CSM Profile')
ax.legend()
plt.tight_layout()