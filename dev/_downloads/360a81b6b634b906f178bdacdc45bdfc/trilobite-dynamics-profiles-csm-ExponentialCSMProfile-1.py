import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import ExponentialCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.linspace(0, 3e16, 500) * u.cm
fig, ax = plt.subplots()
for H, ls in [(3e15, '--'), (6e15, '-'), (1.2e16, ':')]:
    rho = ExponentialCSMProfile.eval(
        r,
        rho_0=1e-18 * u.g / u.cm**3,
        r_0=0.0 * u.cm,
        scale_height=H * u.cm,
        density_floor=1e-23 * u.g / u.cm**3,
    )
    ax.semilogy(r.value, rho.value, ls=ls,
                label=fr'$H = {H:.0e}\ \mathrm{{cm}}$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Exponential CSM Profile')
ax.legend()
plt.tight_layout()