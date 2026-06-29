import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import GaussianShellCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 500) * u.cm
rho = GaussianShellCSMProfile.eval(
    r,
    background_density=1e-23 * u.g / u.cm**3,
    shell_density=5e-19 * u.g / u.cm**3,
    shell_radius=1e16 * u.cm,
    shell_width=2e15 * u.cm,
)
fig, ax = plt.subplots()
ax.loglog(r.value, rho.value)
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Gaussian Shell CSM Profile')
plt.tight_layout()