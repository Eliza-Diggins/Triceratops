import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import WindWithFloorCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 400) * u.cm
rho = WindWithFloorCSMProfile.eval(
    r,
    A=5e11 * u.g / u.cm,
    density_floor=1e-23 * u.g / u.cm**3,
)
fig, ax = plt.subplots()
ax.loglog(r.value, rho.value)
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Wind + Floor CSM Profile')
plt.tight_layout()