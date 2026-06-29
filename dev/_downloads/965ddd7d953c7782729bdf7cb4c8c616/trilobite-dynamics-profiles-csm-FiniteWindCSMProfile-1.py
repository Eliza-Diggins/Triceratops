import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import FiniteWindCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e13, 1e18, 600) * u.cm
rho = FiniteWindCSMProfile.eval(
    r,
    A=5e11 * u.g / u.cm,
    r_min=1e15 * u.cm,
    r_max=3e16 * u.cm,
    density_floor=1e-23 * u.g / u.cm**3,
)
fig, ax = plt.subplots()
ax.loglog(r.value, rho.value)
ax.axvspan(1e15, 3e16, alpha=0.1, color='C0', label='Wind region')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Finite Wind CSM Profile')
ax.legend()
plt.tight_layout()