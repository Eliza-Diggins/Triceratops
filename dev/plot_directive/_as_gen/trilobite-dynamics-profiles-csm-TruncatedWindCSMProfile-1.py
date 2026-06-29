import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import TruncatedWindCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 500) * u.cm
r_max = 3e16 * u.cm
rho = TruncatedWindCSMProfile.eval(
    r,
    A=5e11 * u.g / u.cm,
    r_max=r_max,
    density_floor=1e-23 * u.g / u.cm**3,
)
fig, ax = plt.subplots()
ax.loglog(r.value, rho.value)
ax.axvline(r_max.value, ls='--', color='gray', label=r'$r_{\max}$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Truncated Wind CSM Profile')
ax.legend()
plt.tight_layout()