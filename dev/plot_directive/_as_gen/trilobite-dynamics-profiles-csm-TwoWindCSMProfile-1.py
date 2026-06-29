import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import TwoWindCSMProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

r = np.geomspace(1e14, 1e18, 500) * u.cm

rho = TwoWindCSMProfile.eval(
    r,
    A_star_1=1.0,
    A_star_2=100.0,
    r_transition=1e16 * u.cm,
    transition_width=0.25,
)

fig, ax = plt.subplots()
ax.loglog(r.value, rho.value)
ax.axvline(1e16, ls='--', color='gray', label=r'$r_{\rm transition}$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Smooth Two-Wind CSM Profile')
ax.legend()
plt.tight_layout()