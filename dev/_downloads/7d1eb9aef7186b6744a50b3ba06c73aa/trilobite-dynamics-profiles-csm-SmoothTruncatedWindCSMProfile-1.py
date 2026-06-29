import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import (
    SmoothTruncatedWindCSMProfile, TruncatedWindCSMProfile,
)
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
r = np.geomspace(1e14, 1e18, 500) * u.cm
A = 5e11 * u.g / u.cm
r_max = 3e16 * u.cm
rho_floor = 1e-23 * u.g / u.cm**3

rho_sharp = TruncatedWindCSMProfile.eval(
    r, A=A, r_max=r_max, density_floor=rho_floor,
)
rho_smooth = SmoothTruncatedWindCSMProfile.eval(
    r, A=A, r_max=r_max, transition_width=5e15 * u.cm, density_floor=rho_floor,
)
fig, ax = plt.subplots()
ax.loglog(r.value, rho_sharp.value, ls='--', label='Sharp truncation')
ax.loglog(r.value, rho_smooth.value, label='Smooth truncation')
ax.axvline(r_max.value, ls=':', color='gray', label=r'$r_{\max}$')
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.set_title('Smooth vs. Sharp Truncated Wind')
ax.legend()
plt.tight_layout()