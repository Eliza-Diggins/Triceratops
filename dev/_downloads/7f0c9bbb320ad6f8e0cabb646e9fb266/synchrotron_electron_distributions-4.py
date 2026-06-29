import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from trilobite.radiation.synchrotron.electron_distributions import (
    PowerLaw, MaxwellJuettner,
)

B    = 0.5 * u.G
eps_B = 0.1
eps_E = 0.1

# Power-law: vary the spectral index
p_vals = np.linspace(2.1, 4.0, 30)
j_pl   = np.array([
    PowerLaw.bol_emiss_from_magnetic_field(
        B, eps_B, eps_E, p=p, gamma_min=10.0, gamma_max=1e5
    ).value
    for p in p_vals
])

# Maxwell-Juettner: vary the temperature
Theta_vals = np.logspace(-1, 1, 30)
j_mj       = np.array([
    MaxwellJuettner.bol_emiss_from_magnetic_field(B, eps_B, eps_E, Theta=T).value
    for T in Theta_vals
])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].semilogy(p_vals, j_pl, lw=2, color='C0')
axes[0].set_xlabel(r'Power-law index $p$', fontsize=12)
axes[0].set_ylabel(r'Bolometric emissivity  [erg s$^{-1}$ cm$^{-3}$]', fontsize=12)
axes[0].set_title('Power-law distribution', fontsize=11)
axes[0].grid(True, ls='--', alpha=0.4)

axes[1].loglog(Theta_vals, j_mj, lw=2, color='C1')
axes[1].set_xlabel(r'Dimensionless temperature $\Theta$', fontsize=12)
axes[1].set_title(r'Maxwell-Jüttner distribution', fontsize=11)
axes[1].grid(True, which='both', ls='--', alpha=0.4)

plt.tight_layout()