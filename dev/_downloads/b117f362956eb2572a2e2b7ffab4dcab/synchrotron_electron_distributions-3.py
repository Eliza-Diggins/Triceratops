import numpy as np
import matplotlib.pyplot as plt
from trilobite.radiation.synchrotron.electron_distributions import (
    PowerLaw, BrokenPowerLaw, MaxwellJuettner,
)

gamma = np.logspace(0, 4, 500)
N_total = 1e6

# Set the parameter dictionaries
params = [
    {'p': 2.5, 'gamma_min': 10.0, 'gamma_max': 1e5},
    {'p1': 2.0, 'p2': 3.5, 'gamma_c': 50, 'gamma_min': 10.0, 'gamma_max': 1e5},
    {'Theta': 2.0},
]

# Compute the norms for each distribution.
norms = [
    PowerLaw.normalize_from_n_total(N_total, **params[0]),
    BrokenPowerLaw.normalize_from_n_total(N_total, **params[1]),
    MaxwellJuettner.normalize_from_n_total(N_total, **params[2]),
]

# Compute the CDF
cdf = [
    PowerLaw.cdf(gamma, **params[0], norm=norms[0].value),
    BrokenPowerLaw.cdf(gamma, **params[1], norm=norms[1].value),
    MaxwellJuettner.cdf(gamma, **params[2], norm=norms[2].value),
]


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(gamma, cdf[0], label='Power Law')
ax.plot(gamma, cdf[1], label='Broken Power Law')
ax.plot(gamma, cdf[2], label='Maxwell-Juettner')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$', fontsize=12)
ax.set_ylabel(r'$F(\gamma) / N_{\rm tot}$', fontsize=12)
ax.set_title('Normalized CDFs', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.show()