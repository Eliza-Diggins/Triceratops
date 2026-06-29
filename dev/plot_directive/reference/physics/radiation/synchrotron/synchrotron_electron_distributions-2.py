import numpy as np
import matplotlib.pyplot as plt
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw

gamma = np.logspace(1, 6, 500)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for p in [2.0, 2.5, 3.0]:
    axes[0].loglog(gamma, PowerLaw.pdf(gamma, p=p, gamma_min=10.0), label=f'$p={p}$', lw=2)
axes[0].set_xlabel(r'$\gamma$', fontsize=12)
axes[0].set_ylabel(r'$f(\gamma)$', fontsize=12)
axes[0].set_title('Varying power-law index', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, which='both', ls='--', alpha=0.4)
axes[0].set_xlim(1, 1e6)

for gmax in [1e3, 1e4, 1e5]:
    lbl = rf'$\gamma_{{\max}}={gmax:.0e}$'
    axes[1].loglog(gamma, PowerLaw.pdf(gamma, p=2.5, gamma_min=10.0, gamma_max=gmax), label=lbl, lw=2)
axes[1].set_xlabel(r'$\gamma$', fontsize=12)
axes[1].set_title('Varying upper cutoff', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, which='both', ls='--', alpha=0.4)
axes[1].set_xlim(1, 1e6)

plt.tight_layout()