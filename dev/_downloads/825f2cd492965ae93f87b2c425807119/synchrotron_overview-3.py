import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from trilobite.radiation.synchrotron.SEDs.numerical import NumericalSynchrotronEngine
from trilobite.radiation.synchrotron.electron_distributions import (
    PowerLaw,
    BrokenPowerLaw,
)

# ---------------------------------------------------------------------
# Initialize synchrotron engine
# ---------------------------------------------------------------------
engine = NumericalSynchrotronEngine()
engine.load_avg_first_kernel()

# ---------------------------------------------------------------------
# Physical parameters (two emitting regions)
# ---------------------------------------------------------------------
B1, R1 = 1.0 * u.G, 1e16 * u.cm
B2, R2 = 10.0 * u.G, 1e13 * u.cm

epsilon_e = 0.1
epsilon_B = 0.01

# Electron distribution parameters
p = 2.5
gamma_c = 1e3    # cooling break Lorentz factor
gamma_min = 1.0
gamma_max = 1e10

# ---------------------------------------------------------------------
# Normalize electron distributions (equipartition closure)
# Component 1: uncooled power law; Component 2: cooling-break BPL
# (p1=injection slope, p2=cooled slope = p1+1)
# ---------------------------------------------------------------------
N0_pl = PowerLaw.normalize_from_magnetic_field(
    B1, epsilon_B, epsilon_e,
    p=p, gamma_min=gamma_min, gamma_max=gamma_max,
)

N0_bpl = BrokenPowerLaw.normalize_from_magnetic_field(
    B2, epsilon_B, epsilon_e,
    p1=p, p2=p + 1, gamma_c=gamma_c,
    gamma_min=gamma_min, gamma_max=gamma_max,
)

# Distribution functions
pl  = PowerLaw.as_callable(
    norm=N0_pl.value, p=p,
    gamma_min=gamma_min, gamma_max=gamma_max,
)

bpl = BrokenPowerLaw.as_callable(
    norm=N0_bpl.value, p1=p, p2=p + 1, gamma_c=gamma_c,
    gamma_min=gamma_min, gamma_max=gamma_max,
)

# ---------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------
gamma = np.geomspace(gamma_min, gamma_max, 500)
nu = np.geomspace(1e-1, 1e5, 500) * u.GHz

# Evaluate distributions
N_pl  = pl(gamma)
N_bpl = bpl(gamma)

# ---------------------------------------------------------------------
# Compute synchrotron intensities
# ---------------------------------------------------------------------
I_pl  = engine.compute_specific_intensity(nu, R1, B1, N_pl, gamma)
I_bpl = engine.compute_specific_intensity(nu, R2, B2, N_bpl, gamma)

I_total = I_pl + I_bpl

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

ax.loglog(nu, I_pl,  label='Component 1 (Power-law)', lw=2)
ax.loglog(nu, I_bpl, label='Component 2 (Broken power-law)', lw=2)
ax.loglog(nu, I_total, '--', label='Combined', lw=2)

ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel(r'$I_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]')
ax.set_title('Multi-component synchrotron SED')

ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.4)
ax.set_ylim(1e-7, 1e-3)

plt.tight_layout()
plt.show()