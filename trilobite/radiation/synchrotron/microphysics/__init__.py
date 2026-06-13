"""
Microphysics routines for synchrotron radiation processes.

These functions provide (a) microphysics calculations for synchrotron radiation and (b) closures for
computing synchrotron from macrophysical dynamical quantities and (c) functions for computing the properties of
relativistic electron distributions.

For a detailed description of the
underlying theory, see :ref:`synchrotron_theory`. For a guide to the usage of these functions, see
:ref:`synchrotron_microphysics`.
"""

# Re-export electron_rest_energy_cgs so tests that import it from microphysics continue to work.
from trilobite.radiation.constants import electron_rest_energy_cgs

# Base utilities
from ._base import (
    _opt_equipart_magnetic_field,
    compute_equipartition_magnetic_field,
)

# Broken power-law family — private backends
# Broken power-law family — public API (new names)
# Broken power-law family — backward-compat aliases for old public names
from ._bpl import (
    _opt_BPL_bol_emissivity_from_magnetic_field,
    _opt_BPL_bol_emissivity_from_thermal,
    _opt_BPL_bol_emissivity_from_thermal_full,
    _opt_BPL_energy_norm_from_magnetic_field,
    _opt_BPL_energy_norm_from_thermal,
    _opt_BPL_moment,
    _opt_BPL_n_eff,
    _opt_BPL_n_total,
    _opt_BPL_norm_energy_to_gamma,
    _opt_BPL_norm_from_magnetic_field,
    _opt_BPL_norm_from_thermal,
    _opt_BPL_norm_gamma_to_energy,
    compute_bol_emissivity_BPL,
    compute_bol_emissivity_BPL_from_thermal_energy_density,
    compute_BPL_bol_emissivity,
    compute_BPL_bol_emissivity_from_thermal_energy_density,
    compute_BPL_effective_number_density,
    compute_BPL_energy_moment,
    compute_BPL_mean_energy,
    compute_BPL_mean_gamma,
    compute_BPL_moment,
    compute_BPL_norm_from_magnetic_field,
    compute_BPL_norm_from_thermal_energy_density,
    compute_BPL_total_number_density,
    compute_electron_energy_BPL_moment,
    compute_electron_gamma_BPL_moment,
    compute_mean_energy_BPL,
    compute_mean_gamma_BPL,
    get_BPL_distribution,
    get_broken_power_law_distribution,
    swap_BPL_normalization,
    swap_electron_BPL_normalization,
)

# Mixed MJD + PL family — private backends
# Mixed MJD + PL family — public API
# Mixed MJD + PL family — backward-compat aliases for old private names
from ._mixed import (
    _opt_mixed_norm_from_magnetic_field,
    _opt_mixed_norm_from_thermal,
    _opt_normalize_MJD_and_PL_from_magnetic_field,
    _opt_normalize_MJD_and_PL_from_thermal_energy_density,
    compute_MJD_and_PL_norm_from_magnetic_field,
    compute_MJD_and_PL_norm_from_thermal_energy_density,
)

# Maxwell-Jüttner family — private backends
# Maxwell-Jüttner family — public API (new names)
# Maxwell-Jüttner family — backward-compat aliases for old public names
from ._mjd import (
    _opt_MJD_bol_emissivity_from_magnetic_field,
    _opt_MJD_bol_emissivity_from_thermal_full,
    _opt_MJD_moment,
    _opt_MJD_n_eff,
    _opt_MJD_norm_from_magnetic_field,
    _opt_MJD_norm_from_thermal,
    compute_bol_emissivity_MJD,
    compute_bol_emissivity_MJD_from_thermal_energy_density,
    compute_electron_gamma_MJD_moment,
    compute_mean_energy_MJD,
    compute_mean_gamma_MJD,
    compute_MJD_bol_emissivity,
    compute_MJD_bol_emissivity_from_thermal_energy_density,
    compute_MJD_effective_number_density,
    compute_MJD_mean_energy,
    compute_MJD_mean_gamma,
    compute_MJD_moment,
    compute_MJD_norm_from_magnetic_field,
    compute_MJD_norm_from_thermal_energy_density,
    compute_MJD_total_number_density,
    get_maxwell_juttner_distribution,
    get_MJD_distribution,
)

# Power-law family — private backends
# Power-law family — public API (new names)
# Power-law family — backward-compat aliases for old public names
from ._pl import (
    _opt_PL_bol_emissivity_from_magnetic_field,
    _opt_PL_bol_emissivity_from_thermal,
    _opt_PL_bol_emissivity_from_thermal_full,
    _opt_PL_energy_norm_from_magnetic_field,
    _opt_PL_energy_norm_from_thermal,
    _opt_PL_moment,
    _opt_PL_n_eff,
    _opt_PL_n_total,
    _opt_PL_norm_energy_to_gamma,
    _opt_PL_norm_from_magnetic_field,
    _opt_PL_norm_from_thermal,
    _opt_PL_norm_gamma_to_energy,
    compute_bol_emissivity,
    compute_bol_emissivity_from_thermal_energy_density,
    compute_electron_energy_PL_moment,
    compute_electron_gamma_PL_moment,
    compute_mean_energy_PL,
    compute_mean_gamma_PL,
    compute_PL_bol_emissivity,
    compute_PL_bol_emissivity_from_thermal_energy_density,
    compute_PL_effective_number_density,
    compute_PL_energy_moment,
    compute_PL_mean_energy,
    compute_PL_mean_gamma,
    compute_PL_moment,
    compute_PL_norm_from_magnetic_field,
    compute_PL_norm_from_thermal_energy_density,
    compute_PL_total_number_density,
    get_PL_distribution,
    get_power_law_distribution,
    swap_electron_PL_normalization,
    swap_PL_normalization,
)

# ============================================================= #
# Old private name aliases for backward compatibility           #
# (so any code that still uses the old _opt_compute_* names     #
#  continues to work)                                           #
# ============================================================= #
_opt_compute_PL_moment = _opt_PL_moment
_opt_compute_BPL_moment = _opt_BPL_moment
_opt_compute_PL_n_total = _opt_PL_n_total
_opt_compute_BPL_n_total = _opt_BPL_n_total
_opt_compute_PL_n_eff = _opt_PL_n_eff
_opt_compute_BPL_n_eff = _opt_BPL_n_eff
_opt_convert_PL_norm_gamma_to_energy = _opt_PL_norm_gamma_to_energy
_opt_convert_BPL_norm_gamma_to_energy = _opt_BPL_norm_gamma_to_energy
_opt_compute_convert_PL_norm_energy_to_gamma = _opt_PL_norm_energy_to_gamma
_opt_compute_convert_BPL_norm_energy_to_gamma = _opt_BPL_norm_energy_to_gamma
_opt_normalize_PL_from_magnetic_field = _opt_PL_norm_from_magnetic_field
_opt_normalize_BPL_from_magnetic_field = _opt_BPL_norm_from_magnetic_field
_opt_normalize_MJD_from_magnetic_field = _opt_MJD_norm_from_magnetic_field
_opt_normalize_PL_from_thermal_energy_density = _opt_PL_norm_from_thermal
_opt_normalize_BPL_from_thermal_energy_density = _opt_BPL_norm_from_thermal
_opt_normalize_MJD_from_thermal_energy_density = _opt_MJD_norm_from_thermal
_opt_normalize_energy_PL_from_magnetic_field = _opt_PL_energy_norm_from_magnetic_field
_opt_normalize_energy_BPL_from_magnetic_field = _opt_BPL_energy_norm_from_magnetic_field
_opt_normalize_energy_PL_from_thermal_energy_density = _opt_PL_energy_norm_from_thermal
_opt_normalize_energy_BPL_from_thermal_energy_density = _opt_BPL_energy_norm_from_thermal
_opt_compute_bol_emiss_from_magnetic_field = _opt_PL_bol_emissivity_from_magnetic_field
_opt_compute_bol_emiss_BPL_from_magnetic_field = _opt_BPL_bol_emissivity_from_magnetic_field
_opt_compute_bol_emiss_MJD_from_magnetic_field = _opt_MJD_bol_emissivity_from_magnetic_field
_opt_compute_bol_emiss_from_thermal_energy_density = _opt_PL_bol_emissivity_from_thermal
_opt_compute_bol_emiss_BPL_from_thermal_energy_density = _opt_BPL_bol_emissivity_from_thermal
_opt_compute_bol_emiss_from_thermal_energy_density_full = _opt_PL_bol_emissivity_from_thermal_full
_opt_compute_bol_emiss_BPL_from_thermal_energy_density_full = _opt_BPL_bol_emissivity_from_thermal_full
_opt_compute_bol_emiss_MJD_from_thermal_energy_density_full = _opt_MJD_bol_emissivity_from_thermal_full
_opt_compute_equipart_magnetic_field = _opt_equipart_magnetic_field
_opt_compute_MJD_moment = _opt_MJD_moment
_opt_compute_MJD_n_eff = _opt_MJD_n_eff

__all__ = [
    # Constants
    "electron_rest_energy_cgs",
    # Base
    "_opt_equipart_magnetic_field",
    "compute_equipartition_magnetic_field",
    # Broken power-law — private backends
    "_opt_BPL_bol_emissivity_from_magnetic_field",
    "_opt_BPL_bol_emissivity_from_thermal",
    "_opt_BPL_bol_emissivity_from_thermal_full",
    "_opt_BPL_energy_norm_from_magnetic_field",
    "_opt_BPL_energy_norm_from_thermal",
    "_opt_BPL_moment",
    "_opt_BPL_n_eff",
    "_opt_BPL_n_total",
    "_opt_BPL_norm_energy_to_gamma",
    "_opt_BPL_norm_from_magnetic_field",
    "_opt_BPL_norm_from_thermal",
    "_opt_BPL_norm_gamma_to_energy",
    # Broken power-law — public API
    "compute_bol_emissivity_BPL",
    "compute_bol_emissivity_BPL_from_thermal_energy_density",
    "compute_BPL_bol_emissivity",
    "compute_BPL_bol_emissivity_from_thermal_energy_density",
    "compute_BPL_effective_number_density",
    "compute_BPL_energy_moment",
    "compute_BPL_mean_energy",
    "compute_BPL_mean_gamma",
    "compute_BPL_moment",
    "compute_BPL_norm_from_magnetic_field",
    "compute_BPL_norm_from_thermal_energy_density",
    "compute_BPL_total_number_density",
    "compute_electron_energy_BPL_moment",
    "compute_electron_gamma_BPL_moment",
    "compute_mean_energy_BPL",
    "compute_mean_gamma_BPL",
    "get_BPL_distribution",
    "get_broken_power_law_distribution",
    "swap_BPL_normalization",
    "swap_electron_BPL_normalization",
    # Mixed MJD+PL — private backends
    "_opt_mixed_norm_from_magnetic_field",
    "_opt_mixed_norm_from_thermal",
    "_opt_normalize_MJD_and_PL_from_magnetic_field",
    "_opt_normalize_MJD_and_PL_from_thermal_energy_density",
    # Mixed MJD+PL — public API
    "compute_MJD_and_PL_norm_from_magnetic_field",
    "compute_MJD_and_PL_norm_from_thermal_energy_density",
    # Maxwell-Jüttner — private backends
    "_opt_MJD_bol_emissivity_from_magnetic_field",
    "_opt_MJD_bol_emissivity_from_thermal_full",
    "_opt_MJD_moment",
    "_opt_MJD_n_eff",
    "_opt_MJD_norm_from_magnetic_field",
    "_opt_MJD_norm_from_thermal",
    # Maxwell-Jüttner — public API
    "compute_bol_emissivity_MJD",
    "compute_bol_emissivity_MJD_from_thermal_energy_density",
    "compute_electron_gamma_MJD_moment",
    "compute_mean_energy_MJD",
    "compute_mean_gamma_MJD",
    "compute_MJD_bol_emissivity",
    "compute_MJD_bol_emissivity_from_thermal_energy_density",
    "compute_MJD_effective_number_density",
    "compute_MJD_mean_energy",
    "compute_MJD_mean_gamma",
    "compute_MJD_moment",
    "compute_MJD_norm_from_magnetic_field",
    "compute_MJD_norm_from_thermal_energy_density",
    "compute_MJD_total_number_density",
    "get_maxwell_juttner_distribution",
    "get_MJD_distribution",
    # Power-law — private backends
    "_opt_PL_bol_emissivity_from_magnetic_field",
    "_opt_PL_bol_emissivity_from_thermal",
    "_opt_PL_bol_emissivity_from_thermal_full",
    "_opt_PL_energy_norm_from_magnetic_field",
    "_opt_PL_energy_norm_from_thermal",
    "_opt_PL_moment",
    "_opt_PL_n_eff",
    "_opt_PL_n_total",
    "_opt_PL_norm_energy_to_gamma",
    "_opt_PL_norm_from_magnetic_field",
    "_opt_PL_norm_from_thermal",
    "_opt_PL_norm_gamma_to_energy",
    # Power-law — public API
    "compute_bol_emissivity",
    "compute_bol_emissivity_from_thermal_energy_density",
    "compute_electron_energy_PL_moment",
    "compute_electron_gamma_PL_moment",
    "compute_mean_energy_PL",
    "compute_mean_gamma_PL",
    "compute_PL_bol_emissivity",
    "compute_PL_bol_emissivity_from_thermal_energy_density",
    "compute_PL_effective_number_density",
    "compute_PL_energy_moment",
    "compute_PL_mean_energy",
    "compute_PL_mean_gamma",
    "compute_PL_moment",
    "compute_PL_norm_from_magnetic_field",
    "compute_PL_norm_from_thermal_energy_density",
    "compute_PL_total_number_density",
    "get_PL_distribution",
    "get_power_law_distribution",
    "swap_electron_PL_normalization",
    "swap_PL_normalization",
    # Backward-compat aliases
    "_opt_compute_PL_moment",
    "_opt_compute_BPL_moment",
    "_opt_compute_PL_n_total",
    "_opt_compute_BPL_n_total",
    "_opt_compute_PL_n_eff",
    "_opt_compute_BPL_n_eff",
    "_opt_convert_PL_norm_gamma_to_energy",
    "_opt_convert_BPL_norm_gamma_to_energy",
    "_opt_compute_convert_PL_norm_energy_to_gamma",
    "_opt_compute_convert_BPL_norm_energy_to_gamma",
    "_opt_normalize_PL_from_magnetic_field",
    "_opt_normalize_BPL_from_magnetic_field",
    "_opt_normalize_MJD_from_magnetic_field",
    "_opt_normalize_PL_from_thermal_energy_density",
    "_opt_normalize_BPL_from_thermal_energy_density",
    "_opt_normalize_MJD_from_thermal_energy_density",
    "_opt_normalize_energy_PL_from_magnetic_field",
    "_opt_normalize_energy_BPL_from_magnetic_field",
    "_opt_normalize_energy_PL_from_thermal_energy_density",
    "_opt_normalize_energy_BPL_from_thermal_energy_density",
    "_opt_compute_bol_emiss_from_magnetic_field",
    "_opt_compute_bol_emiss_BPL_from_magnetic_field",
    "_opt_compute_bol_emiss_MJD_from_magnetic_field",
    "_opt_compute_bol_emiss_from_thermal_energy_density",
    "_opt_compute_bol_emiss_BPL_from_thermal_energy_density",
    "_opt_compute_bol_emiss_from_thermal_energy_density_full",
    "_opt_compute_bol_emiss_BPL_from_thermal_energy_density_full",
    "_opt_compute_bol_emiss_MJD_from_thermal_energy_density_full",
    "_opt_compute_equipart_magnetic_field",
    "_opt_compute_MJD_moment",
    "_opt_compute_MJD_n_eff",
]
