"""Light curve data structures and utilities."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy import units as u
from astropy.table import Table

from triceratops.utils.log import triceratops_logger


class RadioLightCurveContainer:
    """
    Immutable container for single-band radio light curve data.

    This class represents time-series radio flux density measurements taken
    at a *fixed observing band or frequency*. Unlike
    :class:`RadioPhotometryContainer`, which supports heterogeneous observing
    frequencies, this container assumes all observations belong to the same
    band and are intended to be modeled as a single light curve.

    The container provides a validated, unit-aware, read-only interface to
    light curve data and supports detections, non-detections (upper limits),
    and conversion to NumPy arrays for numerical backends.

    Intended use
    ------------
    - Likelihood evaluation for time-domain models
    - Light curve fitting and inference
    - Comparison to model-predicted light curves
    - Visualization of single-band radio evolution

    Notes
    -----
    - Detection status is inferred from ``flux_upper_limit`` being NaN
      for detections and finite for non-detections.
    - All observations are assumed to correspond to the same observing band.
    """

    # ========================= SCHEMA ========================= #
    COLUMNS = [
        {"name": "time", "dtype": float, "unit": u.day, "required": True, "description": "Observation time."},
        {
            "name": "flux_density",
            "dtype": float,
            "unit": u.Jy,
            "required": True,
            "description": "Measured flux density for detections.",
        },
        {
            "name": "flux_density_error",
            "dtype": float,
            "unit": u.Jy,
            "required": True,
            "description": "1-sigma uncertainty on flux density.",
        },
        {
            "name": "flux_upper_limit",
            "dtype": float,
            "unit": u.Jy,
            "required": True,
            "description": "Upper limit for non-detections.",
        },
        {"name": "obs_name", "dtype": str, "required": False, "description": "Optional observation identifier."},
        {"name": "comments", "dtype": str, "required": False, "description": "Optional metadata or comments."},
    ]

    # ========================= INIT ========================= #
    def __init__(
        self,
        table: Table,
        *,
        frequency: Union[float, u.Quantity],
        band: Optional[str] = None,
    ):
        """
        Instantiate a radio light curve container.

        Parameters
        ----------
        table : astropy.table.Table
            Table containing the light curve data.
        frequency : float or astropy.units.Quantity
            Observing frequency of the light curve. If unitless, assumed GHz.
        band : str, optional
            Optional human-readable band label (e.g., 'VLA C-band').
        """
        self.__table__ = self._validate_table(table.copy())

        # --- metadata ---
        if isinstance(frequency, u.Quantity):
            self.__frequency__ = frequency.to(u.GHz)
        else:
            self.__frequency__ = frequency * u.GHz

        self.__band__ = band

        # --- detection logic ---
        self.__detection_mask__ = np.isnan(np.asarray(self.__table__["flux_upper_limit"]))
        self.__non_detection_mask__ = ~self.__detection_mask__

    # ========================= VALIDATION ========================= #
    def _validate_table(self, table: Table) -> Table:
        for spec in self.COLUMNS:
            name = spec["name"]
            required = spec.get("required", False)

            if required and name not in table.colnames:
                raise ValueError(f"Missing required column '{name}'")

            if name not in table.colnames:
                continue

            # dtype coercion
            try:
                table[name] = table[name].astype(spec["dtype"])
            except Exception as e:
                raise TypeError(f"Failed to cast column '{name}' to {spec['dtype']}: {e}") from e

            # unit handling
            unit = spec.get("unit", None)
            if unit is not None:
                unit = u.Unit(unit)
                col = table[name]
                if col.unit is None:
                    triceratops_logger.warning(f"Column '{name}' has no unit. Assigning '{unit}'.")
                    col.unit = unit
                elif not col.unit.is_equivalent(unit):
                    raise u.UnitsError(f"Column '{name}' has unit '{col.unit}', expected '{unit}'.")

        return table

    # ========================= CORE PROPERTIES ========================= #
    @property
    def table(self) -> Table:
        """Return a copy of the underlying table."""
        return self.__table__.copy()

    @property
    def frequency(self) -> u.Quantity:
        """Return the observing frequency."""
        return self.__frequency__

    @property
    def band(self) -> Optional[str]:
        """Return the band label, if provided."""
        return self.__band__

    @property
    def time(self) -> u.Quantity:
        """Observation times."""
        return self.__table__["time"].quantity

    @property
    def flux_density(self) -> u.Quantity:
        """Flux density measurements."""
        return self.__table__["flux_density"].quantity

    @property
    def flux_density_error(self) -> u.Quantity:
        """Flux density uncertainties."""
        return self.__table__["flux_density_error"].quantity

    @property
    def flux_upper_limit(self) -> u.Quantity:
        """Flux density upper limits."""
        return self.__table__["flux_upper_limit"].quantity

    @property
    def detection_mask(self) -> np.ndarray:
        """Boolean mask selecting detections."""
        return self.__detection_mask__

    @property
    def non_detection_mask(self) -> np.ndarray:
        """Boolean mask selecting upper limits."""
        return self.__non_detection_mask__

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return len(self.__table__)

    # ========================= NUMERICAL BACKENDS ========================= #
    def to_cgs_array(self) -> np.ndarray:
        """
        Convert the light curve to a dense NumPy array in CGS units.

        Returns
        -------
        ndarray
            Array of shape (n_obs, 4) containing
            [time, flux, flux_err, flux_ul] in CGS units.
        """
        return np.vstack(
            [
                self.time.to(u.s).value,
                self.flux_density.to(u.erg / u.s / u.cm**2 / u.Hz).value,
                self.flux_density_error.to(u.erg / u.s / u.cm**2 / u.Hz).value,
                self.flux_upper_limit.to(u.erg / u.s / u.cm**2 / u.Hz).value,
            ]
        ).T

    # ========================= IO ========================= #
    @classmethod
    def from_table(
        cls,
        table: Table,
        *,
        frequency: Union[float, u.Quantity],
        band: Optional[str] = None,
        column_map: Optional[dict] = None,
    ):
        """
        Construct a :class:`RadioLightCurveContainer` from an existing Astropy table.

        This classmethod provides a flexible interface for building a light curve
        container from tabular data that may not exactly match the internal column
        naming conventions. An optional ``column_map`` may be supplied to rename
        columns prior to validation.

        Parameters
        ----------
        table : astropy.table.Table
            Input table containing the light curve data. At minimum, the table must
            contain columns corresponding to time, flux density, flux uncertainty,
            and flux upper limits (after applying ``column_map`` if provided).

        frequency : float or astropy.units.Quantity
            Observing frequency of the light curve. If provided as a float, units
            of GHz are assumed. If provided as a Quantity, it will be converted
            to GHz internally.

        band : str, optional
            Optional human-readable band label (e.g., ``"VLA C-band"`` or
            ``"ALMA Band 3"``). This value is stored as metadata and does not
            affect numerical calculations.

        column_map : dict, optional
            Mapping from column names in the input table to the canonical column
            names expected by the container. Keys correspond to existing column
            names in ``table``, and values correspond to target names defined in
            :attr:`COLUMNS`.

            For example::

                column_map = {
                    "t_days": "time",
                    "flux": "flux_density",
                    "flux_err": "flux_density_error",
                    "ulim": "flux_upper_limit",
                }

        Returns
        -------
        RadioLightCurveContainer
            A validated, immutable radio light curve container.

        Notes
        -----
        This method does not modify the input table in-place. A copy of the table
        is created internally before validation and column renaming.
        """
        if column_map is not None:
            table = table.copy()
            table.rename_columns(list(column_map.keys()), list(column_map.values()))
        return cls(table, frequency=frequency, band=band)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        frequency: Union[float, u.Quantity],
        band: Optional[str] = None,
        **kwargs,
    ):
        """
        Construct a :class:`RadioLightCurveContainer` from a file on disk.

        This method is a convenience wrapper around
        :meth:`astropy.table.Table.read` followed by
        :meth:`RadioLightCurveContainer.from_table`. It supports any file format
        readable by Astropy (e.g., ECSV, FITS, ASCII).

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the input data file containing the light curve table.

        frequency : float or astropy.units.Quantity
            Observing frequency of the light curve. If provided as a float, units
            of GHz are assumed. If provided as a Quantity, it will be converted
            to GHz internally.

        band : str, optional
            Optional human-readable band label for the dataset.

        **kwargs
            Additional keyword arguments passed directly to
            :meth:`astropy.table.Table.read`, such as ``format=`` or
            ``delimiter=``.

        Returns
        -------
        RadioLightCurveContainer
            A validated, immutable radio light curve container.

        Notes
        -----
        This method performs no special handling beyond reading the file and
        validating the resulting table. If column renaming is required, users
        should instead call :meth:`from_table` with a ``column_map``.
        """
        table = Table.read(path, **kwargs)
        return cls(table, frequency=frequency, band=band)
