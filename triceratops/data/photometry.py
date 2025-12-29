"""
Data containers for standardizing the format of radio data.

This module defines core container classes used to enforce a standardized,
unit-aware representation of radio photometry data for transient and
supernova modeling.

The primary abstraction is the ``RadioPhotometryContainer``, which wraps
an Astropy FITS table and provides:

- Strict column validation
- Unit-safe accessors
- Detection / non-detection handling
- Convenience plotting utilities
- Controlled mutability with optional persistence to disk
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy import units as u
from astropy.table import Table


class RadioPhotometryContainer:
    """
    Container for radio photometry observations stored in a FITS table.

    This class provides a standardized, unit-aware interface to radio
    photometric data used in transient and supernova modeling. It wraps
    an Astropy FITS table and enforces a consistent schema, supports
    detection / non-detection logic, and provides convenience accessors
    for time-, frequency-, and flux-related quantities.

    The container may operate in either a read-only or modifiable mode.
    When initialized with ``modify=True``, changes to the underlying
    table can be persisted back to disk.

    Parameters
    ----------
    path : pathlib.Path
        Path to the FITS file containing the radio photometry table.
    modify : bool, optional
        If True, modifications to the table are allowed and may be written
        back to disk. If False (default), the container is read-only.

    Notes
    -----
    **Table schema**

    The underlying FITS table is required to contain the following columns:

    - ``obs_name`` : str
      Observation identifier (e.g. telescope + epoch).

    - ``flux_density`` : float, with unit
      Measured flux density for detections.

    - ``flux_density_error`` : float, with unit
      1σ uncertainty on ``flux_density``.

    - ``flux_upper_limit`` : float, with unit
      Upper limit on flux density for non-detections.
      This column must be NaN for true detections.

    - ``time_start`` : float, with unit
      Start time of the observation.

    - ``obs_time`` : float, with unit
      Total integration time of the observation.

    - ``freq`` : float, with unit
      Central observing frequency.

    - ``band`` : int
      Integer band identifier (instrument-specific).

    - ``comments`` : str
      Free-form comments or metadata.

    **Time conventions**

    Observation mid-times and end-times are derived quantities computed as:

    - ``time_mid = time_start + 0.5 * obs_time``
    - ``time_end = time_start + obs_time``

    All time quantities are returned as Astropy ``Quantity`` objects
    with the units defined in the table.

    **Detections vs. upper limits**

    Detections and non-detections are distinguished using
    ``flux_upper_limit``:

    - A row is considered a *detection* if ``flux_upper_limit`` is NaN.
    - A row is considered a *non-detection* if ``flux_upper_limit`` is finite.

    This convention allows detection masks to be computed without
    additional bookkeeping.

    **Units**

    All numeric, physical columns are exposed as Astropy
    ``Quantity`` objects via property accessors. String and categorical
    columns (e.g. ``obs_name``, ``band``, ``comments``) are returned as
    NumPy arrays.

    Users should prefer working with the provided properties rather than
    accessing the underlying table directly.

    **Mutability**

    When ``modify=False`` (default), the container is read-only and
    attempts to modify the table or write to disk will raise an error.
    When ``modify=True``, changes may be persisted using ``save()`` or
    via assignment to the ``table`` property.

    See Also
    --------
    astropy.table.Table : Underlying table representation.
    astropy.units.Quantity : Unit-aware numerical arrays.
    """

    # ------------------------------------------------------------------
    # Required column names for a valid radio photometry table
    # ------------------------------------------------------------------
    __REQUIRED_COLUMNS__ = {
        "obs_name",
        "flux_density",
        "flux_density_error",
        "flux_upper_limit",
        "time_start",
        "obs_time",
        "freq",
        "band",
        "comments",
    }

    # ------------------------------------------------------------------
    # Canonical dtypes enforced when ingesting CSV data
    # ------------------------------------------------------------------
    __COLUMN_DTYPES__ = {
        "obs_name": str,
        "flux_density": np.float32,
        "flux_density_error": np.float32,
        "flux_upper_limit": np.float32,
        "time_start": np.float64,
        "obs_time": np.float32,
        "freq": np.float32,
        "band": int,
        "comments": str,
    }

    # ========================= Initialization ========================= #

    def __init__(self, path: Path, *_, modify: bool = False):
        """
        Initialize a RadioPhotometryContainer instance.

        Parameters
        ----------
        path : Path
            The path to the ``.fits`` file containing the data.
        modify: bool, optional
            If ``True``, then the :class:`RadioPhotometryContainer` instance will allow
            modification of the underlying ``.fits`` file. Otherwise (default), the view is
            read only.
        """
        self._path: Path = Path(path)

        if not self._path.exists():
            raise FileNotFoundError(f"FITS file not found: {self._path}")

        self._allow_modifications: bool = modify

        # Load the FITS table
        self._table: Table = Table.read(
            str(self._path.resolve()),
            format="fits",
        )

    # ========================= Dunder Methods ========================= #

    def __len__(self) -> int:
        """Return the number of observations."""
        return len(self._table)

    def __getitem__(self, key: Union[int, str]) -> Union[Table, np.void]:
        """
        Index into the container.

        Parameters
        ----------
        key : int or str
            - int: return the corresponding table row
            - str: return all rows with matching ``obs_name``

        Returns
        -------
        astropy.table.Row or astropy.table.Table
        """
        if isinstance(key, int):
            return self._table[key]

        if isinstance(key, str):
            mask = np.asarray(self._table["obs_name"] == key)
            if not mask.any():
                raise KeyError(f"No observation with obs_name='{key}'")
            return self._table[mask]

        raise TypeError("Indices must be int (row index) or str (obs_name)")

    def __iter__(self) -> Iterator:
        """Iterate over table rows."""
        return iter(self._table)

    def __enter__(self):
        """Enter context-manager mode."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context-manager mode."""
        return False  # propagate exceptions

    def __repr__(self) -> str:
        mode = "modifiable" if self._allow_modifications else "read-only"
        return f"<RadioPhotometryContainer n_obs={len(self)} mode={mode} path='{self._path.name}'>"

    def __str__(self) -> str:
        """Human-readable summary."""
        return "\n".join(
            [
                "RadioPhotometryContainer",
                f"  Path        : {self._path}",
                f"  Observations: {len(self)}",
                f"  Mode        : {'modifiable' if self._allow_modifications else 'read-only'}",
                f"  Columns     : {', '.join(self._table.colnames)}",
            ]
        )

    # ========================= Core Properties ========================= #

    @property
    def table(self) -> Table:
        """
        Return a *copy* of the underlying Astropy table.

        This prevents accidental mutation when the container is read-only.
        """
        return self._table.copy()

    @table.setter
    def table(self, value: Table):
        """Replace the underlying table and persist to disk if allowed."""
        if not self._allow_modifications:
            raise RuntimeError("Cannot modify table: container was initialized with modify=False.")
        self._table = value
        self.save(overwrite=True)

    @property
    def path(self) -> Path:
        """Path to the backing FITS file."""
        return self._path

    # ========================= Column Accessors ========================= #
    # Numeric columns return Quantities; string columns return arrays.

    @property
    def obs_names(self) -> np.ndarray:
        """Return observation names."""
        return self._table["obs_name"].data

    @property
    def flux_densities(self) -> u.Quantity:
        """Return flux densities as an Astropy Quantity array."""
        return self._table["flux_density"].quantity

    @property
    def flux_density_errors(self) -> u.Quantity:
        """Return flux density errors as an Astropy Quantity array."""
        return self._table["flux_density_error"].quantity

    @property
    def flux_upper_limits(self) -> u.Quantity:
        """Return flux upper limits as an Astropy Quantity array."""
        return self._table["flux_upper_limit"].quantity

    @property
    def time_starts(self) -> u.Quantity:
        """Return observation start times as an Astropy Quantity array."""
        return self._table["time_start"].quantity

    @property
    def obs_times(self) -> u.Quantity:
        """Return observation durations as an Astropy Quantity array."""
        return self._table["obs_time"].quantity

    @property
    def time_ends(self) -> u.Quantity:
        """Return observation end times as an Astropy Quantity array."""
        return self.time_starts + self.obs_times

    @property
    def time_mids(self) -> u.Quantity:
        """Return observation mid-times as an Astropy Quantity array."""
        return self.time_starts + 0.5 * self.obs_times

    @property
    def freqs(self) -> u.Quantity:
        """Return observation frequencies as an Astropy Quantity array."""
        return self._table["freq"].quantity

    @property
    def bands(self) -> np.ndarray:
        """Return observation band identifiers."""
        return self._table["band"].data

    @property
    def comments(self) -> np.ndarray:
        """Return observation comments."""
        return self._table["comments"].data

    # ========================= Detection Masks ========================= #

    @property
    def detection_mask(self) -> np.ndarray:
        """Boolean mask selecting detections (i.e. non-upper-limits)."""
        return np.isnan(np.asarray(self._table["flux_upper_limit"].data))

    @property
    def non_detection_mask(self) -> np.ndarray:
        """Boolean mask selecting upper limits."""
        return ~self.detection_mask

    @property
    def detection_count(self) -> int:
        """Return the number of detections."""
        return int(self.detection_mask.sum())

    @property
    def non_detection_count(self) -> int:
        """Return the number of non-detections (upper limits)."""
        return int(self.non_detection_mask.sum())

    # ========================= Persistence ========================= #

    def save(self, path: Optional[Path] = None, overwrite: bool = True):
        """
        Write the table to disk.

        Parameters
        ----------
        path : Path or None
            Output path. If None, overwrite the original FITS file.
        overwrite : bool
            Whether to overwrite existing file.
        """
        if not self._allow_modifications:
            raise RuntimeError("Cannot save: container was initialized with modify=False.")

        outpath = self._path if path is None else Path(path)
        self._table.write(outpath, format="fits", overwrite=overwrite)

    # ========================= Constructors ========================= #
    @classmethod
    def from_csv(
        cls,
        path: Path,
        output_path: Optional[Path] = None,
        column_map: Optional[dict[str, str]] = None,
        unit_map: Optional[dict[str, str]] = None,
        modify: bool = True,
    ) -> "RadioPhotometryContainer":
        """
        Construct a container from a CSV file.

        By default, the first row is interpreted as column names and the
        second row as unit strings. If ``unit_map`` is provided, all rows
        are treated as data.

        Parameters
        ----------
        path : Path
            Path to CSV file.
        output_path : Path, optional
            Output FITS file path.
        column_map : dict, optional
            Mapping from CSV column names to standardized column names.
        unit_map : dict, optional
            Mapping from standardized column names to Astropy unit strings.
        modify : bool
            Whether the resulting container is modifiable.

        Returns
        -------
        RadioPhotometryContainer
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        raw = Table.read(path, format="ascii.csv")

        # Rename columns if requested
        if column_map:
            for std_name, csv_name in column_map.items():
                if csv_name not in raw.colnames:
                    raise ValueError(f"Column '{csv_name}' not found in CSV.")
                raw.rename_column(csv_name, std_name)

        missing = cls.__REQUIRED_COLUMNS__ - set(raw.colnames)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Unit handling
        if unit_map is None:
            if len(raw) < 2:
                raise ValueError("CSV must contain unit row + data.")

            for col in raw.colnames:
                unit_str = raw[col][0]
                if unit_str:
                    raw[col].unit = u.Unit(unit_str)

            table = raw[1:]
        else:
            for col, unit_str in unit_map.items():
                raw[col].unit = u.Unit(unit_str)
            table = raw

        # Enforce canonical dtypes
        for col, dtype in cls.__COLUMN_DTYPES__.items():
            table[col] = table[col].astype(dtype)

        # Write FITS
        output_path = output_path or path.with_suffix(".fits")
        table.write(output_path, format="fits", overwrite=True)

        return cls(output_path, modify=modify)
