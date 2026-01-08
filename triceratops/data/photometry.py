"""
Data container classes for radio photometry.

This module provides the :class:`RadioPhotometryContainer` class,
which encapsulates radio photometric observations stored in FITS tables and read into memory as
Astropy Tables. The container enforces a standardized schema, supports detection / non-detection logic,
and provides unit-aware accessors for time-, frequency-, and flux-related quantities.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy import units as u
from astropy.table import Table

from triceratops.utils.log import triceratops_logger


class RadioPhotometryContainer:
    """
    Immutable container for radio photometric observations.

    This class provides a **validated, unit-aware, read-only interface** to radio
    photometry data, typically originating from FITS tables. Internally, the data
    are stored as an :class:`astropy.table.Table`, but access is mediated through a
    standardized schema that enforces consistent column names, dtypes, and units.

    The container is designed to serve as a **clean boundary object** between raw
    observational data and downstream modeling, inference, and visualization
    routines. It supports explicit handling of detections versus non-detections
    (upper limits), optional epoch grouping, and safe conversion to NumPy arrays
    for numerical backends.

    Key features
    ------------
    - **Schema enforcement**
        Input tables are validated against a predefined schema specifying required
        and optional columns, expected dtypes, and physical units. Missing or
        incompatible columns raise informative errors.

    - **Unit awareness**
        All physical quantities are exposed as :class:`astropy.units.Quantity`
        objects. Unitless input columns are automatically assigned expected units
        (with a warning), while incompatible units raise an exception.

    - **Detection / non-detection logic**
        Observations are automatically classified as detections or upper limits
        based on the presence of ``flux_upper_limit`` values, with boolean masks
        and filtered table views provided.

    - **Epoch support**
        Observations may be grouped into epochs via an explicit ``epoch_id`` column
        or generated dynamically using time gaps or fixed time bins. Epochs are
        treated as metadata and do not modify the underlying data.

    - **Immutability**
        The container does not permit in-place mutation of the underlying table.
        All table accessors return copies, ensuring reproducibility and preventing
        accidental side effects during analysis.

    - **Numerical backend compatibility**
        The container can be coerced into dense NumPy arrays, either in schema units
        (via ``np.asarray(container)``) or explicitly converted to CGS base units
        (via :meth:`to_cgs_array`), for use in likelihoods, samplers, or compiled
        backends.

    Intended use
    ------------
    This class is intended to represent **observed radio photometry**, not model
    predictions or simulated data. It deliberately avoids embedding any source-
    specific physical interpretation (e.g., supernovae, jets, or shocks), and
    instead focuses on providing a robust, transparent data interface for
    higher-level emission models and inference pipelines.

    Typical workflows include:
    - Reading and validating radio photometry from FITS files
    - Grouping observations into epochs for joint modeling
    - Passing data to likelihood functions or samplers
    - Quick-look visualization of multi-frequency light curves

    Notes
    -----
    - Detection status is inferred from ``flux_upper_limit`` being NaN for
      detections and finite for non-detections.
    - Equality comparisons between containers compare the underlying tables.
    - This class intentionally does **not** subclass :class:`astropy.table.Table`
      in order to control mutability and enforce invariants.

    See Also
    --------
    astropy.table.Table
        Underlying data structure used for storage.
    triceratops.inference.likelihood
        Likelihood implementations that consume this container.
    """

    # ========================= SCHEMA DEFINITION ========================= #
    # This ``COLUMNS`` dictionary contains the core schema requirements for the input
    # table in order to ensure that the radio photometry input is valid. This is then
    # enforced in ``_validate_table``.
    COLUMNS = [
        {
            "name": "obs_name",
            "dtype": str,
            "description": "Observation identifier (e.g. telescope + epoch).",
            "required": False,
        },
        {
            "name": "flux_density",
            "dtype": float,
            "unit": u.Jy,
            "description": "Measured flux density for detections.",
            "required": True,
        },
        {
            "name": "flux_density_error",
            "dtype": float,
            "unit": u.Jy,
            "description": "1-Sigma uncertainty on ``flux_density``.",
            "required": True,
        },
        {
            "name": "flux_upper_limit",
            "dtype": float,
            "unit": u.Jy,
            "description": "Upper limit on flux density for non-detections.",
            "required": True,
        },
        {
            "name": "time",
            "dtype": float,
            "unit": u.day,
            "description": "The canonical time of the observation. This is what is used in analysis.",
            "required": True,
        },
        {
            "name": "obs_time",
            "dtype": float,
            "unit": u.day,
            "description": "Total integration time of the observation.",
            "required": False,
        },
        {
            "name": "freq",
            "dtype": float,
            "unit": u.GHz,
            "description": "Central observing frequency.",
            "required": True,
        },
        {
            "name": "band",
            "dtype": int,
            "description": "Integer band identifier (instrument-specific).",
            "required": False,
        },
        {
            "name": "comments",
            "dtype": str,
            "description": "Free-form comments or metadata.",
            "required": False,
        },
        {
            "name": "epoch_id",
            "dtype": int,
            "description": "Integer epoch identifier for grouping observations.",
            "required": False,
        },
    ]

    # ========================= Initialization ========================= #
    def __init__(self, table: Table):
        # Validate the input table and then set the self.__table__ attribute
        self.__table__ = self._validate_table(table.copy())

        # With the table validation complete, identify the detection and non-detection masks
        self.__detection_mask__ = np.isnan(np.asarray(self.__table__["flux_upper_limit"].data))
        self.__non_detection_mask__ = ~self.__detection_mask__

        # Generate the internal epochs object. If the epoch_id column is not present,
        # we set ``self.__epochs__`` to ``None``, otherwise, we extract the unique epoch IDs, order them,
        # and then store the indices of each row corresponding to each epoch ID.
        self.__epoch_ids__ = None
        if "epoch_id" in self.__table__.colnames:
            self.__epoch_ids__ = np.asarray(self.__table__["epoch_id"].data, dtype=int)

    def _validate_table(self, table: Table):
        """Ensure that the input table conforms to the required schema."""
        for column in self.__class__.COLUMNS:
            name = column["name"]
            required = column.get("required", False)

            if required and name not in table.colnames:
                raise ValueError(f"Missing required column '{name}' in input table.")

            if name not in table.colnames:
                continue

            # --- dtype coercion ---
            expected_dtype = np.dtype(column["dtype"])
            try:
                table[name] = table[name].astype(expected_dtype)
            except Exception as e:
                raise TypeError(
                    f"Column '{name}' has dtype '{table[name].dtype}', expected '{expected_dtype}'. Failed to cast: {e}"
                ) from e

            # --- unit validation ---
            expected_unit = column.get("unit", None)
            if expected_unit is not None:
                expected_unit = u.Unit(expected_unit)
                col = table[name]
                if col.unit is None:
                    triceratops_logger.warning(
                        f"Column '{name}' has no unit. Assigning expected unit '{expected_unit}'."
                    )
                    col.unit = expected_unit
                elif not col.unit.is_equivalent(expected_unit):
                    raise u.UnitsError(
                        f"Column '{name}' has unit '{col.unit}', "
                        f"which is not compatible with expected unit '{expected_unit}'."
                    )

        return table

    # ========================= Dunder Methods ========================= #
    # The scheme for dunder implementation in this class is as follows:
    #
    # - Effectively, we delegate everything to the underlying Astropy Table, this
    #   ensures users are not caught off-guard by missing functionality.
    # - However, we need to set a high __array_priority__ to ensure that
    #   operations involving both RadioPhotometryContainer and numpy arrays
    #   defer to the container's methods.
    # - We also implement __eq__ to allow for equality comparisons between
    #   two RadioPhotometryContainer instances.
    # - Finally, we are now an immutable container, so we do not implement
    #   any methods that would allow mutation of the underlying table.
    __array_priority__ = 1000

    def __len__(self):
        return self.__table__.__len__()

    def __getitem__(self, item):
        result = self.__table__.__getitem__(item)
        if isinstance(result, Table):
            return result.copy()
        else:
            return result.copy()

    def __iter__(self):
        return iter(self.__table__.copy())

    def __str__(self):
        return str(self.__table__)

    def __repr__(self):
        return f"<RadioPhotometryContainer n_obs={len(self)}>"

    def __contains__(self, item):
        return self.__table__.__contains__(item)

    def __bool__(self):
        return len(self) > 0

    def __eq__(self, other):
        if not isinstance(other, RadioPhotometryContainer):
            return NotImplemented
        return self.__table__ == other.__table__

    def __array__(self):
        """
        Convert the container to a NumPy structured array.

        This method is invoked by NumPy when coercing the container via
        ``np.asarray(container)``. Only columns marked as ``required=True`` in the
        schema and having numerical dtypes are included. Values are returned in the
        units specified by the schema (i.e. *not* automatically converted to CGS).

        The returned array is a copy and does not share writable memory with the
        internal table.

        Returns
        -------
        numpy.ndarray
            Structured NumPy array containing required numerical columns in schema
            units.
        """
        # Allocate a list to store each of the subarrays as we process them.
        _array_items = []

        # For each column in the dataset, we're going to go and
        # process it to a numerical array if it's required.
        for colspec in self.__class__.COLUMNS:
            if not colspec.get("required", False):
                continue

            name = colspec["name"]
            dtype = colspec["dtype"]

            # Skip non-numerical required columns (e.g. strings, identifiers)
            if not np.issubdtype(np.dtype(dtype), np.number):
                continue

            col = self.__table__[name]

            if col.unit is not None:
                values = col.quantity.to_value(col.unit)
            else:
                values = np.asarray(col)

            _array_items.append(values)

        return np.stack(_array_items, axis=-1)

    # ========================= Core Properties ========================= #
    # These are core properties and basic task methods.
    @property
    def table(self) -> Table:
        """
        Return a copy of the underlying Astropy table.

        This prevents mutation of the internal container state.
        """
        return self.__table__.copy()

    @property
    def detection_table(self) -> Table:
        """Return a table containing only detections."""
        return self.__table__[self.__detection_mask__].copy()

    @property
    def non_detection_table(self) -> Table:
        """Return a table containing only non-detections (upper limits)."""
        return self.__table__[self.__non_detection_mask__].copy()

    @property
    def detection_mask(self) -> np.ndarray:
        """Boolean mask selecting detections (i.e. non-upper-limits)."""
        return self.__detection_mask__

    @property
    def non_detection_mask(self) -> np.ndarray:
        """Boolean mask selecting upper limits."""
        return self.__non_detection_mask__

    @property
    def n_obs(self) -> int:
        """Return the number of observations."""
        return len(self.__table__)

    @property
    def has_epochs(self) -> bool:
        """Return True if epochs have been defined."""
        return self.__epoch_ids__ is not None

    @property
    def epoch_ids(self) -> np.ndarray:
        """Return a copy of the epoch IDs array."""
        if self.__epoch_ids__ is None:
            raise AttributeError("No epochs defined.")
        return self.__epoch_ids__.copy()

    @property
    def n_epochs(self) -> Optional[int]:
        """Return the number of unique epochs, or None if epochs are not defined."""
        if self.__epoch_ids__ is None:
            return None
        return int(len(np.unique(self.__epoch_ids__)))

    def get_epoch_mask(self, epoch: int) -> np.ndarray:
        """Return a boolean mask selecting observations in the specified epoch."""
        if self.__epoch_ids__ is None:
            raise AttributeError("No epochs defined.")
        return self.__epoch_ids__ == epoch

    def get_epoch_table(self, epoch: int) -> Table:
        """Return a table containing only observations in the specified epoch."""
        return self.__table__[self.get_epoch_mask(epoch)].copy()

    @property
    def n_detections(self) -> int:
        """Return the number of detections."""
        return int(self.__detection_mask__.sum())

    @property
    def n_non_detections(self) -> int:
        """Return the number of non-detections (upper limits)."""
        return int(self.__non_detection_mask__.sum())

    @property
    def time(self) -> u.Quantity:
        """Return observation times as an Astropy Quantity array."""
        return self.__table__["time"].quantity

    @property
    def freq(self) -> u.Quantity:
        """Return observation frequencies as an Astropy Quantity array."""
        return self.__table__["freq"].quantity

    @property
    def flux_density(self) -> u.Quantity:
        """Return flux densities as an Astropy Quantity array."""
        return self.__table__["flux_density"].quantity

    @property
    def flux_density_error(self) -> u.Quantity:
        """Return flux density errors as an Astropy Quantity array."""
        return self.__table__["flux_density_error"].quantity

    @property
    def flux_upper_limit(self) -> u.Quantity:
        """Return flux upper limits as an Astropy Quantity array."""
        return self.__table__["flux_upper_limit"].quantity

    @property
    def obs_name(self) -> np.ndarray:
        """Return observation names."""
        if "obs_name" not in self.__table__.colnames:
            raise AttributeError("Column 'obs_name' not found in the table.")
        return self.__table__["obs_name"].data

    @property
    def obs_time(self) -> u.Quantity:
        """Return observation durations as an Astropy Quantity array."""
        if "obs_time" not in self.__table__.colnames:
            raise AttributeError("Column 'obs_time' not found in the table.")
        return self.__table__["obs_time"].quantity

    @property
    def band(self) -> np.ndarray:
        """Return observation band identifiers."""
        if "band" not in self.__table__.colnames:
            raise AttributeError("Column 'band' not found in the table.")
        return self.__table__["band"].data

    @property
    def comments(self) -> np.ndarray:
        """Return observation comments."""
        if "comments" not in self.__table__.colnames:
            raise AttributeError("Column 'comments' not found in the table.")
        return self.__table__["comments"].data

    def copy(self):
        """Create a copy of this container."""
        return RadioPhotometryContainer(self.__table__.copy())

    def to_cgs_array(self):
        """
        Convert the container to a NumPy array, converted to CGS base units.

        This method mirrors ``__array__`` but explicitly converts all unit-bearing
        columns to their CGS equivalents. The output is intended for numerical backends such as
        likelihood evaluation and samplers.

        The returned array is a copy and does not share writable memory with the
        internal table.

        Returns
        -------
        numpy.ndarray
            Dense NumPy array of shape (n_obs, n_required_numeric_columns) with all
            values expressed in CGS base units.
        """
        import numpy as np
        from astropy import units as u

        _array_items = []

        for colspec in self.__class__.COLUMNS:
            if not colspec.get("required", False):
                continue

            name = colspec["name"]
            dtype = colspec["dtype"]

            # Skip non-numerical required columns
            if not np.issubdtype(np.dtype(dtype), np.number):
                continue

            col = self.__table__[name]

            if col.unit is not None:
                # Convert to CGS base units
                values = col.quantity.to(u.Unit(col.unit).cgs).value
            else:
                values = np.asarray(col)

            _array_items.append(np.asarray(values))

        return np.stack(_array_items, axis=-1)

    # ========================= Epoch Generation ========================= #
    def set_epochs_from_indices(self, indices):
        """
        Define epochs explicitly by assigning an epoch index to each observation.

        This method sets the internal epoch structure using a user-provided array
        of integer epoch identifiers, one per observation. The provided indices
        need not be contiguous or start at zero; they will be internally normalized
        to a contiguous range ``[0, n_epochs - 1]`` while preserving grouping.

        Parameters
        ----------
        indices : array-like of int, shape (n_obs,)
            Integer epoch identifiers for each observation. Observations with the
            same value are grouped into the same epoch.

        Raises
        ------
        ValueError
            If the shape of ``indices`` does not match the number of observations.
        TypeError
            If ``indices`` does not contain integer values.

        Notes
        -----
        This method overwrites any existing epoch definition. After calling this
        method, epoch-based accessors such as ``n_epochs``, ``epoch_ids``,
        ``epoch_table``, and ``epoch_times`` become available.
        """
        indices = np.asarray(indices)
        if indices.shape != (self.n_obs,):
            raise ValueError("Epoch indices must have shape (n_obs,)")

        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Epoch indices must be integers")

        # Normalize to 0..N-1
        unique = np.unique(indices)
        remap = {old: new for new, old in enumerate(unique)}
        self.__epoch_ids__ = np.vectorize(remap.get)(indices)

    def set_epochs_from_time_gaps(self, max_gap: u.Quantity):
        """
        Define epochs by grouping observations separated by time gaps.

        Observations are first sorted by observation time. A new epoch is started
        whenever the time difference between consecutive observations exceeds
        ``max_gap``. All observations separated by gaps smaller than or equal to
        ``max_gap`` are grouped into the same epoch.

        Parameters
        ----------
        max_gap : astropy.units.Quantity
            Maximum allowed time gap between consecutive observations within the
            same epoch. Must be convertible to the units of the ``time`` column.

        Raises
        ------
        astropy.units.UnitsError
            If ``max_gap`` is not compatible with the time units of the data.

        Notes
        -----
        - Epoch assignment is invariant under reordering of the input table.
        - Epoch IDs are assigned in increasing order of time.
        - This method overwrites any existing epoch definition.
        """
        times = self.time.to(max_gap.unit).value
        order = np.argsort(times)

        epoch_ids = np.zeros(len(times), dtype=int)
        current = 0

        for i in range(1, len(order)):
            if times[order[i]] - times[order[i - 1]] > max_gap.value:
                current += 1
            epoch_ids[order[i]] = current

        self.__epoch_ids__ = epoch_ids

    def set_epochs_from_bins(self, bins):
        """
        Define epochs by binning observations into fixed time intervals.

        Each observation is assigned to an epoch based on which time bin it falls
        into. Epoch IDs correspond to bin indices in the range
        ``[0, len(bins) - 2]``.

        Parameters
        ----------
        bins : array-like or astropy.units.Quantity
            Monotonically increasing array of bin edges. If a Quantity is provided,
            it must be convertible to the units of the ``time`` column.

        Raises
        ------
        ValueError
            If ``bins`` is not one-dimensional, has fewer than two edges, or if any
            observations fall outside the bin range.

        Notes
        -----
        - Bins follow NumPy's ``digitize`` convention: bins are left-inclusive and
          right-exclusive, except for the final bin.
        - This method overwrites any existing epoch definition.
        """
        if isinstance(bins, u.Quantity):
            bins = bins.to(self.time.unit).value
        bins = np.asarray(bins)

        if bins.ndim != 1 or len(bins) < 2:
            raise ValueError("bins must be 1D with at least two edges")

        epoch_ids = np.digitize(self.time.value, bins) - 1
        if np.any(epoch_ids < 0) or np.any(epoch_ids >= len(bins) - 1):
            raise ValueError("Some observations fall outside bin range")

        self.__epoch_ids__ = epoch_ids

    # ========================= IO Methods ========================= #
    @classmethod
    def from_table(cls, table: Table, column_map: Optional[dict] = None, time_starts: u.Quantity = None):
        """
        Create a :class:`RadioPhotometryContainer` from an Astropy Table.

        This method allows for optional column renaming via the `column_map` parameter,
        enabling flexibility in input table schemas.

        Parameters
        ----------
        table : astropy.table.Table
            The input table containing radio photometry data.
        column_map : dict, optional
            A mapping from existing column names (key) to expected column names (value).
            If provided, the table's columns will be renamed accordingly before validation. If
            a column is not present in the mapping, it will retain its original name.
        time_starts : astropy.time.Time, optional
            If provided, this time will be subtracted from the 'time' column to convert
            absolute times to relative times.

        Returns
        -------
        RadioPhotometryContainer
            The constructed RadioPhotometryContainer instance.
        """
        if column_map is not None:
            # Rename columns according to the provided mapping
            table = table.copy()
            table.rename_columns(list(column_map.keys()), list(column_map.values()))

        # Handle the provided time_starts adjustment.
        if time_starts is not None:
            # Ensure that we have a time column.
            if "time" not in table.colnames:
                raise ValueError("Column 'time' not found in the table for time adjustment.")

            # Ensure that the time column has a valid dtype or attempt to coerce it.
            if not np.issubdtype(table["time"].dtype, np.number):
                try:
                    table["time"] = table["time"].astype(float)
                except Exception as e:
                    raise TypeError(
                        f"Column 'time' has dtype '{table['time'].dtype}', "
                        f"expected a numerical dtype. Failed to cast: {e}"
                    ) from e

            # Ensure that the time column has a unit. If it doesn't have one, we assume days.
            time_col = table["time"]
            if time_col.unit is None:
                triceratops_logger.warning("Column 'time' has no unit. Assuming unit of days for time adjustment.")
                time_col.unit = u.day

            # Now proceed to make the adjustment.
            time_quantity = time_col.quantity
            adjusted_time = time_quantity - time_starts.to(time_quantity.unit)
            table["time"] = adjusted_time.value * time_quantity.unit

        return cls(table)

    @classmethod
    def from_file(
        cls, path: Union[str, Path], column_map: Optional[dict] = None, time_starts: u.Quantity = None, **kwargs
    ):
        """
        Create a :class:`RadioPhotometryContainer` from a FITS file.

        This method reads a FITS table from the specified file path and constructs
        a RadioPhotometryContainer. Optional column renaming can be performed via
        the `column_map` parameter.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path to the FITS file containing the radio photometry data.
        column_map : dict, optional
            A mapping from existing column names (key) to expected column names (value).
            If provided, the table's columns will be renamed accordingly before validation. If
            a column is not present in the mapping, it will retain its original name.
        time_starts : astropy.units.Quantity, optional
            If provided, this time will be subtracted from the 'time' column to convert
            absolute times to relative times.
        **kwargs:
            Additional keyword arguments to pass to `astropy.table.Table.read`.

        Returns
        -------
        RadioPhotometryContainer
            The constructed RadioPhotometryContainer instance.
        """
        table = Table.read(path, **kwargs)
        return cls.from_table(table, column_map=column_map, time_starts=time_starts)

    def to_file(self, path: Union[str, Path], **kwargs):
        """
        Write the RadioPhotometryContainer's table to a FITS file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path where the FITS file will be saved.
        **kwargs:
            Additional keyword arguments to pass to `astropy.table.Table.write`.
        """
        self.__table__.write(path, **kwargs)

    # ======================== QUICK VISUALIZATION ======================== #
    def plot_photometry(
        self,
        fig=None,
        axes=None,
        show_upper_limits=True,
        **kwargs,
    ):
        """
        Plot the radio photometry data.

        Parameters
        ----------
        fig: matplotlib.figure.Figure, optional
            Figure to plot on. If None, a new figure will be created.
        axes: matplotlib.axes.Axes, optional
            Axes to plot on. If None, axes will be created or retrieved from the figure
        show_upper_limits: bool, optional
            Whether to show upper limits in the plot. Default is True.
        kwargs:
            Additional keyword arguments for customizing the plot appearance.
            Supported kwargs include:

                - detection_fmt: Format string for detection markers (default: 'o').
                - detection_capsize: Capsize for detection error bars (default: 3).
                - detection_ms: Marker size for detections (default: 5).
                - detection_mec: Marker edge color for detections (default: 'none').
                - upper_limit_fmt: Format string for upper limit markers (default: 'v').
                - upper_limit_capsize: Capsize for upper limit error bars (default: 3).
                - upper_limit_ms: Marker size for upper limits (default: 5).
                - upper_limit_mec: Marker edge color for upper limits (default: 'none').
                - colorbar_label: Label for the colorbar (default: 'Observation Time (days)').
                - xlabel: Label for the x-axis (default: 'Frequency (GHz)').
                - ylabel: Label for the y-axis (default: 'Flux Density (mJy)').

        Returns
        -------
        fig: matplotlib.figure.Figure
            The figure containing the plot.
        axes: matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        from triceratops.utils.plot_utils import (
            get_cmap,
            get_default_cmap,
            resolve_fig_axes,
            set_plot_style,
        )

        # Set the plot style and resolve the figure and the axes.
        set_plot_style()
        fig, axes = resolve_fig_axes(fig=fig, axes=axes, fig_size=(8, 6))

        # Convert the times to colors with normalization to the top and bottom of
        # the time range.
        times = self.time.to(u.day).value
        min_time, max_time = times.min(), times.max()
        norm = plt.Normalize(min_time, max_time)

        # Determine the colormap and get the colors.
        cmap = kwargs.get("cmap", get_default_cmap())
        cmap = get_cmap(cmap)  # Ensure we have an actual colormap instance.
        colors = cmap(norm(times))

        # Plot the detections with the specified markers and colors determined by the times.
        detection_mask = self.detection_mask
        # Each detection point needs to be done separately because
        # we cannot have varying colors in a single errorbar call.
        for i in np.where(detection_mask)[0]:
            axes.errorbar(
                self.freq[i].to(u.GHz).value,
                self.flux_density[i].to(u.mJy).value,
                yerr=self.flux_density_error[i].to(u.mJy).value,
                fmt=kwargs.get("detection_fmt", "o"),
                color=colors[i],
                capsize=kwargs.get("detection_capsize", 3),
                ms=kwargs.get("detection_ms", 5),
                mec=kwargs.get("detection_mec", "none"),
            )

        # Plot the upper limits if requested.
        if show_upper_limits:
            non_detection_mask = self.non_detection_mask
            for i in np.where(non_detection_mask)[0]:
                axes.errorbar(
                    self.freq[i].to(u.GHz).value,
                    self.flux_upper_limit[i].to(u.mJy).value,
                    yerr=0.2 * self.flux_upper_limit[i].to(u.mJy).value,
                    uplims=True,
                    fmt=kwargs.get("upper_limit_fmt", "v"),
                    color=colors[i],
                    capsize=kwargs.get("upper_limit_capsize", 3),
                    ms=kwargs.get("upper_limit_ms", 5),
                    mec=kwargs.get("upper_limit_mec", "none"),
                )

        # configure the colorbar.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=axes, label=kwargs.get("colorbar_label", r"Observation Time (days)"))

        # Configure the axes and the labels.
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel(kwargs.get("xlabel", r"Frequency (GHz)"))
        axes.set_ylabel(kwargs.get("ylabel", r"Flux Density (mJy)"))

        return fig, axes
