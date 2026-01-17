"""Generic / core data container structures for Triceratops.

These containers form the backbone of data handling in Triceratops, providing
a structured, validated interface to observational and synthetic datasets.
They enforce schema compliance, unit consistency, and immutability, ensuring
robustness and reproducibility throughout the modeling and inference pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from astropy import units as u
from astropy.table import Table

from triceratops.utils.log import triceratops_logger


class DataContainer(ABC):
    """
    Abstract base class for validated, immutable data containers in Triceratops.

    A :class:`DataContainer` provides a **structured, unit-aware, read-only interface**
    to observational or synthetic datasets used throughout the Triceratops modeling
    and inference pipeline. It acts as a *boundary object* between raw input data
    (typically stored as :class:`astropy.table.Table` instances) and downstream
    numerical, statistical, and physical modeling components.

    The core responsibilities of a DataContainer are to:

    - enforce a **well-defined schema** via the class-level :attr:`COLUMNS` specification,
    - validate column presence, dtypes, and physical units,
    - provide safe, immutable access to the underlying data,
    - and support conversion to NumPy arrays for performance-critical backends
      such as likelihood evaluation and samplers.

    Unlike raw Astropy tables, DataContainer instances are **immutable by design**: all
    accessors return copies of the underlying data to prevent accidental modification
    during analysis or inference. This design choice ensures reproducibility and avoids
    subtle bugs in long-running sampling workflows.

    Subclasses must define a concrete schema by overriding the :attr:`COLUMNS` class
    attribute and implement the :meth:`from_table` constructor to handle any
    domain-specific preprocessing (e.g., detection logic, epoch grouping, masking,
    or derived quantities).

    Design principles
    -----------------
    - **Schema-driven**:
      Each subclass declares a formal column schema describing required and optional
      columns, expected dtypes, and physical units. Validation occurs eagerly at
      construction time.

    - **Unit-aware**:
      Physical quantities are validated and exposed using Astropy units. Missing but
      expected units are assigned with a warning; incompatible units raise errors.

    - **Immutable**:
      The underlying table cannot be modified in place. All slicing and accessors
      return copies.

    - **Inference-friendly**:
      Containers support NumPy coercion (``np.asarray(container)``) and explicit CGS
      conversion via :meth:`to_cgs_array` for efficient numerical evaluation.

    Role in the Triceratops pipeline
    --------------------------------
    DataContainer objects are typically consumed by likelihood classes, which
    define how model predictions are statistically compared to data. Models and
    samplers **never interact with raw tables directly**—all data access flows
    through a validated container.

    This abstraction allows Triceratops to support a wide range of data modalities
    (e.g., photometry, spectra, time series, generic ``(x, y)`` datasets) while
    maintaining a consistent and robust inference API.

    Notes
    -----
    - This class is abstract and cannot be instantiated directly.
    - Subclasses should call ``super().__init__`` to ensure schema validation.
    - Equality comparisons compare the underlying tables.
    - This class deliberately does **not** subclass :class:`astropy.table.Table`
      in order to strictly control mutability and invariants.

    See Also
    --------
    triceratops.inference.likelihood.base.Likelihood
        Likelihood classes that consume DataContainer instances.
    astropy.table.Table
        Underlying data structure used for storage.
    """

    # ========================= SCHEMA DEFINITION ========================= #
    # This ``COLUMNS`` dictionary contains the core schema requirements for the input
    # table in order to ensure that the radio photometry input is valid. This is then
    # enforced in ``_validate_table``.
    #
    # DEVELOPERS: YOU NEED TO OVERWRITE THIS!
    COLUMNS = [
        {
            "name": "COLUMN_NAME",
            "dtype": str,
            "description": "Description of the column.",
            "unit": None,
            "required": True,
        }
    ]

    # ========================= SPECIAL COLUMNS =========================== #
    # In subclasses in this module, it is common to see this section filled by
    # declarations about certain special columns (i.e. X_COL = ..., Y_COL = ...).
    # This is to facilitate easy access to these columns in downstream code.
    #
    # In this base class, we do not define any special columns.

    # ====================================================================== #
    # Initialization and validation methods
    # ====================================================================== #
    def __init__(self, table: Table, **kwargs):
        """
        Instantiate the data container.

        This method should be overridden in subclasses to ensure proper behavior
        downstream. The core responsibilities of this class are to

        1. Assign the ``self.__table__`` attribute to a :class:`~astropy.table.Table` instance.
        2. Validate that the table is consistent with the schema.
        3. Any additional configuration / setup required by the subclass.

        In the default implementation, we assign the table and offload to the
        ``self._validate_table`` method for validation. New implementations should
        call ``super().__init__()`` to ensure these responsibilities are fulfilled and
        then add any additional setup required.

        Parameters
        ----------
        table: ~astropy.table.Table
            The data table to be contained. This must conform to the schema defined in the
            :attr:`COLUMNS`` class attribute.
        kwargs:
            Additional keyword arguments for subclass-specific configuration.
        """
        # Assign the table to the data class.
        if not isinstance(table, Table):
            raise TypeError("The 'table' parameter must be an instance of astropy.table.Table.")

        # Hand off to the validation method. DEVELOPERS: you should modify validation by
        # intercepting in your subclass's ``_validate_table`` method.
        self.__table__: Table = self._validate_table(table.copy())

        # Proceed with any additional management.

    def _validate_table(self, table: Table) -> Table:
        """
        Validate that the provided table conforms to the schema defined in the :attr:`COLUMNS` class attribute.

        This method should be overridden in subclasses to implement specific validation
        logic. The default implementation checks for the presence of required columns.

        Parameters
        ----------
        table: ~astropy.table.Table
            The data table to validate.

        Returns
        -------
        ~astropy.table.Table
            The validated data table.

        Raises
        ------
        ValueError
            If the table does not conform to the schema.
        """
        for column in self.__class__.COLUMNS:
            # Extract necessary fields.
            name = column["name"]
            required = column.get("required", False)

            # Check for required columns.
            if required and name not in table.colnames:
                raise ValueError(f"Missing required column '{name}' in input table.")

            # If this column just isn't in the table, we can break
            # out now. This can happen for non-required columns.
            if name not in table.colnames:
                continue

            # Now ensure that we can coerce to the correct dtype
            # before we proceed
            expected_dtype = column.get("dtype", None)
            if expected_dtype is not None:
                try:
                    table[name] = table[name].astype(expected_dtype)
                except Exception as e:
                    raise ValueError(
                        f"Column '{name}' cannot be coerced to expected dtype '{expected_dtype}': {e}"
                    ) from e

            # Now handle the units. For units, a "" is a specifically dimensionless unit,
            # while "None" means we just don't care what unit is on the data.
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
                try:
                    table[name].unit = expected_unit
                except Exception as e:
                    raise ValueError(f"Column '{name}' cannot be assigned expected unit '{expected_unit}': {e}") from e

        return table

    # ====================================================================== #
    # Dunder Methods / Specialty Methods
    # ====================================================================== #
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
        return f"<{self.__class__.__name__} n_obs={len(self)}>"

    def __contains__(self, item):
        return self.__table__.__contains__(item)

    def __bool__(self):
        return len(self) > 0

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
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

    # ====================================================================== #
    # Properties
    # ====================================================================== #
    @property
    def table(self) -> Table:
        """
        Return a copy of the underlying Astropy table.

        This prevents mutation of the internal container state.
        """
        return self.__table__.copy()

    @property
    def size(self) -> int:
        """int: The number of observations in the container."""
        return len(self)

    # ====================================================================== #
    # Utility Methods
    # ====================================================================== #
    def copy(self):
        """Create a copy of this container."""
        return self.__class__(self.__table__.copy())

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

    @classmethod
    @abstractmethod
    def from_table(cls, table: Table, **kwargs):
        """
        Create a DataContainer from an Astropy Table.

        This is an abstract method that must be implemented by subclasses.

        Parameters
        ----------
        table : ~astropy.table.Table
            The input table containing the data.
        kwargs:
            Additional keyword arguments for subclass-specific configuration.

        Returns
        -------
        DataContainer
            An instance of the subclass containing the data from the table.
        """
        pass

    @classmethod
    def from_file(cls, path: Union[str, Path], read_kws: dict = None, **kwargs):
        """
        Create a DataContainer from a FITS file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path to the FITS file containing the data.
        read_kws : dict
            Additional keyword arguments to pass to :meth:`astropy.table.Table.read`.
        kwargs:
            Additional keyword arguments to pass to :meth:`from_table`.

        Returns
        -------
        DataContainer
            An instance of the subclass containing the data from the FITS file.
        """
        # Coerce the read kwargs.
        if read_kws is None:
            read_kws = {}

        # Read the table from the file.
        table = Table.read(path, **read_kws)
        return cls.from_table(table, **kwargs)

    def to_file(self, path: Union[str, Path], write_kws: dict = None, **_):
        """
        Write the RadioPhotometryContainer's table to a file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path where the FITS file will be saved.
        write_kws : dict
            Additional keyword arguments to pass to :meth:`astropy.table.Table.write`.
        **kwargs:
            Additional keyword arguments to be used in subclasses.
        """
        if write_kws is None:
            write_kws = {}
        self.__table__.write(path, **write_kws)


# ====================================================================== #
# Generic Containers for (x, y) Data
# ====================================================================== #
class XYDataContainer(DataContainer, ABC):
    """
    Abstract base class for two-dimensional ``(x, y)`` datasets with optional uncertainties and censoring.

    An :class:`XYDataContainer` represents observational or synthetic data that
    can be expressed as pairs of an independent variable ``x`` and a dependent
    variable ``y``. This abstraction is intended to support a wide range of
    inference problems, including curve fitting, spectral modeling, and
    phenomenological regression, while remaining agnostic about the specific
    physical meaning of the axes.

    In addition to core ``(x, y)`` values, this container supports optional:

    - uncertainties on ``x`` and/or ``y``,
    - upper and lower limits (censoring) on either axis,
    - detection and non-detection masking derived from limit columns.

    All axis semantics are defined *declaratively* via class-level column
    attributes (e.g., :attr:`X_COLUMN`, :attr:`Y_ERROR_COLUMN`), allowing
    subclasses to specialize behavior without modifying downstream likelihood
    logic.

    This class does **not** define how uncertainties or limits are treated
    statistically; it only exposes them in a standardized, unit-aware form.
    Interpretation of errors and censoring is entirely delegated to likelihood
    implementations.

    Key features
    ------------
    - **Axis abstraction**:
    Independent and dependent variables are identified via class attributes
    rather than hard-coded column names.

    - **Optional uncertainties**:
    Support for symmetric uncertainties on ``x`` and/or ``y`` without
    imposing a statistical model.

    - **Censoring-aware**:
    Upper and lower limits on either axis are detected automatically and
    exposed through boolean masks.

    - **Unit-safe**:
    All values are validated and returned as Astropy quantities.

    - **Immutable**:
    Like all :class:`DataContainer` subclasses, this container is read-only.

    Detection and censoring model
    -----------------------------
    Censoring is inferred purely from the presence of *finite values* in
    declared limit columns:

    - Upper limits correspond to finite values in ``*_UPPER_LIMIT_COLUMN``.
    - Lower limits correspond to finite values in ``*_LOWER_LIMIT_COLUMN``.

    The container constructs independent boolean masks for:

    - x-axis upper limits,
    - x-axis lower limits,
    - y-axis upper limits,
    - y-axis lower limits,

    as well as composite masks indicating whether any limit applies to a given
    observation.

    No assumptions are made about how these limits should enter a likelihood
    (e.g., one-sided Gaussian, survival analysis, hard truncation).

    Intended use
    ------------
    :class:`XYDataContainer` is intended to serve as a *common interface* between
    diverse data products and inference machinery, including:

    - spectral energy distributions,
    - light curve slices at fixed frequency,
    - generic ``(x, y)`` curve-fitting problems,
    - censored regression with upper or lower limits.

    Subclasses should specialize this container by:

    - defining a concrete :attr:`COLUMNS` schema,
    - setting axis column attributes (``X_COLUMN``, ``Y_COLUMN``, etc.),
    - adding domain-specific convenience properties if needed.

    Notes
    -----
    - This class is abstract and cannot be instantiated directly.
    - Axis uncertainties are assumed to be symmetric unless otherwise
    interpreted by the likelihood.
    - If both upper and lower limits are present for the same axis and
    observation, the data point is considered interval-censored.

    See Also
    --------
    DataContainer
     Base class providing schema validation and immutability.
    triceratops.inference.likelihood.base.Likelihood
     Likelihood classes that interpret uncertainties and censoring.
    """

    # ========================= SCHEMA DEFINITION ========================= #
    # This ``COLUMNS`` dictionary contains the core schema requirements for the input
    # table in order to ensure that the radio photometry input is valid. This is then
    # enforced in ``_validate_table``.
    #
    # DEVELOPERS: YOU NEED TO OVERWRITE THIS!
    COLUMNS = [
        {
            "name": "COLUMN_NAME",
            "dtype": str,
            "description": "Description of the column.",
            "unit": None,
            "required": True,
        }
    ]

    # ========================= SPECIAL COLUMNS =========================== #
    # In subclasses in this module, it is common to see this section filled by
    # declarations about certain special columns (i.e. X_COL = ..., Y_COL = ...).
    # This is to facilitate easy access to these columns in downstream code.
    #
    # In this base class, we do not define any special columns.
    X_COLUMN = None
    """str: Name of the x-axis column.

    For :class:`XYDataContainer` subclasses, this should be overridden to specify
    the name of the column representing the x-axis data. In upstream processing,
    this is how the independent variable is identified.
    """
    Y_COLUMN = None
    """str: Name of the y-axis column.

    For :class:`XYDataContainer` subclasses, this should be overridden to specify
    the name of the column representing the y-axis data. In upstream processing,
    this is how the dependent variable is identified.
    """
    Y_ERROR_COLUMN = None
    """str: Name of the y-axis error column.

    This should be overridden to specify the name of the column representing
    the y-axis error data. This is generally the 1-sigma error, but that is
    left for the Likelihood function. If ``None``, then it is assumed that
    no y-axis errors are provided.
    """
    X_ERROR_COLUMN = None
    """str: Name of the x-axis error column.

    This should be overridden to specify the name of the column representing
    the x-axis error data. This is generally the 1-sigma error, but that is
    left for the Likelihood function. If ``None``, then it is assumed that
    no y-axis errors are provided.
    """
    Y_UPPER_LIMIT_COLUMN = None
    """str: Name of the y-axis upper limit column.

    This should be overridden to specify the name of the column representing
    the y-axis upper limit data. This is generally used to indicate non-detections.
    """
    Y_LOWER_LIMIT_COLUMN = None
    """str: Name of the y-axis lower limit column.

    This should be overridden to specify the name of the column representing
    the y-axis lower limit data. This is generally used to indicate non-detections.
    """
    X_UPPER_LIMIT_COLUMN = None
    """str: Name of the x-axis upper limit column.

    This should be overridden to specify the name of the column representing
    the x-axis upper limit data. This is generally used to indicate non-detections.
    """
    X_LOWER_LIMIT_COLUMN = None
    """str: Name of the x-axis lower limit column.

    This should be overridden to specify the name of the column representing
    the x-axis lower limit data. This is generally used to indicate non-detections.
    """

    # ====================================================================== #
    # Initialization and validation methods
    # ====================================================================== #
    def __init__(self, table: Table, **kwargs):
        """
        Instantiate the data container.

        This method should be overridden in subclasses to ensure proper behavior
        downstream. The core responsibilities of this class are to

        1. Assign the ``self.__table__`` attribute to a :class:`~astropy.table.Table` instance.
        2. Validate that the table is consistent with the schema.
        3. Generate limit masks.
        4. Any additional configuration / setup required by the subclass.

        In the default implementation, we assign the table and offload to the
        ``self._validate_table`` method for validation. New implementations should
        call ``super().__init__()`` to ensure these responsibilities are fulfilled and
        then add any additional setup required.

        Parameters
        ----------
        table: ~astropy.table.Table
            The data table to be contained. This must conform to the schema defined in the
            :attr:`COLUMNS`` class attribute.
        kwargs:
            Additional keyword arguments for subclass-specific configuration.
        """
        # Pass off to the super class to instantiate the
        # core data container functionality.
        super().__init__(table, **kwargs)

        # Now address the potential detection / non-detection logic.
        # This is done by checking for upper limit columns.
        (
            self.__x_ul_mask__,
            self.__x_ll_mask__,
            self.__y_ul_mask__,
            self.__y_ll_mask__,
        ) = self._construct_detection_mask()

    def _construct_detection_mask(self) -> np.ndarray:
        """
        Construct the detection mask based on upper limit columns.

        Returns
        -------
        numpy.ndarray
            Boolean array where True indicates a detection and False indicates
            a non-detection.

        Notes
        -----
        For each of X and Y, we first (a) check if we even have a column for the upper
        limit defined. If we do, we (b) check if that column is in the table. If it is, we (c)
        check if the value is finite. If it is finite, that indicates a non-detection
        in that axis. We combine the non-detection masks for both axes to get the final
        detection mask.
        """
        n_obs = len(self)
        _xumask, _xlmask, _yumask, _ylmask = (
            np.zeros(n_obs, dtype=bool),
            np.zeros(n_obs, dtype=bool),
            np.zeros(n_obs, dtype=bool),
            np.zeros(n_obs, dtype=bool),
        )

        # For each of the potential upper limit columns, we need to
        # check if they are defined and present in the table.
        if self.X_UPPER_LIMIT_COLUMN is not None:
            if self.X_UPPER_LIMIT_COLUMN in self.__table__.colnames:
                x_ul_col = self.__table__[self.X_UPPER_LIMIT_COLUMN]
                _xumask = np.isfinite(x_ul_col)

        if self.X_LOWER_LIMIT_COLUMN is not None:
            if self.X_LOWER_LIMIT_COLUMN in self.__table__.colnames:
                x_ll_col = self.__table__[self.X_LOWER_LIMIT_COLUMN]
                _xlmask = np.isfinite(x_ll_col)

        if self.Y_UPPER_LIMIT_COLUMN is not None:
            if self.Y_UPPER_LIMIT_COLUMN in self.__table__.colnames:
                y_ul_col = self.__table__[self.Y_UPPER_LIMIT_COLUMN]
                _yumask = np.isfinite(y_ul_col)

        if self.Y_LOWER_LIMIT_COLUMN is not None:
            if self.Y_LOWER_LIMIT_COLUMN in self.__table__.colnames:
                y_ll_col = self.__table__[self.Y_LOWER_LIMIT_COLUMN]
                _ylmask = np.isfinite(y_ll_col)

        return _xumask, _xlmask, _yumask, _ylmask

    # ====================================================================== #
    # Properties
    # ====================================================================== #
    @property
    def x(self) -> u.Quantity:
        """astropy.units.Quantity: The x-axis data."""
        if self.X_COLUMN is None:
            raise NotImplementedError(f"X_COLUMN is not defined in {self.__class__.__name__}.")

        col = self.__table__[self.X_COLUMN]
        return col.quantity.copy()

    @property
    def y(self) -> u.Quantity:
        """astropy.units.Quantity: The y-axis data."""
        if self.Y_COLUMN is None:
            raise NotImplementedError(f"Y_COLUMN is not defined in {self.__class__.__name__}.")

        col = self.__table__[self.Y_COLUMN]
        return col.quantity.copy()

    @property
    def y_error(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The y-axis error data, if provided."""
        if self.Y_ERROR_COLUMN is None:
            return None

        col = self.__table__[self.Y_ERROR_COLUMN]
        return col.quantity.copy()

    @property
    def y_upper_limit(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The y-axis upper limit data, if provided."""
        if self.Y_UPPER_LIMIT_COLUMN is None:
            return None

        col = self.__table__[self.Y_UPPER_LIMIT_COLUMN]
        return col.quantity.copy()

    def y_lower_limit(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The y-axis lower limit data, if provided."""
        if self.Y_LOWER_LIMIT_COLUMN is None:
            return None

        col = self.__table__[self.Y_LOWER_LIMIT_COLUMN]
        return col.quantity.copy()

    @property
    def x_error(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The x-axis error data, if provided."""
        if self.X_ERROR_COLUMN is None:
            return None

        col = self.__table__[self.X_ERROR_COLUMN]
        return col.quantity.copy()

    @property
    def x_upper_limit(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The x-axis upper limit data, if provided."""
        if self.X_UPPER_LIMIT_COLUMN is None:
            return None

        col = self.__table__[self.X_UPPER_LIMIT_COLUMN]
        return col.quantity.copy()

    @property
    def x_lower_limit(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The x-axis lower limit data, if provided."""
        if self.X_LOWER_LIMIT_COLUMN is None:
            return None

        col = self.__table__[self.X_LOWER_LIMIT_COLUMN]
        return col.quantity.copy()

    @property
    def x_upper_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating x-axis upper limits."""
        return self.__x_ul_mask__.copy()

    @property
    def x_lower_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating x-axis lower limits."""
        return self.__x_ll_mask__.copy()

    @property
    def x_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating x-axis limits."""
        return self.__x_ul_mask__.copy() | self.__x_ll_mask__.copy()

    @property
    def y_upper_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating y-axis upper limits."""
        return self.__y_ul_mask__.copy()

    @property
    def y_lower_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating y-axis lower limits."""
        return self.__y_ll_mask__.copy()

    @property
    def y_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating y-axis limits."""
        return self.__y_ul_mask__.copy() | self.__y_ll_mask__.copy()

    @property
    def limit_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating any limits."""
        return (
            self.__x_ul_mask__.copy()
            | self.__x_ll_mask__.copy()
            | self.__y_ul_mask__.copy()
            | self.__y_ll_mask__.copy()
        )

    # ====================================================================== #
    # Utility Functions
    # ====================================================================== #
    @classmethod
    def class_has_x_column(cls) -> bool:
        """Check if the class defines an x-axis column."""
        return cls.X_COLUMN is not None

    @classmethod
    def class_has_y_column(cls) -> bool:
        """Check if the class defines a y-axis column."""
        return cls.Y_COLUMN is not None

    @classmethod
    def class_has_y_error_column(cls) -> bool:
        """Check if the class defines a y-axis error column."""
        return cls.Y_ERROR_COLUMN is not None

    @classmethod
    def class_has_y_upper_limit_column(cls) -> bool:
        """Check if the class defines a y-axis upper limit column."""
        return cls.Y_UPPER_LIMIT_COLUMN is not None

    @classmethod
    def class_has_x_error_column(cls) -> bool:
        """Check if the class defines an x-axis error column."""
        return cls.X_ERROR_COLUMN is not None

    @classmethod
    def class_has_x_upper_limit_column(cls) -> bool:
        """Check if the class defines an x-axis upper limit column."""
        return cls.X_UPPER_LIMIT_COLUMN is not None
