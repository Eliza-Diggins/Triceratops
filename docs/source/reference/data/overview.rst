.. _data_overview:
=========================================
Data Loading, Handling, and Visualization
=========================================

The data modules in Triceratops provide tools for loading, processing, and visualizing observational data. Most
importantly, these structures are the entry point to the library, providing a consistent interface for working
with different types of data in our model and inference pipelines.

There are **three core data types** that appear in Triceratops:

1. **Light Curves**: Time series data representing the brightness of an object over time at one or more frequencies.
   Light curves are typically used to study the temporal evolution of radio sources. These are implemented in the
   :mod:`data.light_curve` module.
2. **Spectra**: Frequency-dependent data representing the flux density of an object at a specific time or over a
   range of times. Spectra are used to analyze the frequency characteristics of radio sources. These are implemented in the
   :mod:`data.spectra`.
3. **Photometry Tables**: Tabular data containing measurements of flux densities at various times and frequencies.
   Photometry tables provide a structured way to store and access observational data. These are implemented in the
   :mod:`data.photometry` module.

Each of these data types comes with a set of methods for loading data from common file formats (e.g., CSV, FITS),
processing the data (e.g., filtering, interpolation), and visualizing the results (e.g., plotting light curves and spectra).


Photometric Data
-----------------

Photometry data in Triceratops is handled through the :class:`data.photometry.RadioPhotometryContainer` class, which
is effectively a wrapper around a standard :class:`astropy.table.Table` object with an enforced schema dictating the
required columns and their meanings. This structure allows for easy loading, manipulation, and access to photometric
data. The photometry container includes methods for common operations such as filtering data by time or frequency,
interpolating missing values, and exporting data to various formats. It is also immediately compatible with
the inference pipelines in Triceratops, allowing users to seamlessly integrate their observational data into
model fitting and analysis workflows.

The Photometry Table
^^^^^^^^^^^^^^^^^^^^

Underlying the photometry container is an :class:`astropy.table.Table` object with a specific schema. This schema breaks
columns into 3 categories:

1. **Required Columns**: These columns must be present in the table for it to be considered valid photometry data.
   They include essential information such as time, frequency, flux density, and measurement uncertainties.
2. **Optional Columns**: These columns provide additional information that can enhance the analysis but are not strictly
   necessary. Examples include upper limits, measurement methods, and observational metadata.
3. **Auxiliary Columns**: These are columns that you, as the user, may wish to include for your own purposes. They are not
   interpreted by Triceratops in any way, but are preserved when saving and loading photometry data.

The set of **required** and **optional** columns are as follows:

.. list-table::
    :header-rows: 1
    :widths: 20 50 15 15

    * - Column Name
      - Description
      - CGS-Equivalent Unit
      - Data Type
    * - ``time``
      - Canonical time of the observation used in analysis.
      - ``s``
      - float
    * - ``freq``
      - Central observing frequency.
      - ``Hz``
      - float
    * - ``flux_density``
      - Measured flux density for detections. Should be ``np.nan`` for non-detections.
      - ``erg s^-1 cm^-2 Hz^-1``
      - float
    * - ``flux_density_error``
      - 1σ uncertainty on ``flux_density``.
      - ``erg s^-1 cm^-2 Hz^-1``
      - float
    * - ``flux_upper_limit``
      - Upper limit on flux density for non-detections. Should be ``np.nan`` for detections.
      - ``erg s^-1 cm^-2 Hz^-1``
      - float
    * - ``obs_time``
      - Total integration time of the observation.
      - ``s``
      - float
    * - ``obs_name``
      - Observation identifier (e.g. telescope + epoch).
      - ``None``
      - str
    * - ``band``
      - Integer band identifier (instrument-specific).
      - ``None``
      - int
    * - ``comments``
      - Free-form comments or metadata.
      - ``None``
      - str

.. hint::

    There are a couple of important notes regarding the photometry table schema:

    - **Time** is always a relative measurement with respect to some reference time (e.g., explosion time, trigger time).
      The actual reference time is not stored in the photometry table itself, but should be tracked separately by the user.
    - **Units**: Columns must be *compatible* with the specified CGS-equivalent units, but do not need to be in those exact units.
      For example, frequency can be provided in GHz as long as it can be converted to Hz.
    - **Non-detections**: For non-detections, the ``flux_density`` and ``flux_density_error`` columns should be set to ``np.nan``,
      and the ``flux_upper_limit`` column should contain the upper limit value.

Once the photometry table has been created, it is **immutable**; that is, you cannot add or remove rows or columns directly.
You can, of course, modify the progenitor :class:`astropy.table.Table` before creating the photometry container, or create a new
photometry container from modified data. The reason for the immutability is to ensure data integrity and consistency when using
the photometry container in analysis and modeling.

Light Curves
------------

.. important::

    Not yet implemented.

Spectra
-------

.. important::

    Not yet implemented.
