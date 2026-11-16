****************
Design decisions
****************

.. _observation-roadmap:

Observation-handling roadmap
============================

``muse`` will support MUSE context-imager and spectrograph observations.
Planned data containers will hold observations with uncertainties, masks, world coordinate systems, and metadata.
Calibration tools will cover exposure-time correction, conversion between data numbers and photons, and radiometric calibration, with custom routines that make level 3 data from level 2.
Only the conversions exist yet, for synthetic spectra.
`MUSE observations will be public together with co-aligned SDO/AIA data <https://muse.lmsal.com/search/>`__.

Units are attrs
===============

Every physical quantity carries an ``astropy.units`` unit, stored as a string in the relevant ``.attrs["units"]``.
We have converters which normalize to a canonical unit on construction (arcsec, Angstrom, km/s, etc), so downstream code can assume the canonical unit without re-checking.

**Why.** ``astropy.units`` does not yet play well with xarray, and MUSE mixes wavelengths, Doppler velocities, and spatial scales.
A unit in the ``attrs`` lets a boundary catch a mismatch (nm vs Angstrom, km/s vs m/s) that would otherwise give wrong numbers silently.

**Consequence.** Keep the units in the ``attrs`` and normalize before arithmetic on raw arrays.
Validate input units with :func:`muse.utils.require_unit`, which returns the unit so the caller can rescale:

.. code-block:: python

    wavelength_unit = require_unit(
        response,
        "detector_wavelength",
        "response.detector_wavelength",
        coord_only=True,
        convertible_to=u.AA,
    )
    detector_wavelength = response.coords["detector_wavelength"] * wavelength_unit.to(u.AA)

One function does these checks, so every unit error reads the same.
Domain checks call it rather than live in it, as ``muse.synthesis.synthesis._validate_inputs`` does.

Datasets should be immutable
============================

Treat every input :class:`~xarray.Dataset` as read-only.
``assign``, ``assign_coords`` and arithmetic return a *new* dataset that **shares** the underlying arrays, so adding a coordinate or attr is cheap.

**Why.** The data arrays (``vdem``, ``detector_response``, ``flux``) are large and the coordinates and attrs tiny.
``ds.copy(deep=True)`` to tweak one coordinate copies *everything*, and in-place mutation (``ds.coords[...] = ...``) silently changes the caller's object.

**Rules.**

- Never mutate an input in place; return a new object.
- Deep-copy only the array you overwrite, not the whole dataset:

  .. code-block:: python

      ds = ds.assign(detector_response=ds.detector_response.copy(deep=True))

- ``.attrs`` are shared on a shallow copy.
  Set attrs on a freshly computed ``DataArray`` *before* ``assign_coords``, or use ``.assign_attrs(...)``; mutating ``ds.var.attrs[...]`` on a shared object leaks back to the original.

**Exception: finalizers.**
:func:`muse.utils.add_history` and :func:`muse.utils.update_attrs` mutate their argument in place and return `None`.
Call them only on a new output the calling function owns, named before the call so the ownership shows.

Importing ``muse`` must not configure the host
==============================================

``import muse`` leaves the host process untouched: no ``xarray.set_options`` call and no Loguru handler replacement.
Reconfiguring process-wide state would change the host application and every other library.

**Consequences.** ``muse`` code runs under the host's xarray options, so:

- Never assume attrs survive a reduction or arithmetic (the host may set ``keep_attrs=False``, and the test suite does).
  Set the attrs you need on the result, or pass ``keep_attrs=True`` to that call.
  :func:`muse.utils.require_unit` re-validates the units at every boundary.
- Pass ``data_vars``, ``coords``, ``compat``, ``join`` and ``combine_attrs`` to every ``xarray.concat`` and ``xarray.merge``.
  Their defaults depend on the host's ``use_new_combine_kwarg_defaults``, and ambiguous calls raise a ``FutureWarning``, an error in the test suite.
- Logging configuration is the application's job.
  :func:`muse.log.change_logging_level` replaces every Loguru sink and is for scripts and notebooks; library code must never call it.

Lineage
=======

Functions that return a dataset record the call in its history with :func:`muse.utils.add_history`, a readable trail on the data itself.
``add_history`` alone sets the provenance attributes (``history``, ``date_created``, ``date_modified``, ``muse_version``).

A new dataset keeps its own attrs and is finalized in two steps: :func:`muse.utils.update_attrs` copies the other attrs of its sources, then ``add_history(..., sources=...)`` inherits their histories and records the call.
Concatenation must choose ``combine_attrs`` deliberately rather than keep only the attrs of the first input.
