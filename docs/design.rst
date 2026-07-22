****************
Design decisions
****************

This page contains my latest ramblings on the design decisions behind ``muse`` that I have made.

Units are attrs
===============

Every physical quantity carries an ``astropy.units`` unit, stored as a string in the relevant ``.attrs["units"]``.
We have converters which normalize to a canonical unit on construction (arcsec, Angstrom, km/s, etc), so downstream code can assume the canonical unit without re-checking.

**Why** For now, ``astropy.units`` does not play well with xarray, and MUSE mixes wavelengths, Doppler velocities, and spatial scales.
A silent unit mismatch (nm vs Angstrom, km/s vs m/s) will produce incorrect numbers that we can not catch.
I hope by adding the units to the ``attrs``, we can at least catch these errors at a boundary.

**Consequence** Don't strip units from the ``attrs`` and don't do unit arithmetic on raw arrays without first normalizing.
Validate input units with :func:`muse.utils.require_unit`, which checks presence, parseability, and (optionally) convertibility, then returns the parsed unit so the caller can rescale:

.. code-block:: python

    wavelength_unit = require_unit(
        response,
        "detector_wavelength",
        "response.detector_wavelength",
        coord_only=True,
        convertible_to=u.AA,
    )
    detector_wavelength = response.coords["detector_wavelength"] * wavelength_unit.to(u.AA)

Input Validation
================

There is a single presence-and-units function, :func:`muse.utils.require_unit`, which checks presence, ``sum_over`` membership, and the per-field unit checks into one call.

**Why** I want to avoid re-implementing the same checks across modules and start to enforce a consistent error message and behavior on input validation.

Datasets should be immutable
============================

Treat every input :class:`~xarray.Dataset` as read-only.
Produce results with ``assign`` / ``assign_coords`` / arithmetic, which return a *new* dataset that **shares** the underlying arrays.
So adding a coordinate or attr is cheap and never duplicates the large data variables.

**Why.** We will have large data arrays (e.g., ``vdem``, ``detector_response``, ``flux``) whereas the coordinates and attrs are tiny.
Avoiding ``ds.copy(deep=True)`` to tweak one coordinate copies *everything*, which does not scale. In-place mutation (``ds.coords[...] = ...``) silently changes the caller's object, this is something we want to avoid.

**Rules.**

- Never mutate an input in place.
  Return a new object.
- Deep-copy only the single array you actually overwrite:

  .. code-block:: python

      ds = ds.assign(detector_response=ds.detector_response.copy(deep=True))

  not the whole dataset.

- ``.attrs`` are shared on a shallow copy.
  Set attrs on a freshly computed ``DataArray`` *before* ``assign_coords``, or use ``.assign_attrs(...)``; mutating ``ds.var.attrs[...]`` on a shared object leaks back to the original.

Importing ``muse`` must not configure the host
==============================================

``import muse`` leaves the host process untouched: no ``xarray.set_options`` call and no Loguru handler replacement.
A library import that silently reconfigures process-wide state changes the behavior of the host application and of every other imported library.

**Consequences for contributors.** ``muse`` code runs under whatever xarray options the host application set, so:

- Never assume attrs survive a reduction or arithmetic operation (the host may run with ``keep_attrs=False``, and the test suite does).
  Set the attrs you need explicitly on the object you return, or pass ``keep_attrs=True`` to that one call.
  Units are the load-bearing attrs and are re-validated at every trust boundary by :func:`muse.utils.require_unit`.
- Pass the combine keyword arguments (``data_vars``, ``coords``, ``compat``, ``join``) explicitly to every ``xarray.concat`` / ``xarray.merge`` call.
  With no explicit values, the behavior depends on the host's ``use_new_combine_kwarg_defaults`` setting, and ambiguous calls raise a ``FutureWarning``, which the test suite treats as an error.
- Logging configuration is the application's job.
  For scripts and notebooks, :func:`muse.log.change_logging_level` sets the overall level explicitly at call time; it replaces every Loguru sink, so applications that manage their own Loguru configuration should configure Loguru directly instead.
  Library code must never call it.

Lineage
=======

Functions that return a dataset record the call that produced it via :func:`muse.utils.add_history`.
This keeps a human-readable trail of the operations on the data itself.
There is a similar one for ``attrs``, :func:`muse.utils.update_attrs`.
