****************
Design decisions
****************

This page contains my latest ramblings on the design decisions behind ``muse`` that I have made.

Units are attrs
===============

Every physical quantity carries an ``astropy.units`` unit, stored as a string in the relevant ``.attrs["units"]``.
We have converters which normalize to a canonical unit on construction (arcsec, Angstrom, km/s, etc), so downstream code can assume the canonical unit without re-checking.

**Why** For now, ``astropy.units`` does not play well with xarray and since MUSE mixes wavelengths, Doppler velocities, and spatial scales.
A silent unit mismatch (nm vs Angstrom, km/s vs m/s) will produce incorrect numbers that we can not catch.
I hope by adding the units to the ``attrs``, we can at least catch these errors at a boundary.

**Consequence** Don't strip units from the ``attrs`` and don't do unit arithmetic on raw arrays without first normalizing.
Validate input units with :func:`muse.utils.require_unit`, which checks presence, parseability, and (optionally) convertibility, then returns the parsed unit so the caller can rescale:

.. code-block:: python

    sg_unit = require_unit(response, "SG_wvl", "response.SG_wvl", coord_only=True, convertible_to=u.AA)
    sg_wvl = response.coords["SG_wvl"] * sg_unit.to(u.AA)  # now it is guaranteed to be Angstrom

Input Validation
================

There is a single presence-and-units function, :func:`muse.utils.require_unit`, which checks presence, ``sum_over`` membership, and the per-field unit checks into one call.

**Why** I want to avoid re-implementing the same checks across modules and start to enforce a consistent error message and behavior on input validation.

Datasets should be immutable
============================

Treat every input :class:`~xarray.Dataset` as read-only.
Produce results with ``assign`` / ``assign_coords`` / arithmetic, which return a *new* dataset that **shares** the underlying arrays.
So adding a coordinate or attr is cheap and never duplicates the large data variables.

**Why.** We will have large data arrays (e.g., ``vdem``, ``SG_resp``, ``flux``) where as the coordinates and attrs are tiny.
Avoiding ``ds.copy(deep=True)`` to tweak one coordinate copies *everything*, which does not scale. In-place mutation (``ds.coords[...] = ...``) silently changes the caller's object, this is something we want to avoid.

**Rules.**

- Never mutate an input in place.
  Return a new object.
- Deep-copy only the single array you actually overwrite:

  .. code-block:: python

      ds = ds.assign(SG_resp=ds.SG_resp.copy(deep=True))

  not the whole dataset.

- ``.attrs`` are shared on a shallow copy.
  Set attrs on a freshly computed ``DataArray`` *before* ``assign_coords``, or use ``.assign_attrs(...)``; mutating ``ds.var.attrs[...]`` on a shared object leaks back to the original.

Lineage
=======

Functions that return a dataset record the call that produced it via :func:`muse.utils.add_history`.
This keeps a human-readable trail of the operations on the data itself.
There is a similar one for ``attrs``, :func:`muse.utils.update_attrs`.
