0.2.0 (2026-09-07)
==================

Breaking Changes
----------------

- Rename the in-memory MUSE response fields ``SG_resp``, ``SG_wvl``,
  ``SG_xpixel``, and ``line_wvl`` to ``detector_response``,
  ``detector_wavelength``, ``detector_x_pixel``, and ``line_wavelength``.
  :func:`~muse.instrument.read_response` continues to load existing
  response files that use the legacy names. The derived ``dopp_vel`` coordinate
  is now named ``doppler_velocity``. (`#42 <https://github.com/LM-SAL/muse/pull/42>`__)
- Importing ``muse`` no longer configures the host process: the import-time
  ``xarray.set_options(keep_attrs=True, use_new_combine_kwarg_defaults=True)``
  call and the Loguru handler replacement were removed.
  ``muse.log.change_logging_level`` remains available as an explicit,
  user-callable way to set the overall logging level (it replaces all Loguru
  sinks and is intended for scripts and notebooks); it is simply never called
  at import time, and the ``MUSE_DEBUG`` environment variable no longer does
  anything.
  Operations inside ``muse`` now pass the required ``xarray`` keyword arguments
  explicitly and set attributes on their outputs directly, so results no longer
  depend on host-level ``xarray`` options. (`#48 <https://github.com/LM-SAL/muse/pull/48>`__)
- Matplotlib is no longer a runtime dependency of ``muse``; it moved to the
  ``docs`` and ``tests`` extras since no library module imports it. (`#50 <https://github.com/LM-SAL/muse/pull/50>`__)
- The Torch array-conversion helpers (``torch_to_numpy``, ``numpy_to_torch``)
  and backend resolution moved from ``muse.utils`` into a private ``muse.synthesis`` module;
  synthesis is their only consumer and they are no longer part of the public API. (`#55 <https://github.com/LM-SAL/muse/pull/55>`__)
- The density dimension on line lists from `muse.instrument.create_chianti_line_list` was renamed from ``log_density`` to ``logD``, matching the ``logT`` naming convention. (`#56 <https://github.com/LM-SAL/muse/pull/56>`__)
- :func:`~muse.instrument.map_response_to_sg_detector` no longer converts energy
  to photons or steradians to detector pixels. (`#59 <https://github.com/LM-SAL/muse/pull/59>`__)
- `muse.instrument.read_response` no longer falls back to the ``channel`` coordinate when a
  response defines no ``line_wavelength``. (`#60 <https://github.com/LM-SAL/muse/pull/60>`__)
- The Doppler-velocity axis is now called ``doppler_velocity`` everywhere, replacing the legacy
  ``vdop`` name. (`#61 <https://github.com/LM-SAL/muse/pull/61>`__)
- ``DEFAULTS_MUSE.ccd_gain_sg`` is now a per-channel `xarray.DataArray` (channels
  108, 171, 284) instead of a scalar `~astropy.units.Quantity`; every channel
  currently carries the same 10 electron / DN value, but the gains can now be
  calibrated independently. (`#68 <https://github.com/LM-SAL/muse/pull/68>`__)
- Channel arguments for radiometric conversion and detector mapping
  now require scalar wavelength quantities. (`#69 <https://github.com/LM-SAL/muse/pull/69>`__)
- Replace the ``restype`` argument to `muse.transforms.match_fov` with ``tile``, rename
  ``DEFAULTS_MUSE.fov_restype`` to ``fov_tile``, and make all optional arguments keyword-only.
  Inputs wider than the MUSE field of view are still cropped when tiling is disabled,
  one-column inputs can now be tiled, and rotation no longer resamples grids that already have
  the requested pixel sizes. (`#72 <https://github.com/LM-SAL/muse/pull/72>`__)
- Replace the ``logT_method`` and ``doppler_velocity_method`` arguments to
  `muse.instrument.align_response_and_vdem` with a single ``coord_methods`` mapping of coordinate
  name to a ``(resampling method, unit)`` tuple. The aligned response and VDEM now also share
  identical coordinate values, so downstream exact joins no longer fail on floating-point
  differences. (`#75 <https://github.com/LM-SAL/muse/pull/75>`__)
- Replace the ``tile`` argument to `muse.transforms.match_fov` with ``x_extent`` and rename
  ``DEFAULTS_MUSE.fov_tile`` to ``fov_x_extent``. ``x_extent="tile"`` matches ``tile=True``,
  ``"crop"`` matches ``tile=False``, and the new ``"keep"`` leaves the x width untouched and only
  matches the spatial resolution, for instruments whose width is not tied to the MUSE raster. (`#75 <https://github.com/LM-SAL/muse/pull/75>`__)
- :func:`~muse.instrument.map_response_to_sg_detector` now takes ``channel`` as an
  optional keyword-only argument — pass it for the MUSE SG calibration, or give the
  detector geometry explicitly. ``muse.instrument.utils`` was renamed to
  ``muse.instrument.response_io``. (`#81 <https://github.com/LM-SAL/muse/pull/81>`__)
- ``InstrumentDefaults`` names now capitalize the SG/CI detector shorthands consistently:
  ``main_line_effective_area_sg``/``_ci`` → ``main_line_effective_area_SG``/``_CI``,
  ``ccd_gain_sg``/``_ci`` → ``ccd_gain_SG``/``_CI``,
  ``pair_creation_energy_sg``/``_ci`` → ``pair_creation_energy_SG``/``_CI``,
  ``instrumental_width_sg`` → ``instrumental_width_SG``,
  ``pixels_between_slits`` → ``pixels_between_slits_SG``,
  ``channel_spectral_order`` → ``channel_spectral_order_SG``. The CI calibration
  DataArrays now use the plain ``channel`` dimension instead of ``ci_channel``, matching
  the SG calibrations (each detector keeps its own channel set). The ``detector`` selector of
  `~muse.instrument.transform_response_units` is now case-insensitive (``"SG"``/``"CI"``).
  No compatibility aliases; saved response files and their generic ``channel`` coordinate
  are unchanged. (`#82 <https://github.com/LM-SAL/muse/pull/82>`__)
- ``muse.synthesis.vdem_synthesis`` now aligns the contraction by coordinate labels
  instead of purely by position. Shared indexed coordinates that do not match
  exactly (for example a response file storing ``logT`` as float32 against a
  float64 VDEM, or grids offset by interpolation drift), and shared dimensions
  that carry an index coordinate on only one input, now raise a ``ValueError``
  instead of silently contracting positionally. Regrid the response onto the
  VDEM's grid with ``muse.instrument.align_response_and_vdem`` before
  synthesizing. Output coordinates, their units, and attrs are unchanged. (`#83 <https://github.com/LM-SAL/muse/pull/83>`__)


Removals
--------

- Remove the RTD-only ``n_x_chunks`` argument from ``create_simple_vdem``. (`#43 <https://github.com/LM-SAL/muse/pull/43>`__)
- The JAX synthesis backend was removed along with the ``jax`` extra; future
  accelerator-specific solvers will target Torch only. ``vdem_synthesis`` now
  accepts ``backend="numpy"`` (default) or ``"torch"``. (`#55 <https://github.com/LM-SAL/muse/pull/55>`__)


New Features
------------

- Add CHIANTI line-list generation, including optional ion and element restrictions. (`#39 <https://github.com/LM-SAL/muse/pull/39>`__)
- Add `muse.instrument.create_spectral_response` for unit-aware,
  instrument-neutral wavelength-space response generation from CHIANTI line
  lists. (`#41 <https://github.com/LM-SAL/muse/pull/41>`__)
- Add :func:`~muse.instrument.map_response_to_sg_detector` to convert
  unit-bearing spectral responses into MUSE slit and detector geometry ready for
  synthesis, using the descriptive ``detector_response``,
  ``detector_wavelength``, ``detector_x_pixel``, and ``line_wavelength`` names. (`#42 <https://github.com/LM-SAL/muse/pull/42>`__)
- Add benchmark-backed Zarr and NetCDF response saving and a function to migrate legacy response files to the canonical schema. (`#46 <https://github.com/LM-SAL/muse/pull/46>`__)
- Added basic default effective areas for each band. (`#49 <https://github.com/LM-SAL/muse/pull/49>`__)
- Each user-facing subpackage now exposes an explicit public facade:
  ``muse.instrument``, ``muse.synthesis``, and ``muse.utils`` re-export their
  intentional public entry points in ``__all__``, the API reference is generated
  from the facades, and the examples import through them. (`#51 <https://github.com/LM-SAL/muse/pull/51>`__)
- ``read_response`` and ``load_and_concat_responses`` gained a ``chunked``
  keyword that opens response files as dask-backed arrays, so resampling and
  synthesis stay lazy and peak memory stays bounded by the on-disk chunks.
  ``map_response_to_sg_detector`` now preserves a dask-backed spectral response,
  letting ``save_response`` stream the detector response to disk; the synthesis
  tutorial uses both to cut its peak memory. ``vdem_synthesis`` rechunks
  dask-backed operands so the contracted dimensions are single-chunk, keeping
  the task graph and peak memory bounded. (`#54 <https://github.com/LM-SAL/muse/pull/54>`__)
- With the default NumPy backend, ``vdem_synthesis`` now keeps dask-backed
  inputs lazy by contracting through ``dask.array.einsum``, so synthesizing
  from a chunked zarr VDEM stays memory-bounded and writing the flux streams
  chunk by chunk. ``create_simple_vdem`` also computes its temperature-bin
  overlap in log space, avoiding two full-cube ``log10`` passes per bin. (`#54 <https://github.com/LM-SAL/muse/pull/54>`__)
- Add :func:`~muse.instrument.transform_response_units`, which converts the
  radiometric units of a wavelength-space response along the detector chain
  energy -> photon -> electron -> data number, and between per steradian and per
  detector pixel. (`#59 <https://github.com/LM-SAL/muse/pull/59>`__)
- ``DEFAULTS_MUSE`` gained ``pair_creation_energy_sg``, the mean energy that frees
  one electron-hole pair in the SG detector, stored per channel like
  ``channel_spectral_order``. (`#59 <https://github.com/LM-SAL/muse/pull/59>`__)
- Add CI response support with detector-specific calibration defaults, radiometric
  conversion, and band-integrated detector mapping. (`#69 <https://github.com/LM-SAL/muse/pull/69>`__)
- Add `muse.data.fetch_example_data` to download and cache the example data files used by the documentation gallery; the gallery examples use it instead of inlining download URLs. (`#70 <https://github.com/LM-SAL/muse/pull/70>`__)
- `muse.synthesis.vdem_synthesis` now accepts the wavelength-space response names produced by `muse.instrument.create_spectral_response` (``spectral_response``, ``wavelength_grid``) as aliases for the detector-style names, and drops ``"slit"`` from the default ``sum_over`` when neither input has a slit dimension, so single-slit responses work without extra arguments. (`#70 <https://github.com/LM-SAL/muse/pull/70>`__)
- Add `muse.instrument.align_response_and_vdem` to place response and VDEM datasets on common temperature and velocity grids. (`#71 <https://github.com/LM-SAL/muse/pull/71>`__)
- `muse.instrument.map_response_to_ci_detector` now accepts any positive channel wavelength
  instead of only the MUSE CI channels, and accepts spectral responses in emission-measure bases
  per cm3 as well as per cm5, so imaging bands of other instruments (e.g., AIA with a CHIANTI
  ``gofnt`` line list) can be band-integrated directly. (`#75 <https://github.com/LM-SAL/muse/pull/75>`__)
- The ``channel`` argument to `muse.instrument.transform_response_units` is now optional when both
  ``gain`` and ``pair_energy`` are given explicitly, so non-MUSE instruments can supply their own
  calibration without referencing a MUSE channel. (`#75 <https://github.com/LM-SAL/muse/pull/75>`__)
- The SG instrumental FWHM is now a named calibration field,
  ``InstrumentDefaults.instrumental_fwhm_SG`` (normalized to Angstroms), instead of a constant
  baked into the ``instrumental_width_SG`` property. The property now derives the sigma from the
  field and raises `ValueError` when the field is unset. (`#85 <https://github.com/LM-SAL/muse/pull/85>`__)


Bug Fixes
---------

- Correct the wavelength registration produced by :func:`~muse.instrument.map_response_to_sg_detector`
  so adjacent detector-pixel centers are separated by exactly the requested dispersion. (`#47 <https://github.com/LM-SAL/muse/pull/47>`__)
- ``add_history`` now solely owns the provenance attributes (``HISTORY``,
  ``date created``, ``date modified``, ``version``) and has gained a ``sources``
  keyword so multi-input results inherit their inputs' histories explicitly;
  ``update_attrs`` copies only non-provenance attributes. Both helpers are
  documented as finalizers that must only be applied to newly constructed
  outputs. (`#53 <https://github.com/LM-SAL/muse/pull/53>`__)
- `muse.transforms.match_fov` now restores the caller's dimension order after resampling (the internal unstack used to move the resampled axis last, transposing downstream plots relative to the input), `muse.instrument.create_chianti_line_list` now records units on the ``logT`` (``dex(K)``) and ``log_density`` (``dex(1 / cm3)``) coordinates, spectral-response creation warns when a flat effective area is combined with contaminant lines (contaminant counts are then unreliable), main lines with a single transition skip a needless concat/reduce pass (about 2x faster response creation), and ``print(DEFAULTS_MUSE)`` now lists one field per line. (`#56 <https://github.com/LM-SAL/muse/pull/56>`__)
- ``InstrumentDefaults`` now checks the channel coordinate of every per-channel
  ``DataArray`` field. (`#59 <https://github.com/LM-SAL/muse/pull/59>`__)
- A scalar ``effective_area`` whose ``attrs["units"]`` contradicts the unit its data already
  carries (for example a ``cm2`` quantity tagged ``units="s"``) now raises `ValueError` instead of
  silently ignoring the attribute. (`#65 <https://github.com/LM-SAL/muse/pull/65>`__)
- `muse.transforms.match_fov` with ``mode="constant"`` now pads the field of view with zeros, as
  its documentation always stated.
  `xarray.Dataset.pad` defaults ``constant_values`` to NaN, so the padded columns were NaN and
  propagated through synthesis into every derived moment. (`#66 <https://github.com/LM-SAL/muse/pull/66>`__)
- `muse.synthesis.vdem_synthesis` evaluates its contraction with an optimized einsum path (BLAS-backed) on both the numpy and dask backends, roughly 30x faster for the tutorial-sized synthesis. (`#70 <https://github.com/LM-SAL/muse/pull/70>`__)


Documentation
-------------

- Add a staged synthesis tutorial that prepares reusable CHIANTI line lists
  separately, creates MUSE detector responses, synthesizes observations, and
  analyzes moments from downloadable NetCDF artifacts. (`#43 <https://github.com/LM-SAL/muse/pull/43>`__)
- The documentation landing page now separates implemented capabilities from
  the mission roadmap, and the contributor guide documents the development
  environment contract and which CI system owns which jobs. (`#50 <https://github.com/LM-SAL/muse/pull/50>`__)
- Corrected the MUSE mission overview (35 slits, payload wording, Fe XXI 108 Å in the main line list, Decadal Survey link), reframed the landing page around synthesis since no MUSE data exists yet, fixed the density-dependent VDEM equation in `muse.synthesis.create_simple_vdem`, and refreshed the synthesis tutorial descriptions of VDEMs and response functions. (`#56 <https://github.com/LM-SAL/muse/pull/56>`__)
- Add a tutorial showing how the 171 Angstrom SG response varies with electron
  density and nonthermal velocity. (`#58 <https://github.com/LM-SAL/muse/pull/58>`__)
- Default values injected into docstrings by ``format_docstring`` now link to the
  matching field on :class:`~muse.variables_schema.InstrumentDefaults`. (`#59 <https://github.com/LM-SAL/muse/pull/59>`__)
- Add AIA and Hinode/EIS response and synthesis examples using instrument-neutral wavelength-space responses, and a tutorial example creating the MUSE context-imager response. (`#70 <https://github.com/LM-SAL/muse/pull/70>`__)
- Revamp the AIA 94 Angstrom synthesis example to band-integrate the wavelength-space response
  with `muse.instrument.map_response_to_ci_detector` instead of hand-rolling the integration. (`#75 <https://github.com/LM-SAL/muse/pull/75>`__)
- The installation guide moved from ``tutorial/installation`` to a top-level ``installation``
  page and the tutorial wrapper page is gone. (`#86 <https://github.com/LM-SAL/muse/pull/86>`__)
- Document the MUSE spectrograph and context-imager fields of view, add the Fe X 174.531/175.263 Å density diagnostic to the Hinode/EIS response example, and note that the default effective areas are calibrated only at the main-line wavelength. (`#99 <https://github.com/LM-SAL/muse/pull/99>`__)


Internal Changes
----------------

- Import the logger directly from `muse.log` in `muse.synthesis` instead of via the package ``__init__``. (`#44 <https://github.com/LM-SAL/muse/pull/44>`__)
- Low-level modules no longer import the ``muse`` package root:
  ``muse.utils`` reads the version from ``muse.version`` and
  ``muse.variables`` imports ``InstrumentDefaults`` directly from
  ``muse.variables_schema``. (`#53 <https://github.com/LM-SAL/muse/pull/53>`__)
- `muse.transforms.match_fov` now sources its ``mode`` and ``sub_interpolation`` defaults from
  ``DEFAULTS_MUSE`` (``fov_mode`` and ``fov_sub_interpolation``) instead of hardcoded literals;
  the values are unchanged. (`#56 <https://github.com/LM-SAL/muse/pull/56>`__)
- Internal simplifications with no change in behaviour. (`#62 <https://github.com/LM-SAL/muse/pull/62>`__)
- Removed dead configuration and documentation scaffolding. (`#63 <https://github.com/LM-SAL/muse/pull/63>`__)
- ``match_fov`` decides whether the input already sits on the MUSE pixel grid with one condition
  instead of three nested copies of the same comparison. (`#64 <https://github.com/LM-SAL/muse/pull/64>`__)
- A scalar ``effective_area`` is kept zero-dimensional instead of being expanded into a two-point
  flat curve and interpolated back onto the wavelength grid.
  The scaling it produces is unchanged. (`#65 <https://github.com/LM-SAL/muse/pull/65>`__)
- Building a spectral response for all lines in a line list now logs a warning when that means
  more than ten lines, since the computation may be slow and memory-hungry. (`#75 <https://github.com/LM-SAL/muse/pull/75>`__)
- Function signatures across the package now carry type annotations, public API and
  private helpers alike. (`#93 <https://github.com/LM-SAL/muse/pull/93>`__)
- Skip contaminant transitions whose line window lies entirely outside the wavelength grid in ``create_spectral_response``; the output is unchanged and broad-band responses build about twice as fast. (`#99 <https://github.com/LM-SAL/muse/pull/99>`__)

0.1.0 (unreleased)
==================

Previous muse PyPI package which existed before LMSAL took over the namespace on PyPI.
