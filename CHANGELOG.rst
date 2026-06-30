0.1.0 (unreleased)
==================

First pre-alpha release of ``muse``, a Python library to read, analyze, and reduce data from the Multi-slit Solar Explorer (MUSE) mission.

Synthesis
---------

- ``muse.synthesis.vdem_synthesis`` — synthesize observables from a VDEM raster and one or more response functions via a tensor product, with NumPy, PyTorch, or JAX backends (optional GPU).
- ``muse.synthesis.create_simple_vdem`` — build a velocity differential emission measure (DEM as a function of temperature and line-of-sight velocity) from a simulation box.
- ``muse.synthesis.calculate_moments`` — compute the zeroth, first, and second spectral moments of a spectrum, with optional velocity masking. Flux and Doppler coordinate names are configurable.
- ``muse.synthesis.wavelength_to_doppler`` / ``doppler_to_wavelength`` — convert between wavelength and Doppler-velocity coordinates.

Instrument responses
--------------------

- ``muse.instrument.read_response`` — read a response function into an `xarray.Dataset`, interpolating over temperature (``logT``) and velocity (``vdop``) as needed.
- ``muse.instrument.load_and_concat_responses`` — load multiple response functions and concatenate them along the ``line`` axis.

Transforms
----------

- ``muse.transforms.match_fov`` — match data to the MUSE field of view by resampling to the instrument resolution and tiling along the raster axis.
- ``muse.transforms.reshape_x_to_slit_step`` / ``reshape_slit_step_to_x`` — convert between a single ``x`` spatial axis and the slit / raster-step layout.

Instrument defaults
-------------------

- ``muse.variables.DEFAULTS_MUSE`` — frozen MUSE instrument constants (pixel sizes, slit geometry, diffraction parameters) with units attached.

0.1.0 (unreleased)
==================

Previous muse PyPI package which existed before LMSAL took over the namespace on PyPI.
