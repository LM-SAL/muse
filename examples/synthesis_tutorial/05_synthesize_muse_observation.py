"""
==================================
05 - Synthesize a MUSE observation
==================================

This tutorial demonstrates how to synthesize the MUSE detector spectra.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import align_response_and_vdem, load_and_concat_responses
from muse.synthesis import vdem_synthesis
from muse.transforms import match_fov, reshape_slit_step_to_x, reshape_x_to_slit_step

##############################################################################
# First we will load and reshape the VDEM. This is the used in
# :ref:`sphx_glr_generated_gallery_synthesis_tutorial_02_vdem_to_muse_fov.py`.

vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))
vdem_raster = reshape_x_to_slit_step(match_fov(vdem))
# We need to keep the tutorial spectrum manageable.
# Remove this selection so you can have the  full-resolution y axis.
vdem_raster = vdem_raster.isel(y=slice(None, None, 8))

print(vdem_raster)

##############################################################################
# For multi-line analysis, we load the response functions for several spectral
# lines and concatenate them.

output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)
response_files = [
    fetch_example_data(fname)
    for fname in (
        "muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.zarr",
        "muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.zarr",
        "muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.zarr",
    )
]

response = load_and_concat_responses(
    response_directory=response_files[0].parent,
    response_files=[path.name for path in response_files],
    channels=[108, 171, 284],
    slit=vdem_raster.slit,
    # Keep the responses dask-backed so the synthesis below stays lazy and the
    # spectrum streams to disk instead of being materialized in memory at once.
    chunked=True,
)

print(response)

##############################################################################
# We also need to ensure that both the response function and vdem are on the
# same grid when it comes to doppler and temperature, otherwise the
# synthesis will output incorrectly.
#
# We use :func:`muse.instrument.align_response_and_vdem` to ensure that
# the VDEM and response function share the same temperature and velocity grids.

response, vdem_raster = align_response_and_vdem(
    response,
    vdem_raster,
    coord_methods={"logT": ("nearest", u.dex(u.K)), "doppler_velocity": ("linear", u.km / u.s)},
)

print(response)
print("\n\n")
print(vdem_raster)

##############################################################################
# Now we can perform the synthesis using :func:`muse.synthesis.vdem_synthesis`
# It computes synthetic spectra by convolving the VDEM with the response
# functions. The operation consists of:
#
# 1. Multiplies VDEM by response function at each (logT, doppler_velocity) point
# 2. Sums over the specified dimensions (typically logT and doppler_velocity)
# 3. Returns synthetic spectra with spatial, line, and detector-pixel dimensions
#
# Where:
#
# - **line**: spectral line identifier
# - **y, slit, step**: spatial and raster coordinates
# - **detector_x_pixel**: spectral detector pixels
# - **detector_wavelength**: wavelength at each detector pixel and slit
#
# If you find numpy too slow, we have an optional Torch backend for speeding
# up the calculation. Please install it and set ``backend="torch"``.
#
# Note that :func:`muse.synthesis.vdem_synthesis` will work
# for any response function (MUSE, EIS, EUVST etc).
# Similarly with any VDEM (or DEM for broadband filters, e.g., AIA)
# This includes VDEM with original resolution, MUSE resolution or
# with raster/step instead of x-axis.

# With dask-backed inputs the contraction stays lazy; peak memory is roughly
# one chunk's temporaries per dask worker; if you run out of RAM, cap
# dask.config.set(num_workers=...).
spectrum = vdem_synthesis(
    vdem_raster,
    response,
    backend="numpy",
    sum_over=("logT", "doppler_velocity"),
)

output = output_dir / "muse_synthetic_spectra.nc"
encoding = {"flux": {"zlib": True, "complevel": 5}}
spectrum.to_netcdf(output, engine="h5netcdf", encoding=encoding)

print(spectrum)
print(f"Saved {output}")

##############################################################################
# Finally, a quick look at the synthesized field of view: the spectrally
# integrated Fe IX 171.073 intensity, with the slit and raster-step axes
# collapsed back onto the x axis by
# :func:`muse.transforms.reshape_slit_step_to_x`.
# We read the spectrum back from the file we just saved rather than
# recomputing the lazy synthesis graph a second time.

saved_spectrum = xr.open_dataset(output, engine="h5netcdf")
intensity = reshape_slit_step_to_x(saved_spectrum.sum(dim="detector_x_pixel"))
plt.figure(figsize=(10, 4))
intensity.flux.sel(line="Fe IX 171.073").isel(pressure=0).plot(
    x="x",
    y="y",
    norm=colors.LogNorm(vmin=0.3),
    cmap="inferno",
)
plt.title("Synthesized Fe IX 171.073 intensity over the full FOV")
plt.show()
