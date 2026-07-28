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

from muse.data import fetch_example_data
from muse.instrument import load_and_concat_responses, match_responses_and_vdems
from muse.synthesis import vdem_synthesis
from muse.transforms import match_fov, reshape_slit_step_to_x, reshape_x_to_slit_step

##############################################################################
# First we will load and reshape the VDEM. This is the used in Example 02.

vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))
vdem_raster = reshape_x_to_slit_step(match_fov(vdem))
# We need to keep the tutorial spectrum manageable.
# Remove this selection so you can have the  full-resolution y axis.
vdem_raster = vdem_raster.isel(y=slice(None, None, 8))

##############################################################################
# For multi-line analysis, we load the response functions for several spectral
# lines and concatenate them.
#
# We then use :func:`muse.instrument.match_responses_and_vdems` to ensure that
# the VDEM and response function share the same temperature and velocity grids.

output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)
response_files = [
    fetch_example_data(fname)
    for fname in (
        "muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.nc",
        "muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.nc",
        "muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.nc",
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
response, vdem_raster = match_responses_and_vdems(
    response,
    vdem_raster,
    logT_method="nearest",
    doppler_velocity_method="nearest",
)

print(response)

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
