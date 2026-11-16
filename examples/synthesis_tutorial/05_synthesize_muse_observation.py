"""
==================================
05 - Synthesize a MUSE observation
==================================

Synthesize the MUSE detector spectra from the VDEM and the SG responses.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import align_response_and_vdem, read_responses
from muse.synthesis import vdem_synthesis
from muse.transforms import match_fov, reshape_slit_step_to_x, reshape_x_to_slit_step

##############################################################################
# Load the VDEM and reshape it to the MUSE raster, as in
# :ref:`sphx_glr_generated_gallery_synthesis_tutorial_02_vdem_to_muse_fov.py`.

vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))
vdem_raster = reshape_x_to_slit_step(match_fov(vdem))

print(vdem_raster)

##############################################################################
# :func:`muse.instrument.read_responses` concatenates the responses of the three bands.

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

# Each file gives its channel and its camera gain.
response = read_responses(
    response_files,
    number_of_slits=vdem_raster.sizes["slit"],
    # Lazy, so the spectrum streams to disk instead of filling memory.
    chunked=True,
)

print(response)

##############################################################################
# The synthesis needs the response and the VDEM on the same temperature and
# velocity grids, which :func:`muse.instrument.align_response_and_vdem` gives.

response, vdem_raster = align_response_and_vdem(
    response,
    vdem_raster,
    coordinate_methods={"logT": ("nearest", u.dex(u.K)), "doppler_velocity": ("linear", u.km / u.s)},
)

print(response)
print("\n\n")
print(vdem_raster)

##############################################################################
# :func:`muse.synthesis.vdem_synthesis` multiplies the VDEM by the response at
# each (logT, doppler_velocity) point and sums over ``sum_over``. The spectrum
# has ``line``, the spatial ``y``, ``slit`` and ``step``, and ``detector_x_pixel``
# dimensions, with a ``detector_wavelength`` per pixel and slit.
# ``backend="torch"`` uses the faster, optional Torch backend.
#
# It works for any response (MUSE, EIS, ...) and any VDEM, or DEM for a
# broadband filter such as AIA, at any resolution, with x or with slit and step.

# Peak memory is about one chunk per dask worker; lower num_workers in dask.config if RAM runs out.
spectrum = vdem_synthesis(
    vdem_raster,
    response,
    backend="numpy",
    sum_over=("logT", "doppler_velocity"),
)

output = output_dir / "muse_synthetic_spectra.nc"
temporary_output = output.with_suffix(".tmp.nc")
encoding = {"flux": {"zlib": True, "complevel": 5}}
spectrum.to_netcdf(temporary_output, engine="h5netcdf", encoding=encoding)
temporary_output.replace(output)

print(spectrum)
print(f"Saved {output}")

##############################################################################
# The spectrally integrated Fe IX 171.073 intensity, with slit and step put
# back onto x by :func:`muse.transforms.reshape_slit_step_to_x`. Reading the
# saved file dask-backed avoids running the lazy synthesis twice or loading
# the whole flux at once.

saved_spectrum = xr.open_dataset(output, engine="h5netcdf", chunks="auto")
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
