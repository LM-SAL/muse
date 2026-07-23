"""
==================================
05 - Synthesize a MUSE observation
==================================

This tutorial demonstrates how to synthesize the MUSE detector spectra.
"""

import os
from pathlib import Path

import pooch
import xarray as xr

from muse.instrument import load_and_concat_responses
from muse.log import change_logging_level
from muse.synthesis import vdem_synthesis
from muse.transforms import match_fov, reshape_x_to_slit_step

# muse logs at DEBUG level by default; raise it to INFO to reduce the noise.
change_logging_level("INFO")

##############################################################################
# First we will load and reshape the VDEM. This is the used in Example 02.

extract_path = Path(pooch.os_cache("muse")) / "muse_example_vdem"
pooch.retrieve(
    "https://www.dropbox.com/scl/fi/xb2f6pvs4cn1yg54n0pdg/muse_example_vdem.zarr.tar.gz?rlkey=u5y19c5lydrw9kur9bzahkvsv&st=t5vltlk8&dl=1",
    known_hash="ab6c8a3fe4f30de6906f75165f19ccc8730040527f6b9b0cccbdd9a09c28a71c",
    fname="muse_example_vdem.zarr.tar.gz",
    path=extract_path.parent,
    processor=pooch.Untar(extract_dir=extract_path.name),
)
vdem = xr.open_zarr(extract_path / "muse_example_vdem.zarr")
vdem_raster = reshape_x_to_slit_step(match_fov(vdem))
# We need to keep the tutorial spectrum manageable.
# Remove this selection so you can have the  full-resolution y axis.
vdem_raster = vdem_raster.isel(y=slice(None, None, 8))

##############################################################################
# For multi-line analysis, we load the response functions for several spectral
# lines and concatenate them. Each response function is interpolated to
# match the VDEM's ``logT`` and ``vdop`` grids.
#
# To do this, we will use :func:`muse.instrument.load_and_concat_responses`
# This ensures that the VDEM and response function share the same temperature
# and velocity grids.

abundance = "sun_coronal_2021_chianti"
tutorial_cache = Path(pooch.os_cache("muse")) / "synthesis_tutorial"
output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)
response_artifacts = [
    (
        f"muse_sg_response_108_FeXIX108.355_FeXXI108.117_{abundance}_effarea.nc",
        "https://www.dropbox.com/scl/fi/j9a783mb7cgfgvv32dvyg/muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.nc?rlkey=6l10a72999tunxabjmaj6todk&st=j25qujz2&dl=1",
        "sha256:7cc72056ff8f5da547a8ebab649742b3463b0c1598bb1956826f73b0db893090",
    ),
    (
        f"muse_sg_response_171_FeIX171.073_{abundance}_effarea.nc",
        "https://www.dropbox.com/scl/fi/kyc9fisbxl8hxxl1um4ok/muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.nc?rlkey=h8coyvfvi35yzefqn9r0atgbl&st=3pelvukg&dl=1",
        "sha256:caa78779b27e9a0d1e1b71ccd81c0d15403053ae551d5d4013d58b7a2a4680d1",
    ),
    (
        f"muse_sg_response_284_FeXV284.163_{abundance}_effarea.nc",
        "https://www.dropbox.com/scl/fi/7urj1qr4csm92apusgik6/muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.nc?rlkey=qer5d0hw94rpvk7l4bemignmv&st=kcue7mqi&dl=1",
        "sha256:753b9bd0472d102431dc0ab9637ff9e2626098ac093a276634d0ad4025336839",
    ),
]
response_files = [
    Path(pooch.retrieve(url=url, known_hash=known_hash, fname=fname, path=tutorial_cache)).name
    for fname, url, known_hash in response_artifacts
]

response = load_and_concat_responses(
    response_directory=tutorial_cache,
    response_files=response_files,
    channels=[108, 171, 284],
    logT=vdem_raster.logT,
    vdop=vdem_raster.vdop,
    slit=vdem_raster.slit,
    logT_method="nearest",
    vdop_method="nearest",
    # Keep the responses dask-backed so the synthesis below stays lazy and the
    # spectrum streams to disk instead of being materialized in memory at once.
    chunked=True,
)

print(response)

##############################################################################
# Now we can perform the synthesis using :func:`muse.synthesis.vdem_synthesis`
# It computes synthetic spectra by convolving the VDEM with the response
# functions. The operation consists of:
#
# 1. Multiplies VDEM by response function at each (logT, vdop) point
# 2. Sums over the specified dimensions (typically logT and vdop)
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
    sum_over=("logT", "vdop"),
)

output = output_dir / "muse_synthetic_spectra.nc"
encoding = {"flux": {"zlib": True, "complevel": 5}}
spectrum.to_netcdf(output, engine="h5netcdf", encoding=encoding)

print(spectrum)
print(f"Saved {output}")
