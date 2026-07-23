"""
=================================
Synthesize an EIS Fe XII spectrum
=================================

This tutorial builds on the
:ref:`EIS response example
<sphx_glr_generated_gallery_other_instruments_skip_eis_fe_xii_response.py>`
and synthesizes an EIS Fe XII 195 Å raster from the same VDEM used in
the :ref:`MUSE synthesis tutorial
<sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`,
using :func:`muse.synthesis.vdem_synthesis`.

It requires a local CHIANTI database configured with ``XUVTOP``

.. code-block::

    export XUVTOP=/path/to/CHIANTI_11.0.2_database
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.instrument import create_chianti_line_list, create_spectral_response
from muse.synthesis import calculate_moments, vdem_synthesis, wavelength_to_doppler

##############################################################################
# We first confirm that the local environment is working.

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

##############################################################################
# We fetch the VDEM used by the MUSE synthesis tutorial. Its ``logT`` and
# ``vdop`` grids will define the response axes below, so no interpolation is
# needed before the synthesis.

extract_path = Path(pooch.os_cache("muse")) / "muse_example_vdem"
pooch.retrieve(
    "https://www.dropbox.com/scl/fi/xb2f6pvs4cn1yg54n0pdg/muse_example_vdem.zarr.tar.gz?rlkey=u5y19c5lydrw9kur9bzahkvsv&st=t5vltlk8&dl=1",
    known_hash="ab6c8a3fe4f30de6906f75165f19ccc8730040527f6b9b0cccbdd9a09c28a71c",
    fname="muse_example_vdem.zarr.tar.gz",
    path=extract_path.parent,
    processor=pooch.Untar(extract_dir=extract_path.name),
)
vdem = xr.open_zarr(extract_path / "muse_example_vdem.zarr")
# We need to keep the tutorial spectrum manageable.
# Remove this selection so you can have the full-resolution y axis.
vdem = vdem.isel(y=slice(None, None, 8))
print(vdem)

##############################################################################
# We recreate the EIS Fe XII response from the
# :ref:`response example
# <sphx_glr_generated_gallery_other_instruments_skip_eis_fe_xii_response.py>`,
# this time on the VDEM's temperature and velocity grids.

abundance = "sun_coronal_2021_chianti"
temperature = xr.DataArray(10**vdem.logT.data * u.K, dims="logT")
pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")

line_list = create_chianti_line_list(
    temperature=temperature,
    pressure=pressure,
    abundance=abundance,
    wavelength_range=[194.5, 195.7] * u.AA,
    ion_list=["fe_12"],
)

dispersion = 0.0223 * u.AA
instrumental_fwhm = 0.056 * u.AA
response = create_spectral_response(
    line_list,
    np.arange(194.5, 195.7, dispersion.to_value(u.AA)) * u.AA,
    main_lines=["Fe XII 195.119", "Fe XII 195.179"],
    instrumental_width=instrumental_fwhm / (2 * np.sqrt(2 * np.log(2))),
    doppler_velocity=vdem.vdop.data * u.km / u.s,
    effective_area=xr.DataArray(1 * u.cm**2),
)

##############################################################################
# :func:`muse.synthesis.vdem_synthesis` contracts the dimensions the VDEM and
# response share by name. It accepts the wavelength-space names produced by
# :func:`muse.instrument.create_spectral_response` directly (renaming them to
# the detector-style names internally), and drops the MUSE ``slit`` dimension
# from the default ``sum_over`` since neither input here has one.

spectrum = vdem_synthesis(vdem, response).compute()
print(spectrum)

##############################################################################
# The synthesized spectrum along one column of the raster shows both Fe XII
# lines.

plt.figure(figsize=(10, 5))
spectrum_image = spectrum.flux.isel(x=128, pressure=0).sum(dim="line")
# The line wings decay to numerically zero, so anchor the log scale to the
# brightest pixel and show four decades below it.
spectrum_image.plot(
    x="detector_wavelength",
    y="y",
    norm=colors.LogNorm(vmin=spectrum_image.max().item() / 1e4, vmax=spectrum_image.max().item()),
    cmap="inferno",
)
plt.title("Synthesized EIS Fe XII spectrum at one raster position")

##############################################################################
# Finally we compute the spectral moments of the main line:
# total intensity, Doppler shift, and line width.
# :func:`muse.synthesis.wavelength_to_doppler` adds the velocity coordinate
# required by :func:`muse.synthesis.calculate_moments`.

velocity_spectrum = wavelength_to_doppler(spectrum)
moments = calculate_moments(velocity_spectrum, moment_dim="wavelength_bin")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
line_moments = moments.sel(line="Fe XII 195.119").isel(pressure=0)
intensity = line_moments["0th"]
# Anchor the log scale to the data: median to max spans the quiet
# background and the flare core without washing either out.
intensity.plot(ax=axes[0], norm=colors.LogNorm(vmin=intensity.quantile(0.5).item(), vmax=intensity.max().item()))
axes[0].set_title("0th moment (intensity)")
line_moments["1st"].plot(ax=axes[1], cmap="RdBu_r", robust=True)
axes[1].set_title("1st moment (Doppler shift)")
line_moments["2nd"].plot(ax=axes[2], robust=True)
axes[2].set_title("2nd moment (line width)")
plt.tight_layout()

plt.show()
