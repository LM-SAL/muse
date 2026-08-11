"""
============================
Synthesize an AIA 94 Å image
============================

This tutorial builds on the
:ref:`AIA response example
<sphx_glr_generated_gallery_other_instruments_aia_94_response.py>`
and synthesizes an AIA 94 Å image from the same VDEM used in the
:ref:`MUSE synthesis tutorial
<sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`,
using :func:`muse.synthesis.vdem_synthesis`.

AIA runs through the same pipeline stages as MUSE, with AIA's own
calibration supplied at each step. As in the response example, this includes
only the five strongest iron lines (no other lines or continuum).

It requires `aiapy` (``pip install aiapy``) for the instrument response.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

import astropy.constants as const
import astropy.units as u

import sunpy.visualization.colormaps  # NOQA: F401 -- registers the SDO colormaps with matplotlib

from muse.data import fetch_example_data
from muse.instrument import (
    align_response_and_vdem,
    create_spectral_response,
    map_response_to_ci_detector,
    transform_response_units,
)
from muse.synthesis import vdem_synthesis
from muse.transforms import match_fov

try:
    from aiapy.response import Channel
except ImportError:
    msg = "aiapy is required for this example, install it with `pip install aiapy`"
    raise ImportError(msg) from None

##############################################################################
# We fetch the VDEM used by the MUSE synthesis tutorial. Its ``logT`` and
# ``doppler_velocity`` grids will define the response axes below, so no
# interpolation is needed before the synthesis.

output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))

print(vdem)

##############################################################################
# We recreate the AIA 94 Å response instead of reusing the one from the
# previous example.
#
# :func:`muse.synthesis.vdem_synthesis` pairs the VDEM and response grid
# point by grid point without interpolating, so the response must be
# evaluated on the VDEM's exact ``logT`` and ``doppler_velocity`` grids,
# while the response example used wider display grids to show the full
# response shape.

channel = Channel(94 * u.angstrom)
# With no ``obstime``, this uses the baseline calibration without a
# time-dependent degradation correction.

aia_pixel = 0.6 * u.arcsec  # AIA plate scale per pixel side
pair_energy = 3.65 * u.eV / u.electron  # silicon pair-creation energy
# aiapy packages the camera gain as DN per photon
electrons_per_photon = (const.h * const.c / channel.wavelength / u.ph / pair_energy).to(u.electron / u.ph)
camera_gain = (electrons_per_photon / channel.gain).to(u.electron / u.DN).mean()

effective_area = xr.DataArray(
    channel.effective_area.to_value(u.cm**2),
    dims="wavelength",
    coords={"wavelength": ("wavelength", channel.wavelength.to_value(u.AA), {"units": str(u.AA)})},
    attrs={"units": str(u.cm**2)},
).sel(wavelength=slice(84, 106))

line_list_file = fetch_example_data("aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf").sel(logT=vdem.logT, method="nearest", tolerance=0.05)
line_list = line_list.assign_coords(logT=vdem.logT)
line_list = line_list.assign(wavelength=line_list.wavelength.assign_attrs(units=str(u.AA)))

area_at_lines = effective_area.interp(wavelength=line_list.wavelength).fillna(0.0).drop_vars("wavelength")
peak_weight = (line_list.gofnt.isel(pressure=0) * area_at_lines).max(dim="logT")
ranked = line_list.full_name.values[np.argsort(-peak_weight.values)]
main_lines = list(dict.fromkeys(str(name) for name in ranked))[:5]
print(f"Strongest contributors: {main_lines}")

##############################################################################
# Now we will create the AIA response. In addition, we use
# :func:`muse.instrument.transform_response_units` and AIA's own camera
# gain, silicon pair energy, and plate scale.

response = create_spectral_response(
    line_list,
    np.arange(91.0, 97.0, 0.05) * u.AA,
    main_lines=main_lines,
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
    effective_area=effective_area,
)
response = transform_response_units(
    response,
    "1e-27 cm5 DN / (Angstrom s)",
    gain=camera_gain,
    pair_energy=pair_energy,
    pixel_width=aia_pixel,
    pixel_height=aia_pixel,
)

##############################################################################
# AIA is an imager, so we do not need Doppler-resolved spectra: the
# wavelength-space response is integrated over the band, leaving a
# response per (line, logT, doppler_velocity) grid point.

aia_response = map_response_to_ci_detector(response, 94 * u.AA)

##############################################################################
# The VDEM is resampled onto AIA's 0.6 arcsec plate scale. AIA's field of
# view is not tied to the MUSE raster width, so ``x_extent="keep"`` matches
# the resolution while leaving the x extent alone. Afterwards
# :func:`muse.instrument.align_response_and_vdem` puts the response and the
# VDEM on shared temperature and velocity grids.

vdem_aia = match_fov(vdem, dx_pix=aia_pixel, dy_pix=aia_pixel, x_extent="keep")
aia_response, vdem_aia = align_response_and_vdem(aia_response, vdem_aia)

##############################################################################
# The band-integrated response feeds straight into
# :func:`muse.synthesis.vdem_synthesis`; summing over temperature and
# velocity leaves one band-integrated count-rate image per line.

spectrum = vdem_synthesis(vdem_aia, aia_response, sum_over=("logT", "doppler_velocity"))
print(spectrum)

per_line = spectrum.flux.isel(pressure=0).compute()
image = per_line.sum(dim="line", keep_attrs=True)
plt.figure()
# Anchor the log scale to the data: median to max spans the background
# arcade and the flare core without washing either out.
image.plot(norm=colors.LogNorm(vmin=image.quantile(0.5).item(), vmax=image.max().item()), cmap="sdoaia94")
plt.title("Synthesized AIA 94 Å image (five Fe lines)")

output_dir.mkdir(parents=True, exist_ok=True)
output = output_dir / "aia_94_synthetic_image.nc"
image.to_dataset(name="flux").assign_attrs(spectrum.attrs).to_netcdf(output)
print(f"Saved {output}")

##############################################################################
# The per-line images separate the hot and cool channel components:
# Fe XVIII picks out the hottest plasma while Fe X and Fe VIII show the
# cooler background.

nrows = -(-len(main_lines) // 2)
fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows))
for ax, line in zip(axes.flat, per_line.line.values, strict=False):
    data = per_line.sel(line=line)
    norm = colors.LogNorm(vmin=data.quantile(0.5).item(), vmax=data.max().item())
    data.plot(ax=ax, norm=norm, cmap="sdoaia94", add_colorbar=False)
    ax.set_title(str(line))
for ax in axes.flat[len(main_lines) :]:
    ax.set_visible(False)
plt.tight_layout()

plt.show()
