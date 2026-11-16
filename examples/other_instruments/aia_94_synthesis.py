"""
============================
Synthesize an AIA 94 Å image
============================

Synthesize an AIA 94 Å image from the VDEM of the
:ref:`MUSE synthesis tutorial
<sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`
with the response of the
:ref:`AIA response example
<sphx_glr_generated_gallery_other_instruments_aia_94_response.py>`.
The five strongest iron lines stay separate so their images can be compared;
other lines and the continuum are left out.

.. warning::

    Requires the optional ``aiapy`` package: ``pip install "muse[aia]"``.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

import astropy.units as u

import sunpy.visualization.colormaps  # NOQA: F401 -- registers the SDO colormaps with matplotlib

from muse.data import fetch_example_data
from muse.instrument import (
    align_response_and_vdem,
    create_aia_effective_area,
    create_spectral_response,
    map_response_to_ci_detector,
    rank_line_list,
    transform_response_units,
)
from muse.synthesis import vdem_synthesis
from muse.transforms import match_fov

output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))

print(vdem)

##############################################################################
# We recreate the response on the ``doppler_velocity`` grid of the VDEM and the
# ``logT`` grid of the line list, where the response example used the rest
# wavelength on a wide display grid.

# As in the response example: the baseline calibration, with no time-dependent correction.
channel = 94 * u.AA
effective_area = create_aia_effective_area(channel, include_eve_correction=False)
detector = {
    name: u.Quantity(effective_area.attrs[name]) for name in ("pixel_width", "pixel_height", "gain", "pair_energy")
}

line_list_file = fetch_example_data("aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf")

ranked = rank_line_list(line_list, effective_area)
main_lines = list(dict.fromkeys(str(name) for name in ranked.full_name.values))[:5]
print(f"Strongest contributors: {main_lines}")

##############################################################################
# The response in DN per AIA pixel. The default ``bin_method`` integrates each
# line over its bin: iron lines are narrower than the 0.05 Å bins, and sampled
# at the bin centers they would make the image several times too faint.

response = create_spectral_response(
    line_list,
    np.arange(91.0, 97.0, 0.05) * u.AA,
    main_lines=main_lines,
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
    effective_area=effective_area,
)
response = transform_response_units(response, "1e-27 cm5 DN / (Angstrom s)", channel=channel, detector="CI", **detector)

##############################################################################
# An imager needs no spectra: integrating over the band leaves a response per
# (line, logT, doppler_velocity) point.

aia_response = map_response_to_ci_detector(response, channel=channel)

##############################################################################
# Resample the VDEM onto the 0.6 arcsec AIA pixels; ``extent="keep"`` leaves
# the x extent alone, as the AIA field of view is not the MUSE raster.
# :func:`muse.instrument.align_response_and_vdem` then puts both on the same
# temperature and velocity grids, since :func:`muse.synthesis.vdem_synthesis`
# does not interpolate.

vdem_aia = match_fov(vdem, pixel_width=detector["pixel_width"], pixel_height=detector["pixel_height"], extent="keep")
aia_response, vdem_aia = align_response_and_vdem(aia_response, vdem_aia)

##############################################################################
# Summing over temperature and velocity leaves one count-rate image per line.

spectrum = vdem_synthesis(vdem_aia, aia_response, sum_over=("logT", "doppler_velocity"))
print(spectrum)

per_line = spectrum.flux.isel(pressure=0).compute()
image = per_line.sum(dim="line", keep_attrs=True)
plt.figure()
# Median to max spans the background arcade and the flare core.
image.plot(
    x="x",
    y="y",
    norm=colors.LogNorm(vmin=image.quantile(0.5).item(), vmax=image.max().item()),
    cmap="sdoaia94",
)
plt.title("Synthesized AIA 94 Å image (five Fe lines)")

output_dir.mkdir(parents=True, exist_ok=True)
output = output_dir / "aia_94_synthetic_image.nc"
image.to_dataset(name="flux").assign_attrs(spectrum.attrs).to_netcdf(output)
print(f"Saved {output}")

##############################################################################
# The per-line images separate the hot and cool components: Fe XVIII shows the
# hottest plasma, Fe X and Fe VIII the cooler background.

nrows = -(-len(main_lines) // 2)
fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows))
for ax, line in zip(axes.flat, per_line.line.values, strict=False):
    data = per_line.sel(line=line)
    norm = colors.LogNorm(vmin=data.quantile(0.5).item(), vmax=data.max().item())
    data.plot(x="x", y="y", ax=ax, norm=norm, cmap="sdoaia94", add_colorbar=False)
    ax.set_title(str(line))
for ax in axes.flat[len(main_lines) :]:
    ax.set_visible(False)
plt.tight_layout()

plt.show()
