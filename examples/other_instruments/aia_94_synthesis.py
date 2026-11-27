"""
============================
Synthesize an AIA 94 Å image
============================

Here we will synthesize an AIA 94 Å image from the VDEM of the
:ref:`MUSE synthesis tutorial
<sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`
with the response of the
:ref:`AIA response example
<sphx_glr_generated_gallery_other_instruments_aia_94_response.py>`.

.. warning::

    Requires the optional ``aiapy`` package: ``pip install "muse[aia]"``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import astropy.units as u

import sunpy.visualization.colormaps  # NOQA: F401 - registers the SDO colormaps with matplotlib

from muse.data import open_example_dataset, save_example_dataset
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

##############################################################################
# First item is to load the VDEM (either from the MUSE synthesis tutorial or your own).

vdem = open_example_dataset("muse_example_vdem.zarr.zip", chunked=True)

print(vdem)

##############################################################################
# We will recreate the response on the ``doppler_velocity`` grid of the VDEM
# and the ``logT`` grid of the line list, where the response example used the rest
# wavelength on a wide display grid.

channel = 94 * u.AA
effective_area = create_aia_effective_area(channel)
detector = {
    name: u.Quantity(effective_area.attrs[name]) for name in ("pixel_width", "pixel_height", "gain", "pair_energy")
}

line_list = open_example_dataset("aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.zarr.zip")

ranked = rank_line_list(line_list, effective_area)
main_lines = list(dict.fromkeys(str(name) for name in ranked.full_name.values))[:5]
print(f"Strongest contributors: {main_lines}")

response = create_spectral_response(
    line_list,
    np.arange(91.0, 97.0, 0.05) * u.AA,
    main_lines=main_lines,
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
    effective_area=effective_area,
)
response = transform_response_units(response, "1e-27 cm5 DN / (Angstrom s)", channel=channel, detector="CI", **detector)

##############################################################################
# Now we will integrate over the band to leave a response per
# ``(line, logT, doppler_velocity)`` point.

aia_response = map_response_to_ci_detector(response, channel=channel)

##############################################################################
# Now we can resample the VDEM onto the 0.6 arcsec AIA pixels; ``extent="keep"``
# leaves the x extent alone, as the AIA field of view is not the MUSE raster.
# :func:`muse.instrument.align_response_and_vdem` then puts both on the same
# temperature and velocity grids, since :func:`muse.synthesis.vdem_synthesis`
# does not interpolate.

vdem_aia = match_fov(vdem, pixel_width=detector["pixel_width"], pixel_height=detector["pixel_height"], extent="keep")
aia_response, vdem_aia = align_response_and_vdem(aia_response, vdem_aia)

##############################################################################
# By summing over temperature and velocity, it leaves one image for the entire
# response.

spectrum = vdem_synthesis(vdem_aia, aia_response, sum_over=("logT", "doppler_velocity"))
print(spectrum)

per_line = spectrum.flux.isel(pressure=0).compute()
image = per_line.sum(dim="line", keep_attrs=True)
fig, ax = plt.subplots()
# Median to max spans the background arcade and the flare core.
image.plot(
    x="x",
    y="y",
    ax=ax,
    norm=colors.LogNorm(vmin=image.quantile(0.5).item(), vmax=image.max().item()),
    cmap="sdoaia94",
)
ax.set_title("Synthesized AIA 94 Å image (five Fe lines)")

output = save_example_dataset(
    image.to_dataset(name="flux").assign_attrs(spectrum.attrs), "aia_94_synthetic_image.zarr.zip"
)
print(f"Saved {output}")

##############################################################################
# One then can separate the hot and cool components:

nrows = -(-len(main_lines) // 2)
fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows), layout="constrained")
for ax, line in zip(axes.flat, per_line.line.values, strict=False):
    data = per_line.sel(line=line)
    norm = colors.LogNorm(vmin=data.quantile(0.5).item(), vmax=data.max().item())
    data.plot(x="x", y="y", ax=ax, norm=norm, cmap="sdoaia94", add_colorbar=False)
    ax.set_title(str(line) + " Å")
for ax in axes.flat[len(main_lines) :]:
    ax.set_visible(False)

plt.show()
