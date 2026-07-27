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

An imager integrates over its whole bandpass, so after synthesizing the
Doppler-resolved spectra of the strongest contributors we integrate over
wavelength and sum over line to form the image. As in the response example,
this includes only the five strongest iron lines (no other lines or continuum).

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
from muse.instrument import create_spectral_response
from muse.synthesis import vdem_synthesis

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
# We need to keep the tutorial spectrum manageable.
# Remove this selection so you can have the full-resolution y axis.
vdem = vdem.isel(y=slice(None, None, 8))
print(vdem)

##############################################################################
# We recreate the AIA 94 Å response from the
# :ref:`response example
# <sphx_glr_generated_gallery_other_instruments_aia_94_response.py>`
# on the VDEM's temperature and velocity grids: the `aiapy` wavelength
# response and plate scale, a precomputed iron-only line list, and the five
# strongest radiometrically weighted contributors.

channel = Channel(94 * u.angstrom)
# With no ``obstime``, this uses the baseline calibration without a
# time-dependent degradation correction.
photon_energy = (const.h * const.c / channel.wavelength / u.ph).to(u.erg / u.ph)
radiometric_conversion_quantity = (channel.wavelength_response() * channel.plate_scale / photon_energy).to(
    u.cm**2 * u.DN * u.sr / (u.erg * u.pix)
)
radiometric_conversion = xr.DataArray(
    radiometric_conversion_quantity.value,
    dims="wavelength",
    coords={"wavelength": ("wavelength", channel.wavelength.to_value(u.AA), {"units": str(u.AA)})},
    attrs={"units": str(radiometric_conversion_quantity.unit)},
).sel(wavelength=slice(84, 106))

line_list_file = fetch_example_data("aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf").sel(logT=vdem.logT, method="nearest", tolerance=0.05)
line_list = line_list.assign_coords(logT=vdem.logT)
line_list = line_list.assign(wavelength=line_list.wavelength.assign_attrs(units=str(u.AA)))

conversion_at_lines = radiometric_conversion.interp(wavelength=line_list.wavelength).fillna(0.0).drop_vars("wavelength")
peak_weight = (line_list.gofnt.isel(pressure=0) * conversion_at_lines).max(dim="logT")
ranked = line_list.full_name.values[np.argsort(-peak_weight.values)]
main_lines = list(dict.fromkeys(str(name) for name in ranked))[:5]
print(f"Strongest contributors: {main_lines}")

response = create_spectral_response(
    line_list,
    np.arange(91.0, 97.0, 0.05) * u.AA,
    main_lines=main_lines,
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
)
conversion_on_grid = (
    radiometric_conversion.interp(wavelength=response.wavelength_grid).fillna(0.0).drop_vars("wavelength")
)
response_unit = u.Unit(response.spectral_response.attrs["units"]) * radiometric_conversion_quantity.unit
scaled_response = (response.spectral_response * conversion_on_grid).assign_attrs(
    {**response.spectral_response.attrs, "units": str(response_unit)}
)
response = response.assign(spectral_response=scaled_response)

##############################################################################
# As in the :ref:`EIS synthesis example
# <sphx_glr_generated_gallery_other_instruments_eis_fe_xii_synthesis.py>`,
# the wavelength-space response feeds straight into
# :func:`muse.synthesis.vdem_synthesis`.

spectrum = vdem_synthesis(vdem, response)
print(spectrum)

##############################################################################
# Integrating over the physical wavelength coordinate and summing over line
# gives the band-integrated count rate. ``integrate`` includes the wavelength
# bin width; a plain sum would make the result depend on the chosen grid step.

per_line = spectrum.flux.isel(pressure=0).integrate("detector_wavelength")
per_line.attrs["units"] = str(u.Unit(spectrum.flux.attrs["units"]) * u.AA)
per_line = per_line.compute()
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
