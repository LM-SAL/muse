"""
=================================
Synthesize an EUVST Fe X spectrum
=================================

This tutorial builds on the
:ref:`EUVST response example
<sphx_glr_generated_gallery_other_instruments_euvst_fe_x_response.py>`
and synthesizes an EUVST Fe X 174/175 Å raster from the same VDEM used
in the :ref:`MUSE synthesis tutorial
<sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`,
using :func:`muse.synthesis.vdem_synthesis`.

The VDEM carries no density axis, so here we select a single electron density
from the grid used in the response example; a density-dependent synthesis
needs a VDEM with a matching ``logD`` dimension.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import create_spectral_response
from muse.synthesis import vdem_synthesis

##############################################################################
# We fetch the VDEM used by the MUSE synthesis tutorial. Its ``logT`` and
# ``doppler_velocity`` grids will define the response axes below, so no
# interpolation is needed before the synthesis.

vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))
# We need to keep the tutorial spectrum manageable.
# Remove this selection so you can have the full-resolution y axis.
vdem = vdem.isel(y=slice(None, None, 8))
print(vdem)

##############################################################################
# We recreate the EUVST Fe X response.
#
# :func:`muse.synthesis.vdem_synthesis` pairs the VDEM and response grid
# point by grid point without interpolating, so the response must be
# evaluated on the VDEM's exact ``logT`` and ``doppler_velocity`` grids,
# while the response example used wider display grids to show the full
# response shape.

line_list_file = fetch_example_data("euvst_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf").sel(
    logT=vdem.logT,
    logD=[9.0],
    method="nearest",
    tolerance=0.05,
)
line_list = line_list.assign_coords(logT=vdem.logT, logD=[9.0])
line_list = line_list.assign(wavelength=line_list.wavelength.assign_attrs(units=str(u.AA)))

dispersion = 0.017 * u.AA
response = create_spectral_response(
    line_list,
    np.arange(174.0, 175.6, dispersion.to_value(u.AA)) * u.AA,
    main_lines=["Fe X 174.531", "Fe X 175.263"],
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
    # Placeholder unity area; substitute the calibrated EUVST curve when available.
    effective_area=xr.DataArray(1 * u.cm**2),
)
# Convert the per-Å response to one value per EUVST spectral detector pixel.
response_unit = u.Unit(response.spectral_response.attrs["units"]) * u.AA
binned_response = (response.spectral_response * dispersion.to_value(u.AA)).assign_attrs(
    {**response.spectral_response.attrs, "units": str(response_unit)}
)
response = response.assign(spectral_response=binned_response)

##############################################################################
# As in the :ref:`EIS synthesis example
# <sphx_glr_generated_gallery_other_instruments_eis_fe_xii_synthesis.py>`,
# the wavelength-space response feeds straight into
# :func:`muse.synthesis.vdem_synthesis`.

spectrum = vdem_synthesis(vdem, response).compute()
print(spectrum)

##############################################################################
# The synthesized spectrum along one column of the raster shows both Fe X
# lines.

plt.figure(figsize=(10, 5))
spectrum_image = spectrum.flux.isel(x=128, logD=0).sum(dim="line")
# The line wings decay to numerically zero, so anchor the log scale to the
# brightest pixel and show four decades below it.
spectrum_image.plot(
    x="detector_wavelength",
    y="y",
    norm=colors.LogNorm(vmin=spectrum_image.max().item() / 1e4, vmax=spectrum_image.max().item()),
    cmap="inferno",
)
plt.title("Synthesized EUVST Fe X spectrum at one raster position")

##############################################################################
# Finally, the intensity maps of the density-sensitive line pair: the
# per-pixel spectrum summed over the spectral axis.

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# Anchor each log scale to its own data: median to max spans the quiet
# background and the flare core without washing either out.
for ax, line in zip(axes, ["Fe X 174.531", "Fe X 175.263"], strict=True):
    intensity = spectrum.flux.sel(line=line).isel(logD=0).sum(dim="wavelength_bin", keep_attrs=True)
    intensity.plot(ax=ax, norm=colors.LogNorm(vmin=intensity.quantile(0.5).item(), vmax=intensity.max().item()))
    ax.set_title(f"{line} intensity")
plt.tight_layout()

plt.show()
