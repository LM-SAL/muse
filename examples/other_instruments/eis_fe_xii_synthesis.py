"""
=================================
Synthesize an EIS Fe XII spectrum
=================================

Here we will synthesize an EIS Fe XII 195 Å raster from the VDEM of the
:ref:`MUSE synthesis tutorial <sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.data import fetch_example_data, open_example_dataset
from muse.instrument import create_spectral_response
from muse.synthesis import calculate_moments, vdem_synthesis, wavelength_to_doppler

##############################################################################
# First item is to load the VDEM (either from the MUSE synthesis tutorial or your own).

vdem = open_example_dataset("muse_example_vdem.zarr.zip", chunked=True)

print(vdem)

##############################################################################
# :func:`muse.synthesis.vdem_synthesis` does not interpolate or align grids,
# so we will recreate the response on the ``logT`` and ``doppler_velocity`` grids
# of the VDEM.

line_list = open_example_dataset("eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.zarr.zip").sel(
    logT=vdem.logT, method="nearest", tolerance=0.05
)
line_list = line_list.assign_coords(logT=vdem.logT)

dispersion = 0.0223 * u.AA
instrumental_fwhm = 0.056 * u.AA
# Pre-flight SW-band effective area (no in-flight sensitivity decay).
ea_table = np.loadtxt(fetch_example_data("EIS_EffArea_B.005"))
effective_area = xr.DataArray(
    ea_table[:, 1],
    dims="wavelength",
    coords={"wavelength": ("wavelength", ea_table[:, 0], {"units": str(u.AA)})},
    attrs={"units": str(u.cm**2)},
)
response = create_spectral_response(
    line_list,
    np.arange(194.5, 195.7, dispersion.to_value(u.AA)) * u.AA,
    main_lines=["Fe XII 195.119", "Fe XII 195.179"],
    instrumental_fwhm=instrumental_fwhm,
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
    effective_area=effective_area,
)
# Per Å times the dispersion gives one value per EIS spectral pixel.
response_unit = u.Unit(response.spectral_response.attrs["units"]) * u.AA
binned_response = (response.spectral_response * dispersion.to_value(u.AA)).assign_attrs(
    {**response.spectral_response.attrs, "units": str(response_unit)}
)
response = response.assign(spectral_response=binned_response)

##############################################################################
# Now we will call :func:`muse.synthesis.vdem_synthesis`
# to produce the synthesized spectrum.

spectrum = vdem_synthesis(vdem, response).compute()
print(spectrum)

##############################################################################
# The synthesized spectrum along one column of the raster shows both Fe XII
# lines.

fig, ax = plt.subplots(figsize=(10, 5))
spectrum_image = spectrum.flux.isel(x=128, pressure=0).sum(dim="line")
spectrum_image.plot(
    x="detector_wavelength",
    y="y",
    ax=ax,
    norm=colors.LogNorm(vmin=spectrum_image.max().item() / 1e4, vmax=spectrum_image.max().item()),
    cmap="inferno",
)
ax.set_title("Synthesized EIS Fe XII spectrum at one raster position")

##############################################################################
# Finally, we will plot the spectral moments of the main line.
#
# :func:`muse.synthesis.wavelength_to_doppler` adds the velocity coordinate
# that :func:`muse.synthesis.calculate_moments` needs.

velocity_spectrum = wavelength_to_doppler(spectrum)
moments = calculate_moments(velocity_spectrum, moment_dim="wavelength_bin")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
line_moments = moments.sel(line="Fe XII 195.119").isel(pressure=0)
intensity = line_moments["0th"]
intensity.plot(
    x="x",
    y="y",
    ax=axes[0],
    # Median to max spans the quiet background and the flare core.
    norm=colors.LogNorm(vmin=intensity.quantile(0.5).item(), vmax=intensity.max().item()),
)
axes[0].set_title("0th moment (intensity)")
line_moments["1st"].plot(x="x", y="y", ax=axes[1], cmap="RdBu_r", robust=True)
axes[1].set_title("1st moment (Doppler shift)")
line_moments["2nd"].plot(x="x", y="y", ax=axes[2], robust=True)
axes[2].set_title("2nd moment (line width)")

plt.show()
