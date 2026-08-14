"""
=============================================
08 - Create and synthesize a MUSE CI response
=============================================

This tutorial creates a main-line response for the MUSE context imager (CI)
195 Angstrom channel and uses it directly to synthesize a CI image.

CI is an imager, so
:func:`muse.instrument.map_response_to_ci_detector` integrates the
wavelength-space response over the band.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import create_spectral_response, map_response_to_ci_detector, transform_response_units
from muse.synthesis import vdem_synthesis
from muse.variables import DEFAULTS_MUSE

##############################################################################
# Fetch the VDEM used throughout the synthesis tutorial.

vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))

##############################################################################
# Fetch a CHIANTI Fe XII line list around 195 Angstrom. CHIANTI emissivities
# describe the plasma rather than a particular instrument, so the same
# precomputed line list used by the EIS example is also valid here. The
# instrument-specific calibration enters below.

line_list_file = fetch_example_data("eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf").sel(logT=vdem.logT, method="nearest", tolerance=0.05)
line_list = line_list.assign_coords(logT=vdem.logT)
line_list = line_list.assign(wavelength=line_list.wavelength.assign_attrs(units=str(u.AA)))

##############################################################################
# Build the wavelength-space response using a scalar CI effective area.

channel = 195 * u.AA
waveband_response = create_spectral_response(
    line_list,
    np.arange(194.5, 195.7, 0.01) * u.AA,
    main_lines=["Fe XII 195.119", "Fe XII 195.179"],
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
    effective_area=DEFAULTS_MUSE.main_line_effective_area_CI.sel(channel=channel.to_value(u.AA)),
)

##############################################################################
# Convert radiance to DN per CI pixel using the CI-specific pixel geometry,
# CCD gain, and pair-creation energy, then integrate over wavelength.

waveband_response = transform_response_units(
    waveband_response,
    "1e-27 cm5 DN / (Angstrom s)",
    channel,
    detector="ci",
)
response = map_response_to_ci_detector(waveband_response, channel)
print(response)

plt.figure()
response.detector_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="line").plot()
plt.title("MUSE CI 195 main-line temperature response")

##############################################################################
# The band-integrated response feeds directly into the same synthesis
# function as the SG and other-instrument responses.

spectrum = vdem_synthesis(vdem, response)
image = spectrum.flux.isel(pressure=0).sum(dim="line", keep_attrs=True).compute()
print(spectrum)

plt.figure()
image.plot(
    norm=colors.LogNorm(vmin=image.quantile(0.5).item(), vmax=image.max().item()),
    cmap="inferno",
)
plt.title("Synthesized MUSE CI 195 image")

plt.show()
