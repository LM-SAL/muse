"""
=============================================
08 - Create and synthesize a MUSE CI response
=============================================

Create a main-line response for the 195 Angstrom channel of the MUSE context
imager (CI) and synthesize a CI image with it. CI is an imager, so
:func:`muse.instrument.map_response_to_ci_detector` integrates the
wavelength-space response over the band.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import (
    align_response_and_vdem,
    create_spectral_response,
    map_response_to_ci_detector,
    transform_response_units,
)
from muse.synthesis import vdem_synthesis
from muse.variables import DEFAULTS_MUSE

vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))

##############################################################################
# A line list describes the plasma, not the instrument, so the Fe XII 195
# Angstrom list of the EIS example serves here too.

line_list_file = fetch_example_data("eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf")

##############################################################################
# The wavelength-space response, with the scalar CI effective area.

channel = 195 * u.AA
waveband_response = create_spectral_response(
    line_list,
    np.arange(194.5, 195.7, 0.01) * u.AA,
    main_lines=["Fe XII 195.119", "Fe XII 195.179"],
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
    effective_area=DEFAULTS_MUSE.main_line_effective_area_CI.sel(channel=channel.to_value(u.AA)),
)

##############################################################################
# Convert to DN per CI pixel, from the CI pixel size, gain and pair-creation
# energy, then integrate over wavelength.

waveband_response = transform_response_units(
    waveband_response,
    "1e-27 cm5 DN / (Angstrom s)",
    channel=channel,
    detector="ci",
)
response = map_response_to_ci_detector(waveband_response, channel=channel)
print(response)

plt.figure()
response.detector_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="line").plot()
plt.title("MUSE CI 195 main-line temperature response")

##############################################################################
# Once :func:`muse.instrument.align_response_and_vdem` puts it on the
# temperature and velocity grids of the VDEM, the response synthesizes like any other.

response, vdem = align_response_and_vdem(response, vdem)
spectrum = vdem_synthesis(vdem, response)
image = spectrum.flux.isel(pressure=0).sum(dim="line", keep_attrs=True).compute()
print(spectrum)

plt.figure()
image.plot(
    x="x",
    y="y",
    norm=colors.LogNorm(vmin=image.quantile(0.5).item(), vmax=image.max().item()),
    cmap="inferno",
)
plt.title("Synthesized MUSE CI 195 image")

plt.show()
