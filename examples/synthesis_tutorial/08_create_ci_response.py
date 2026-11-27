"""
=============================================
08 - Create and synthesize a MUSE CI response
=============================================

Here we will create a main-line response for the 195 Å channel
of the MUSE context imager (CI) and synthesize a CI image with it.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import astropy.units as u

from muse.data import open_example_dataset
from muse.instrument import (
    align_response_and_vdem,
    create_spectral_response,
    map_response_to_ci_detector,
    transform_response_units,
)
from muse.synthesis import vdem_synthesis
from muse.variables import DEFAULTS_MUSE

##############################################################################
# First we will load a VDEM example dataset.

vdem = open_example_dataset("muse_example_vdem.zarr.zip", chunked=True)

##############################################################################
# The same for the line list for the 195 Å channel of the MUSE CI,
# using the one created for EIS.

line_list = open_example_dataset("eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.zarr.zip")

##############################################################################
# Now we can compute the CI response.

channel = 195 * u.AA
waveband_response = create_spectral_response(
    line_list,
    np.arange(194.5, 195.7, 0.01) * u.AA,
    main_lines=["Fe XII 195.119", "Fe XII 195.179"],
    doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
    effective_area=DEFAULTS_MUSE.main_line_effective_area_CI.sel(channel=channel.to_value(u.AA)),
)

##############################################################################
# Next, we need to convert to DN and map the response to the CI detector.

waveband_response = transform_response_units(
    waveband_response,
    "1e-27 cm5 DN / (Angstrom s)",
    channel=channel,
    detector="ci",
)
response = map_response_to_ci_detector(waveband_response, channel=channel)
print(response)

fig, ax = plt.subplots()
response.detector_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="line").plot(ax=ax)
ax.set_title("MUSE CI 195 Å main-line temperature response")

##############################################################################
# Finally, we can synthesize a CI image with the response and the VDEM.

response, vdem = align_response_and_vdem(response, vdem)
spectrum = vdem_synthesis(vdem, response)
image = spectrum.flux.isel(pressure=0).sum(dim="line", keep_attrs=True).compute()
print(spectrum)

fig, ax = plt.subplots()
image.plot(
    x="x",
    y="y",
    ax=ax,
    norm=colors.LogNorm(vmin=image.quantile(0.5).item(), vmax=image.max().item()),
    cmap="inferno",
)
ax.set_title("Synthesized MUSE CI 195 Å image")

plt.show()
