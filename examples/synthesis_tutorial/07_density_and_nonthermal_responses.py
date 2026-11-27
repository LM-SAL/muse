"""
===============================================
07 - Density and non-thermal velocity responses
===============================================

Here we will explore variants of the 171 Å SG response using
a CHIANTI line list with an electron-density grid instead of a
fixed pressure, and with non-thermal broadening (e.g., from
unresolved Alfvén-wave turbulence).
"""

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from muse.data import open_example_dataset
from muse.instrument import create_spectral_response, map_response_to_sg_detector, transform_response_units
from muse.variables import DEFAULTS_MUSE

##############################################################################
# We fetch two 171 Å line lists made as in
# :ref:`tutorial 03 <sphx_glr_generated_gallery_synthesis_tutorial_03_prepare_chianti_line_lists_skip.py>`:
# one at a fixed pressure, and one with ``density`` instead of ``pressure``:
#
# .. code-block:: python
#
#     temperature = np.arange(4.5, 8.0, 0.1) * u.dex(u.K)
#     density = 10 ** np.arange(7.5, 12.5, 0.5) * u.cm**-3
#     line_list = create_chianti_line_list(
#         temperature=temperature,
#         density=density,
#         abundance="sun_coronal_2021_chianti",
#         wavelength_range=wavelength_range,
#         ion_list=["fe_9"],
#     )
#
# The density-grid list, and everything derived from it, carries a ``logD``
# dimension whose coordinate is ``log10(density)``.

line_lists = {
    variant: open_example_dataset(f"muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti{suffix}.zarr.zip")
    for variant, suffix in (("pressure", ""), ("density", "_density"))
}

##############################################################################
# Both variants share the wavelength grid, instrumental width and effective area.

band = 171
main_lines = ["Fe IX 171.073"]
spectral_order = DEFAULTS_MUSE.channel_spectral_order_SG.sel(channel=band).item()
lower = band - 35 / spectral_order
upper = band + 35 / spectral_order
wavelength_grid = np.arange(lower, upper + 0.0049, 0.0049) * u.AA
instrumental_fwhm = DEFAULTS_MUSE.instrumental_fwhm_SG.sel(channel=band).data
effective_area = DEFAULTS_MUSE.main_line_effective_area_SG.sel(channel=band)
# One Doppler velocity, the only one plotted, saves time and memory.
doppler_velocity = [0] * u.km / u.s

##############################################################################
# **Density-dependent response**

waveband_response = create_spectral_response(
    line_lists["density"],
    wavelength_grid,
    main_lines=main_lines,
    instrumental_fwhm=instrumental_fwhm,
    doppler_velocity=doppler_velocity,
    effective_area=effective_area,
)
waveband_response = transform_response_units(waveband_response, "1e-27 cm5 ph / (Angstrom s)", channel=band * u.AA)
response = map_response_to_sg_detector(waveband_response, channel=band * u.AA)
print(response)

fig, ax = plt.subplots()
response.detector_response.sum(dim=["logT", "slit"]).sel(doppler_velocity=0).squeeze().plot(ax=ax)
ax.set_title("171 Å response with electron density")

##############################################################################
# **Non-thermal broadening**
#
# On the fixed-pressure line list, ``nonthermal_velocity`` adds an axis.
# It is the 1/e (most probable) velocity ξ, which broadens a line at wavelength λ
# by a Gaussian standard deviation of λξ / (c√2), added in quadrature to the
# thermal and instrumental widths.
# The goal is to mimic the effect of unresolved motions such as Alfvén-wave
# turbulence in the Alfvén Wave Solar Model (AWSoM; e.g. :cite:t:`shi_2022`).

waveband_response = create_spectral_response(
    line_lists["pressure"],
    wavelength_grid,
    main_lines=main_lines,
    instrumental_fwhm=instrumental_fwhm,
    doppler_velocity=doppler_velocity,
    nonthermal_velocity=np.arange(0, 100, 20) * u.km / u.s,
    effective_area=effective_area,
)
waveband_response = transform_response_units(waveband_response, "1e-27 cm5 ph / (Angstrom s)", channel=band * u.AA)
response = map_response_to_sg_detector(waveband_response, channel=band * u.AA)
print(response)

fig, ax = plt.subplots()
response.detector_response.sum(dim=["logT", "slit"]).sel(doppler_velocity=0).squeeze().plot(ax=ax)
ax.set_title("171 Å response with non-thermal velocity")

plt.show()
