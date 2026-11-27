"""
===========================
Create an AIA 94 Å response
===========================

Here we will create the spectral response of SDO/AIA 94 Å
using a CHIANTI line list and the instrument properties from `aiapy`.

.. warning::

    Requires the optional ``aiapy`` package: ``pip install "muse[aia]"``.
"""

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from muse.data import open_example_dataset
from muse.instrument import (
    create_aia_effective_area,
    create_spectral_response,
    map_response_to_ci_detector,
    transform_response_units,
)

##############################################################################
# :func:`muse.instrument.create_aia_effective_area` returns the effective area
# in the form the rest of the functions expect.

channel = 94 * u.AA
effective_area = create_aia_effective_area(channel)
detector = {
    name: u.Quantity(effective_area.attrs[name]) for name in ("pixel_width", "pixel_height", "gain", "pair_energy")
}
print(detector)

fig, ax = plt.subplots()
effective_area.effective_area.sel(wavelength=slice(84, 106)).plot(ax=ax)
ax.set(yscale="log", title="AIA 94 Å effective area")

##############################################################################
# Iron lines dominate the 94 Å channel (Fe XVIII in flares and hot active
# regions or Fe VIII-Fe X in cooler plasma for example), so the line list of the
# :ref:`CHIANTI preparation example
# <sphx_glr_generated_gallery_other_instruments_00_prepare_chianti_line_lists_skip.py>`
# keeps only the iron lines. Primarily, done to reduce the size of the line list.

line_list = open_example_dataset("aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.zarr.zip")
print(line_list)

##############################################################################
# If you want the lines of every sufficiently abundant element, you can do the
# following:
#
# .. code-block:: python
#
#     temperature = np.arange(4.5, 8.0, 0.1) * u.dex(u.K)
#     pressure = [3e15] * u.K / u.cm**3
#     abundance = "sun_coronal_2021_chianti"
#     line_list = create_chianti_line_list(
#         temperature=temperature,
#         pressure=pressure,
#         abundance=abundance,
#         wavelength_range=[84, 106] * u.AA,
#         minimum_abundance=1e-6,
#     )
#
# Please note, this code block is not ran.

##############################################################################
# Below we define a helper function to create the full response of the AIA 94
# Å channel to a given set of lines.

wavelengths = np.arange(84.0, 106.01, 0.02) * u.AA


def imager_response(lines):
    """
    Return the wavelength-space response to ``lines`` in DN per pixel, and its band
    integral.
    """
    waveband = create_spectral_response(
        lines,
        wavelengths,
        main_lines=None,
        include_contaminants=True,
        effective_area=effective_area,
        doppler_velocity=[0.0] * u.km / u.s,
    )
    waveband = transform_response_units(
        waveband, "1e-27 cm5 DN / (Angstrom s)", channel=channel, detector="CI", **detector
    )
    band = map_response_to_ci_detector(waveband, channel=channel)
    return waveband, band.detector_response.isel(line=0, doppler_velocity=0, pressure=0)


##############################################################################
# We will build it ion by ion.

ions = np.unique(line_list.spectroscopic_name.values)
by_ion = {
    str(ion): imager_response(line_list.isel(transition=line_list.spectroscopic_name.values == ion)) for ion in ions
}
temperature_response = sum(response for _, response in by_ion.values())
units = temperature_response.attrs["units"] = by_ion[str(ions[0])][1].attrs["units"]

##############################################################################
# The outputted response shows two peaks. The hot peak, near logT 6.9, is
# almost all Fe XVIII 93.932  whereas the cool peak, near logT 6.0, is
# the Fe VIII-Fe XII lines, here are the cool plasma response, not contaminants.

# We want to name the ions above 2% of the peak, in order of peak temperature.
named = [ion for ion, (_, response) in by_ion.items() if response.max() > temperature_response.max() / 50]
named.sort(key=lambda ion: float(by_ion[ion][1].idxmax(dim="logT")))
fig, ax = plt.subplots()
temperature_response.plot(ax=ax, color="k", lw=2, label="all lines")
for ion in named:
    by_ion[ion][1].plot(ax=ax, label=ion)
sum(response for ion, (_, response) in by_ion.items() if ion not in named).plot(
    ax=ax, ls="--", color="gray", label="other ions"
)
ax.set(
    yscale="log",
    ylim=(temperature_response.max().item() / 1e4, None),
    ylabel=f"Line-only response [{units}]",
    title="AIA 94 Å temperature response by ion",
)
ax.legend(fontsize=8)

##############################################################################
# Before we integrate over the band, the response is a spectrum in DN per Angstrom.
# At the hot peak is Fe XVIII 93.932 Å above a collection of weaker lines.

spectrum = sum(waveband.spectral_response for waveband, _ in by_ion.values())
fig, ax = plt.subplots()
spectrum.isel(line=0, doppler_velocity=0, pressure=0).sel(logT=6.9, method="nearest").plot(ax=ax, x="wavelength_grid")
ax.set(
    yscale="log",
    ylabel=f"Response [{by_ion[str(ions[0])][0].spectral_response.attrs['units']}]",
    title="AIA 94 Å response at logT = 6.9",
)

plt.show()
