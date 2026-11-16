"""
===========================
Create an AIA 94 Å response
===========================

Create the response of the SDO/AIA 94 Å channel from a CHIANTI line list and
the effective area, pixel size and camera of the channel from `aiapy`.

An imager integrates over its band, so the product is a temperature response.
As for the MUSE context imager, :func:`muse.instrument.create_spectral_response`
folds the lines through the effective area into wavelength bins,
:func:`muse.instrument.transform_response_units` converts them to DN per
detector pixel, and :func:`muse.instrument.map_response_to_ci_detector`
integrates them over the band.

.. warning::

    Requires the optional ``aiapy`` package: ``pip install "muse[aia]"``.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import (
    create_aia_effective_area,
    create_spectral_response,
    map_response_to_ci_detector,
    transform_response_units,
)

##############################################################################
# :func:`muse.instrument.create_aia_effective_area` gives the effective area in
# cm2, with the gain, pair energy and pixel size that
# :func:`muse.instrument.transform_response_units` needs for DN in its
# attributes. Without the EVE normalization, it is the baseline calibration with
# no time-dependent correction.

channel = 94 * u.AA
effective_area = create_aia_effective_area(channel, include_eve_correction=False)
detector = {
    name: u.Quantity(effective_area.attrs[name]) for name in ("pixel_width", "pixel_height", "gain", "pair_energy")
}
print(detector)

plt.figure()
effective_area.effective_area.sel(wavelength=slice(84, 106)).plot()
plt.yscale("log")
plt.title("AIA 94 Å effective area")

##############################################################################
# Iron lines dominate the 94 Å channel (Fe XVIII in flares and hot active
# regions, Fe VIII-Fe X in cooler plasma), so the line list of the
# :ref:`CHIANTI preparation example
# <sphx_glr_generated_gallery_other_instruments_skip_prepare_chianti_line_lists.py>`
# keeps only iron.

line_list_file = fetch_example_data("aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf")
print(line_list)

##############################################################################
# For the lines of every sufficiently abundant element, generate the line list
# with CHIANTI (not run here):
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

##############################################################################
# ``main_lines=None`` with ``include_contaminants=True`` folds every line into
# one component, as an imager sees them. The default ``bin_method="average"``
# integrates each line over its bin: a cool iron line, about 0.004 Å wide, is
# five times narrower than the bins, and sampled at the bin centers the cool
# lines would be miscounted, losing about half their signal on this grid. The
# band integral does not depend on the Doppler velocity, so one is enough.

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
# Built ion by ion, the response shows two peaks of different origin. The hot
# peak, near logT 6.9, is almost all Fe XVIII 93.932 Å; the cool peak, near
# logT 6.0, is thousands of weaker Fe VIII-Fe XII lines, which for an imager
# are the cool response, not contaminants. Lines only: a full calculation also
# needs the free-free, free-bound and two-photon continua.

ions = np.unique(line_list.spectroscopic_name.values)
by_ion = {
    str(ion): imager_response(line_list.isel(transition=line_list.spectroscopic_name.values == ion)) for ion in ions
}
temperature_response = sum(response for _, response in by_ion.values())
units = temperature_response.attrs["units"] = by_ion[str(ions[0])][1].attrs["units"]

plt.figure()
temperature_response.plot()
plt.yscale("log")
plt.ylim(temperature_response.max().item() / 1e4, None)
plt.ylabel(f"Line-only response [{units}]")
plt.title("AIA 94 Å temperature response (Fe lines only)")

# Name the ions above 2% of the peak, in order of peak temperature.
named = [ion for ion, (_, response) in by_ion.items() if response.max() > temperature_response.max() / 50]
named.sort(key=lambda ion: float(by_ion[ion][1].idxmax(dim="logT")))
plt.figure()
temperature_response.plot(color="k", lw=2, label="all lines")
for ion in named:
    by_ion[ion][1].plot(label=ion)
sum(response for ion, (_, response) in by_ion.items() if ion not in named).plot(
    ls="--", color="gray", label="other ions"
)
plt.yscale("log")
plt.ylim(temperature_response.max().item() / 1e4, None)
plt.ylabel(f"Line-only response [{units}]")
plt.legend(fontsize=8)
plt.title("AIA 94 Å temperature response by ion")

##############################################################################
# Before the band integral, the response is a spectrum in DN per Angstrom, which
# a synthesis can sample. At the hot peak it is Fe XVIII 93.932 Å above a forest
# of weaker lines.

spectrum = sum(waveband.spectral_response for waveband, _ in by_ion.values())
plt.figure()
spectrum.isel(line=0, doppler_velocity=0, pressure=0).sel(logT=6.9, method="nearest").plot(x="wavelength_grid")
plt.yscale("log")
plt.ylabel(f"Response [{by_ion[str(ions[0])][0].spectral_response.attrs['units']}]")
plt.title("AIA 94 Å response at logT = 6.9")

plt.show()
