"""
===========================
Create an AIA 94 Å response
===========================

This example demonstrates how to create a broadband imaging response for the
SDO/AIA 94 Å channel by combining a CHIANTI line list with the AIA
effective-area curve from `aiapy`.

Unlike a spectrograph, an imager integrates over its whole bandpass, so the
main product is a temperature response rather than a line profile. We build
that directly from the line list, then use
:func:`muse.instrument.create_spectral_response` to resolve the dominant
contributors in wavelength and Doppler velocity.

It requires a local CHIANTI database configured with ``XUVTOP``

.. code-block::

    export XUVTOP=/path/to/CHIANTI_11.0.2_database

and `aiapy` (``pip install aiapy``) for the effective area.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from aiapy.response import Channel

import astropy.units as u

from muse.instrument import create_chianti_line_list, create_spectral_response

##############################################################################
# We first confirm that the local environment is working.

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

##############################################################################
# `aiapy` provides the AIA wavelength-dependent effective area. We convert
# it into the one-dimensional ``xarray.DataArray`` (with a unit-bearing
# ``wavelength`` coordinate) that ``muse`` expects, keeping only the region
# around the 94 Å bandpass.
#
# `~aiapy.response.Channel.wavelength_response` additionally folds in the detector gain if
# you want a response in DN rather than photons.

channel = Channel(94 * u.angstrom)
area = channel.effective_area
effective_area = xr.DataArray(
    area.to_value(u.cm**2),
    dims="wavelength",
    coords={"wavelength": ("wavelength", channel.wavelength.to_value(u.AA), {"units": str(u.AA)})},
    attrs={"units": str(u.cm**2)},
).sel(wavelength=slice(84, 106))

plt.figure()
effective_area.plot()
plt.yscale("log")
plt.title("AIA 94 Å effective area")

##############################################################################
# The 94 Å channel is dominated by iron lines (Fe XVIII in flares and
# hot active-region plasma, Fe VIII-Fe X at cooler temperatures), so we keep
# the line list fast by restricting it to iron with ``element_list``. For
# completeness you can drop that and use, e.g., ``minimum_abundance=1e-5``
# to sweep all abundant elements.

abundance = "sun_coronal_2021_chianti"
temperature = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
wavelength_range = [85, 105] * u.AA

line_list = create_chianti_line_list(
    temperature=temperature,
    pressure=pressure,
    abundance=abundance,
    wavelength_range=wavelength_range,
    element_list=["fe"],
)
print(line_list)

##############################################################################
# The temperature response of the channel is the sum over every line of its
# contribution function weighted by the effective area at that line's
# wavelength. Note that this is the line contribution only: the continuum
# (free-free, free-bound, two-photon) is not included, so for quantitative
# work use the full spectral models in SolarSoft or `aiapy`.
#
# The result shows the familiar bimodal shape of the 94 Å channel,
# peaking around logT of about 6.0 (Fe X) and about 6.8 (Fe XVIII).

area_at_lines = effective_area.interp(wavelength=line_list.wavelength).fillna(0.0).drop_vars("wavelength")
temperature_response = (line_list.gofnt * area_at_lines).sum(dim="trans_index")
plt.figure()
temperature_response.isel(pressure=0).plot()
plt.yscale("log")
plt.ylim(temperature_response.max().item() / 1e4, None)
plt.ylabel("Line-only response [erg cm5 / s x cm2]")
plt.title("AIA 94 Å temperature response (Fe lines only)")

##############################################################################
# To see which lines drive that shape, we rank the transitions by their
# effective-area-weighted peak contribution and build a Doppler-resolved
# spectral response for the strongest ones. Repeated transitions sharing a
# ``full_name`` are summed by ``create_spectral_response``.

peak_weight = (line_list.gofnt.isel(pressure=0) * area_at_lines).max(dim="logT")
ranked = line_list.full_name.values[np.argsort(-peak_weight.values)]
main_lines = list(dict.fromkeys(str(name) for name in ranked))[:5]
print(f"Strongest contributors: {main_lines}")

response = create_spectral_response(
    line_list,
    np.arange(91.0, 97.0, 0.02) * u.AA,
    main_lines=main_lines,
    doppler_velocity=np.arange(-300, 320, 20) * u.km / u.s,
    effective_area=effective_area,
)
print(response)

##############################################################################
# Finally, the per-line temperature sensitivity within the bandpass.

line_total = response.spectral_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="wavelength_bin")
plt.figure()
for line in line_total.line.values:
    line_total.sel(line=line).plot(label=str(line))
plt.yscale("log")
plt.ylim(line_total.max().item() / 1e4, None)
plt.title("AIA 94 Å per-line temperature sensitivity")
plt.legend()

plt.show()
