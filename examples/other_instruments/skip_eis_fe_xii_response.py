"""
=============================
Create an EIS Fe XII response
=============================

This tutorial demonstrates how to create a CHIANTI line list and a
wavelength-space response for the Hinode/EIS Fe XII 195.119 Å window.

:func:`muse.instrument.create_spectral_response` is instrument-neutral:
everything MUSE-specific lives in the detector mapping
(:func:`muse.instrument.map_response_to_sg_detector`), which we do not use here.
Instead we sample the response on the EIS spectral-pixel grid directly.

It requires a local CHIANTI database configured with ``XUVTOP``

.. code-block::

    export XUVTOP=/path/to/CHIANTI_11.0.2_database
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument import create_chianti_line_list, create_spectral_response

##############################################################################
# We first confirm that the local environment is working.

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

##############################################################################
# EIS observes two EUV bands (SW: 166-212, LW: 245-291 Å) and
# Fe XII 195.119 Å is the strongest line in the SW band.
# We select a narrow window that also contains the density-sensitive
# Fe XII 195.179 Å blend.
#
# For a worked example of a line list computed on an electron-density grid
# (which you would want to actually exploit that blend), see
# :ref:`sphx_glr_generated_gallery_other_instruments_skip_euvst_fe_x_response.py`.

abundance = "sun_coronal_2021_chianti"
temperature = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
wavelength_range = [194.5, 195.7] * u.AA

line_list = create_chianti_line_list(
    temperature=temperature,
    pressure=pressure,
    abundance=abundance,
    wavelength_range=wavelength_range,
    ion_list=["fe_12"],
)
print(line_list)

##############################################################################
# Now we describe the instrument. The numbers below are representative EIS
# values: a spectral-pixel size of 22.3 mÅ and an instrumental width
# of about 56 mÅ FWHM (see the EIS instrument paper,
# Culhane et al. 2007, and EIS Software Note #7 for the calibrated,
# CCD-position-dependent values).
#
# ``instrumental_width`` is a Gaussian sigma, so we convert from FWHM.
#
# We use a unity effective area here; for quantitative work, pass the
# calibrated EIS effective-area curve (available via SolarSoft or
# `eispac <https://eispac.readthedocs.io/en/stable/>`__)
# as a one-dimensional ``xarray.DataArray`` with a unit-bearing ``wavelength``
# coordinate instead.

dispersion = 0.0223 * u.AA
instrumental_fwhm = 0.056 * u.AA
instrumental_width = instrumental_fwhm / (2 * np.sqrt(2 * np.log(2)))
wavelength_grid = np.arange(194.5, 195.7, dispersion.to_value(u.AA)) * u.AA
doppler_velocity = np.arange(-300, 310, 10) * u.km / u.s
effective_area_unity = xr.DataArray(1 * u.cm**2)

response = create_spectral_response(
    line_list,
    wavelength_grid,
    main_lines=["Fe XII 195.119", "Fe XII 195.179"],
    instrumental_width=instrumental_width,
    doppler_velocity=doppler_velocity,
    effective_area=effective_area_unity,
)
print(response)

##############################################################################
# The line profile on the EIS pixel grid, summed over temperature, shifts
# with the Doppler-velocity axis.

profile = response.spectral_response.isel(pressure=0).sum(dim=["logT", "line"])
plt.figure()
for velocity in [-200, 0, 200]:
    plt.plot(
        response.wavelength_grid,
        profile.sel(doppler_velocity=velocity, method="nearest"),
        label=f"{velocity} km/s",
    )
plt.xlabel("Wavelength [Å]")
plt.ylabel(f"Response [{response.spectral_response.attrs['units']}]")
plt.title("EIS Fe XII 195 response, summed over logT")
plt.legend()

##############################################################################
# Integrating over wavelength instead gives the temperature sensitivity of
# each retained line.

temperature_sensitivity = response.spectral_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="wavelength_bin")
plt.figure()
for line in temperature_sensitivity.line.values:
    temperature_sensitivity.sel(line=line).plot(label=str(line))
plt.title("EIS Fe XII temperature sensitivity")
plt.legend()

plt.show()
