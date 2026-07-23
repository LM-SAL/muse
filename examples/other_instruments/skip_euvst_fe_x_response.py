"""
=============================
Create an EUVST Fe X response
=============================

This tutorial demonstrates how to create a density-dependent CHIANTI line
list and a wavelength-space response for the Solar-C/EUVST short-wavelength
band around the Fe X 174.531 and 175.263 Å pair, whose ratio is a
well-known electron-density diagnostic.

Unlike the :ref:`EIS example
<sphx_glr_generated_gallery_other_instruments_skip_eis_fe_xii_response.py>`,
which computes the line list at a fixed electron pressure, here we compute it
on an electron-density grid, so the response carries a ``logD`` dimension.

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
# We compute the Fe X line list on an electron-density grid by passing
# ``density`` instead of ``pressure``. The output then carries a ``logD``
# dimension whose coordinate is ``log10(density)``, and that dimension flows
# through the spectral response unchanged.

abundance = "sun_coronal_2021_chianti"
temperature = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
density = xr.DataArray(10 ** np.arange(7.5, 12.5, 0.5) * u.cm**-3, dims="logD")
wavelength_range = [174.0, 175.6] * u.AA

line_list = create_chianti_line_list(
    temperature=temperature,
    density=density,
    abundance=abundance,
    wavelength_range=wavelength_range,
    ion_list=["fe_10"],
)
print(line_list)

##############################################################################
# Now the instrument. EUVST is still pre-flight, so the numbers below are
# representative design values: a spectral-pixel size of about 17 mÅ in
# the short-wavelength band. We leave the instrumental width at zero and use a
# unity effective area; substitute the real values once the calibrated curves
# are available (as a one-dimensional ``xarray.DataArray`` with a unit-bearing
# ``wavelength`` coordinate).

dispersion = 0.017 * u.AA
wavelength_grid = np.arange(174.0, 175.6, dispersion.to_value(u.AA)) * u.AA
doppler_velocity = np.arange(-500, 510, 10) * u.km / u.s
effective_area_unity = xr.DataArray(1 * u.cm**2)

response = create_spectral_response(
    line_list,
    wavelength_grid,
    main_lines=["Fe X 174.531", "Fe X 175.263"],
    doppler_velocity=doppler_velocity,
    effective_area=effective_area_unity,
)
print(response)

##############################################################################
# The spectrum, summed over temperature, changes shape with electron density:
# Fe X 175.263 grows relative to Fe X 174.531.

profile = response.spectral_response.sel(doppler_velocity=0).sum(dim=["logT", "line"])
plt.figure()
for logD in [8, 10, 12]:
    curve = profile.sel(logD=logD, method="nearest")
    plt.plot(
        response.wavelength_grid,
        curve / curve.max(),
        label=f"logD = {curve.logD.values:.1f}",
    )
plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized response")
plt.title("EUVST Fe X 174/175 response, summed over logT")
plt.legend()

##############################################################################
# Integrating each line over wavelength and taking their ratio gives the
# density diagnostic directly, here shown at the temperature where Fe X
# formation peaks.

line_total = response.spectral_response.sel(doppler_velocity=0).sum(dim="wavelength_bin")
peak_logT = line_total.sel(line="Fe X 174.531").mean(dim="logD").idxmax(dim="logT")
ratio = line_total.sel(line="Fe X 175.263", logT=peak_logT) / line_total.sel(line="Fe X 174.531", logT=peak_logT)
plt.figure()
ratio.plot(marker="o")
plt.ylabel("Fe X 175.263 / 174.531")
plt.title(f"Density diagnostic at logT = {peak_logT.values:.1f}")

plt.show()
