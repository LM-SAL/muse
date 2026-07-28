"""
===============================
03 - Prepare CHIANTI line lists
===============================

This tutorial demonstrates how to compute the CHIANTI line lists needed to
create a MUSE response.

It requires a local CHIANTI database configured with ``XUVTOP``

.. code-block::

    export XUVTOP=/path/to/CHIANTI_11.0.2_database
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument import create_chianti_line_list
from muse.variables import DEFAULTS_MUSE

##############################################################################
# We first confirm that the local environment is working.

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

##############################################################################
# Now we need to configure the line-list calculations.
# For MUSE the options are:
#
# - Ion: fe_9, fe_15, fe_19 or fe_21
# - Wavelength range of the location of the main line.
#   You want to make sure the wavelength range is small to select a
#   single line and this would be:
#
#   - For fe_9: [171.0,171.2]
#   - For fe_15: [284.1, 284.2]
#   - For fe_19: [108.3, 108.4]
#   - For fe_21: [108.1, 108.2]
#
# - The temperature axis in K (min, max and step),
# - LOS velocity in km/s (min, max and step)
# - Abundances, e.g., "sun_coronal_2021_chianti"
# - Pressure or density (see below).
#
# For a worked example of a line list computed on an electron-density grid,
# see :ref:`sphx_glr_generated_gallery_synthesis_tutorial_07_density_and_nonthermal_responses.py`.

abundance = "sun_coronal_2021_chianti"
logT = np.arange(4.5, 8.0, 0.1)
temperature = xr.DataArray(10**logT * u.K, dims="logT")
pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
# You can use density too and pass this instead into create_chianti_line_list
# eDensity = xr.DataArray(10 ** np.arange(7.5, 12.5, 0.5), dims="logD")
line_list_directory = Path(
    os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts")
)
line_list_directory.mkdir(parents=True, exist_ok=True)
bands = {
    108: {"ions": ["fe_19", "fe_21"], "label": "FeXIX108.355_FeXXI108.117"},
    171: {"ions": ["fe_9"], "label": "FeIX171.073"},
    284: {"ions": ["fe_15"], "label": "FeXV284.163"},
}

##############################################################################
# Finally we can compute and save the line lists.
#
# The generated datasets record their CHIANTI versions and physical coordinates
# and we use netCDF to save out the expensive results so they can be reused.
#
# Note that if you want to speed up the calculation, please make sure to install
# ``muse[chianti]`` such that the dependencies required for this function are installed.

for band, config in bands.items():
    spectral_order = DEFAULTS_MUSE.channel_spectral_order.sel(channel=band).item()
    wavelength_range = [band - 35 / spectral_order, band + 35 / spectral_order] * u.AA
    line_list = create_chianti_line_list(
        temperature=temperature,
        pressure=pressure,
        abundance=abundance,
        wavelength_range=wavelength_range,
        ion_list=config["ions"],
    )
    output = line_list_directory / f"muse_chianti_line_list_{band}_{config['label']}_{abundance}.nc"
    encoding = {name: {"zlib": True, "complevel": 5} for name in line_list.data_vars}
    line_list.to_netcdf(output, engine="h5netcdf", encoding=encoding)

    print(line_list)
    print(f"Line list ready: {output.resolve()}")
