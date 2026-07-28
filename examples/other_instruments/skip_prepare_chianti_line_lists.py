"""
==========================
Prepare CHIANTI line lists
==========================

This example computes the CHIANTI line lists used by the response examples in
this section. It requires a local CHIANTI database configured with ``XUVTOP``:

.. code-block::

    export XUVTOP=/path/to/CHIANTI_11.0.2_database
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument import create_chianti_line_list

##############################################################################
# We first confirm that the local environment is working.

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

##############################################################################
# Now we need to configure the line-list calculations.

abundance = "sun_coronal_2021_chianti"
temperature = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)

##############################################################################
# AIA 94 Angstrom: retain iron lines across the channel bandpass.

line_list = create_chianti_line_list(
    temperature=temperature,
    pressure=pressure,
    abundance=abundance,
    wavelength_range=[85, 105] * u.AA,
    element_list=["fe"],
)
output = output_dir / f"aia_chianti_line_list_94_Fe_{abundance}.nc"
encoding = {name: {"zlib": True, "complevel": 5} for name in line_list.data_vars}
line_list.to_netcdf(output, engine="h5netcdf", encoding=encoding)
print(f"AIA line list ready: {output.resolve()}")

##############################################################################
# Hinode/EIS: the Fe XII 195.119 and 195.179 Angstrom window.

line_list = create_chianti_line_list(
    temperature=temperature,
    pressure=pressure,
    abundance=abundance,
    wavelength_range=[194.5, 195.7] * u.AA,
    ion_list=["fe_12"],
)
output = output_dir / f"eis_chianti_line_list_195_FeXII_{abundance}.nc"
encoding = {name: {"zlib": True, "complevel": 5} for name in line_list.data_vars}
line_list.to_netcdf(output, engine="h5netcdf", encoding=encoding)
print(f"EIS line list ready: {output.resolve()}")

##############################################################################
# Solar-C/EUVST: the density-sensitive Fe X 174.531/175.263 Angstrom pair.

density = xr.DataArray(10 ** np.arange(7.5, 12.5, 0.5) * u.cm**-3, dims="logD")
line_list = create_chianti_line_list(
    temperature=temperature,
    density=density,
    abundance=abundance,
    wavelength_range=[174.0, 175.6] * u.AA,
    ion_list=["fe_10"],
)
output = output_dir / f"euvst_chianti_line_list_174_175_FeX_{abundance}_density.nc"
encoding = {name: {"zlib": True, "complevel": 5} for name in line_list.data_vars}
line_list.to_netcdf(output, engine="h5netcdf", encoding=encoding)
print(f"EUVST line list ready: {output.resolve()}")
