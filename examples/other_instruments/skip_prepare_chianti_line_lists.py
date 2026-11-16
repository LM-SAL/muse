"""
==========================
Prepare CHIANTI line lists
==========================

Compute the CHIANTI line lists of the response examples in this section.

.. warning::

    Requires the optional CHIANTI dependencies, ``pip install "muse[chianti]"``,
    and a local CHIANTI database in ``XUVTOP``.
"""

import os
from pathlib import Path

import numpy as np

import astropy.units as u

from muse.instrument import create_chianti_line_list

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

abundance = "sun_coronal_2021_chianti"
temperature = np.arange(4.5, 8.0, 0.1) * u.dex(u.K)
pressure = [3e15] * u.K / u.cm**3
output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)

##############################################################################
# AIA 94 Å: the iron lines across the band.

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
# Hinode/EIS: the Fe XII 195.119 and 195.179 Å window.

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
# Hinode/EIS: the density-sensitive Fe X 174.531/175.263 Å pair.

density = 10 ** np.arange(7.5, 12.5, 0.5) * u.cm**-3
line_list = create_chianti_line_list(
    temperature=temperature,
    density=density,
    abundance=abundance,
    wavelength_range=[174.0, 175.6] * u.AA,
    ion_list=["fe_10"],
)
output = output_dir / f"eis_chianti_line_list_174_175_FeX_{abundance}_density.nc"
encoding = {name: {"zlib": True, "complevel": 5} for name in line_list.data_vars}
line_list.to_netcdf(output, engine="h5netcdf", encoding=encoding)
print(f"EIS density line list ready: {output.resolve()}")
