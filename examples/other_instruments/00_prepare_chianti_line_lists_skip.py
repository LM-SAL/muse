"""
==========================
Prepare CHIANTI line lists
==========================

Here we will compute the CHIANTI line lists of needed for the response
examples in this section.

.. warning::

    Requires the optional CHIANTI dependencies, ``pip install "muse[chianti]"``,
    and a local CHIANTI database in ``XUVTOP``.
"""

import os

import numpy as np

import astropy.units as u

from muse.data import save_example_dataset
from muse.instrument import create_chianti_line_list

##############################################################################
# First we will do some setup for the CHIANTI line list generation.

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

abundance = "sun_coronal_2021_chianti"
temperature = np.arange(4.5, 8.0, 0.1) * u.dex(u.K)
pressure = [3e15] * u.K / u.cm**3

##############################################################################
# SDO/AIA 94 Å line list.

line_list = create_chianti_line_list(
    temperature=temperature,
    pressure=pressure,
    abundance=abundance,
    wavelength_range=[85, 105] * u.AA,
    element_list=["fe"],
)
output = save_example_dataset(line_list, f"aia_chianti_line_list_94_Fe_{abundance}.zarr.zip")
print(f"AIA line list ready: {output.resolve()}")

##############################################################################
# Hinode/EIS narrow band line list.

line_list = create_chianti_line_list(
    temperature=temperature,
    pressure=pressure,
    abundance=abundance,
    wavelength_range=[194.5, 195.7] * u.AA,
    ion_list=["fe_12"],
)
output = save_example_dataset(line_list, f"eis_chianti_line_list_195_FeXII_{abundance}.zarr.zip")
print(f"EIS line list ready: {output.resolve()}")

##############################################################################
# Hinode/EIS density-sensitive Fe X 174.531/175.263 Å line list.

density = 10 ** np.arange(7.5, 12.5, 0.5) * u.cm**-3
line_list = create_chianti_line_list(
    temperature=temperature,
    density=density,
    abundance=abundance,
    wavelength_range=[174.0, 175.6] * u.AA,
    ion_list=["fe_10"],
)
output = save_example_dataset(line_list, f"eis_chianti_line_list_174_175_FeX_{abundance}_density.zarr.zip")
print(f"EIS density line list ready: {output.resolve()}")
