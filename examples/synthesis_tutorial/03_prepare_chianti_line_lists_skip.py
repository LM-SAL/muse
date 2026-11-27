"""
===============================
03 - Prepare CHIANTI line lists
===============================

Compute the CHIANTI line lists that the MUSE responses need.

.. warning::

    Requires the optional CHIANTI dependencies, ``pip install "muse[chianti]"``,
    and a local CHIANTI database in setup in the environment variable ``XUVTOP``.
"""

import os

import numpy as np

import astropy.units as u

from muse.data import save_example_dataset
from muse.instrument import create_chianti_line_list
from muse.variables import DEFAULTS_MUSE

##############################################################################
# First we will do some setup for the CHIANTI line list generation.
#
# We will create one line list for each MUSE band.

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_DATABASE` first."
    raise OSError(msg)

abundance = "sun_coronal_2021_chianti"
temperature = np.arange(4.5, 8.0, 0.1) * u.dex(u.K)
pressure = [3e15] * u.K / u.cm**3
bands = {
    108: {"ions": ["fe_19", "fe_21"], "label": "FeXIX108.355_FeXXI108.117"},
    171: {"ions": ["fe_9"], "label": "FeIX171.073"},
    284: {"ions": ["fe_15"], "label": "FeXV284.163"},
}

##############################################################################
# The line lists are expensive, so we save them as zipped Zarr stores for reuse.

for band, config in bands.items():
    spectral_order = DEFAULTS_MUSE.channel_spectral_order_SG.sel(channel=band).item()
    wavelength_range = [band - 35 / spectral_order, band + 35 / spectral_order] * u.AA
    line_list = create_chianti_line_list(
        temperature=temperature,
        pressure=pressure,
        abundance=abundance,
        wavelength_range=wavelength_range,
        ion_list=config["ions"],
    )
    output = save_example_dataset(line_list, f"muse_chianti_line_list_{band}_{config['label']}_{abundance}.zarr.zip")

    print(line_list)
    print(f"Line list ready: {output.resolve()}")
