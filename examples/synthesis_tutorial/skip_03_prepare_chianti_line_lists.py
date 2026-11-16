"""
===============================
03 - Prepare CHIANTI line lists
===============================

Compute the CHIANTI line lists that the MUSE responses need.

.. warning::

    Requires the optional CHIANTI dependencies, ``pip install "muse[chianti]"``,
    which also speed up the calculation, and a local CHIANTI database in ``XUVTOP``.
"""

import os
from pathlib import Path

import numpy as np

import astropy.units as u

from muse.instrument import create_chianti_line_list
from muse.variables import DEFAULTS_MUSE

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

##############################################################################
# One line list per MUSE band, over the wavelength window of the band, with the
# ions of its main lines. Pass ``density`` instead of ``pressure`` for an
# electron-density grid, as in
# :ref:`sphx_glr_generated_gallery_synthesis_tutorial_07_density_and_nonthermal_responses.py`.

abundance = "sun_coronal_2021_chianti"
temperature = np.arange(4.5, 8.0, 0.1) * u.dex(u.K)
pressure = [3e15] * u.K / u.cm**3
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
# The line lists are expensive, so we save them to netCDF for reuse; each
# records its CHIANTI version.

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
    output = line_list_directory / f"muse_chianti_line_list_{band}_{config['label']}_{abundance}.nc"
    encoding = {name: {"zlib": True, "complevel": 5} for name in line_list.data_vars}
    line_list.to_netcdf(output, engine="h5netcdf", encoding=encoding)

    print(line_list)
    print(f"Line list ready: {output.resolve()}")
