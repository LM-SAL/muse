"""
=============================
04 - Create MUSE SG responses
=============================

Create the main-line responses of the 108, 171 and 284 Angstrom bands from the
CHIANTI line lists of the
:ref:`previous step <sphx_glr_generated_gallery_synthesis_tutorial_skip_03_prepare_chianti_line_lists.py>`,
and map them onto the 35 MUSE spectrograph slits.
:func:`muse.instrument.generate_sg_responses` runs these steps for every channel.

.. note::

    The default effective areas (``DEFAULTS_MUSE.main_line_effective_area_SG``)
    hold at the wavelength of each main line and apply across the band, so
    contaminant counts away from the main line are approximate.

:ref:`Tutorial 07 <sphx_glr_generated_gallery_synthesis_tutorial_07_density_and_nonthermal_responses.py>`
builds the response on an electron-density grid and with nonthermal broadening.
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import (
    create_spectral_response,
    map_response_to_sg_detector,
    save_response,
    transform_response_units,
)
from muse.log import change_logging_level
from muse.variables import DEFAULTS_MUSE

# muse logs at DEBUG by default; INFO is quieter.
change_logging_level("INFO")

##############################################################################
# We fetch the published line lists of the previous step; point the code at
# yours if you generated them.

abundance = "sun_coronal_2021_chianti"
output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)
bands = {
    108: {
        "main_lines": ["Fe XIX 108.355", "Fe XXI 108.117"],
        "output_label": "FeXIX108.355_FeXXI108.117",
    },
    171: {
        "main_lines": ["Fe IX 171.073"],
        "output_label": "FeIX171.073",
    },
    284: {
        "main_lines": ["Fe XV 284.163"],
        "output_label": "FeXV284.163",
    },
}

##############################################################################
# Each response takes three steps:
#
# - :func:`muse.instrument.create_spectral_response` gives it in
#   ``1e-27 erg cm5 / (Angstrom s sr)``.
# - :func:`muse.instrument.transform_response_units` converts energy to photons
#   and per steradian to per detector pixel: ``1e-27 cm5 ph / (Angstrom s)``.
#   Ask for ``DN`` instead of ``ph`` for data numbers.
# - :func:`muse.instrument.map_response_to_sg_detector` samples it at each
#   detector pixel center and multiplies by the wavelength width of a pixel:
#   ``1e-27 cm5 ph / s``.

for band, config in bands.items():
    line_list_file = fetch_example_data(f"muse_chianti_line_list_{band}_{config['output_label']}_{abundance}.nc")
    line_list = xr.load_dataset(line_list_file, engine="h5netcdf")
    # Count the samples: np.arange with a float step can gain or lose the last one to round-off.
    spectral_order = DEFAULTS_MUSE.channel_spectral_order_SG.sel(channel=band).item()
    half_width = (DEFAULTS_MUSE.waveband_half_width_SG / spectral_order).to_value(u.AA)
    step = DEFAULTS_MUSE.wavelength_step_SG.to_value(u.AA)
    wavelength_grid = (band - half_width + step * np.arange(round(2 * half_width / step) + 1)) * u.AA
    waveband_response = create_spectral_response(
        line_list,
        wavelength_grid,
        main_lines=config["main_lines"],
        instrumental_fwhm=DEFAULTS_MUSE.instrumental_fwhm_SG.sel(channel=band).data,
        doppler_velocity=np.arange(-1000, 1010, 10) * u.km / u.s,
        effective_area=DEFAULTS_MUSE.main_line_effective_area_SG.sel(channel=band),
    )
    # Chunk over Doppler velocity so the detector mapping stays lazy
    waveband_response = waveband_response.chunk({"doppler_velocity": 10})
    waveband_response = transform_response_units(waveband_response, "1e-27 cm5 ph / (Angstrom s)", channel=band * u.AA)
    waveband_response = waveband_response.assign(
        spectral_response=waveband_response.spectral_response.astype(np.float32)
    )
    response = map_response_to_sg_detector(waveband_response, channel=band * u.AA)

    print(response)

    if band == 171:
        integrated_response = response.detector_response.sum(dim=["logT", "slit"]).sel(doppler_velocity=0).squeeze()
        integrated_response.plot(x="detector_x_pixel")
        plt.title("Integrated 171 Angstrom response at zero Doppler velocity")

    response = response.assign(detector_response=response.detector_response.astype(np.float32))
    output = output_dir / f"muse_sg_response_{band}_{config['output_label']}_{abundance}_effarea.zarr"
    # save_response refuses to overwrite, so clear the artifact of a previous run.
    shutil.rmtree(output, ignore_errors=True)
    save_response(response, output)
    print(f"Saved {output}")

    # Lowers the peak memory of the example.
    del line_list, waveband_response, response

plt.show()
