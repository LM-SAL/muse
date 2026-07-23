"""
=============================
04 - Create MUSE SG responses
=============================

This tutorial demonstrates how to create the main-line responses for
the 108, 171, and 284 Angstrom bands and map them onto all
35 MUSE spectrograph slits.

We will use the default per-channel effective areas
(``DEFAULTS_MUSE.main_line_effective_area``) and CHIANTI line lists from the
:ref:`previous step <sphx_glr_generated_gallery_synthesis_tutorial_skip_03_prepare_chianti_line_lists.py>`.

To see how the response changes with other parameters — an electron-density
grid instead of a fixed pressure, or nonthermal broadening from unresolved
motions such as Alfven-wave turbulence — see
:ref:`sphx_glr_generated_gallery_synthesis_tutorial_07_density_and_nonthermal_responses.py`.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
import xarray as xr

import astropy.units as u

from muse.instrument import create_spectral_response, map_response_to_sg_detector, save_response
from muse.log import change_logging_level
from muse.variables import DEFAULTS_MUSE

# muse logs at DEBUG level by default; raise it to INFO to reduce the noise.
change_logging_level("INFO")

##############################################################################
# We will fetch the line lists saved from the
# :ref:`previous step <sphx_glr_generated_gallery_synthesis_tutorial_skip_03_prepare_chianti_line_lists.py>`.
# If you generate them locally, you can update the code to point to those instead.

abundance = "sun_coronal_2021_chianti"
cache_dir = Path(pooch.os_cache("muse")) / "chianti_line_lists"
output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)
bands = {
    108: {
        "url": (
            "https://www.dropbox.com/scl/fi/dn0s3dvqe6d9misrlo1e8/"
            "muse_chianti_line_list_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti.nc"
            "?rlkey=egzb1ciuapro22qc2jbqwzqgx&st=jnezcgkr&dl=1"
        ),
        "hash": "sha256:aad92894196920d236fe515e89dd06e4aad1b8dea7b34cce6cff2b43e098ae7d",
        "fname": f"muse_chianti_line_list_108_FeXIX108.355_FeXXI108.117_{abundance}.nc",
        "main_lines": ["Fe XIX 108.355", "Fe XXI 108.117"],
        "output_label": "FeXIX108.355_FeXXI108.117",
    },
    171: {
        "url": (
            "https://www.dropbox.com/scl/fi/blch1e23rj34prqik3mcl/"
            "muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti.nc"
            "?rlkey=a6x99xdh2cl1ri649ojbkrly8&st=cu3vz09h&dl=1"
        ),
        "hash": "sha256:170993ad7bd2ebfc843561b9d0d46b2fffe04025617bc63eff39e55d69446cbf",
        "fname": f"muse_chianti_line_list_171_FeIX171.073_{abundance}.nc",
        "main_lines": ["Fe IX 171.073"],
        "output_label": "FeIX171.073",
    },
    284: {
        "url": (
            "https://www.dropbox.com/scl/fi/riiexsfu9u7shiowo7hn8/"
            "muse_chianti_line_list_284_FeXV284.163_sun_coronal_2021_chianti.nc"
            "?rlkey=wf3wxlh392znzhwu5m5gqccsb&st=kwcxfrwx&dl=1"
        ),
        "hash": "sha256:a86280eed528550186c446d1fb566c4cc35705a064afe665164ee77582a24517",
        "fname": f"muse_chianti_line_list_284_FeXV284.163_{abundance}.nc",
        "main_lines": ["Fe XV 284.163"],
        "output_label": "FeXV284.163",
    },
}

##############################################################################
# Now we can create and save the SG response functions.
#
# Response functions describe how the instrument responds to different wavelengths
# , temperatures, and velocities. For MUSE, these include:
#
# - **Spectral lines**: Temperature-dependent emission line profiles
# - **Slit characteristics**: Spatial response of each slit position
# - **Spectral broadening**: Instrumental and thermal line broadening
# - **Wavelength calibration**: Mapping from detector pixels to wavelengths
#
# :func:`muse.instrument.create_spectral_response` produces an response in
# ``1e-27 erg cm5 / (Angstrom s sr)``.
#
# :func:`muse.instrument.map_response_to_sg_detector` then
# converts energy to photons, applies the detector-pixel solid angle, and
# integrates over each pixel's wavelength width. The mapped response is
# therefore in ``1e-27 cm5 ph / s`` rather than per Angstrom.

for band, config in bands.items():
    line_list_file = Path(
        pooch.retrieve(
            url=config["url"],
            known_hash=config["hash"],
            fname=config["fname"],
            path=cache_dir,
        )
    )
    line_list = xr.load_dataset(line_list_file, engine="h5netcdf")
    line_list = line_list.assign(wavelength=line_list.wavelength.assign_attrs(units=str(u.AA)))
    spectral_order = DEFAULTS_MUSE.channel_spectral_order.sel(channel=band).item()
    lower = band - 35 / spectral_order
    upper = band + 35 / spectral_order
    wavelength_grid = np.arange(lower, upper + 0.0049, 0.0049) * u.AA
    waveband_response = create_spectral_response(
        line_list,
        wavelength_grid,
        main_lines=config["main_lines"],
        instrumental_width=u.Quantity(DEFAULTS_MUSE.instrumental_width_sg.sel(channel=band).data),
        doppler_velocity=np.arange(-1000, 1010, 10) * u.km / u.s,
        effective_area=DEFAULTS_MUSE.main_line_effective_area.sel(channel=band),
    )
    # Chunk over Doppler velocity so the detector mapping stays lazy (dask):
    # save_response then streams it to disk chunk by chunk instead of
    # materializing the full detector response (~18 GB for the 108 band) in
    # memory. Peak memory is roughly one chunk's interpolation temporaries per
    # dask worker; if you run out of RAM, cap dask.config.set(num_workers=...).
    waveband_response = waveband_response.chunk({"doppler_velocity": 20})
    response = map_response_to_sg_detector(waveband_response, band)

    print(response)

    if band == 171:
        integrated_response = response.detector_response.sum(dim=["logT", "slit"]).sel(vdop=0).squeeze()
        integrated_response.plot(x="detector_x_pixel")
        plt.title("Integrated 171 Angstrom response at zero Doppler velocity")

    output = output_dir / f"muse_sg_response_{band}_{config['output_label']}_{abundance}_effarea.nc"
    # save_response refuses to overwrite, so clear the artifact of a previous run.
    output.unlink(missing_ok=True)
    save_response(response, output)
    print(f"Saved {output}")

    # This is to reduce the peak memory in the example, you might not need this
    del line_list, waveband_response, response

plt.show()
