"""
=======================================================
07 - Density and nonthermal-velocity response variants
=======================================================

This aside shows how the 171 Angstrom SG response changes when the CHIANTI
line list is computed on an electron-density grid instead of at a fixed
pressure, and when nonthermal broadening (e.g. from unresolved Alfven-wave
turbulence) is added to the spectral response.

Both line lists are downloaded precomputed, so no CHIANTI database is needed
to run this example. The density-grid list was generated with the same
workflow as
:ref:`tutorial 03 <sphx_glr_generated_gallery_synthesis_tutorial_skip_03_prepare_chianti_line_lists.py>`,
passing ``density`` instead of ``pressure``:

.. code-block:: python

    temperature = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
    density = xr.DataArray(10 ** np.arange(7.5, 12.5, 0.5) * u.cm**-3, dims="logD")
    line_list = create_chianti_line_list(
        temperature=temperature,
        density=density,
        abundance="sun_coronal_2021_chianti",
        wavelength_range=wavelength_range,
        ion_list=["fe_9"],
    )

The line list, and everything derived from it, then carries a ``logD``
dimension whose coordinate is ``log10(density)``.

Every plot below is at ``vdop=0``, so we pass a single Doppler-velocity
point; this keeps the detector responses roughly 200 times smaller than the
:ref:`tutorial 04 <sphx_glr_generated_gallery_synthesis_tutorial_04_create_sg_responses.py>`
velocity grid and the whole example comfortably in memory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
import xarray as xr

import astropy.units as u

from muse.instrument import create_spectral_response, map_response_to_sg_detector
from muse.log import change_logging_level
from muse.variables import DEFAULTS_MUSE

# muse logs at DEBUG level by default; raise it to INFO to reduce the noise.
change_logging_level("INFO")

##############################################################################
# We fetch the two precomputed CHIANTI line lists for the 171 Angstrom band:
# the fixed-pressure one already used in
# :ref:`tutorial 04 <sphx_glr_generated_gallery_synthesis_tutorial_04_create_sg_responses.py>`,
# and the density-grid one generated with the snippet above.

abundance = "sun_coronal_2021_chianti"
cache_dir = Path(pooch.os_cache("muse")) / "chianti_line_lists"
line_lists = {
    "pressure": {
        "url": (
            "https://www.dropbox.com/scl/fi/blch1e23rj34prqik3mcl/"
            "muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti.nc"
            "?rlkey=a6x99xdh2cl1ri649ojbkrly8&st=cu3vz09h&dl=1"
        ),
        "hash": "sha256:170993ad7bd2ebfc843561b9d0d46b2fffe04025617bc63eff39e55d69446cbf",
        "fname": f"muse_chianti_line_list_171_FeIX171.073_{abundance}.nc",
    },
    "density": {
        "url": (
            "https://www.dropbox.com/scl/fi/s6eiikmyxcmp99cfp3dod/"
            "muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti_density.nc"
            "?rlkey=tgst4vndfy2jhfhw8upja8w7a&st=l23hdoxq&dl=1"
        ),
        "hash": "sha256:47205ccd3cf3c7f315bd9f7c38e78410a92611f2ce4779c39d70f34572232d0d",
        "fname": f"muse_chianti_line_list_171_FeIX171.073_{abundance}_density.nc",
    },
}
for config in line_lists.values():
    file = Path(pooch.retrieve(url=config["url"], known_hash=config["hash"], fname=config["fname"], path=cache_dir))
    line_list = xr.load_dataset(file, engine="h5netcdf")
    config["line_list"] = line_list.assign(wavelength=line_list.wavelength.assign_attrs(units=str(u.AA)))

##############################################################################
# Both variants share the 171 Angstrom band configuration: wavelength grid,
# instrumental width, and effective area.

band = 171
main_lines = ["Fe IX 171.073"]
spectral_order = DEFAULTS_MUSE.channel_spectral_order.sel(channel=band).item()
lower = band - 35 / spectral_order
upper = band + 35 / spectral_order
wavelength_grid = np.arange(lower, upper + 0.0049, 0.0049) * u.AA
instrumental_width = u.Quantity(DEFAULTS_MUSE.instrumental_width_sg.sel(channel=band).data)
effective_area = DEFAULTS_MUSE.main_line_effective_area.sel(channel=band)
doppler_velocity = [0] * u.km / u.s

##############################################################################
# **Density-dependent response.** The ``logD`` dimension of the line list
# flows through the spectral response and the detector mapping unchanged.

waveband_response = create_spectral_response(
    line_lists["density"]["line_list"],
    wavelength_grid,
    main_lines=main_lines,
    instrumental_width=instrumental_width,
    doppler_velocity=doppler_velocity,
    effective_area=effective_area,
)
response = map_response_to_sg_detector(waveband_response, band)
print(response)

plt.figure()
integrated = response.detector_response.sum(dim=["logT", "slit"]).sel(vdop=0).squeeze()
integrated.plot.line(x="detector_x_pixel", hue="logD")
plt.title("171 Angstrom response at zero Doppler velocity per electron density")

##############################################################################
# **Nonthermal broadening.** Back on the fixed-pressure line list, we add a
# ``nonthermal_velocity`` axis. Each value adds in quadrature to the thermal
# width, mimicking unresolved motions such as Alfven-wave turbulence.

waveband_response = create_spectral_response(
    line_lists["pressure"]["line_list"],
    wavelength_grid,
    main_lines=main_lines,
    instrumental_width=instrumental_width,
    doppler_velocity=doppler_velocity,
    nonthermal_velocity=np.arange(0, 100, 20) * u.km / u.s,
    effective_area=effective_area,
)
response = map_response_to_sg_detector(waveband_response, band)
print(response)

plt.figure()
integrated = response.detector_response.sum(dim=["logT", "slit"]).sel(vdop=0).squeeze()
integrated.plot.line(x="detector_x_pixel", hue="nonthermal_velocity")
plt.title("171 Angstrom response at zero Doppler velocity per nonthermal velocity")

plt.show()
