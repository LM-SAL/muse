"""
==================================
05 - Synthesize a MUSE observation
==================================

Here we will synthesize the MUSE detector spectra from the VDEM and the SG responses.
"""

import matplotlib.pyplot as plt
from matplotlib import colors

import astropy.units as u

from muse.data import fetch_example_data, open_example_dataset, save_example_dataset
from muse.instrument import align_response_and_vdem, read_responses
from muse.synthesis import vdem_synthesis
from muse.transforms import match_fov, reshape_slit_step_to_x, reshape_x_to_slit_step

##############################################################################
# First we will load the VDEM and reshape it to the MUSE raster, as in
# :ref:`sphx_glr_generated_gallery_synthesis_tutorial_02_vdem_to_muse_fov.py`.

vdem = open_example_dataset("muse_example_vdem.zarr.zip", chunked=True)
vdem_raster = reshape_x_to_slit_step(match_fov(vdem))

##############################################################################
# next we will use :func:`muse.instrument.read_responses` to load and concatenate
# the responses of the three bands.

response_files = [
    fetch_example_data(fname)
    for fname in (
        "muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.zarr.zip",
        "muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.zarr.zip",
        "muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.zarr.zip",
    )
]

response = read_responses(
    response_files,
    number_of_slits=vdem_raster.sizes["slit"],
    chunked=True,
)

##############################################################################
# The synthesis needs the response and the VDEM on the same temperature and
# velocity grids, which :func:`muse.instrument.align_response_and_vdem` provides.

response, vdem_raster = align_response_and_vdem(
    response,
    vdem_raster,
    coordinate_methods={"logT": ("nearest", u.dex(u.K)), "doppler_velocity": ("linear", u.km / u.s)},
)

print(response)
print("\n\n")
print(vdem_raster)

##############################################################################
# :func:`muse.synthesis.vdem_synthesis` multiplies the VDEM by the response at
# each ``(logT, doppler_velocity)`` point and sums the ``sum_over`` dimension.
# The spectrum will then have ``line``, the spatial ``y``, ``slit`` and ``step``,
# and ``detector_x_pixel`` dimensions, with a ``detector_wavelength`` per pixel and slit.
#
# This works for any response (MUSE, EIS, EUVST, etc) and any VDEM, or DEM for a
# broadband filter such as SDO/AIA, at any resolution, with x or with slit and step.

spectrum = vdem_synthesis(
    vdem_raster,
    response,
    backend="numpy",
    sum_over=("logT", "doppler_velocity"),
)

# One line and slit per chunk lets the next tutorial read a single spectrogram.
output = save_example_dataset(spectrum.chunk({"line": 1, "slit": 1}), "muse_synthetic_spectra.zarr.zip")

print(spectrum)
print(f"Saved {output}")

##############################################################################
# Finally, we will plot the Fe IX 171.073 intensity over the full FOV,
# summing over the detector pixels and reshaping to the x dimension.

saved_spectrum = open_example_dataset(output.name, chunked=True)
intensity = reshape_slit_step_to_x(saved_spectrum.sum(dim="detector_x_pixel"))
fig, ax = plt.subplots(figsize=(10, 4))
intensity.flux.sel(line="Fe IX 171.073").isel(pressure=0).plot(
    x="x",
    y="y",
    ax=ax,
    norm=colors.LogNorm(vmin=0.3),
    cmap="inferno",
)
ax.set_title("Synthesized Fe IX 171.073 intensity over the full FOV")
plt.show()
