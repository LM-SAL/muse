"""
===================================
06 - Analyze synthetic MUSE spectra
===================================

This tutorial demonstrates how to visualize the detector spectra created by
:ref:`example 05 <sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`.
In addition to calculating the spectral moments as well.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pooch
import xarray as xr
from matplotlib import colors

from muse.log import change_logging_level
from muse.synthesis import calculate_moments, wavelength_to_doppler
from muse.transforms import reshape_slit_step_to_x

# muse logs at DEBUG level by default; raise it to INFO to reduce the noise.
change_logging_level("INFO")

##############################################################################
# Download the saved MUSE spectrum from the
# :ref:`previous example <sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`.
#
# If you have your own version locally, you need to change the ``spectrum_path``.

tutorial_dir = Path(pooch.os_cache("muse")) / "synthesis_tutorial"
spectrum_path = Path(
    pooch.retrieve(
        url=(
            "https://www.dropbox.com/scl/fi/jekc9ol4tjmaazbs0zii5/muse_synthetic_spectra.nc"
            "?rlkey=rqe45g5f82mrkgyg5l3fy4ssg&st=lxupqq0h&dl=1"
        ),
        known_hash="sha256:cb0a2b4ad3f496a1b1bc5c224c5d84b0f268299efca964252e590befd6339009",
        fname="muse_synthetic_spectra.nc",
        path=tutorial_dir,
    )
)

spectrum = xr.open_dataset(spectrum_path, engine="h5netcdf", chunks="auto")

##############################################################################
# Visualizing synthetic spectra
#
# The synthetic spectra can be visualized as:
#
# - **Spectrograms**: intensity as a function of wavelength and spatial position
# - **Line profiles**: spectra at specific spatial locations
# - **Spatial maps**: intensity integrated over wavelength
#
# **Spectrogram**
#
# This image shows Fe IX 171.073 along slit 15 at raster step 2. The wavelength
# coordinate depends on the selected slit.

spectrogram = spectrum.flux.sel(line="Fe IX 171.073").isel(step=2, slit=15, pressure=0)
spectrogram_plot = spectrogram.plot.imshow(
    x="detector_wavelength",
    y="y",
    norm=colors.PowerNorm(0.3),
    cmap="jet",
    figsize=(12, 6),
)
spectrogram_plot.axes.set_xlim(170.2, 172)
spectrogram_plot.axes.set_title("Fe IX 171.073: raster step 2, slit 15")

##############################################################################
# **Line profile**
#
# Selecting one y position from the spectrogram produces a conventional line
# profile.

profile = spectrogram.isel(y=spectrogram.sizes["y"] // 2)
fig, ax = plt.subplots(figsize=(8, 4))
profile.plot.line(x="detector_wavelength", ax=ax)
ax.set_xlim(170.2, 172)
ax.set_title("Fe IX 171.073 at one spatial location")

##############################################################################
# **Spatial map**
#
# Integrating over the detector pixels produces the 171 Angstrom intensity.
# Restacking the slit and raster-step dimensions reconstructs the full spatial
# field of view.

intensity = (
    spectrum[["flux"]].where((spectrum.channel == 171).compute(), drop=True).sum(dim=["line", "detector_x_pixel"])
)
image = reshape_slit_step_to_x(intensity).flux.isel(pressure=0)
fig, ax = plt.subplots(figsize=(8, 5))
image.plot(ax=ax, norm=colors.LogNorm())
ax.set_title("Synthesized 171 Angstrom intensity")

##############################################################################
# Finally we can calculate the spectral moments.
#
# Spectral moments characterize the line properties:
#
# - **0th moment**: Total intensity (integrated flux)
# - **1st moment**: Line-of-sight velocity (Doppler shift)
# - **2nd moment**: Line width (thermal + non-thermal broadening)
#
# :func:`muse.synthesis.wavelength_to_doppler` adds the velocity coordinate
# required by :func:`muse.synthesis.calculate_moments`.

velocity_spectrum = wavelength_to_doppler(spectrum)
moments = calculate_moments(velocity_spectrum)
moment_maps = reshape_slit_step_to_x(moments).isel(pressure=0)

fig, axes = plt.subplots(
    moment_maps.sizes["line"],
    3,
    figsize=(15, 15),
    constrained_layout=True,
    squeeze=False,
)
for row, line in enumerate(moment_maps.line.values):
    line_maps = moment_maps.sel(line=line)
    line_maps["0th"].plot.imshow(ax=axes[row, 0], norm=colors.LogNorm(vmin=1))
    line_maps["1st"].plot.imshow(ax=axes[row, 1], vmin=-100, vmax=100, cmap="bwr")
    line_maps["2nd"].plot.imshow(ax=axes[row, 2], vmin=0, vmax=100)

    axes[row, 0].set_title(f"{line} 0th moment")
    axes[row, 1].set_title(f"{line} 1st moment")
    axes[row, 2].set_title(f"{line} 2nd moment")

plt.show()
