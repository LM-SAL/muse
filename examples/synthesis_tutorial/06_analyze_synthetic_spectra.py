"""
===================================
06 - Analyze synthetic MUSE spectra
===================================

Here we will plot the detector spectra of
:ref:`example 05 <sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`
and calculate their spectral moments.
"""

import matplotlib.pyplot as plt
from matplotlib import colors

from muse.data import open_example_dataset
from muse.synthesis import calculate_moments, wavelength_to_doppler
from muse.transforms import reshape_slit_step_to_x

##############################################################################
# First, we will load the synthetic spectra from the previous tutorial.

spectrum = open_example_dataset("muse_synthetic_spectra.zarr.zip", chunked=True)

##############################################################################
# **Spectrogram**

spectrogram = spectrum.flux.sel(line="Fe IX 171.073").isel(step=2, slit=15, pressure=0)
fig, ax = plt.subplots(figsize=(12, 6))
spectrogram.plot.imshow(x="detector_wavelength", y="y", ax=ax, norm=colors.LogNorm(vmin=0.3), cmap="jet")
ax.set(xlim=(170.2, 172), title="Fe IX 171.073: raster step 2, slit 15")

##############################################################################
# **Line profile**

profile = spectrogram.isel(y=spectrogram.sizes["y"] // 2)
fig, ax = plt.subplots(figsize=(8, 4))
profile.plot.line(x="detector_wavelength", ax=ax)
ax.set_xlim(170.2, 172)
ax.set_title("Fe IX 171.073 at one spatial location")

##############################################################################
# **Spatial map**

intensity = (
    spectrum[["flux"]].where((spectrum.channel == 171).compute(), drop=True).sum(dim=["line", "detector_x_pixel"])
)
image = reshape_slit_step_to_x(intensity).flux.isel(pressure=0)
fig, ax = plt.subplots(figsize=(8, 5))
image.plot(ax=ax, norm=colors.LogNorm())
ax.set_title("Synthesized 171 Angstrom intensity")

##############################################################################
# **Spectral moments**

velocity_spectrum = wavelength_to_doppler(spectrum)
moments = calculate_moments(velocity_spectrum)
# Compute once, or every panel below recomputes the lazy moments from the spectrum.
moment_maps = reshape_slit_step_to_x(moments).isel(pressure=0).compute()

fig, axes = plt.subplots(
    moment_maps.sizes["line"],
    3,
    figsize=(15, 15),
    layout="constrained",
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
