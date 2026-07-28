"""
==================================
02 - Matching a VDEM to MUSE's FOV
==================================

This tutorial demonstrates how to match a Velocity-Differential Emission Measure (VDEM) to MUSE's Field of View (FOV).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

from muse.data import fetch_example_data
from muse.transforms import match_fov, reshape_slit_step_to_x, reshape_x_to_slit_step

##############################################################################
# Loading a VDEM
#
# In the :ref:`previous tutorial <sphx_glr_generated_gallery_synthesis_tutorial_skip_01_create_vdem.py>`,
# we created a simple VDEM using ``create_simple_vdem``.
# We will download this VDEM (from a personal data archive) using
# :func:`muse.data.fetch_example_data`, which caches it locally via
# `pooch <https://www.fatiando.org/pooch/latest/>`__, and then load it using xarray.

vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))

##############################################################################
# First, let's print the VDEM to see what it looks like.

print(vdem)

##############################################################################
# Now we can do several transforms to the VDEM, such as matching the FOV and
# reshaping to include/exclude the slits.
#
# First we will match the MUSE FOV to the VDEM and compare them.
#
# MUSE samples its field of view with 35 slits at 11 raster positions.
# :func:`muse.transforms.match_fov` crops or interpolates the VDEM to
# the instrument pixel scale and required spatial extent.

vdem_muse_fov = match_fov(vdem)
original_intensity = vdem.vdem.sum(dim=["logT", "doppler_velocity"], skipna=False)
full_intensity = vdem_muse_fov.vdem.sum(dim=["logT", "doppler_velocity"], skipna=False).compute()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
original_intensity.plot(norm=colors.LogNorm(vmin=1), ax=ax1)
full_intensity.plot(norm=colors.LogNorm(vmin=1), ax=ax2)

##############################################################################
# MUSE is a slit-scanning spectrograph.
#
# :func:`muse.transforms.reshape_x_to_slit_step` transforms the x-axis into
# slit and raster-step dimensions to match how MUSE observes.

vdem_muse_fov_slit = reshape_x_to_slit_step(vdem_muse_fov)

##############################################################################
# Now when we print the reshaped VDEM, we can see that it has been reshaped to
# match the MUSE slit step configuration.

print(vdem_muse_fov_slit)

##############################################################################
# Compare one raster step with one slit
#
# Plotting directly against the compact ``slit`` index would hide the physical
# gaps between slits. The left panel restores the spatial x-axis and masks the
# positions not sampled at step 0. The right panel shows the central slit as it
# moves through all 11 raster positions.

slit_step_intensity = reshape_x_to_slit_step(full_intensity.to_dataset(name="vdem")).vdem
selected_step = 0
step_mask = xr.DataArray(
    np.arange(full_intensity.sizes["x"]) % slit_step_intensity.sizes["step"] == selected_step,
    dims="x",
    coords={"x": full_intensity.x},
)
selected_slit = slit_step_intensity.sizes["slit"] // 2

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
full_intensity.where(step_mask).plot(ax=axes[0], norm=colors.LogNorm(vmin=1e-3))
axes[0].set_title(f"All slits at raster step {selected_step}")
slit_step_intensity.isel(slit=selected_slit).plot(ax=axes[1], norm=colors.LogNorm(vmin=1e-3))
axes[1].set_title(f"Slit {selected_slit} across raster steps")

##############################################################################
# If we want to go back to the original VDEM, we can use
# :func:`muse.transforms.reshape_slit_step_to_x` to reshape the reshaped VDEM
# back to the original shape.

vdem_muse_fov_original = reshape_slit_step_to_x(vdem_muse_fov_slit)

print(vdem_muse_fov_original)

plt.show()
