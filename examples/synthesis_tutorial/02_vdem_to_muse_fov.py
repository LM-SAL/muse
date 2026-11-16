"""
==================================
02 - Matching a VDEM to MUSE's FOV
==================================

Match a velocity-differential emission measure (VDEM) to the MUSE field of view (FOV).
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
# :func:`muse.data.fetch_example_data` downloads and caches the VDEM of the
# :ref:`previous tutorial <sphx_glr_generated_gallery_synthesis_tutorial_skip_01_create_vdem.py>`.

vdem = xr.open_zarr(fetch_example_data("muse_example_vdem.zarr"))

print(vdem)

##############################################################################
# MUSE samples its field of view with 35 slits at 11 raster positions.
# :func:`muse.transforms.match_fov` crops or interpolates the VDEM to the MUSE
# pixel scale and extent.

vdem_muse_fov = match_fov(vdem)
original_intensity = vdem.vdem.sum(dim=["logT", "doppler_velocity"], skipna=False)
full_intensity = vdem_muse_fov.vdem.sum(dim=["logT", "doppler_velocity"], skipna=False).compute()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
original_intensity.plot(x="x", y="y", norm=colors.LogNorm(vmin=1), ax=ax1)
ax1.set_title("Original VDEM")
full_intensity.plot(x="x", y="y", norm=colors.LogNorm(vmin=1), ax=ax2)
ax2.set_title("VDEM matched to MUSE FOV")

##############################################################################
# MUSE is a slit-scanning spectrograph:
# :func:`muse.transforms.reshape_x_to_slit_step` splits the x axis into slit
# and raster-step dimensions. Here the steps have no time evolution.

vdem_muse_fov_slit = reshape_x_to_slit_step(vdem_muse_fov)

print(vdem_muse_fov_slit)

##############################################################################
# Compare one raster step with one slit
#
# Left: the positions sampled at step 0, on the spatial x axis, which keeps
# the gaps between slits. Right: the central slit through all 11 raster positions.

slit_step_intensity = reshape_x_to_slit_step(full_intensity.to_dataset(name="vdem")).vdem
selected_step = 0
step_mask = xr.DataArray(
    np.arange(full_intensity.sizes["x"]) % slit_step_intensity.sizes["step"] == selected_step,
    dims="x",
    coords={"x": full_intensity.x},
)
selected_slit = slit_step_intensity.sizes["slit"] // 2

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
full_intensity.where(step_mask).plot(x="x", y="y", ax=axes[0], norm=colors.LogNorm(vmin=1e-3))
axes[0].set_title(f"All slits at raster step {selected_step}")
slit_step_intensity.isel(slit=selected_slit).plot(ax=axes[1], x="step", y="y", norm=colors.LogNorm(vmin=1e-3))
axes[1].set_title(f"Slit {selected_slit} across raster steps")

##############################################################################
# :func:`muse.transforms.reshape_slit_step_to_x` reverts the reshape.

vdem_muse_fov_original = reshape_slit_step_to_x(vdem_muse_fov_slit)

print(vdem_muse_fov_original)

plt.show()
