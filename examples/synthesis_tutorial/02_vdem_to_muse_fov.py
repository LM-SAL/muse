"""
==================================
02 - Matching a VDEM to MUSE's FOV
==================================

Here we will show how to crop or interpolate a
Velocity-Differential Emission Measure (VDEM) to the MUSE Field of View (FOV).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

from muse.data import open_example_dataset
from muse.transforms import match_fov, reshape_slit_step_to_x, reshape_x_to_slit_step

##############################################################################
# First we will load a VDEM. For this set of tutorials, we have pre-computed
# and saved the outputs from each stage.

vdem = open_example_dataset("muse_example_vdem.zarr.zip", chunked=True)

print(vdem)

##############################################################################
# Since MUSE samples its field of view with 35 slits at 11 raster positions.
# To match the VDEM to MUSE's FOV and instrument parameters,
# we will use :func:`muse.transforms.match_fov`.

vdem_muse_fov = match_fov(vdem)

vdem_intensity = vdem.vdem.sum(dim=["logT", "doppler_velocity"], skipna=False)
muse_fov_intensity = vdem_muse_fov.vdem.sum(dim=["logT", "doppler_velocity"], skipna=False).compute()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
vdem_intensity.plot(x="x", y="y", norm=colors.LogNorm(vmin=1), ax=ax1)
ax1.set_title("Original VDEM")
muse_fov_intensity.plot(x="x", y="y", norm=colors.LogNorm(vmin=1), ax=ax2)
ax2.set_title("VDEM matched to MUSE FOV")

##############################################################################
# As MUSE is a slit-based spectrograph, :func:`muse.transforms.reshape_x_to_slit_step`
# splits the x axis into slit and raster-step dimensions.
# Here the steps have no time evolution.

vdem_muse_fov_slit = reshape_x_to_slit_step(vdem_muse_fov)

print(vdem_muse_fov_slit)

##############################################################################
# Now we will visualize the reshaped data.

slit_step_intensity = reshape_x_to_slit_step(muse_fov_intensity.to_dataset(name="vdem")).vdem
selected_step = 0
step_mask = xr.DataArray(
    np.arange(muse_fov_intensity.sizes["x"]) % slit_step_intensity.sizes["step"] == selected_step,
    dims="x",
    coords={"x": muse_fov_intensity.x},
)
selected_slit = slit_step_intensity.sizes["slit"] // 2

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
muse_fov_intensity.where(step_mask).plot(x="x", y="y", ax=axes[0], norm=colors.LogNorm(vmin=1e-3))
axes[0].set_title(f"All slits at raster step {selected_step}")
slit_step_intensity.isel(slit=selected_slit).plot(ax=axes[1], x="step", y="y", norm=colors.LogNorm(vmin=1e-3))
axes[1].set_title(f"Slit {selected_slit} across raster steps")

##############################################################################
# Finally, if you want to undo the reshape,
# :func:`muse.transforms.reshape_slit_step_to_x` reverts it.

vdem_muse_fov_original = reshape_slit_step_to_x(vdem_muse_fov_slit)

print(vdem_muse_fov_original)

plt.show()
