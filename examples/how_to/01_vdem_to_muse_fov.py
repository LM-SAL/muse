"""
=============================
Matching a VDEM to MUSE's FOV
=============================

This how-to demonstrates how to match a Velocity-Differential Emission Measure (VDEM) to MUSE's Field of View (FOV).
"""

import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pooch
import xarray as xr
from matplotlib import colors

from muse.transforms import match_fov, reshape_slit_step_to_x, reshape_x_to_slit_step

##############################################################################
# Loading a VDEM
#
# In the previous how-to, we created a simple VDEM using ``create_simple_vdem``.
# We will download this VDEM (from a personal data archive) and then load it using xarray.
#
# To download the data, we will use `pooch <https://www.fatiando.org/pooch/latest/>`__.
# To avoid downloading individual files, we will use a tar-ed snapshot.

tar_path = pooch.retrieve(
    "https://www.dropbox.com/scl/fi/xb2f6pvs4cn1yg54n0pdg/muse_example_vdem.zarr.tar.gz?rlkey=u5y19c5lydrw9kur9bzahkvsv&st=t5vltlk8&dl=1",
    known_hash="bc05a8b074b2c7994e0075ccfd21ba748e40b6efdab1d8f967885a2b9fc34c0d",
    fname="muse_example_vdem.zarr.tar.gz",
    processor=pooch.Untar(),
)
vdem = xr.open_zarr(Path(os.path.commonpath(tar_path))).load()

##############################################################################
# First, let's print the VDEM to see what it looks like.

print(vdem)

##############################################################################
# Now we can do several transforms to the VDEM, such as matching the FOV and
# reshaping to include/exclude the slits.
#
# First we will match the MUSE FOV to the VDEM and compare them.
#
# MUSE has a specific field of view (FOV) of 35x11 arcsec:math:`^2`. The
# :func:`muse.transforms.match_fov` function crops or interpolates the VDEM to match this instrumental FOV. This ensures the synthetic observations accurately represent what MUSE would observe.

vdem_muse_fov = match_fov(vdem)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
vdem.vdem.sum(dim=["logT", "vdop"], skipna=False).plot(norm=colors.LogNorm(vmin=1), ax=ax1)
vdem_muse_fov.vdem.sum(dim=["logT", "vdop"], skipna=False).plot(norm=colors.LogNorm(vmin=1), ax=ax2)
del vdem
gc.collect()

##############################################################################
# Now we can also introduce the :func:`muse.transforms.reshape_x_to_slit_step` function to reshape the VDEM to match the MUSE slit step configuration.
#
# This function takes the VDEM and reshapes it to match the MUSE slit step configuration, which includes the number of slits and steps per raster.
#
# The result is a reshaped VDEM that can be used to create a synthetic MUSE observation.

vdem_muse_fov_slit = reshape_x_to_slit_step(vdem_muse_fov)
del vdem_muse_fov
gc.collect()

##############################################################################
# Now when we print the reshaped VDEM, we can see that it has been reshaped to
# match the MUSE slit step configuration.

print(vdem_muse_fov_slit)

##############################################################################
# Finally, we can plot the VDEM for a specific step to see the reshaped VDEM.

fig, ax = plt.subplots()
vdem_muse_fov_slit.isel(step=0).vdem.sum(dim=["logT", "vdop"]).plot(norm=colors.LogNorm(vmin=1e-3), ax=ax)

##############################################################################
# If we want to go back to the original VDEM, we can use
# the :func:`muse.transforms.reshape_slit_step_to_x` function to reshape
# the reshaped VDEM back to the original shape.

vdem_muse_fov_original = reshape_slit_step_to_x(vdem_muse_fov_slit)
del vdem_muse_fov_slit
gc.collect()

print(vdem_muse_fov_original)

plt.show()
