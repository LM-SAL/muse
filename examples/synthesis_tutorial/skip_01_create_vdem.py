"""
==================
01 - Create a VDEM
==================

This tutorial demonstrates how to create a Velocity-Differential Emission Measure (VDEM) for MUSE.

A VDEM is the emission measure of the solar atmosphere as a function of temperature, velocity, and spatial structure.
"""

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
from matplotlib import colors
from PlasmaCalcs.hookups.muram.muram_calculator import MuramCalculator

from muse.synthesis import calculate_moments, create_simple_vdem

##############################################################################
# Creating a VDEM
#
# The MUSE library includes a simple function to create VDEMs.
# But we will use `PlasmaCalcs <https://plasmacalcs.readthedocs.io/>`__ to derive the VDEM properties from a MURaM simulation snapshot.
#
# PlasmaCalcs includes the possibility to create VDEMs with many more options: absorption, other interpolation methods, TRAC if applies, limb view options, etc.
#
# We will use a MURaM simulation snapshot of a flare.
# `The data is stored online at Stanford. <https://purl.stanford.edu/dv883vb9686>`__
# It was used in `A comprehensive three-dimensional radiative magnetohydrodynamic simulation of a solar flare. <https://www.nature.com/articles/s41550-018-0629-3#article-info>`__
#
# To download the data, we will use `pooch <https://www.fatiando.org/pooch/latest/>`__.
# To avoid downloading individual files, we will use a tar-ed snapshot.

simulation_path = Path(pooch.os_cache("muse")) / "flare_nature_astro"
pooch.retrieve(
    "https://www.dropbox.com/scl/fi/tpkscbv2jq0slpz5hbupe/flare_nature_astro.tar.gz?rlkey=egmnsk2u8y17sdx4d6rcl8brm&st=kllq9izh&e=1&dl=1",
    known_hash="4ddc37682e65ee343657929beb8ddc50f472411ebd9fca66ec6ee18afeaf68c9",
    fname="flare_nature_astro.tar.gz",
    path=simulation_path.parent,
    processor=pooch.Untar(extract_dir=simulation_path.parent),
)

# Due to a bug in the MURaM reader, we need to change the working directory to the simulation path.
with contextlib.chdir(simulation_path):
    muram_calc = MuramCalculator(dir=simulation_path, snap="0310000", units="cgs")
    temperature = muram_calc("T")  # Temperature array in K
    # Mass per hydrogen nucleus in g, hardcoded to avoid depending on the Bifrost abundance tables.
    r_per_nH_tot = 2.383931923587366e-24
    ne_nh = (muram_calc("r") / r_per_nH_tot) ** 2  # Emission measure 1/cm^6
    ne_nh = ne_nh.where(np.isfinite(ne_nh), 0.0)  # Zero out non-finite voxels so they cannot poison the VDEM
    velocity = muram_calc("u", component="z") * 1e-5  # LOS velocity in km/s
    cell_length = muram_calc("dz") + muram_calc("maindims_z_coord") * 0.0  # Grid spacing along the line of sight in cm
    x_coord = muram_calc("maindims_x_coord")
    y_coord = muram_calc("maindims_y_coord")
velocity_axis = np.arange(-500, 510, 10)  # Velocity axis in km/s
log_temperature_axis = np.arange(5.5, 7.6, 0.1)

vdem = create_simple_vdem(
    temperature.values,
    velocity.values,
    ne_nh,
    cell_length,
    x_coord,
    y_coord,
    velocity_axis,
    log_temperature_axis,
)

##############################################################################
# The VDEM contains information about:
#
# - Temperature distribution (logT dimension)
# - Doppler velocity distribution (vdop dimension)
# - Spatial distribution (x, y dimensions)
# - Temporal evolution (time dimension)
#
# We can now print and plot the VDEM we created.

print(vdem)

# Calculate the VDEM moments
#
# The velocity moments can be calculated directly from the VDEM without first
# synthesizing spectra. ``calculate_moments`` integrates one spectral axis, so
# we first sum over temperature and then calculate the moments along ``vdop``.

velocity_distribution = vdem[["vdem"]].sum(dim="logT", skipna=False)
vdem_moments = calculate_moments(
    velocity_distribution,
    integration_name="vdem",
    doppler_name="vdop",
    moment_dim="vdop",
)

fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
vdem_moments["0th"].plot.imshow(ax=axes[0], norm=colors.LogNorm(vmin=1))
vdem_moments["1st"].plot.imshow(ax=axes[1], vmin=-100, vmax=100, cmap="bwr")
vdem_moments["2nd"].plot.imshow(ax=axes[2], vmin=0, vmax=100)
axes[0].set_title("VDEM 0th moment")
axes[1].set_title("VDEM 1st moment")
axes[2].set_title("VDEM 2nd moment")


##############################################################################
# Now that you have a VDEM, you can use it to create a synthetic MUSE observation.

plt.show()
