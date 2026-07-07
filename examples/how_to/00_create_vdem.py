"""
=============
Create a VDEM
=============

This how-to demonstrates how to create a Velocity-Differential Emission Measure (VDEM) for MUSE.

A VDEM contains the physical properties of the solar atmosphere (temperature, velocity, spatial structure).
"""

import gc
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
from matplotlib import colors
from PlasmaCalcs.hookups.muram.muram_calculator import MuramCalculator

from muse.synthesis.utils import create_simple_vdem

##############################################################################
# Creating a VDEM
#
# The MUSE library includes a simple function to create VDEMs.
# But we will use PlasmaCalcs to derive the VDEM properties from a MURaM simulation snapshot.
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
    processor=pooch.Untar(extract_dir=simulation_path.name),
)

# Due to a bug in the MURaM reader, we need to change the working directory to the simulation path.
with contextlib.chdir(simulation_path):
    # In your case, you may need to change the snapshot number if you have your own simulation.
    muram_calc = MuramCalculator(dir=simulation_path, snap="0310000", units="cgs")
    temperature = muram_calc("T")  # Temperature array in K
    r_per_nH_tot = (muram_calc.elements.n_per_nH() * muram_calc.elements.m * muram_calc.u("amu")).sum()
    ne_nh = (muram_calc("r") / r_per_nH_tot) ** 2  # Emission measure 1/cm^6
    velocity = muram_calc("u", component="z") * 1e-5  # velocity along the line of sight in km/s
    cell_length = muram_calc("dz") + muram_calc("maindims_z_coord") * 0.0  # grid spacing along the line of sight in cm
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
    # Used to split the x axis into this many contiguous blocks and process them one at a time.
    # Required for the online documentation build.
    n_x_chunks=8,
)
# Due to tight memory constraints, the online documentation build requires deleting a few large variables.
del (
    temperature,
    velocity,
    ne_nh,
    cell_length,
    x_coord,
    y_coord,
    velocity_axis,
    log_temperature_axis,
    MuramCalculator,
    muram_calc,
)
gc.collect()

##############################################################################
# The VDEM contains information about:
# - Temperature distribution (logT dimension)
# - Doppler velocity distribution (vdop dimension)
# - Spatial distribution (x, y dimensions)
# - Temporal evolution (time dimension)
#
# We can now print and plot the VDEM we created.

print(vdem)
# Moment 0 from VDEM
vdem.vdem.sum(dim=["logT", "vdop"], skipna=False).plot(norm=colors.LogNorm(vmin=1))

plt.show()

##############################################################################
# Now that you have a VDEM, you can use it to create a synthetic MUSE observation.
