"""
==================
01 - Create a VDEM
==================

This tutorial demonstrates how to create a Velocity-Differential Emission Measure (VDEM) for MUSE.

A VDEM is the emission measure of the solar atmosphere as a function of temperature, velocity, and spatial structure.

.. warning::

    This example requires the optional ``PlasmaCalcs`` package. Install it with
    ``pip install "plasmacalcs[most]"`` before running the script.

.. note::

    This example is skipped due to amount of RAM it requires to generate the VDEM.
    This is why there are no plots.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PlasmaCalcs.hookups.muram.muram_calculator import MuramCalculator

import astropy.units as u

from muse.data import fetch_example_data
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
# `muse.data.fetch_example_data` downloads a tar-ed snapshot of it.

simulation_path = fetch_example_data("flare_nature_astro")

# PlasmaCalcs returns the simulation in cgs units; each input of create_simple_vdem carries its
# unit, and the function converts it to the one it works in.
muram_calc = MuramCalculator(dir=simulation_path, snap="0297000", units="cgs")
muram_calc.eos_mode = "aux"
temperature = muram_calc("T").values * u.K
# Mass per hydrogen nucleus in g, hardcoded to avoid depending on the Bifrost abundance tables.
r_per_nH_tot = 2.383931923587366e-24
ne_nh = (muram_calc("r") / r_per_nH_tot) ** 2  # Emission measure
# Zero out non-finite voxels so they cannot poison the VDEM.
ne_nh = ne_nh.where(np.isfinite(ne_nh), 0.0).values * u.cm**-6
velocity = muram_calc("u", component="z").values * u.cm / u.s  # LOS velocity
# Grid spacing along the line of sight.
cell_length = (muram_calc("dz") + muram_calc("maindims_z_coord") * 0.0).values * u.cm
x_coord = muram_calc("maindims_x_coord").values * u.cm
y_coord = muram_calc("maindims_y_coord").values * u.cm

vdem = create_simple_vdem(
    temperature=temperature,
    velocity=velocity,
    ne_nh=ne_nh,
    cell_length=cell_length,
    x=x_coord,
    y=y_coord,
    velocity_grid=np.arange(-500, 510, 10) * u.km / u.s,
    logT_grid=np.arange(5.5, 7.6, 0.1) * u.dex(u.K),
    # This is the axis along which the integration is performed,
    # in this case along the line of sight (z-axis)
    integration_axis=2,
)

##############################################################################
# The VDEM contains information about:
#
# - Temperature distribution (logT dimension)
# - Doppler velocity distribution (doppler_velocity dimension)
# - Spatial distribution (x, y dimensions)
# - Temporal evolution (time dimension)
#
# We can now print the VDEM we created.

print(vdem)

##############################################################################
# Calculate the VDEM moments
#
# The velocity moments can be calculated directly from the VDEM without first
# synthesizing spectra. ``calculate_moments`` integrates one spectral axis, so
# we first sum over temperature and then calculate the moments along ``doppler_velocity``.

velocity_distribution = vdem[["vdem"]].sum(dim="logT", skipna=False)
vdem_moments = calculate_moments(
    velocity_distribution,
    variable_name="vdem",
    doppler_name="doppler_velocity",
    moment_dim="doppler_velocity",
)

fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
vdem_moments["0th"].plot.imshow(ax=axes[0], norm=colors.LogNorm(vmin=1))
vdem_moments["1st"].plot.imshow(ax=axes[1], vmin=-100, vmax=100, cmap="bwr")
vdem_moments["2nd"].plot.imshow(ax=axes[2], vmin=0, vmax=100)
axes[0].set_title("VDEM 0th moment")
axes[1].set_title("VDEM 1st moment")
axes[2].set_title("VDEM 2nd moment")


##############################################################################
# Finally we save the VDEM so the later tutorials can reuse it.
# This is the artifact hosted online for examples 02 and 05.

output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)
vdem.to_zarr(output_dir / "muse_example_vdem.zarr", mode="w", zarr_format=3)

##############################################################################
# Now that you have a VDEM, you can use it to create a synthetic MUSE observation.

plt.show()
