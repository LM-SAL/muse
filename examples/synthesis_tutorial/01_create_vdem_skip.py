"""
==================
01 - Create a VDEM
==================

Here we will create a Velocity-Differential Emission Measure (VDEM), the emission measure
as a function of temperature, velocity and position, from a MURaM simulation.

.. warning::

    Requires the optional ``PlasmaCalcs`` package: ``pip install "plasmacalcs[most]"``.

.. note::

    This example is not run in the gallery, as has high RAM requirements,
    this is the reason there is no plot.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PlasmaCalcs.hookups.muram.muram_calculator import MuramCalculator

import astropy.units as u

from muse.data import fetch_example_data, save_example_dataset
from muse.synthesis import calculate_moments, create_simple_vdem

##############################################################################
# First we will fetch the MURaM flare snapshot, which is a 3D cube of temperature, density and velocity.
#
# From here, we will need to load the correct parameters to create the VDEM.
# `PlasmaCalcs <https://plasmacalcs.readthedocs.io/>`__ :cite:p:`plasmacalcs` reads the plasma of a
# MURaM :cite:p:`vogler_2005,rempel_2017` flare snapshot
# (`stored at Stanford <https://purl.stanford.edu/dv883vb9686>`__, from :cite:t:`cheung_2019`).
#
# PlasmaCalcs can also one to add more options such as absorption, other interpolation methods,
# TRAC, limb views.

simulation_path = fetch_example_data("flare_nature_astro")

# PlasmaCalcs returns cgs; create_simple_vdem converts each input from its unit.
muram_calc = MuramCalculator(dir=simulation_path, snap="0297000", units="cgs")
muram_calc.eos_mode = "aux"
temperature = muram_calc("T").values * u.K
# Mass per hydrogen nucleus in g, hardcoded to avoid depending on the Bifrost abundance tables.
r_per_nH_tot = 2.383931923587366e-24
n_H = muram_calc("r") / r_per_nH_tot
# The contribution functions are per n_e n_H, and the snapshot has no electron density.
# Where the MUSE lines form, Hydrogen and Helium are fully ionized:
# n_e = (1 + 2 A_He) n_H, with A_He of the responses' sun_coronal_2021_chianti abundances.
ne_nh = (1 + 2 * 10 ** (10.914 - 12)) * n_H**2
# Zero out non-finite voxels so they cannot poison the VDEM.
ne_nh = ne_nh.where(np.isfinite(ne_nh), 0.0).values * u.cm**-6
velocity = muram_calc("u", component="z").values * u.cm / u.s  # LOS velocity
cell_length = (muram_calc("dz") + muram_calc("maindims_z_coord") * 0.0).values * u.cm
x_coord = muram_calc("maindims_x_coord").values * u.cm
y_coord = muram_calc("maindims_y_coord").values * u.cm

##############################################################################
# Now we can create a VDEM using :func:`muse.synthesis.create_simple_vdem`.

vdem = create_simple_vdem(
    temperature=temperature,
    velocity=velocity,
    ne_nh=ne_nh,
    cell_length=cell_length,
    x=x_coord,
    y=y_coord,
    velocity_grid=np.arange(-500, 510, 10) * u.km / u.s,
    logT_grid=np.arange(5.5, 7.6, 0.1) * u.dex(u.K),
    # The line of sight, the z axis.
    integration_axis=2,
)

print(vdem)

##############################################################################
# Next, we can calculate the VDEM moments using :func:`muse.synthesis.calculate_moments`.

velocity_distribution = vdem[["vdem"]].sum(dim="logT", skipna=False)
vdem_moments = calculate_moments(
    velocity_distribution,
    variable_name="vdem",
    doppler_name="doppler_velocity",
    moment_dim="doppler_velocity",
)

fig, axes = plt.subplots(1, 3, figsize=(15, 7), layout="constrained")
vdem_moments["0th"].plot.imshow(ax=axes[0], norm=colors.LogNorm(vmin=1))
vdem_moments["1st"].plot.imshow(ax=axes[1], vmin=-100, vmax=100, cmap="bwr")
vdem_moments["2nd"].plot.imshow(ax=axes[2], vmin=0, vmax=100)
axes[0].set_title("VDEM 0th moment")
axes[1].set_title("VDEM 1st moment")
axes[2].set_title("VDEM 2nd moment")

##############################################################################
# Finally, we will save the VDEM for the later tutorials.

save_example_dataset(vdem, "muse_example_vdem.zarr.zip")

plt.show()
