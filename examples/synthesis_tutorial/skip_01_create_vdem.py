"""
==================
01 - Create a VDEM
==================

Create a velocity-differential emission measure (VDEM), the emission measure
as a function of temperature, velocity and position, from a MURaM simulation.

.. warning::

    Requires the optional ``PlasmaCalcs`` package: ``pip install "plasmacalcs[most]"``.

.. note::

    Not run in the gallery, as it needs too much RAM, so it has no plots.
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
# `PlasmaCalcs <https://plasmacalcs.readthedocs.io/>`__ reads the plasma of a MURaM flare snapshot
# (`stored at Stanford <https://purl.stanford.edu/dv883vb9686>`__, from
# `Cheung et al. 2019 <https://www.nature.com/articles/s41550-018-0629-3#article-info>`__), and
# :func:`muse.synthesis.create_simple_vdem` builds the VDEM from it.
# PlasmaCalcs can also build VDEMs with more options: absorption, other interpolation methods, TRAC, limb views.
# :func:`muse.data.fetch_example_data` downloads the snapshot.

simulation_path = fetch_example_data("flare_nature_astro")

# PlasmaCalcs returns cgs; create_simple_vdem converts each input from its unit.
muram_calc = MuramCalculator(dir=simulation_path, snap="0297000", units="cgs")
muram_calc.eos_mode = "aux"
temperature = muram_calc("T").values * u.K
# Mass per hydrogen nucleus in g, hardcoded to avoid depending on the Bifrost abundance tables.
r_per_nH_tot = 2.383931923587366e-24
# The contribution functions are per n_e n_H, and n_e comes from the same aux EOS output as T.
ne_nh = muram_calc("ne") * muram_calc("r") / r_per_nH_tot
# Zero out non-finite voxels so they cannot poison the VDEM.
ne_nh = ne_nh.where(np.isfinite(ne_nh), 0.0).values * u.cm**-6
velocity = muram_calc("u", component="z").values * u.cm / u.s  # LOS velocity
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
    # The line of sight, the z axis.
    integration_axis=2,
)

print(vdem)

##############################################################################
# Calculate the VDEM moments
#
# The velocity moments come straight from the VDEM, without synthesizing spectra.
# :func:`muse.synthesis.calculate_moments` integrates one axis, so we first sum over temperature.

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
# Save the VDEM for the later tutorials, which fetch the hosted copy when there is no local one.

output_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
output_dir.mkdir(parents=True, exist_ok=True)
vdem.to_zarr(output_dir / "muse_example_vdem.zarr", mode="w", zarr_format=3)

plt.show()
