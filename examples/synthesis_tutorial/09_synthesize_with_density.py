"""
============================
09 - Synthesize with density
============================

Here we will synthesize the Fe IX 171.073 intensity using a VDEM which has
an emission-measure-weighted mean electron density, following
:cite:t:`mckevitt_2026`.
"""

import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.data import open_example_dataset
from muse.instrument import (
    align_response_and_vdem,
    create_spectral_response,
    map_response_to_ci_detector,
    transform_response_units,
)
from muse.synthesis import vdem_synthesis
from muse.variables import DEFAULTS_MUSE

##############################################################################
# First we will load the VDEM and the emission-measure-weighted mean electron
# density of each of its bins, both saved by
# :ref:`tutorial 01 <sphx_glr_generated_gallery_synthesis_tutorial_01_create_vdem_skip.py>`.
#
# Next we will load two 171 Å line lists made as in
# :ref:`tutorial 03 <sphx_glr_generated_gallery_synthesis_tutorial_03_prepare_chianti_line_lists_skip.py>`:
# one at a fixed pressure, and one with ``density`` instead of ``pressure``:
#
# .. code-block:: python
#
#     temperature = np.arange(4.5, 8.0, 0.1) * u.dex(u.K)
#     density = 10 ** np.arange(7.5, 12.5, 0.5) * u.cm**-3
#     line_list = create_chianti_line_list(
#         temperature=temperature,
#         density=density,
#         abundance="sun_coronal_2021_chianti",
#         wavelength_range=wavelength_range,
#         ion_list=["fe_9"],
#     )
#
# The density-grid list, and everything derived from it, carries a ``logD``
# dimension whose coordinate is ``log10(density)``.

vdem = open_example_dataset("muse_example_vdem.zarr.zip", chunked=True)
density = open_example_dataset("muse_example_vdem_density.zarr.zip", chunked=True)
# join="exact" refuses a density that is not on the grid of this VDEM.
vdem = xr.merge([vdem, density], join="exact", compat="no_conflicts", combine_attrs="override")
line_lists = {
    variant: open_example_dataset(f"muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti{suffix}.zarr.zip")
    for variant, suffix in (("pressure", ""), ("density", "_density"))
}
print(vdem)

##############################################################################
# Now we can create a response from each line list. As we only need the
# intensity of the line, we will integrate each over the band with
# :func:`muse.instrument.map_response_to_ci_detector`.

band = 171 * u.AA
responses = {}
for variant, line_list in line_lists.items():
    response = create_spectral_response(
        line_list,
        np.arange(170.5, 171.65, 0.02) * u.AA,
        main_lines=["Fe IX 171.073"],
        doppler_velocity=vdem.doppler_velocity.data * u.km / u.s,
        effective_area=DEFAULTS_MUSE.main_line_effective_area_SG.sel(channel=171),
    )
    response = transform_response_units(response, "1e-27 cm5 ph / (Angstrom s)", channel=band)
    responses[variant] = map_response_to_ci_detector(response, channel=band)
print(responses["density"])

##############################################################################
# Next we will plot the response at four densities. The line is fainter in
# denser plasma, and the fixed-pressure response assumes a density of ``P / T``.

at_rest = {variant: response.detector_response.sel(doppler_velocity=0) for variant, response in responses.items()}
fig, ax = plt.subplots()
for logD in (8.0, 9.0, 10.0, 11.0):
    at_rest["density"].sel(logD=logD).plot(ax=ax, x="logT", label=f"$10^{{{logD:.0f}}}$ cm$^{{-3}}$")
at_rest["pressure"].plot(ax=ax, x="logT", color="black", linestyle="--", label="fixed pressure")
ax.set(
    xlim=(5.4, 6.5),
    xlabel="log T",
    ylabel=f"response [{at_rest['density'].attrs['units']}]",
    title="Fe IX 171.073 response by electron density",
)
ax.legend()

##############################################################################
# Next, we need the responses on the temperature and velocity grids of the VDEM,
# as in
# :ref:`tutorial 05 <sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`.

# We chunk to try and bound the memory use.
vdem = vdem.chunk(x=32, y=32).assign_coords({name: (vdem[name] / 1e8).assign_attrs(units="Mm") for name in ("x", "y")})
for variant, response in responses.items():
    responses[variant], vdem = align_response_and_vdem(response, vdem)

##############################################################################
# Now we can synthesize the line two ways: with the fixed-pressure response, and
# with the density response at the emission-measure-weighted density.

sum_over = ("logT", "doppler_velocity")
fixed_pressure, with_density = dask.compute(
    vdem_synthesis(vdem, responses["pressure"], sum_over=sum_over).flux.squeeze(drop=True),
    vdem_synthesis(vdem, responses["density"], density=vdem.density, sum_over=sum_over).flux.squeeze(drop=True),
)

fig, ax = plt.subplots(figsize=(7.5, 3.9), layout="constrained")
with_density.plot(
    ax=ax,
    x="x",
    y="y",
    norm=colors.LogNorm(vmin=with_density.quantile(0.5).item(), vmax=with_density.max().item()),
    cmap="inferno",
)
ax.set(
    xlabel="x [Mm]",
    ylabel="y [Mm]",
    title="Fe IX 171.073 at the emission-measure-weighted density",
    aspect="equal",
)

##############################################################################
# Finally, we will compare the image at the emission-measure-weighted density
# with the fixed-pressure one.

# Pixels fainter than 1% of the 99th percentile are left grey.
bright = fixed_pressure >= 0.01 * fixed_pressure.quantile(0.99)
ratio = (with_density / fixed_pressure).where(bright)
cmap = plt.get_cmap("RdBu_r").with_extremes(bad="0.7")
fig, ax = plt.subplots(figsize=(7.5, 3.9), layout="constrained")
ratio.plot(
    ax=ax,
    x="x",
    y="y",
    cmap=cmap,
    vmin=0.7,
    vmax=1.3,
    cbar_kwargs={"label": "emission-measure-weighted density\n/ fixed pressure", "extend": "both"},
)
ax.set(xlabel="x [Mm]", ylabel="y [Mm]", title="Fe IX 171.073 flux ratio", aspect="equal")

plt.show()

# sphinx_gallery_thumbnail_number = -1
