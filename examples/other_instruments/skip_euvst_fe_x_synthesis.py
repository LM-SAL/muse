"""
=================================
Synthesize an EUVST Fe X spectrum
=================================

This tutorial builds on the
:ref:`EUVST response example
<sphx_glr_generated_gallery_other_instruments_skip_euvst_fe_x_response.py>`
and synthesizes an EUVST Fe X 174/175 Å raster from the same VDEM used
in the :ref:`MUSE synthesis tutorial
<sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`,
using :func:`muse.synthesis.vdem_synthesis`.

The VDEM carries no density axis, so here we compute the line list at a
single electron density instead of the density grid used in the response
example; a density-dependent synthesis needs a VDEM with a matching ``logD``
dimension.

It requires a local CHIANTI database configured with ``XUVTOP``

.. code-block::

    export XUVTOP=/path/to/CHIANTI_11.0.2_database
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.instrument import create_chianti_line_list, create_spectral_response
from muse.synthesis import calculate_moments, vdem_synthesis, wavelength_to_doppler

##############################################################################
# We first confirm that the local environment is working.

if not os.environ.get("XUVTOP"):
    msg = "XUVTOP is not set. Run `export XUVTOP=/path/to/CHIANTI_11.0.2_database` first."
    raise OSError(msg)

##############################################################################
# We fetch the VDEM used by the MUSE synthesis tutorial. Its ``logT`` and
# ``vdop`` grids will define the response axes below, so no interpolation is
# needed before the synthesis.

extract_path = Path(pooch.os_cache("muse")) / "muse_example_vdem"
pooch.retrieve(
    "https://www.dropbox.com/scl/fi/xb2f6pvs4cn1yg54n0pdg/muse_example_vdem.zarr.tar.gz?rlkey=u5y19c5lydrw9kur9bzahkvsv&st=t5vltlk8&dl=1",
    known_hash="ab6c8a3fe4f30de6906f75165f19ccc8730040527f6b9b0cccbdd9a09c28a71c",
    fname="muse_example_vdem.zarr.tar.gz",
    path=extract_path.parent,
    processor=pooch.Untar(extract_dir=extract_path.name),
)
vdem = xr.open_zarr(extract_path / "muse_example_vdem.zarr")
# We need to keep the tutorial spectrum manageable.
# Remove this selection so you can have the full-resolution y axis.
vdem = vdem.isel(y=slice(None, None, 8))
print(vdem)

##############################################################################
# We recreate the EUVST Fe X response from the
# :ref:`response example
# <sphx_glr_generated_gallery_other_instruments_skip_euvst_fe_x_response.py>`
# on the VDEM's temperature and velocity grids, at a single electron density.

abundance = "sun_coronal_2021_chianti"
temperature = xr.DataArray(10**vdem.logT.data * u.K, dims="logT")
density = xr.DataArray([1e9] * u.cm**-3, dims="logD")

line_list = create_chianti_line_list(
    temperature=temperature,
    density=density,
    abundance=abundance,
    wavelength_range=[174.0, 175.6] * u.AA,
    ion_list=["fe_10"],
)

dispersion = 0.017 * u.AA
response = create_spectral_response(
    line_list,
    np.arange(174.0, 175.6, dispersion.to_value(u.AA)) * u.AA,
    main_lines=["Fe X 174.531", "Fe X 175.263"],
    doppler_velocity=vdem.vdop.data * u.km / u.s,
    effective_area=xr.DataArray(1 * u.cm**2),
)

##############################################################################
# As in the :ref:`EIS synthesis example
# <sphx_glr_generated_gallery_other_instruments_skip_eis_fe_xii_synthesis.py>`,
# the wavelength-space response feeds straight into
# :func:`muse.synthesis.vdem_synthesis`.

spectrum = vdem_synthesis(vdem, response).compute()
print(spectrum)

##############################################################################
# The synthesized spectrum along one column of the raster shows both Fe X
# lines.

plt.figure(figsize=(10, 5))
spectrum_image = spectrum.flux.isel(x=128, logD=0).sum(dim="line")
# The line wings decay to numerically zero, so anchor the log scale to the
# brightest pixel and show four decades below it.
spectrum_image.plot(
    x="detector_wavelength",
    y="y",
    norm=colors.LogNorm(vmin=spectrum_image.max().item() / 1e4, vmax=spectrum_image.max().item()),
    cmap="inferno",
)
plt.title("Synthesized EUVST Fe X spectrum at one raster position")

##############################################################################
# Finally, the per-line intensity maps and the Doppler map of the stronger
# line.

velocity_spectrum = wavelength_to_doppler(spectrum)
moments = calculate_moments(velocity_spectrum, moment_dim="wavelength_bin")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Anchor each log scale to its own data: median to max spans the quiet
# background and the flare core without washing either out.
for ax, line in zip(axes[:2], ["Fe X 174.531", "Fe X 175.263"], strict=True):
    intensity = moments["0th"].sel(line=line).isel(logD=0)
    intensity.plot(ax=ax, norm=colors.LogNorm(vmin=intensity.quantile(0.5).item(), vmax=intensity.max().item()))
    ax.set_title(f"{line} intensity")
moments["1st"].sel(line="Fe X 174.531").isel(logD=0).plot(ax=axes[2], cmap="RdBu_r", robust=True)
axes[2].set_title("Fe X 174.531 Doppler shift")
plt.tight_layout()

plt.show()
