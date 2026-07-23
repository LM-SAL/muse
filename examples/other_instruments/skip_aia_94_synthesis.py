"""
============================
Synthesize an AIA 94 Å image
============================

This tutorial builds on the
:ref:`AIA response example
<sphx_glr_generated_gallery_other_instruments_skip_aia_94_response.py>`
and synthesizes an AIA 94 Å image from the same VDEM used in the
:ref:`MUSE synthesis tutorial
<sphx_glr_generated_gallery_synthesis_tutorial_05_synthesize_muse_observation.py>`,
using :func:`muse.synthesis.vdem_synthesis`.

An imager integrates over its whole bandpass, so after synthesizing the
Doppler-resolved spectra of the strongest contributors we sum over wavelength
and line to form the image. As in the response example, this is the iron-line
contribution only (no continuum, no other elements).

It requires a local CHIANTI database configured with ``XUVTOP``

.. code-block::

    export XUVTOP=/path/to/CHIANTI_11.0.2_database

and `aiapy` (``pip install aiapy``) for the effective area.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
import sunpy.visualization.colormaps  # NOQA: F401 -- registers the SDO colormaps with matplotlib
import xarray as xr
from matplotlib import colors

import astropy.units as u

from muse.instrument import create_chianti_line_list, create_spectral_response
from muse.synthesis import vdem_synthesis

try:
    from aiapy.response import Channel
except ImportError:
    msg = "aiapy is required for this example, install it with `pip install aiapy`"
    raise ImportError(msg) from None

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
# We recreate the AIA 94 Å response from the
# :ref:`response example
# <sphx_glr_generated_gallery_other_instruments_skip_aia_94_response.py>`
# on the VDEM's temperature and velocity grids: the `aiapy` effective area,
# an iron-only line list, and the five strongest effective-area-weighted
# contributors.

channel = Channel(94 * u.angstrom)
effective_area = xr.DataArray(
    channel.effective_area.to_value(u.cm**2),
    dims="wavelength",
    coords={"wavelength": ("wavelength", channel.wavelength.to_value(u.AA), {"units": str(u.AA)})},
    attrs={"units": str(u.cm**2)},
).sel(wavelength=slice(84, 106))

abundance = "sun_coronal_2021_chianti"
temperature = xr.DataArray(10**vdem.logT.data * u.K, dims="logT")
pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")

line_list = create_chianti_line_list(
    temperature=temperature,
    pressure=pressure,
    abundance=abundance,
    wavelength_range=[85, 105] * u.AA,
    element_list=["fe"],
)

area_at_lines = effective_area.interp(wavelength=line_list.wavelength).fillna(0.0).drop_vars("wavelength")
peak_weight = (line_list.gofnt.isel(pressure=0) * area_at_lines).max(dim="logT")
ranked = line_list.full_name.values[np.argsort(-peak_weight.values)]
main_lines = list(dict.fromkeys(str(name) for name in ranked))[:5]
print(f"Strongest contributors: {main_lines}")

response = create_spectral_response(
    line_list,
    np.arange(91.0, 97.0, 0.05) * u.AA,
    main_lines=main_lines,
    doppler_velocity=vdem.vdop.data * u.km / u.s,
    effective_area=effective_area,
)

##############################################################################
# As in the :ref:`EIS synthesis example
# <sphx_glr_generated_gallery_other_instruments_skip_eis_fe_xii_synthesis.py>`,
# the wavelength-space response feeds straight into
# :func:`muse.synthesis.vdem_synthesis`.

spectrum = vdem_synthesis(vdem, response).compute()
print(spectrum)

##############################################################################
# Summing over wavelength and line gives the band-integrated image.

image = spectrum.flux.isel(pressure=0).sum(dim=["wavelength_bin", "line"])
plt.figure()
# Anchor the log scale to the data: median to max spans the background
# arcade and the flare core without washing either out.
image.plot(norm=colors.LogNorm(vmin=image.quantile(0.5).item(), vmax=image.max().item()), cmap="sdoaia94")
plt.title("Synthesized AIA 94 Å image (Fe lines only)")

##############################################################################
# The per-line images separate the hot and cool channel components:
# Fe XVIII picks out the hottest plasma while Fe X and Fe VIII show the
# cooler background.

per_line = spectrum.flux.isel(pressure=0).sum(dim="wavelength_bin")
nrows = -(-len(main_lines) // 2)
fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows))
for ax, line in zip(axes.flat, per_line.line.values, strict=False):
    data = per_line.sel(line=line)
    norm = colors.LogNorm(vmin=data.quantile(0.5).item(), vmax=data.max().item())
    data.plot(ax=ax, norm=norm, cmap="sdoaia94", add_colorbar=False)
    ax.set_title(str(line))
for ax in axes.flat[len(main_lines) :]:
    ax.set_visible(False)
plt.tight_layout()

plt.show()
