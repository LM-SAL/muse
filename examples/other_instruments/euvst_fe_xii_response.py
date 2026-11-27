"""
===============================
Create an EUVST Fe XII response
===============================

Here we will create a SOLAR-C/EUVST response for the Fe XII 195.119 Å
using the EUVST instrument model from `ECLIPSE <https://solarc-eclipse.readthedocs.io>`__.

.. warning::

    Requires the optional ``eclipse`` package: ``pip install eclipse``.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from eclipse import Detector_SWC, Telescope_EUVST, spectral_psf_fwhm

import astropy.units as u

from muse.data import fetch_example_data, open_example_dataset
from muse.instrument import create_spectral_response

##############################################################################
# First we will load the line list for Fe XII 195.119 Å, which is the same as used in the
# :ref:`EIS response example <sphx_glr_generated_gallery_other_instruments_eis_fe_xii_response.py>`.
#
# Fe XII 195.119 Å is in EUVST's short-wave (SW) channel which ranges from 170 to 212 Å.

line_list = open_example_dataset("eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.zarr.zip")

##############################################################################
# ECLIPSE provides the EUVST instrument model but please be aware that ECLIPSE updates
# the instrument model as EUVST is built, so these numbers follow the installed version of
# ECLIPSE.

telescope = Telescope_EUVST()
detector = Detector_SWC()
slit_width = 0.4 * u.arcsec
dispersion = (1 * u.pix * detector.wvl_res).to(u.AA)
instrumental_fwhm = (spectral_psf_fwhm(telescope, detector, slit_width) * u.pix * detector.wvl_res).to(u.AA)
area_wavelength = np.linspace(170, 212, 421) * u.AA
effective_area = xr.DataArray(
    (telescope.ea_and_throughput(area_wavelength) * detector.qe_euv).to_value(u.cm**2),
    dims="wavelength",
    coords={"wavelength": ("wavelength", area_wavelength.to_value(u.AA), {"units": str(u.AA)})},
    attrs={"units": str(u.cm**2)},
)
print(f"Dispersion {dispersion:.4f}, instrumental FWHM {instrumental_fwhm:.4f}")

##############################################################################
# Now we will compare the EUVST effective area against EIS.

eis_table = np.loadtxt(fetch_example_data("EIS_EffArea_B.005"))
euvst_195 = effective_area.interp(wavelength=195.119).item()
eis_195 = np.interp(195.119, eis_table[:, 0], eis_table[:, 1])
print(f"At 195.119 Å: EUVST {euvst_195:.2f} cm2, EIS {eis_195:.2f} cm2, ratio {euvst_195 / eis_195:.1f}")

fig, ax = plt.subplots()
ax.plot(effective_area.wavelength, effective_area, label="EUVST SW (ECLIPSE)")
ax.plot(eis_table[:, 0], eis_table[:, 1], label="EIS SW, pre-flight")
ax.axvline(195.119, color="gray", linestyle=":")
ax.set(xlim=(170, 212), yscale="log", xlabel="Wavelength [Å]", ylabel="Effective area [cm$^2$]")
ax.legend()

##############################################################################
# Now we will create the response on the EUVST pixel grid.

response = create_spectral_response(
    line_list,
    np.arange(194.5, 195.7, dispersion.to_value(u.AA)) * u.AA,
    main_lines=["Fe XII 195.119", "Fe XII 195.179"],
    instrumental_fwhm=instrumental_fwhm,
    doppler_velocity=np.arange(-300, 310, 10) * u.km / u.s,
    effective_area=effective_area,
)
response_unit = u.Unit(response.spectral_response.attrs["units"]) * u.AA
binned_response = (response.spectral_response * dispersion.to_value(u.AA)).assign_attrs(
    {**response.spectral_response.attrs, "units": str(response_unit)}
)
response = response.assign(spectral_response=binned_response)
print(response)

##############################################################################
# Now we will sum over the detector pixels to give the temperature
# sensitivity of each line.

temperature_sensitivity = (
    response.spectral_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="wavelength_bin", keep_attrs=True)
)
fig, ax = plt.subplots(layout="constrained")
for line in temperature_sensitivity.line.values:
    temperature_sensitivity.sel(line=line).plot(ax=ax, label=str(line))
ax.set_title("EUVST Fe XII temperature sensitivity")
ax.legend()

plt.show()
