"""
===============================
Create an EUVST Fe XII response
===============================

Create a SOLAR-C/EUVST response for the Fe XII 195.119 Å window with the
instrument-neutral :func:`muse.instrument.create_spectral_response`, taking the
effective area, dispersion and instrumental width from
`ECLIPSE <https://solarc-eclipse.readthedocs.io>`__, the EUVST instrument model.
The line list is that of the
:ref:`EIS response example <sphx_glr_generated_gallery_other_instruments_eis_fe_xii_response.py>`,
so the two instruments can be compared line for line.

.. warning::

    Requires the optional ``eclipse`` package: ``pip install eclipse``.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from eclipse import Detector_SWC, Telescope_EUVST, spectral_psf_fwhm

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import create_spectral_response

##############################################################################
# Fe XII 195.119 Å is in EUVST's short-wave (SW) channel, which reaches the
# detector from 170 to 212 Å.

line_list_file = fetch_example_data("eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf")

##############################################################################
# ECLIPSE gives the effective area as the collecting area times the mirror,
# grating and filter throughput, times the CCD quantum efficiency, and the
# width of a spectral pixel. The instrumental width depends on the slit: we
# take the 0.4 arcsec slit, as wide as MUSE's slits. ECLIPSE updates the
# instrument model as EUVST is built, so these numbers follow the installed
# version.

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
# The EUVST effective area against that of EIS, the pre-flight SW-band area
# the EIS examples use.

eis_table = np.loadtxt(fetch_example_data("EIS_EffArea_B.005"))
euvst_195 = effective_area.interp(wavelength=195.119).item()
eis_195 = np.interp(195.119, eis_table[:, 0], eis_table[:, 1])
print(f"At 195.119 Å: EUVST {euvst_195:.2f} cm2, EIS {eis_195:.2f} cm2, ratio {euvst_195 / eis_195:.1f}")
plt.figure()
plt.plot(effective_area.wavelength, effective_area, label="EUVST SW (ECLIPSE)")
plt.plot(eis_table[:, 0], eis_table[:, 1], label="EIS SW, pre-flight")
plt.axvline(195.119, color="gray", linestyle=":")
plt.xlim(170, 212)
plt.yscale("log")
plt.xlabel("Wavelength [Å]")
plt.ylabel("Effective area [cm$^2$]")
plt.legend()

##############################################################################
# The response on the EUVST pixel grid, as for EIS: per Å times the
# dispersion gives one value per spectral pixel.

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
# The line profile, summed over temperature, shifts with the Doppler
# velocity.

profile = response.spectral_response.isel(pressure=0).sum(dim=["logT", "line"])
plt.figure()
for velocity in [-200, 0, 200]:
    plt.plot(
        response.wavelength_grid,
        profile.sel(doppler_velocity=velocity, method="nearest"),
        label=f"{velocity} km/s",
    )
plt.xlabel("Wavelength [Å]")
plt.ylabel(f"Response [{response.spectral_response.attrs['units']}]")
plt.title("EUVST Fe XII 195 response, summed over logT")
plt.legend()

##############################################################################
# Summed over the detector pixels, the response gives the temperature
# sensitivity of each line.

temperature_sensitivity = (
    response.spectral_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="wavelength_bin", keep_attrs=True)
)
plt.figure(layout="constrained")
for line in temperature_sensitivity.line.values:
    temperature_sensitivity.sel(line=line).plot(label=str(line))
plt.title("EUVST Fe XII temperature sensitivity")
plt.legend()

plt.show()
