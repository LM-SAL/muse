"""
====================================
Create EIS Fe XII and Fe X responses
====================================

Here we will create the spectral response of Hinode/EIS EUV lines
using a CHIANTI line list and the instrument properties from SolarSoft.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import astropy.units as u

from muse.data import fetch_example_data, open_example_dataset
from muse.instrument import create_spectral_response

##############################################################################
# Hinode/EIS observes two EUV bands (SW: 166-212 Å & LW: 245-291 Å).
# Fe XII 195.119 Å is the strongest line in the SW band and there is also
# the density-sensitive Fe XII 195.179 Å.

line_list = open_example_dataset("eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.zarr.zip")
print(line_list)

##############################################################################
# We will use several representative EIS values from
# :cite:t:`culhane_2007` and EIS Software Note #7 :cite:p:`young_2011`.
#
# The effective area is the pre-flight SW-band curve ``EIS_EffArea_B.005`` from
# SolarSoft and has no in-flight decay; EIS Software Note #2 :cite:p:`mariska_2010`,
# :cite:t:`del_zanna_2013` and :cite:t:`warren_2014` provide time-dependent calibrations.

dispersion = 0.0223 * u.AA
instrumental_fwhm = 0.056 * u.AA
wavelength_grid = np.arange(194.5, 195.7, dispersion.to_value(u.AA)) * u.AA
doppler_velocity = np.arange(-300, 310, 10) * u.km / u.s
ea_table = np.loadtxt(fetch_example_data("EIS_EffArea_B.005"))
effective_area = xr.DataArray(
    ea_table[:, 1],
    dims="wavelength",
    coords={"wavelength": ("wavelength", ea_table[:, 0], {"units": str(u.AA)})},
    attrs={"units": str(u.cm**2)},
)

response = create_spectral_response(
    line_list,
    wavelength_grid,
    main_lines=["Fe XII 195.119", "Fe XII 195.179"],
    instrumental_fwhm=instrumental_fwhm,
    doppler_velocity=doppler_velocity,
    effective_area=effective_area,
)
# Per Å times the dispersion gives one value per EIS spectral pixel.
response_unit = u.Unit(response.spectral_response.attrs["units"]) * u.AA
binned_response = (response.spectral_response * dispersion.to_value(u.AA)).assign_attrs(
    {**response.spectral_response.attrs, "units": str(response_unit)}
)
response = response.assign(spectral_response=binned_response)
print(response)

##############################################################################
# Here we will plot the line profile summed over the temperature and
# density grid, for three Doppler velocities.

profile = response.spectral_response.isel(pressure=0).sum(dim=["logT", "line"])
fig, ax = plt.subplots()
for velocity in [-200, 0, 200]:
    ax.plot(
        response.wavelength_grid,
        profile.sel(doppler_velocity=velocity, method="nearest"),
        label=f"{velocity} km/s",
    )
ax.set(
    xlabel="Wavelength [Å]",
    ylabel=f"Response [{response.spectral_response.attrs['units']}]",
    title="EIS Fe XII 195 Å response, summed over logT",
)
ax.legend()

##############################################################################
# We can also sum over the detector instead to get the temperature
# sensitivity of each line.

temperature_sensitivity = (
    response.spectral_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="wavelength_bin", keep_attrs=True)
)
fig, ax = plt.subplots()
for line in temperature_sensitivity.line.values:
    temperature_sensitivity.sel(line=line).plot(ax=ax, label=str(line))
ax.set_title("EIS Fe XII temperature sensitivity")
ax.legend()

##############################################################################
# For the Fe X 174.531/175.263 Å line list, an electron-density (``logD``)
# grid was added. Right now, it is not used in the response.

density_line_list = open_example_dataset("eis_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.zarr.zip")
density_response = create_spectral_response(
    density_line_list,
    np.arange(174.0, 175.6, dispersion.to_value(u.AA)) * u.AA,
    main_lines=["Fe X 174.531", "Fe X 175.263"],
    instrumental_fwhm=instrumental_fwhm,
)

##############################################################################
# If we take the per-Å response summed over the grid times the spacing.
# It provides the total intensity of each line.

line_total = (density_response.spectral_response.sum(dim="wavelength_bin") * dispersion.to_value(u.AA)).assign_attrs(
    units=str(u.Unit(density_response.spectral_response.attrs["units"]) * u.AA)
)
is_fe_x_174 = (density_line_list.full_name == "Fe X 174.531").values
peak_logT = density_line_list.logT_peak.isel(transition=is_fe_x_174).median().item()
ratio = line_total.sel(line="Fe X 175.263") / line_total.sel(line="Fe X 174.531")
ratio = ratio.sel(logT=peak_logT, method="nearest").drop_attrs(deep=False)
fig, ax = plt.subplots()
ratio.plot(ax=ax, marker="o")
ax.set(ylabel="Fe X 175.263 / 174.531 intensity ratio", title=f"EIS Fe X density diagnostic at logT = {peak_logT:.1f}")

plt.show()
