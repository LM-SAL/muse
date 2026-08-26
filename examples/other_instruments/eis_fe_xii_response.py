"""
====================================
Create EIS Fe XII and Fe X responses
====================================

This tutorial demonstrates how to create wavelength-space responses for
Hinode/EIS from CHIANTI line lists: the Fe XII 195.119 Å window, and the
density-sensitive Fe X 174.531/175.263 Å pair as a density diagnostic.

:func:`muse.instrument.create_spectral_response` is instrument-neutral.

The CHIANTI line lists are downloaded from the skipped preparation example.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import astropy.units as u

from muse.data import fetch_example_data
from muse.instrument import create_spectral_response

##############################################################################
# EIS observes two EUV bands (SW: 166-212, LW: 245-291 Å) and
# Fe XII 195.119 Å is the strongest line in the SW band.
# We select a narrow window that also contains the density-sensitive
# Fe XII 195.179 Å blend.

line_list_file = fetch_example_data("eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.nc")
line_list = xr.load_dataset(line_list_file, engine="h5netcdf")
print(line_list)

##############################################################################
# Now we describe the instrument. The numbers below are representative EIS
# values: a spectral-pixel size of 22.3 mÅ and an instrumental width
# of about 56 mÅ FWHM (see the EIS instrument paper,
# Culhane et al. 2007, and EIS Software Note #7 for the calibrated,
# CCD-position-dependent values).
#
# ``instrumental_width`` is a Gaussian sigma, so we convert from FWHM.
#
# For the effective area we use the pre-flight SW-band calibration curve
# ``EIS_EffArea_B.005`` from SolarSoft (about 0.30 cm² near 195 Å;
# Culhane et al. 2007), a plain two-column wavelength/area table. It carries
# no in-flight sensitivity decay; for time-dependent calibrated values see
# EIS Software Note #2 and the revised calibrations
# (Del Zanna 2013; Warren et al. 2014).

dispersion = 0.0223 * u.AA
instrumental_fwhm = 0.056 * u.AA
instrumental_width = instrumental_fwhm / (2 * np.sqrt(2 * np.log(2)))
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
    instrumental_width=instrumental_width,
    doppler_velocity=doppler_velocity,
    effective_area=effective_area,
)
# ``create_spectral_response`` is per Å; multiplying by the EIS dispersion
# integrates each sample over one spectral detector pixel.
response_unit = u.Unit(response.spectral_response.attrs["units"]) * u.AA
binned_response = (response.spectral_response * dispersion.to_value(u.AA)).assign_attrs(
    {**response.spectral_response.attrs, "units": str(response_unit)}
)
response = response.assign(spectral_response=binned_response)
print(response)

##############################################################################
# The line profile on the EIS pixel grid, summed over temperature, shifts
# with the Doppler-velocity axis.

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
plt.title("EIS Fe XII 195 response, summed over logT")
plt.legend()

##############################################################################
# Summing the bin-integrated response over detector pixels instead gives the
# temperature sensitivity of each retained line.

temperature_sensitivity = (
    response.spectral_response.isel(pressure=0).sel(doppler_velocity=0).sum(dim="wavelength_bin", keep_attrs=True)
)
plt.figure()
for line in temperature_sensitivity.line.values:
    temperature_sensitivity.sel(line=line).plot(label=str(line))
plt.title("EIS Fe XII temperature sensitivity")
plt.legend()

##############################################################################
# EIS also observes the density-sensitive Fe X 174.531/175.263 Å pair. Its
# line list was computed on an electron-density (``logD``) grid instead of a
# fixed pressure, and that axis flows through the response unchanged.
#
# Density curves in the literature are ratios of line intensities, so this
# response is built without the effective area (the EIS throughput differs by
# about 25% between the two lines and would bias the ratio). Lines stay at
# rest wavelength (``doppler_velocity=None``), and the 195 Å dispersion and
# instrumental width are reused as representative values.

density_line_list_file = fetch_example_data("eis_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.nc")
density_line_list = xr.load_dataset(density_line_list_file, engine="h5netcdf")
density_response = create_spectral_response(
    density_line_list,
    np.arange(174.0, 175.6, dispersion.to_value(u.AA)) * u.AA,
    main_lines=["Fe X 174.531", "Fe X 175.263"],
    instrumental_width=instrumental_width,
)

##############################################################################
# Integrating each line over wavelength (the per-Å response summed over the
# grid, times the grid spacing) gives its total intensity. We take the ratio
# at the peak-formation temperature of Fe X, which the line list stores as
# ``logT_peak``.

line_total = (density_response.spectral_response.sum(dim="wavelength_bin") * dispersion.to_value(u.AA)).assign_attrs(
    units=str(u.Unit(density_response.spectral_response.attrs["units"]) * u.AA)
)
is_fe_x_174 = (density_line_list.full_name == "Fe X 174.531").values
peak_logT = density_line_list.logT_peak.isel(trans_index=is_fe_x_174).median().item()
ratio = line_total.sel(line="Fe X 175.263") / line_total.sel(line="Fe X 174.531")
ratio = ratio.sel(logT=peak_logT, method="nearest").drop_attrs(deep=False)
plt.figure()
ratio.plot(marker="o")
plt.ylabel("Fe X 175.263 / 174.531 intensity ratio")
plt.title(f"EIS Fe X density diagnostic at logT = {peak_logT:.1f}")

plt.show()
