"""
This module stores package wide constants and variables.
"""

import numpy as np
import xarray as xr

import astropy.units as u

from muse.variables_schema import InstrumentDefaults

__all__ = [
    "DEFAULTS_MUSE",
]


# The headings are those of `InstrumentDefaults`. The order of the entries has no effect.
_DEFAULTS_MUSE_VALUES = {
    # Context Imager (CI) parameters
    "pixel_width_CI": 0.143 * u.arcsec,
    "pixel_height_CI": 0.143 * u.arcsec,
    # Spectrograph (SG) parameters
    "pixel_width_SG": 0.4 * u.arcsec,
    "pixel_height_SG": 0.167 * u.arcsec,
    "detector_x_pixels_SG": 1024,
    "number_of_slits_SG": 35,
    "pixels_between_slits_SG": 26.53 * u.pixel,
    # Nominal: 390 mAngstrom at second order, so twice that in the first-order 284 channel.
    "spectral_slit_separation_SG": xr.DataArray(
        np.array([390.0, 390.0, 780.0]) * u.mAA,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    "waveband_half_width_SG": 35.0 * u.AA,
    "wavelength_step_SG": 4.9 * u.mAA,
    "steps_per_raster_SG": 11,
    # Effective areas
    "main_line_effective_area_SG": xr.DataArray(
        np.array([2.6, 4.3, 0.87]) * u.cm**2,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    # Until the private wavelength-dependent throughput is available, both CI
    # channels use the same provisional scalar area.
    "main_line_effective_area_CI": xr.DataArray(
        np.array([4.55, 4.55]) * u.cm**2,
        coords={"channel": [195, 304]},
        dims="channel",
    ),
    # Other Properties
    "gain_SG": xr.DataArray(
        np.array([10.0, 10.0, 10.0]) * u.electron / u.DN,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    "pair_energy_SG": xr.DataArray(
        np.array([3.65, 3.65, 3.65]) * u.eV / u.electron,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    # CI calibrations are independently scoped from the SG hardware.
    "gain_CI": xr.DataArray(
        np.array([10.0, 10.0]) * u.electron / u.DN,
        coords={"channel": [195, 304]},
        dims="channel",
    ),
    "pair_energy_CI": xr.DataArray(
        np.array([3.65, 3.65]) * u.eV / u.electron,
        coords={"channel": [195, 304]},
        dims="channel",
    ),
    # Synthesis/inversions
    "sum_over_dims_synthesis": ("logT", "doppler_velocity", "slit"),
    "main_lines_by_channel_SG": {
        108: ("Fe XIX 108.355", "Fe XXI 108.117"),
        171: ("Fe IX 171.073",),
        284: ("Fe XV 284.163",),
    },
    "fov_pad_mode": "wrap",
    "fov_extent": "extend",
    "fov_oversampling": 2,
    # Response creation
    "response_abundances": ("sun_coronal_2021_chianti", "sun_photospheric_2021_asplund"),
    "response_temperature": xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT"),
    "response_pressure": xr.DataArray([3e14, 3e15, 3e16] * u.K / u.cm**3, dims="pressure"),
    "response_doppler_velocity": np.arange(-1000, 1010, 10) * u.km / u.s,
    "continuum_response_doppler_velocity": np.arange(-1000, 1001, 200) * u.km / u.s,
    # Response creation, continued
    "minimum_abundance": 1e-21,  # low enough to keep every element of a CHIANTI abundance file
    "wavelength_start_SG": xr.DataArray(
        np.array([107.68034, 170.62314, 283.01608]) * u.AA,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    "channel_spectral_order_SG": xr.DataArray(np.array([2, 2, 1]), coords={"channel": [108, 171, 284]}, dims="channel"),
    # Value provided by Paul B.; spectral plate scale baked in: 0.0815 Angstrom at first order, over
    # the spectral order of each channel. Replace with a measured calibration when one exists.
    "instrumental_fwhm_SG": xr.DataArray(
        np.array([0.0815 / 2, 0.0815 / 2, 0.0815]) * u.AA,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
}

DEFAULTS_MUSE = InstrumentDefaults(**_DEFAULTS_MUSE_VALUES)
"""
Default MUSE instrument parameters.

``print(DEFAULTS_MUSE)`` lists every configured value, one field per line; see
`~muse.variables_schema.InstrumentDefaults` for what each field means.

:meta hide-value:
"""
