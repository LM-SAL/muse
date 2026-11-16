"""
Package-wide constants.
"""

import numpy as np
import xarray as xr

import astropy.units as u

from muse.variables_schema import InstrumentDefaults

__all__ = [
    "DEFAULTS_MUSE",
]


# Headings as in `InstrumentDefaults`; the order has no effect.
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
    # 390 mAngstrom at second order, so twice that in the first-order 284 channel.
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
    # Provisional: one area for both CI channels until the wavelength-dependent throughput exists.
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
    "main_lines_by_channel_SG": {
        108: ("Fe XIX 108.355", "Fe XXI 108.117"),
        171: ("Fe IX 171.073",),
        284: ("Fe XV 284.163",),
    },
    "wavelength_start_SG": xr.DataArray(
        np.array([107.68034, 170.62314, 283.01608]) * u.AA,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    "channel_spectral_order_SG": xr.DataArray(np.array([2, 2, 1]), coords={"channel": [108, 171, 284]}, dims="channel"),
    # From Paul B.: 0.0815 Angstrom at first order over the spectral order; replace when measured.
    "instrumental_fwhm_SG": xr.DataArray(
        np.array([0.0815 / 2, 0.0815 / 2, 0.0815]) * u.AA,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
}

DEFAULTS_MUSE = InstrumentDefaults(**_DEFAULTS_MUSE_VALUES)
"""
Default MUSE instrument parameters.

``print(DEFAULTS_MUSE)`` lists them; `~muse.variables_schema.InstrumentDefaults` says
what each means.

:meta hide-value:
"""
