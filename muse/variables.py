"""
This module stores package wide constants and variables.
"""

import numpy as np
import xarray as xr

import astropy.units as u
from astropy.units import imperial

from muse.variables_schema import InstrumentDefaults

__all__ = [
    "DEFAULTS_MUSE",
]


MUSE_DEFAULTS_DICT = {
    # CI
    "dx_pixel_CI": 0.143 * u.arcsec,
    "dy_pixel_CI": 0.143 * u.arcsec,
    "full_well_depth_CI": 120000 * u.DN,
    # SG
    "dx_pixel_SG": 0.4 * u.arcsec,
    "dy_pixel_SG": 0.167 * u.arcsec,
    "pixels_SG": u.Quantity(1024, unit=u.pix, dtype=int),
    "number_of_slits_SG": 35,
    "pixels_between_slits_SG": 26.53 * u.pixel,
    # One nominal value for every channel until a measured one exists for each.
    "spectral_slit_separation_SG": xr.DataArray(
        np.array([390.0, 390.0, 390.0]) * u.mAA,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    "waveband_half_width_SG": 35.0 * u.AA,
    "wavelength_step_SG": 4.9 * u.mAA,
    "steps_per_raster_SG": 11,
    # Diffraction
    "mesh_transmission": {284: 0.81},
    "oversample_x_SG": 7,
    "oversample_y_SG": 3,
    "center_diffraction": False,
    "lpi": {284: 70 / imperial.inch},
    "psf_fwhm_x": 0.25 * u.arcsec,  # 0.25 in x and 0.5 in y.
    "psf_fwhm_y": 0.5 * u.arcsec,
    "psf_fwhm": 0.5 * u.arcsec,
    # Other
    "data_compression": 1,
    "ccd_gain_SG": xr.DataArray(
        np.array([10.0, 10.0, 10.0]) * u.electron / u.DN,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    "pair_creation_energy_SG": xr.DataArray(
        np.array([3.65, 3.65, 3.65]) * u.eV / u.electron,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    # CI calibrations are independently scoped from the SG hardware.
    "ccd_gain_CI": xr.DataArray(
        np.array([10.0, 10.0]) * u.electron / u.DN,
        coords={"channel": [195, 304]},
        dims="channel",
    ),
    "pair_creation_energy_CI": xr.DataArray(
        np.array([3.65, 3.65]) * u.eV / u.electron,
        coords={"channel": [195, 304]},
        dims="channel",
    ),
    # Synthesis/inversions
    "sum_over_dims_synthesis": ("logT", "doppler_velocity", "slit"),
    "main_lines_SG": [["Fe XIX 108.355", "Fe XXI 108.117"], ["Fe IX 171.073"], ["Fe XV 284.163"]],
    "main_lines_SG_wavelength": {
        "Fe XIX 108.355": 108.355 * u.AA,
        "Fe XXI 108.117": 108.117 * u.AA,
        "Fe IX 171.073": 171.073 * u.AA,
        "Fe XV 284.163": 284.163 * u.AA,
    },
    "bands_SG": np.asanyarray([108, 108, 171, 284]) * u.AA,
    "fov_mode": "wrap",
    "fov_x_extent": "tile",
    "fov_sub_interpolation": 2,
    "target_logT": {
        "QS": np.arange(4.8, 6.7, 0.1),
        "AR": np.arange(4.8, 7.2, 0.1),
        "FL": np.arange(4.8, 7.6, 0.1),
    },
    "target_doppler_velocity": {
        "QS": np.arange(-200, 210, 20) * u.km / u.s,
        "AR": np.arange(-300, 310, 20) * u.km / u.s,
        "FL": np.arange(-500, 510, 20) * u.km / u.s,
    },
    # Response
    "response_abundances": ("sun_coronal_2021_chianti", "sun_photospheric_2021_asplund"),
    "response_temperature": xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT"),
    "response_pressure": xr.DataArray([3e14, 3e15, 3e16] * u.K / u.cm**3, dims="pressure"),
    "response_doppler_velocity": np.arange(-1000, 1010, 10) * u.km / u.s,
    "continuum_response_doppler_velocity": np.arange(-1000, 1001, 200) * u.km / u.s,
    "minimum_abundance": 1e-21,  # low enough to keep every element of a CHIANTI abundance file
    "sum_lines": False,
    "initial_wavelength_SG": xr.DataArray(
        np.array([107.68034, 170.62314, 283.01608]) * u.AA,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    "channel_spectral_order_SG": xr.DataArray(np.array([2, 2, 1]), coords={"channel": [108, 171, 284]}, dims="channel"),
    # Value provided by Paul B.; spectral plate scale baked in. Replace with a measured
    # calibration when one exists.
    "instrumental_fwhm_SG": 0.0815 * u.AA,
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
    # Exposures
    "exposure_times_SG": {
        "QS": (2, 6, 18, 60.0) * u.s,
        "plage": (1, 2, 8, 32) * u.s,
        "AR": (1, 2, 8, 32) * u.s,
        "M-flare": (0.1, 0.6, 1.8, 6.0) * u.s,
        "X-flare": (0.1, 0.6, 1.8, 6.0) * u.s,
    },
    "exposure_times_CI": {
        "QS": (1.5, 3.0, 6.0, 12) * u.s,
        "plage": (0.6, 1.2, 2.5, 5) * u.s,
        "AR": (0.6, 1.2, 2.5, 5) * u.s,
        "M-flare": (0.06, 0.15, 0.3, 1.2) * u.s,
        "X-flare": (0.06, 0.15, 0.3, 1.2) * u.s,
    },
}

DEFAULTS_MUSE = InstrumentDefaults(**MUSE_DEFAULTS_DICT)
"""
Default MUSE instrument parameters.

``print(DEFAULTS_MUSE)`` lists every configured value, one field per line; see
`~muse.variables_schema.InstrumentDefaults` for what each field means.

:meta hide-value:
"""
